# Echo 算子接入与三方性能对比

## 1. 代码与入口

新增 `csrc/ops/echo_lru/`，复制同事的 tiling 和内核，提供纯 SIMT direct launch、PyTorch 注册和 Python 状态封装。`peer_code/` 的原始文件保留，原 Host `main()` 不参与扩展构建。

| 实现 | Python 入口 | 状态 / 策略 |
|---|---|---|
| 旧 AIV | `old_lru_compact` / `OldLruState` | 0 基槽，精确顺序 LRU |
| 当前混合 SIMT | `sparse_kv_plan` / `SimtLruState` | 0 基槽，精确顺序 LRU，并验证 source 的可见长度和物理 block |
| 同事纯 SIMT Echo | `echo_lru` / `EchoLruState` | 1 基槽，时间戳优先级；同时间戳按槽号淘汰；空闲栈分配 |

Echo 以 1024 线程/核启动，四参数调用 `<<<blocks,1024,0,stream>>>`，采用编译期单数组 UB 128 KiB；当前 Plan 使用的混合编程三参数启动不适用于此核。CMake 仅给 Echo 编译单元添加 `--enable-simt`。现有两核的计算流程没有修改。

## 2. 接入层修正

- **历史 LRU 转换**：原 `lru_slots` 从最旧到最新；使用 `pri=rank+1` 转换，最旧槽获得最小时间戳。`fifo=capacity+1` 保证本轮新时间戳大于所有历史值。此前 `capacity-rank` 会反转历史顺序，因此 `2f0a2db` 的满缓存对比应在修正后重测。
- **分行**：先算 chunk=ceil(rows/min(rows,cores))，再算实际 blocks=ceil(rows/chunk)。尾块始终有 1..chunk 行。原 Host 在 rows=32、cores=24 时尾行数会下溢。
- **调试区容量**：B=4096 时，next_pow2(B+1)=8192。接入副本的调试 key 段上限从 4096 改成 8192，即使当前关闭 dump，也保证启用调试后不越界；Host 分配与内核偏移同源。
- **依赖**：移除没有使用的 cooperative_groups 头，避免额外要求 CANN 9.2 的该头文件。
- **异步生命周期**：`RunOpApi` 回调按值捕获 tensor、tiling 和 stream。所有 tensor 连续性、类型、形状、设备与 workspace 容量在 Host 校验。
- **索引约束**：核内多处使用 32 位偏移，Host 和 Python 限制形状/布局乘积，避免索引溢出。

原 kernel 的 reset、后缀失效、hit/miss、优先级更新、四趟稳定 radix、release/push/pop、映射更新和 final slots 保持原样。原 Host 的文件 I/O、ACL 分配及同步返回值未检查的问题由 PyTorch 接入替代；它仍是未修改的参考文件，不作为本工程测试入口。

## 3. Echo 输入与输出契约

- 每行 `pos` 的有效前 min(topk,capacity) 个位置，token 必须在 [0,max_token) 内且行内不重复。token 0 对应保留槽 0，普通 token 使用 1..capacity 槽。后续位置 padding 输出 0。
- `spl` 在 [0,max_token]；reset_mask 为 uint8 [rows]。Echo 没有 request ID 输入，由调用方显式传 reset_mask。
- `htd` 是 [rows,max_token] token→slot 稠密表；未驻留为 INT32_MAX，htd[:,0]=0。`dth`/`pri` 是 [rows,capacity+1]，槽 0 保留。
- `free_slots` 是 [rows,capacity+min(topk,capacity)]，有效前 avail 项是空闲槽；fifo 是 int32 时间戳。
- 输出 current_slots、miss_host_pos、miss_alloc_flat 是 int32 [rows,topk]；miss_mask 是 uint8。miss_alloc_flat 按原核保留 `row*capacity+one_based_slot` 约定，**不是 0 基数组偏移**。miss_host_pos/flat 的 hit 和 padding 位为 0，要用 miss_mask 区分。
- `spec_enabled=True` 默认在设备上失效后缀，调用方无需为此读回 NPU 的 spl。只有全部 spl==max_token 时才能显式传 False。
- 状态必须由 create、合法初始转换或上一轮 kernel 产生。行内重复、非法 token、损坏状态和 fifo 溢出不在此核支持范围，不能直接套用当前 Plan 的非法输入/去重测试。接入不会每次同步读回 token 做去重，否则会改变性能口径。

当前源码仍使用 atomic 累计 hit 数，并依赖原块内 barrier/cache 操作完成后续读取；源码注释提到作者历史上排查过 atomic 可见性问题。这条实际路径必须通过 NPU 全状态用例验证，CPU golden 通过不能证明硬件同步正确。

## 4. 为什么不能直接比较所有槽号

当前两核使用精确顺序 LRU；Echo 在同一轮为所有 hit 赋同一个时间戳，为所有 miss 赋下一个相同时间戳，之后平局按槽号处理。空闲栈的分配顺序也不同，因此相同 token 得到不同槽位是可能的；多轮后两种淘汰策略还可能产生不同命中率。

本工程对每个实现检查自己的 golden，全量验证 Echo 的六个持久状态字段和四个输出，并验证双向映射、空闲栈唯一性、miss 地址和最终查询 token。共同单步场景额外核对各实现的 miss 数和历史淘汰 token 集合，benchmark 在计时前也执行这个检查。另用 20 组随机乱序 LRU 满缓存输入，按输入历史独立确定最老的未命中 token，验证转换顺序和三版 golden 的淘汰集合；NPU 满缓存测试也使用打乱的 LRU。连续测试专门续用上一轮 NPU 写出的 Echo 状态，golden 只用于比对，不回灌 NPU。

不能把当前三方结果称为完全等价实现的性能差异；新 Plan 的合法性检查和 Echo 的稠密 htd 成本也属于各自原算法。

## 5. 构建与正确性

建议在服务器的原仓库中创建独立检出，保留已验证工程的工作区与构建产物：

```bash
git fetch origin main
git worktree add --detach ../kv_plan_echo_verify origin/main
cd ../kv_plan_echo_verify
git rev-parse HEAD
```

将 HEAD 与本次修复提交核对后，在匹配的 torch/torch_npu 环境中加载实际安装的 CANN 9.1.0 `set_env.sh`，执行：

```bash
bash build.sh --soc=ascend950
```

显式选择 NPU 4 跑 Echo 专项，避免普通 pytest 使用默认卡 0：

```bash
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0
python3 -X faulthandler - <<'PYTEST'
import torch
import torch_npu
import pytest

assert torch.npu.is_available(), "当前环境没有可用 NPU"
torch.npu.set_device(4)
print("测试设备:", torch.npu.current_device(), flush=True)
raise SystemExit(pytest.main(["tests/test_echo_lru.py", "-v", "-x", "-s"]))
PYTEST
echo "专项测试退出码: $?"
```

专项通过后，将该入口中的测试路径替换为 `"tests/"`，仍选择 device 4，运行全套。确认 NPU 用例实际 PASSED、退出码为 0，再进行下一节性能短跑。

无 NPU 时可运行：

```bash
CPU_ONLY=1 bash tests/run.sh
```

## 6. 生产尺寸性能对比

先短跑，成功后再正式测量；两次均运行三方脚本 `compare_three.py`：

```bash
python3 -X faulthandler bench/compare_three.py \
  --device 4 --warmup 2 --repeat 3 --json results/compare_three_smoke.json
python3 -X faulthandler bench/compare_three.py \
  --device 4 --rows 32 --topk 2048 --capacity 4096 --max-token 262144 \
  --warmup 10 --repeat 50 --json results/compare_three.json
```

使用同一份正数、不重复、source 全合法的查询，四个场景：

| 场景 | 初始状态 | 测量目标 |
|---|---|---|
| cold_reset | 新请求，空缓存，全部 miss | 包含各核 reset 成本；Echo reset 要清理稠密 htd |
| warm_75pct | 初始驻留 query 的前 75%，其余槽空闲 | hit 查询与空闲分配，不需要淘汰 |
| warm_all_hit | query 全部驻留，其余槽空闲 | 全命中路径 |
| full_cache_75pct | 缓存已满，query 的前 75% 命中，其余 resident 不在 query 中 | 真正的淘汰与重新分配 |

每次先恢复同一初始状态并同步，恢复不计时。golden、CPU/NPU 状态转换、结果读回、分配初始状态均不计时。热身后轮换三核顺序，保存每次原始样本、p50/p95/min、状态 tensor 总字节数以及 `peer_vs_new_event_ratio`（Echo p50 / 当前 Plan p50；大于 1 表示 Echo 更慢）。

**NPU event 是公开算子 API 所覆盖的流区间**，包括该调用的 tiling H2D 和可能的 Host 入队空隙，不等于 profiler 的纯 kernel duration。Host wall 则包括调用、事件和等待完成。三个实现的 launch 包装成本可能不同。微秒级纯内核比较需进一步使用服务器上的 profiler，不能将 event 值直接当作此前的纯 kernel 微秒结果。

四个性能场景均不开启 MTP 回滚；MTP 的后缀失效和状态续用由连续正确性测试另外验证。

JSON 记录设备、torch/torch_npu 版本、shape、seed 和重复次数；对比时应使用同一设备、软件栈和构建配置。

生产尺寸下 Echo htd 单项为 32*262144*4=32 MiB；radix UB 为每核 128 KiB，GM workspace 保留原布局的未使用 ping-pong 占位段。此次不删除占位段或优化排序，保证测量的是同事现有计算流程。

## 7. 本机验证范围

本机没有 torch_npu、CANN/bisheng 或 NPU，因此只验证 NumPy golden、共同输入、语义锚点、布局计算和 Python 测试；本机没有实际 NPU 延迟结果。应在服务器执行上述构建和测试后再讨论性能优劣。

初次接入提交 `2f0a2db` 的全量 pytest：**67 passed、36 skipped**。另用本机 torch 的 CPU tensor 验证两种尺寸下 `EchoLruState.create` 的六个初始状态与 NumPy 参考相同；使用临时 ACL/torch_npu 声明桩完成 Host 插件的 g++ 语法检查，并核对 C++/Python 生产尺寸 workspace 均为 2695328 个 int32 元素。这些检查均不代表 CANN 编译或 NPU 执行通过。

历史顺序修复后的本机全量 pytest：**87 passed、36 skipped**；新增 20 组乱序满缓存回归测试。另在内存中复现旧转换，20 组输入全部被新增淘汰集合检查拒绝。Python 编译检查及 `git diff --check` 通过；本次未执行 CANN 编译、NPU 测试或性能计时。
