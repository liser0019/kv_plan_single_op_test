# Echo：跳过零淘汰排序的对照实验

## 1. 实验要回答的问题

原版 Echo 在每行算出 `freeSize` 后，无条件调用 `OpArgtopkRow`。生产容量为 4096 时，排序输入补齐到 8192 个元素，执行四趟 radix 排序，即使本行无需淘汰历史槽也执行这些步骤。

本分支 `feat/echo-skip-zero-sort` 同时构建原版和实验版。实验版仅增加一个条件：**该行 `freeSize == 0`，且没有要求输出该行的排序调试数据时，跳过整次 `OpArgtopkRow`**。

| 版本 | Python / torch 算子名 | 设备 kernel 名 | 行排序行为 |
|---|---|---|---|
| 原版 | `echo_lru` | `echo_lru_kernel` | 每行执行 |
| 实验版 | `echo_lru_skip_sort` | `echo_lru_skip_sort_kernel` | `freeSize == 0` 的普通行跳过 |

`freeSize` 表示本轮必须淘汰的历史槽数，不是缓存容量，也不是空闲槽数。它由 miss 数和当前可用空闲槽数计算；reset 或投机后缀失效后的空闲槽也计入其中。

两版共用相同的张量布局、tiling、workspace、128 KiB UB、线程数和编译选项。reset、投机后缀失效、hit 原子计数、优先级更新、release/push/pop、映射更新及最终输出计算保持相同。原版使用 `Process<false>`，实验版使用 `Process<true>`，编译期选择这一个分支。

### 为什么可以跳过

条件判断位于 `OpFreeSize` 与随后的 `SyncGrid()` 之后。行内所有线程读取同一个已发布且不再修改的 `freeSize`，一起跳过或一起进入排序；不会只有部分线程跳过排序内部的 barrier。

跳过时不清空该行的 `freeIdx` 临时区。后续 `OpFreeRelease` 和 `OpFreePush` 都只读取 `j < freeSize` 的项，因此零淘汰行不消费其中的旧值。需要淘汰的下一行仍重新初始化并执行自己的完整排序。调试行保留排序，以维持原调试 dump 行为。

这些是源码层面的依据；线程同步和 UB 行为仍需通过 NPU 回归验证。

## 2. 用同一份输入测两版

默认生产尺寸为 `rows=32, topk=2048, capacity=4096, max_token=262144`。四个共同场景如下，每次调用前恢复相同的初始状态：

| 场景 | 每行 miss | 初始空闲槽 | 本轮 `freeSize` | 实验版是否排序 |
|---|---:|---:|---:|---|
| `cold_reset` | 2048 | reset 后 4096 | 0 | 跳过 |
| `warm_75pct` | 512 | 2560 | 0 | 跳过 |
| `warm_all_hit` | 0 | 2048 | 0 | 跳过 |
| `full_cache_75pct` | 512 | 0 | 512 | 执行 |

这四个性能场景均关闭投机后缀失效。MTP、reset、多轮状态续用以及混合行由正确性测试另外覆盖。

每版在计时前与同一个独立 CPU golden 比较全部六个持久状态字段和四个输出，并检查双向映射、空闲栈、miss 数及历史淘汰 token 集合。不通过则停止采集。

新增的混合行 NPU 回归将多个 `freeSize=0` 与 `freeSize>0` 的行安排到同一核内，包含 reset 和投机失效，并预先污染 workspace、输出区，检查跳过后的临时数据是否被误读。连续调用回归直接续用上一轮 NPU 状态。

## 3. 在服务器获取并构建

从服务器原仓库创建独立 worktree，使已验证版本和实验版本各自保留构建产物：

```bash
git fetch origin feat/echo-skip-zero-sort
git worktree add --detach ../kv_plan_echo_skip_sort origin/feat/echo-skip-zero-sort
cd ../kv_plan_echo_skip_sort
git rev-parse HEAD
```

进入匹配的 torch / torch_npu 环境，加载服务器实际 CANN 安装的 `set_env.sh`，然后从这个新工作区构建：

```bash
bash build.sh --soc=ascend950
```

该命令把扩展放回源码包目录；下面的测试均从这个新工作区根目录运行。确认两个符号都存在：

```bash
python3 - <<'PY'
import torch
import torch_npu
import sparse_kv_plan_op
print(torch.ops.sparse_kv_plan_op.echo_lru)
print(torch.ops.sparse_kv_plan_op.echo_lru_skip_sort)
PY
```

## 4. 先跑正确性

以下使用本轮测量的逻辑卡 1。如果服务器设备可见性配置不同，将 `1` 改成实际逻辑设备号；两个版本始终使用同一卡。

```bash
unset ASCEND_LAUNCH_BLOCKING
ulimit -c 0
python3 -X faulthandler - <<'PY'
import torch
import torch_npu
import pytest

assert torch.npu.is_available(), "当前环境没有可用 NPU"
torch.npu.set_device(1)
print("测试设备:", torch.npu.current_device(), torch.npu.get_device_name(1), flush=True)
raise SystemExit(pytest.main(["tests/test_echo_lru.py", "-v", "-x", "-s"]))
PY
echo "Echo 专项退出码: $?"
```

专项通过后，将测试路径改为 `"tests/"` 跑全套。确认 NPU 用例实际通过，并保留总数和退出码，再开始性能采集。

## 5. 公共接口 Event 对比

原 `compare_three.py` 默认仍测旧 AIV、新 SIMT、原版 Echo。增加 `--include-echo-skip-sort` 后加入第四路：

```bash
# 短跑
python3 -X faulthandler bench/compare_three.py \
  --device 1 --include-echo-skip-sort --warmup 2 --repeat 3 \
  --json results/echo_sort_event_smoke.json

# 正式测量
python3 -X faulthandler bench/compare_three.py \
  --device 1 --include-echo-skip-sort --warmup 10 --repeat 50 \
  --json results/echo_sort_event.json
```

四路计时轮换顺序，状态恢复及同步在 Event 区间外。JSON 保留全部原始样本，并增加：

- `echo_sort_skip_event_speedup`：原版 Echo P50 / 实验版 Echo P50。
- `echo_sort_skip_event_saved_us`：原版 Echo P50 − 实验版 Echo P50。

Event 区间包含公共接口中的 tiling 上传和可能的下发等待。若要定位 kernel 内部的差异，使用下面的 profiler 入口。

## 6. 同尺寸 kernel profiler 对比

```bash
# 先复现之前的 75% 命中场景
python3 -X faulthandler bench/profile_echo_sort.py \
  --device 1 --scenario warm_75pct

# 交换采集顺序，再独立测一次
python3 -X faulthandler bench/profile_echo_sort.py \
  --device 1 --scenario warm_75pct --reverse-order

# 需要时采集全部四个场景
python3 -X faulthandler bench/profile_echo_sort.py \
  --device 1 --scenario all
```

脚本使用 `torch_npu.profiler`，固定 `wait=0, warmup=5, active=5, repeat=1`，每步恢复状态、执行一次算子并同步，最后调用 `prof.step()`。两版分别导出 trace；每次运行在 `results/echo_sort_profiler/` 下创建新目录，保存：

- `conditions.md`：设备、torch / torch_npu 版本、提交、尺寸、seed、schedule 和采集顺序。
- 各场景各版本的原始 profiler 数据及 `kernel_details.csv`。
- `comparison.md`：两版目标 kernel 的样本数、P50、P95、最小值、最大值、比值、节省量及原始样本。

统计只匹配上述两个独立 kernel 名。恢复产生的 `TensorMove` 和 tiling 上传会出现在 trace 中，但不加到目标 kernel duration。每条路径必须恰好匹配 5 个有效样本，否则脚本报错并保留 trace，避免错误汇总。

如果服务器导出的名称经过改写，先查看原始 CSV，确认各版实际名称后再传入 `--original-kernel-name` / `--skip-kernel-name`。筛选项是唯一名称子串；不要用同时匹配多种任务的通用名称。

Profiler 的 schedule、同步导出选项来自 [Ascend PyTorch profiler API](https://github.com/Ascend/pytorch/blob/master/torch_npu/profiler/profiler.py)，采集配置来自 [experimental_config](https://github.com/Ascend/pytorch/blob/master/torch_npu/profiler/experimental_config.py)。CSV 汇总也支持 [Ascend 导出格式](https://github.com/Ascend/msprof-analyze/blob/master/docs/en/advanced_features/recipe_output_format_introduct.md#kernel_detailscsv)中的 `op_name` / `task_duration` 微秒字段；不接受毫秒列直接作为微秒。

## 7. 怎样解释结果

必须从本分支重新构建并在同一环境同时采集两版。之前得到的 26.68 ms 可作为现象记录；实验差值使用这次重新采集的原版和实验版计算。

对于 `warm_75pct`：

```text
路径净节省 = 原版 kernel P50 − 实验版 kernel P50
加速比     = 原版 kernel P50 / 实验版 kernel P50
```

净节省反映跳过整条 `OpArgtopkRow` 路径的收益，包含排序、相关访存、内部 barrier，以及编译后控制流的影响。这个实验没有进一步拆分这些成本。两版保留相同的 hit 原子计数和其余阶段。

`full_cache_75pct` 两版都需要排序，可作为对照。若该场景也出现很大的差异，应先复查编译、运行条件和采样，再归因。5 个 active 样本用于初步定位，建议交换顺序重复独立采集；更充分的公共接口延迟统计使用 50 次 Event 测量。

## 8. 本机验证范围

本次本地全量 pytest 为 **94 passed、47 skipped**。另通过 Python 语法、CLI 入口及 diff 空白检查。Host 插件使用本机 torch 头文件和临时 ACL / torch_npu 声明桩通过 g++ 语法检查。

本机没有 CANN、bisheng、torch_npu 或 NPU，**没有执行实际 Ascend 编译、NPU 正确性测试或性能采集**。以上本地检查不能证明设备编译、线程同步或性能收益；服务器结果请以本分支重新构建后的实测为准。
