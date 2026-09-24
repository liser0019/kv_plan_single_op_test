# SparseKvPlan 单算子工程(昇腾 950 / dav-3510)

从 vllm-ascend 仓 `csrc/attention/sparse_kv_plan` 抽取的 **SparseKvPlan** 算子
direct launch(直调)单算子工程,含单算子测试与无 NPU 仿真路径。

- 目标芯片:**昇腾 950**(NpuArch `DAV_3510`,`__NPU_ARCH__=3510`,bisheng
  `--npu-arch=dav-3510`,算子仓简写 arch35)
- 算子形态:SIMT VF(2048 线程/AIV block),AIV-only,64 block 上限,
  每行(一条 request 的 Plan)由一个 block 独立完成
- 算子职责:在 NPU 上维护稀疏 KV 的 LRU 驻留状态,输出 resident slot 命中
  (`current_slots`)与压缩后的 miss 描述(`miss_count/miss_tokens/miss_slots`)
- 伴随算子 SparseKvTransfer(把 miss 的 BF16 K/V 从 MemFabric Host DVA 拷回
  驻留 buffer)依赖主机侧 DVA 注册,**不在本单算子工程范围内**

## 目录结构

```
sparse_kv_plan_op/
├── build.sh / setup.py / CMakeLists.txt / cmake/   # direct launch 工程骨架
├── sparse_kv_plan_op/                              # Python 包(torch.ops.sparse_kv_plan_op)
├── csrc/
│   ├── extension.cpp                               # PyInit__C(TORCH_LIBRARY 静态注册)
│   └── ops/sparse_kv_plan/
│       ├── op_kernel/
│       │   ├── sparse_kv_plan_config.h             # 原样拷自 vllm-ascend
│       │   ├── sparse_kv_plan_launch.h             # 共享 POD tiling 结构 + host tiling 纯函数
│       │   └── sparse_kv_plan_kernel.cpp           # 内核(只搬不改)+ 直调入口
│       └── op_plugin/
│           └── sparse_kv_plan_plugin.cpp           # torch.library 注册 + Meta + NPU impl
├── tests/                                          # 单算子测试(三层,见下)
└── sim/                                            # npusim 无 NPU 精度仿真路径
```

## 与 vllm-ascend 原工程的差异(只搬不改清单)

kernel 计算逻辑(`ProcessRuntimeRow`、SIMT VF、scan、hash、LRU 重建)与原
`sparse_kv_plan_apt.cpp` **逐行一致**,仅入口层适配 direct launch:

| 原工程(GE/op-build 注册) | 本工程(direct launch) |
|---|---|
| `REGISTER_TILING_DEFAULT` + `GET_TILING_DATA_WITH_STRUCT` | tiling 结构体由 host `MakeSparseKvPlanTiling()` 填充 → H2D → kernel 显式加载 |
| tiling 在 `op_host/sparse_kv_plan_tiling.cpp` 经 GE 框架执行 | 同源逻辑放在 `sparse_kv_plan_launch.h` 纯函数,plugin 直接调用 |
| `workspace` 形参(GE 分配,从未读取) | 去除;`compactWorkspace` 语义不变 |
| `aclnnSparseKvPlan` + `EXEC_NPU_CMD` | `torch.ops.sparse_kv_plan_op.sparse_kv_plan`(schema 与 aclnn 原型逐参数对应,8 个被写张量带 `(a!)..(h!)` 标注) |
| 编译选项 `--cce-auto-sync=off` 等 | 由顶层 CMakeLists 的 bisheng 命令传入(影响 SIMT VF 一致性,勿删) |

## 单算子测试设计(tests/)

三层结构,前两层**无 NPU 可跑**(当前交付机器已验证:37 项全过),
第三层在 NPU 机器上跑:

| 层 | 文件 | 内容 |
|---|---|---|
| 锚点 | `test_sparse_kv_plan.py::test_golden_anchor_three_steps` | 三步手工推导的精确期望值(reset+重复token → 多slot驻留命中 → stable前移+投机失效) |
| golden 自检 | `test_sparse_kv_plan.py` | 16 个分级用例(L0/L1/L2)结构不变量 + 确定性;`golden.py` 为独立 CPU 参考实现 |
| MTP 投机 | `test_mtp_speculative.py` | 多步有状态 MTP 驱动器(详见下节) |
| NPU 比对 | `test_*_npu*` | 符号探针 + 全用例/全输出 int32 **全等**比对(确定性算法,不允许偏差) |

### 用例分级

- **L0**:l0_basic(T=2,C=4)、l0_single_row
- **L1**:l1_small(T=32)、l1_typical(T=64,C=128,UB 路径)、l1_warm_two_steps(warm 状态续用)
- **L2**:activeRows=0/越界截断、重复 token、全非法 TopK、C=1、
  **UB/GM workspace 精确边界**(T=2048,C=7168 恰好 112 KiB 走 UB;
  C=7169 越界走 GM)、越界 LRU 项、stablePrefix 全失效、请求切换 reset、
  miss > evictable(未分配保持 -1)

### 运行

```bash
# 无 NPU(仅 golden/MTP CPU 层)
CPU_ONLY=1 bash tests/run.sh

# ---- NPU 机器完整流程 ----
# 0) 环境:CANN 已 source;python 环境装好 torch + 匹配版本 torch_npu + pytest + numpy
source /usr/local/Ascend/ascend-toolkit/set_env.sh        # 按实际安装路径
python3 -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.get_device_name(0))"
# 1) 构建并安装(bisheng 编译内核 + g++ 编译插件 → _C.so → wheel)
bash build.sh --soc=ascend950 --install
#    若报"未找到 bisheng 编译器":export BISHENG_CXX=<bisheng_compiler 路径> 后重试
# 2) 符号探针(确认导入的是本次构建)
python3 -c "import torch, sparse_kv_plan_op; print(torch.ops.sparse_kv_plan_op.sparse_kv_plan)"
# 3) 全量测试(CPU 层 + NPU 全等比对,自动跳过不适用的)
python3 -m pytest tests/ -v
# 4) 只跑 MTP 投机推理测试
python3 -m pytest tests/test_mtp_speculative.py -v

# ---- 无 NPU 卡但有 CANN 工具链:npusim 精度仿真(Ascend950)----
bash sim/run_sim.sh                  # 轻量用例
bash sim/run_sim.sh --include-heavy  # 含 GM workspace 大用例
```

> 构建后 `_C.so` 同时解回工程源码包目录:从工程根目录直接 `import
> sparse_kv_plan_op` 用的是本地最新构建,不会被 site-packages 里的旧包
> 遮蔽(也不会反向遮蔽)。若怀疑跑的不是最新代码,重跑第 1 步即可恢复。

## 开 MTP 投机推理的单算子测试 —— 可以,而且不需要模型

结论:**可以**。SparseKvPlan 的 MTP 语义完全由输入表达,单算子层面即可
完整模拟投机解码循环:

1. **投机后缀失效**:kernel 阶段 4 对 resident slot 中
   `token >= stable_prefix_lens[row]` 的项先失效再分类——这就是投机 token
   回滚语义。真实集成里 `stable_prefix_lens = key_len - query_len`
   (`sfa_kv_offload.py`),MTP 一步 query 含 draft token,故 stable 恰为
   已提交前缀。
2. **投机可见性**:`visible_seq_lens[row] = 已提交 + draft`,TopK 允许选中
   投机区 token(它们可作为 miss 正常分配驻留)。
3. **多 token 行展开**:MTP 一步每请求 `(1 + draft_tokens)` 行,
   `token_to_req` 映射回请求,与 fused_overlap_mtp 的 flatten 模式一致。
4. **验证/接受/回滚**:驱动器随机接受 `a ∈ [0, draft]` 个 draft token,
   `context += 1 + a`,被拒 token 回滚;下一步 stable/visible 相应推进。
5. **行重排**:真实 serving 的行压缩会让行换请求,触发 kernel 的整行
   reset 分支(`lastReqIds` 失配),测试每 3 步重排一次。

`tests/test_mtp_speculative.py` 实现了以上全部,并逐步断言 MTP 不变量:
**投机 token(≥ stablePrefix)的驻留只能来自本步 miss 分配,绝不能命中
历史 resident**;被重排的行不允许出现任何历史 resident 命中。CPU 侧先以
golden 验证语义,NPU 侧逐步与 golden 全等比对(状态跨步传递)。

若要进一步逼近端到端:接上 vLLM 的 MTP proposer(`llm_base_proposer.py`)
跑真实 workload 时,本工程的状态张量布局(`SimtLruState`)与 vllm-ascend
`_SparseKVSimtLayerState` 行级字段一致,可直接桥接;Transfer 算子所需的
MemFabric Host DVA 注册是唯一缺口。

## 未在真机验证的部分(诚实声明)

本机无 NPU、无 CANN 工具链,以下内容为按 skills 模板与 CANN 文档编写的
脚手架,真机首跑可能需要微调(均已提供环境变量/参数覆盖):

- `cmake/ascend.cmake` 的 bisheng 发现路径(可用 `-DBISHENG_CXX=` 覆盖)
- `sim/build_sim.sh` 的 bbit 编译选项(不同 CANN 版本叫法可能不同)
- `sim/main.cpp` 假定 npusim 接管标准 aclrt API
- 顶层 CMakeLists 的 bisheng custom command 与 g++ 链接流程
- `KERNEL_TASK_TYPE_DEFAULT` 宏兜底(若工具链已定义则用原生,否则置空,
  直调下任务类型由启动配置决定)

golden/用例生成/MTP 驱动器/锚点用例是纯 CPU 代码,**已在本机全量跑通**。

## 许可

kernel 计算逻辑源自 MemFabric_Hybrid(Mulan PSL v2),保留原始版权声明;
工程骨架与测试代码 Apache-2.0,与 vllm-ascend 仓一致。
