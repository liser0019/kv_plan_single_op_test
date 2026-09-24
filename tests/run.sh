#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SparseKvPlan 单算子测试入口:构建(可选)→ 符号探针 → pytest。
#
# 用法:
#   bash tests/run.sh                  # 构建并安装 wheel,然后跑全部测试
#   SKIP_BUILD=1 bash tests/run.sh     # 跳过构建(已安装)只跑测试
#   CPU_ONLY=1 bash tests/run.sh       # 只跑 CPU 侧(golden 自检 + MTP 语义),无需 NPU
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ "${CPU_ONLY}" != "1" ] && [ "${SKIP_BUILD}" != "1" ]; then
    bash build.sh --soc=ascend950 --install
fi

if [ "${CPU_ONLY}" != "1" ]; then
    # 符号探针:失败说明导入的不是本次构建,先恢复再测,失败结果不作数
    python3 - <<'EOF'
import sys
try:
    import sparse_kv_plan_op
    import torch
    assert hasattr(torch.ops.sparse_kv_plan_op, "sparse_kv_plan"), \
        "符号探针失败:torch.ops.sparse_kv_plan_op.sparse_kv_plan 不存在,请先 bash build.sh --soc=ascend950 --install"
    print("[probe] torch.ops.sparse_kv_plan_op.sparse_kv_plan OK")
except ImportError as e:
    print(f"[probe] 包未安装({e});仅运行 CPU 侧测试")
    sys.exit(0)
EOF
fi

if [ "${CPU_ONLY}" == "1" ]; then
    python3 -m pytest tests/ -v -k "not npu"
else
    python3 -m pytest tests/ -v
fi
