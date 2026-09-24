#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# 编译 npusim 仿真可执行文件(无 NPU 卡,仅需 CANN 工具链 + bisheng)。
#
# ⚠ 未在真机验证的脚手架:不同 CANN/npusim 版本的 bbit 编译选项可能不同,
#   如报 "unrecognized command line option" 请按本机 npusim 文档调整
#   BISHENG_FLAGS(常见变体: --bbit / -bbit / --cce-o-sim,参见 skill
#   ops-simulator troubleshooting Q4)。

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
KERNEL_DIR="${PROJECT_ROOT}/csrc/ops/sparse_kv_plan/op_kernel"

CANN="${ASCEND_CANN_PACKAGE_HOME:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}"
BISHENG_CXX="${BISHENG_CXX:-}"
if [ -z "${BISHENG_CXX}" ]; then
    for cand in "${CANN}/bin/bisheng_compiler" "${CANN}/compiler/bin/bisheng_compiler" \
                "${CANN}/compiler/bisheng/bin/bisheng_compiler" "${CANN}/bin/ccec_compiler"; do
        if [ -x "${cand}" ]; then BISHENG_CXX="${cand}"; break; fi
    done
fi
if [ -z "${BISHENG_CXX}" ]; then
    echo "[ERROR] 未找到 bisheng 编译器,请 export BISHENG_CXX=<path>"
    exit 1
fi

# bbit 仿真模式:kernel 与 host main 同一可执行;SIMT VF 专用选项与 NPU 构建一致
BISHENG_FLAGS="${BISHENG_FLAGS:---bbit --npu-arch=dav-3510}"
COMMON_FLAGS="-O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0"
SIMT_FLAGS="--cce-auto-sync=off -Wno-deprecated-declarations \
-mllvm -cce-vf-remove-membar=false -mllvm -cce-aicore-hoist-movemask=false \
-mllvm -cce-aicore-dcci-before-kernel-end=false"

OUT="${SCRIPT_DIR}/sparse_kv_plan_sim"
echo "[build] BISHENG_CXX=${BISHENG_CXX}"
"${BISHENG_CXX}" ${BISHENG_FLAGS} ${COMMON_FLAGS} ${SIMT_FLAGS} \
    -I"${KERNEL_DIR}" -I"${CANN}/include" \
    "${KERNEL_DIR}/sparse_kv_plan_kernel.cpp" \
    "${SCRIPT_DIR}/main.cpp" \
    -L"${CANN}/lib64" -lascendcl \
    -o "${OUT}"
echo "[build] 产出 ${OUT}"
