#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# npusim 端到端:导出用例 → 编译仿真程序 → npusim record → 与 golden 比对。
#
# 用法:
#   bash sim/run_sim.sh                      # 全部轻量用例
#   bash sim/run_sim.sh --case l0_basic      # 单个用例
#   bash sim/run_sim.sh --include-heavy      # 含 GM workspace 大用例
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-python3}"
CASES_DIR="${SCRIPT_DIR}/cases"

MAKE_ARGS=""
RUN_ARGS=("$@")
for a in "$@"; do
    if [ "$a" == "--include-heavy" ]; then MAKE_ARGS="--include-heavy"; break; fi
done
CASE_NAME=""
if [ "$1" == "--case" ] && [ -n "$2" ]; then CASE_NAME="$2"; fi

echo "=== 1. 导出用例(golden 期望一并生成)==="
${PYTHON} make_case.py --cases-dir "${CASES_DIR}" ${MAKE_ARGS} ${CASE_NAME:+--case ${CASE_NAME}}

echo "=== 2. 编译仿真程序 ==="
bash build_sim.sh

echo "=== 3. npusim record(Ascend950)==="
if ! command -v npusim >/dev/null 2>&1; then
    echo "[ERROR] 未找到 npusim 命令(CANN >= 9.2;旧版叫 cannsim)。请 source CANN 环境。"
    exit 1
fi

FAIL=0
for case_dir in "${CASES_DIR}"/*/; do
    name="$(basename "${case_dir}")"
    [ -f "${case_dir}/meta.txt" ] || continue
    echo "--- 仿真 ${name} ---"
    rm -rf "${case_dir}/output"
    mkdir -p "${case_dir}/output"
    npusim record ./sparse_kv_plan_sim -s Ascend950 --gen-report -u "${case_dir}" -o "${case_dir}/npusim_out" || {
        echo "[FAIL] ${name}: npusim 运行失败,日志: ${case_dir}/npusim_out/npusim.log"; FAIL=1; continue; }
    ${PYTHON} compare_sim_output.py --case-dir "${case_dir}" || FAIL=1
done

if [ "${FAIL}" != "0" ]; then
    echo "=== 存在失败用例 ==="
    exit 1
fi
echo "=== 全部用例仿真比对通过 ==="
