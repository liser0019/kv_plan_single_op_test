#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SparseKvPlan 单算子工程构建入口:SoC 自动检测 + wheel 构建 + 可选安装。
# 本脚本不感知算子,与 direct launch 工程骨架保持一致。

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

detect_soc_version() {
    local torch_soc
    torch_soc=$(python3 -c "import torch, torch_npu; print(torch.npu.get_device_name(0))" 2>/dev/null)
    if [ -n "${torch_soc}" ]; then
        case "${torch_soc}" in
            Ascend910B*)     echo "ascend910b" ; return ;;
            Ascend910_93*)   echo "ascend910_93" ; return ;;
            Ascend950*)      echo "ascend950" ; return ;;
        esac
    fi
    local npu_name
    npu_name=$(npu-smi info 2>/dev/null | grep -oP 'Ascend\S+' | head -1)
    case "${npu_name}" in
        Ascend910B1|Ascend910B2|Ascend910B3|Ascend910B4) echo "ascend910b" ;;
        Ascend910_93*)  echo "ascend910_93" ;;
        Ascend950*)     echo "ascend950" ;;
        *)              echo "" ;;
    esac
}

SOC_VERSION=""
INSTALL=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --soc=*) SOC_VERSION="${1#*=}"; shift ;;
        --install) INSTALL=true; shift ;;
        *) shift ;;
    esac
done

if [ -z "${SOC_VERSION}" ]; then
    SOC_VERSION=$(detect_soc_version)
    if [ -z "${SOC_VERSION}" ]; then
        echo "[ERROR] 无法自动检测 SoC 版本。请使用 --soc=<soc_version> 指定。"
        echo "支持: ascend910b, ascend910_93, ascend950(本算子仅支持昇腾 950)"
        exit 1
    fi
    echo "[INFO] 自动检测 SoC: ${SOC_VERSION}"
fi
export NPU_ARCH="${SOC_VERSION}"

echo "=== 构建 sparse_kv_plan_op wheel 包 ==="
echo "NPU_ARCH: ${NPU_ARCH}"
DIST_DIR="${SCRIPT_DIR}/dist"
rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}"
bash "${SCRIPT_DIR}/scripts/build_wheel.sh"

# 把 _C.so 从 wheel 解回源码包目录:保证从工程根目录直接 import 时
# (python -m pytest 会把 CWD 加进 sys.path)本地包优先于 site-packages,
# 避免"源码包遮蔽已安装包"导致的 ImportError。
python3 - <<'EOF'
import glob, zipfile, shutil, os
wheels = glob.glob("dist/sparse_kv_plan_op*.whl")
assert wheels, "wheel 构建产物不存在"
with zipfile.ZipFile(wheels[0]) as zf:
    member = [n for n in zf.namelist() if n.endswith("sparse_kv_plan_op/_C.so")]
    assert member, "wheel 中未找到 _C.so"
    zf.extract(member[0], ".")
print(f"[build] 已将 {member[0]} 解回工程目录(本地导入上下文可用)")
EOF

if [[ "${INSTALL}" == "true" ]]; then
    echo "=== 安装 wheel 包 ==="
    python3 -m pip install ${DIST_DIR}/sparse_kv_plan_op*.whl --force-reinstall --no-deps
fi

echo "=== 构建完成 ==="
ls -la "${DIST_DIR}"
