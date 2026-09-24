#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# wheel 构建实际执行体,由 build.sh 调用。

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

python3 -m pip wheel \
    --no-build-isolation \
    --no-deps \
    -w dist/ \
    .
