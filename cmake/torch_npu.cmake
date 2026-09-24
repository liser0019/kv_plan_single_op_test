# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# torch_npu 发现模块:取 torch_npu 包目录,导出 include 与 lib 路径。

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch_npu, os; print(os.path.dirname(torch_npu.__file__))"
    RESULT_VARIABLE _tnpu_query_result
    OUTPUT_VARIABLE TORCH_NPU_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT _tnpu_query_result EQUAL 0)
    message(FATAL_ERROR "当前 Python 环境无法 import torch_npu。真实 NPU 构建必须安装与 torch 版本匹配的 torch_npu。")
endif()

set(TORCH_NPU_INCLUDE_DIRS "${TORCH_NPU_DIR}/include")
set(TORCH_NPU_LIB_DIR "${TORCH_NPU_DIR}/lib")
if(NOT EXISTS "${TORCH_NPU_LIB_DIR}")
    set(TORCH_NPU_LIB_DIR "${TORCH_NPU_DIR}")
endif()
message(STATUS "torch_npu: ${TORCH_NPU_DIR}")
