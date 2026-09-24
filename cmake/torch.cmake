# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# PyTorch 发现模块:通过当前 Python 环境的 torch 查询 cmake prefix path。

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE _torch_query_result
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT _torch_query_result EQUAL 0)
    message(FATAL_ERROR "当前 Python 环境无法 import torch,请先安装/激活正确的环境。")
endif()

find_package(Torch REQUIRED PATHS ${TORCH_CMAKE_PREFIX} NO_DEFAULT_PATH)
message(STATUS "PyTorch: ${TORCH_INSTALL_DIR} (${TORCH_VERSION})")
