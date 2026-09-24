# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Python3 解释器与开发头文件发现。

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python3: ${Python3_EXECUTABLE}")
