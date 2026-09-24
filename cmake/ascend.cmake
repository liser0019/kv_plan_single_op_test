# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# CANN / bisheng 工具链发现模块。
#
# 查找优先级(从高到低):
#   1. -DASCEND_CANN_PACKAGE_HOME=<path>   (cmake 变量)
#   2. 环境变量 ASCEND_CANN_PACKAGE_HOME
#   3. 环境变量 ASCEND_HOME_PATH
#   4. /usr/local/Ascend/ascend-toolkit/latest
# bisheng 编译器可用 -DBISHENG_CXX=<path> 显式覆盖。

set(ASCEND_CANN_PACKAGE_HOME "" CACHE PATH "CANN package root (含 include/ lib64/ bin/)")

set(_cann_candidates
    "${ASCEND_CANN_PACKAGE_HOME}"
    "$ENV{ASCEND_CANN_PACKAGE_HOME}"
    "$ENV{ASCEND_HOME_PATH}"
    "/usr/local/Ascend/ascend-toolkit/latest")

set(_cann_found "")
foreach(_cand ${_cann_candidates})
    if(NOT _cann_found AND NOT "${_cand}" STREQUAL "" AND EXISTS "${_cand}/include")
        set(_cann_found "${_cand}")
    endif()
endforeach()

if(_cann_found)
    set(ASCEND_CANN_PACKAGE_HOME "${_cann_found}" CACHE PATH "CANN package root" FORCE)
else()
    message(FATAL_ERROR
        "未找到 CANN 包。请设置环境变量 ASCEND_CANN_PACKAGE_HOME 或 ASCEND_HOME_PATH,"
        "或通过 -DASCEND_CANN_PACKAGE_HOME=/path/to/cann 指定。")
endif()
message(STATUS "CANN 包路径: ${ASCEND_CANN_PACKAGE_HOME}")

# bisheng 编译器(Ascend C kernel 编译入口)
if(NOT BISHENG_CXX)
    find_program(BISHENG_CXX
        NAMES bisheng_compiler bisheng ccec_compiler
        PATHS
            ${ASCEND_CANN_PACKAGE_HOME}/bin
            ${ASCEND_CANN_PACKAGE_HOME}/compiler/bin
            ${ASCEND_CANN_PACKAGE_HOME}/compiler/bisheng/bin
            ${ASCEND_CANN_PACKAGE_HOME}/compiler/tbe/bin
        DOC "bisheng (Ascend C) 编译器")
endif()

if(NOT BISHENG_CXX)
    message(FATAL_ERROR
        "未找到 bisheng 编译器。请在 CANN 安装目录下确认其位置后通过 -DBISHENG_CXX=<path> 指定"
        "(常见位置: \${CANN}/bin/bisheng_compiler 或 \${CANN}/compiler/bisheng/bin/bisheng_compiler)。")
endif()
message(STATUS "bisheng 编译器: ${BISHENG_CXX}")
set(BISHENG_CXX "${BISHENG_CXX}" CACHE FILEPATH "bisheng compiler" FORCE)

# CANN 头文件与运行时库(kernel 侧与插件侧共用)
set(CANN_INCLUDE_DIRS "${ASCEND_CANN_PACKAGE_HOME}/include")
set(CANN_LIB_DIR "${ASCEND_CANN_PACKAGE_HOME}/lib64")
if(NOT EXISTS "${CANN_LIB_DIR}")
    set(CANN_LIB_DIR "${ASCEND_CANN_PACKAGE_HOME}/lib")
endif()
