# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# direct launch 算子自注册宏。
#
# 算子目录(csrc/ops/<op>/)在自己的 CMakeLists.txt 中调用本宏,把 kernel/plugin
# 源文件登记到全局属性;顶层 CMakeLists.txt 在 add_subdirectory(csrc/ops) 之后
# 读取这些全局属性,统一用 bisheng 编 kernel、g++ 编 plugin。

macro(register_direct_launch_op KERNEL_SRCS PLUGIN_SRCS INCLUDE_DIR)
    get_filename_component(_op_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    foreach(_src ${KERNEL_SRCS})
        set_property(GLOBAL APPEND PROPERTY DIRECT_LAUNCH_KERNEL_SRCS "${_src}")
    endforeach()
    foreach(_src ${PLUGIN_SRCS})
        set_property(GLOBAL APPEND PROPERTY DIRECT_LAUNCH_PLUGIN_SRCS "${_src}")
    endforeach()
    set_property(GLOBAL APPEND PROPERTY DIRECT_LAUNCH_INCLUDE_DIRS "${INCLUDE_DIR}")
    message(STATUS "direct launch 算子已注册: ${_op_name}")
endmacro()
