# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SparseKvPlan 算子自注册。

set(SPARSE_KV_PLAN_KERNEL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/sparse_kv_plan_kernel.cpp)
set(SPARSE_KV_PLAN_PLUGIN_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/sparse_kv_plan_plugin.cpp)

register_direct_launch_op(
    "${SPARSE_KV_PLAN_KERNEL_SRCS}"
    "${SPARSE_KV_PLAN_PLUGIN_SRCS}"
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)

# SIMT VF 专用编译选项由顶层 bisheng custom command 传入。
# 原先在本子目录设置 SOURCE 属性不会作用到父目录创建的 kernel 目标。
