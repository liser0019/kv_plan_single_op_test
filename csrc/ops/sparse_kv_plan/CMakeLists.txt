# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SparseKvPlan 算子自注册。

set(SPARSE_KV_PLAN_KERNEL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/sparse_kv_plan_kernel.cpp)
set(SPARSE_KV_PLAN_PLUGIN_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/sparse_kv_plan_plugin.cpp)

register_direct_launch_op(
    "${SPARSE_KV_PLAN_KERNEL_SRCS}"
    "${SPARSE_KV_PLAN_PLUGIN_SRCS}"
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)

# SIMT VF 内核专用编译选项,与 vllm-ascend 仓
# csrc/attention/sparse_kv_plan/op_host/CMakeLists.txt 中 add_ops_compile_options
# 的取值保持一致。这些选项影响 SIMT VF 的 barrier/访存一致性,不得随意删减。
# COMPILE_OPTIONS 追加在顶层 COMPILE_FLAGS(--npu-arch=... -xasc)之后。
set_property(SOURCE ${SPARSE_KV_PLAN_KERNEL_SRCS} APPEND PROPERTY
    COMPILE_OPTIONS
        "--cce-auto-sync=off"
        "-Wno-deprecated-declarations"
        "-mllvm;-cce-vf-remove-membar=false"
        "-mllvm;-cce-aicore-hoist-movemask=false"
        "-mllvm;-cce-aicore-dcci-before-kernel-end=false")
