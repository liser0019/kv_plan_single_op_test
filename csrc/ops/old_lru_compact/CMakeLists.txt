# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

register_direct_launch_op(
    "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/acc_offload_lru_compact.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/old_lru_compact_plugin.cpp"
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)
