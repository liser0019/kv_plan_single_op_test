# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SparseKvPlan 单算子 direct launch 工程顶层构建配置。
#
# 双编译器分工(direct launch 结构):
#   - op_kernel/*.cpp : bisheng 显式编译(--npu-arch=dav-3510 -xasc)
#   - op_plugin/*.cpp : g++ 编译(-O3 -std=c++17),产出 all_plugins_obj
#   - 两者合并为 _C.so,随 sparse_kv_plan_op 包发布
#
# NPU_ARCH → bisheng --npu-arch 映射(本算子仅支持昇腾 950,默认 ascend950):
#   ascend910b / ascend910_93 → dav-2201
#   ascend950                → dav-3510   ← 稀疏 KV LRU 算子的目标芯片
#
# 注意:本工程在无 NPU 的机器上仅做代码交付,未实际编译验证;构建脚本中的
# 工具链路径发现逻辑均已提供环境变量/ cmake 变量覆盖入口,真机上如有路径
# 差异请优先用覆盖参数而非改脚本。

cmake_minimum_required(VERSION 3.16)
project(sparse_kv_plan_op CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- NPU_ARCH 与 bisheng arch 映射 ----
if(NOT DEFINED NPU_ARCH)
    set(NPU_ARCH "$ENV{NPU_ARCH}")
endif()
if(NPU_ARCH STREQUAL "")
    set(NPU_ARCH "ascend950")
endif()

if(NPU_ARCH STREQUAL "ascend910b" OR NPU_ARCH STREQUAL "ascend910_93")
    set(BISHENG_NPU_ARCH "dav-2201")
elseif(NPU_ARCH STREQUAL "ascend950")
    set(BISHENG_NPU_ARCH "dav-3510")
else()
    message(FATAL_ERROR "不支持的 NPU_ARCH=${NPU_ARCH}(可选: ascend910b, ascend910_93, ascend950)")
endif()
message(STATUS "NPU_ARCH=${NPU_ARCH} -> bisheng --npu-arch=${BISHENG_NPU_ARCH}")

# ---- 工具链发现 ----
include(cmake/ascend.cmake)
include(cmake/python.cmake)
include(cmake/torch.cmake)
include(cmake/torch_npu.cmake)
include(cmake/func.cmake)

# ---- 收集算子源文件(csrc/ops/ 下自动发现) ----
add_subdirectory(csrc/ops)

get_property(ALL_KERNEL_SRCS GLOBAL PROPERTY DIRECT_LAUNCH_KERNEL_SRCS)
get_property(ALL_PLUGIN_SRCS GLOBAL PROPERTY DIRECT_LAUNCH_PLUGIN_SRCS)
get_property(OP_INCLUDE_DIRS GLOBAL PROPERTY DIRECT_LAUNCH_INCLUDE_DIRS)

if(NOT ALL_KERNEL_SRCS OR NOT ALL_PLUGIN_SRCS)
    message(FATAL_ERROR "未收集到任何算子源文件,请检查 csrc/ops/ 下的算子注册。")
endif()

# ---- bisheng 编译 kernel ----
# CMAKE_CXX_COMPILER 在 project() 后已确定；临时修改变量不会切换目标编译器。
# 用 custom command 生成外部 object，确保 kernel 不会落到 g++。
set(KERNEL_INCLUDE_ARGS "")
foreach(_include_dir IN LISTS CANN_INCLUDE_DIRS OP_INCLUDE_DIRS)
    list(APPEND KERNEL_INCLUDE_ARGS "-I${_include_dir}")
endforeach()
set(KERNEL_SIMT_FLAGS
    --cce-auto-sync=off
    -Wno-deprecated-declarations
    -mllvm -cce-vf-remove-membar=false
    -mllvm -cce-aicore-hoist-movemask=false
    -mllvm -cce-aicore-dcci-before-kernel-end=false)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(KERNEL_OPT_FLAGS -O0 -g)
else()
    set(KERNEL_OPT_FLAGS -O3)
endif()
set(KERNEL_OBJECTS "")
file(GLOB_RECURSE KERNEL_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/ops/*/op_kernel/*.h")
foreach(_kernel_src IN LISTS ALL_KERNEL_SRCS)
    string(SHA256 _kernel_hash "${_kernel_src}")
    string(SUBSTRING "${_kernel_hash}" 0 12 _kernel_id)
    set(_kernel_obj "${CMAKE_CURRENT_BINARY_DIR}/kernel_${_kernel_id}.o")
    add_custom_command(
        OUTPUT "${_kernel_obj}"
        COMMAND "${BISHENG_CXX}" --npu-arch=${BISHENG_NPU_ARCH} -xasc
                -std=c++17 -fPIC ${KERNEL_OPT_FLAGS} ${KERNEL_SIMT_FLAGS}
                ${KERNEL_INCLUDE_ARGS} -c "${_kernel_src}" -o "${_kernel_obj}"
        DEPENDS "${_kernel_src}" ${KERNEL_HEADERS}
        COMMENT "bisheng 编译 ${_kernel_src}"
        VERBATIM)
    list(APPEND KERNEL_OBJECTS "${_kernel_obj}")
endforeach()
set_source_files_properties(${KERNEL_OBJECTS} PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)

# ---- g++ 编译 plugin ----
add_library(all_plugins_obj OBJECT ${ALL_PLUGIN_SRCS})
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(all_plugins_obj PRIVATE ${TORCH_CXX_FLAGS_LIST})
target_include_directories(all_plugins_obj SYSTEM PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIRS}
    ${CANN_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    ${OP_INCLUDE_DIRS})

# ---- 合并为 _C.so ----
add_library(_C SHARED
    csrc/extension.cpp
    ${KERNEL_OBJECTS}
    $<TARGET_OBJECTS:all_plugins_obj>)
target_compile_options(_C PRIVATE ${TORCH_CXX_FLAGS_LIST})
set_target_properties(_C PROPERTIES PREFIX "" OUTPUT_NAME "_C" SUFFIX ".so")
target_include_directories(_C SYSTEM PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CANN_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    ${OP_INCLUDE_DIRS})
target_link_directories(_C PRIVATE
    ${TORCH_LIB_DIR}
    ${TORCH_NPU_LIB_DIR}
    ${CANN_LIB_DIR})
target_link_libraries(_C PRIVATE
    torch
    torch_cpu
    torch_python
    c10
    torch_npu
    ascendcl)
