/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * Python 扩展入口。加载本 .so 时,链接进来的各算子 plugin 目标文件中的
 * TORCH_LIBRARY 静态初始化器完成 torch.ops 注册,这里无需做任何事。
 */
#include <torch/extension.h>

PYBIND11_MODULE(_C, m) {
  m.doc() = "sparse_kv_plan_op: direct launch C extension (SparseKvPlan, Ascend 950)";
}
