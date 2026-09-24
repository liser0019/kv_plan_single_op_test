# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""SparseKvPlan 单算子直调工程 Python 包。

用法(昇腾 950 + torch_npu 环境):

    import torch
    import sparse_kv_plan_op

    state = sparse_kv_plan_op.SimtLruState(
        max_rows=8, topk=64, capacity=128, device="npu",
    )
    sparse_kv_plan_op.sparse_kv_plan(
        req_ids=..., topk_indices=..., stable_prefix_lens=...,
        visible_seq_lens=..., token_to_req=..., block_table=...,
        active_rows=state.active_rows, state=state,
        max_token=..., block_size=..., host_num_blocks=...,
    )
"""

from typing import NamedTuple

import torch

try:
    from . import _C
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "无法导入 _C:请先在 NPU 机器上执行 `bash build.sh --soc=ascend950 --install` 构建安装。"
        "本包只提供算子注册;golden/用例生成可在无 NPU 环境直接使用 tests/ 目录。"
    ) from e

__all__ = [
    "sparse_kv_plan",
    "SimtLruState",
    "plan_hash_capacity",
    "plan_workspace_elements",
    "make_compact_workspace",
]

_PLAN_ROW_UB_LIMIT_BYTES = 112 * 1024
_MIN_HASH_CAPACITY = 32


def plan_hash_capacity(topk: int) -> int:
    """SparseKvPlan 的 hash 容量:2*topk 向上取 2 的幂,下限 32。"""
    target = 2 * topk
    capacity = _MIN_HASH_CAPACITY
    while capacity < target:
        capacity <<= 1
    return capacity


def plan_workspace_elements(topk: int, capacity: int) -> int:
    """单行临时区元素数(单位 int32),与 vllm-ascend sparse_kv_simt_lru 同源。"""
    return 3 * plan_hash_capacity(topk) + 2 * capacity + topk


def make_compact_workspace(max_rows: int, topk: int, capacity: int, device) -> torch.Tensor:
    """按需分配 compact_workspace:UB 快路径只需 1 个占位元素,GM 路径需完整尺寸。"""
    row_elements = plan_workspace_elements(topk, capacity)
    elements = 1 if row_elements * torch.int32.itemsize <= _PLAN_ROW_UB_LIMIT_BYTES else max_rows * row_elements
    return torch.empty(elements, dtype=torch.int32, device=device)


class SimtLruState(NamedTuple):
    """一次 sparse_kv_plan 调用所需的持久状态张量集合。

    与 vllm-ascend `_SparseKVSimtLayerState` 的行级字段一致(去掉 layer 维度的
    host_cache_bases 等 transfer 专用字段)。current_slots 为调用方持有,
    按需传入;未提供时按全 -1 初始化。
    """

    last_req_ids: torch.Tensor      # int64 [max_rows]
    slot_to_token: torch.Tensor     # int32 [max_rows, capacity]
    lru_slots: torch.Tensor         # int32 [max_rows, capacity]
    current_slots: torch.Tensor     # int32 [max_rows, topk]
    miss_count: torch.Tensor        # int32 [max_rows]
    miss_tokens: torch.Tensor       # int32 [max_rows, topk]
    miss_slots: torch.Tensor        # int32 [max_rows, topk]
    active_rows: torch.Tensor       # int32 [1]
    compact_workspace: torch.Tensor  # int32 [>=1]

    @classmethod
    def create(cls, max_rows: int, topk: int, capacity: int, device) -> "SimtLruState":
        return cls(
            last_req_ids=torch.full((max_rows,), -1, dtype=torch.int64, device=device),
            slot_to_token=torch.full((max_rows, capacity), -1, dtype=torch.int32, device=device),
            lru_slots=torch.arange(capacity, dtype=torch.int32, device=device).repeat(max_rows, 1),
            current_slots=torch.full((max_rows, topk), -1, dtype=torch.int32, device=device),
            miss_count=torch.zeros(max_rows, dtype=torch.int32, device=device),
            miss_tokens=torch.full((max_rows, topk), -1, dtype=torch.int32, device=device),
            miss_slots=torch.full((max_rows, topk), -1, dtype=torch.int32, device=device),
            active_rows=torch.zeros(1, dtype=torch.int32, device=device),
            compact_workspace=make_compact_workspace(max_rows, topk, capacity, device),
        )


def sparse_kv_plan(
    *,
    req_ids: torch.Tensor,
    topk_indices: torch.Tensor,
    stable_prefix_lens: torch.Tensor,
    visible_seq_lens: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    active_rows: torch.Tensor,
    state: SimtLruState,
    max_token: int,
    block_size: int,
    host_num_blocks: int,
) -> None:
    """执行一次 SparseKvPlan(直调),原地更新 state 中的各状态张量。

    参数语义与 vllm-ascend `npu_sparse_kv_plan_transfer` 的 Plan 部分一致;
    Transfer 需 MemFabric Host DVA,不在单算子测试范围内。
    """
    topk = topk_indices.size(1)
    capacity = state.slot_to_token.size(1)
    return torch.ops.sparse_kv_plan_op.sparse_kv_plan(
        req_ids,
        topk_indices,
        stable_prefix_lens,
        visible_seq_lens,
        token_to_req,
        block_table,
        active_rows,
        state.last_req_ids,
        state.slot_to_token,
        state.lru_slots,
        state.current_slots,
        state.miss_count,
        state.miss_tokens,
        state.miss_slots,
        state.compact_workspace,
        topk,
        capacity,
        max_token,
        block_size,
        host_num_blocks,
    )
