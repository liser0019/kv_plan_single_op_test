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
    "old_lru_compact",
    "OldLruState",
    "plan_hash_capacity",
    "plan_workspace_elements",
    "make_compact_workspace",
    "EchoLruState",
    "echo_lru",
    "echo_lru_skip_sort",
    "echo_partition",
    "echo_workspace_elements",
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


class OldLruState(NamedTuple):
    """MemFabric_Hybrid 旧 AIV LRU 核的持久状态与输出缓冲。"""

    last_req_ids: torch.Tensor
    slot_to_token: torch.Tensor
    lru_slots: torch.Tensor
    current_slots: torch.Tensor
    miss_count: torch.Tensor
    miss_tokens: torch.Tensor
    miss_slots: torch.Tensor
    token_mark: torch.Tensor
    token_pos: torch.Tensor
    epochs: torch.Tensor

    @classmethod
    def create(cls, rows: int, topk: int, capacity: int, max_token: int, device) -> "OldLruState":
        return cls(
            last_req_ids=torch.full((rows,), -1, dtype=torch.int64, device=device),
            slot_to_token=torch.full((rows, capacity), -1, dtype=torch.int32, device=device),
            lru_slots=torch.arange(capacity, dtype=torch.int32, device=device).repeat(rows, 1),
            current_slots=torch.full((rows, topk), -1, dtype=torch.int32, device=device),
            miss_count=torch.zeros(rows, dtype=torch.int32, device=device),
            miss_tokens=torch.full((rows, topk), -1, dtype=torch.int32, device=device),
            miss_slots=torch.full((rows, topk), -1, dtype=torch.int32, device=device),
            token_mark=torch.zeros((8, max_token), dtype=torch.int32, device=device),
            token_pos=torch.zeros((8, max_token), dtype=torch.int32, device=device),
            epochs=torch.zeros(8, dtype=torch.int32, device=device),
        )


def old_lru_compact(
    *, req_ids: torch.Tensor, topk_indices: torch.Tensor,
    stable_prefix_lens: torch.Tensor, state: OldLruState,
    max_token: int,
) -> None:
    """调用原始 AIV LRU compact 算法；行号直接对应 request 号。"""
    return torch.ops.sparse_kv_plan_op.old_lru_compact(
        req_ids, topk_indices, stable_prefix_lens,
        state.last_req_ids, state.slot_to_token, state.lru_slots,
        state.current_slots, state.miss_count, state.miss_tokens,
        state.miss_slots, state.token_mark, state.token_pos,
        state.epochs, max_token,
    )


from .echo_layout import I32_MAX, echo_partition, echo_workspace_elements


class EchoLruState(NamedTuple):
    """同事纯 SIMT Echo 的状态。槽位 1..capacity；token 0 对应保留槽 0。"""

    htd: torch.Tensor
    dth: torch.Tensor
    pri: torch.Tensor
    free_slots: torch.Tensor
    avail: torch.Tensor
    fifo: torch.Tensor
    current_slots: torch.Tensor
    miss_host_pos: torch.Tensor
    miss_alloc_flat: torch.Tensor
    miss_mask: torch.Tensor
    workspace: torch.Tensor
    blocks: int

    @classmethod
    def create(cls, rows: int, topk: int, capacity: int, max_token: int, device,
               available_cores=None) -> "EchoLruState":
        if not (1 < max_token <= I32_MAX and rows > 0 and
                rows * max_token <= I32_MAX and
                rows * (capacity + min(topk, capacity)) <= I32_MAX):
            raise ValueError("invalid max_token or 32-bit index overflow")
        if available_cores is None:
            with torch.npu.device(device):
                available_cores = torch.ops.sparse_kv_plan_op.echo_lru_core_count()
        blocks, _, _ = echo_partition(rows, available_cores)
        elements = echo_workspace_elements(rows, topk, capacity, blocks)
        htd = torch.full((rows, max_token), I32_MAX, dtype=torch.int32, device=device)
        htd[:, 0] = 0
        pri = torch.full((rows, capacity + 1), -1, dtype=torch.int32, device=device)
        pri[:, 0] = I32_MAX
        free = torch.zeros((rows, capacity + min(topk, capacity)), dtype=torch.int32, device=device)
        free[:, :capacity] = torch.arange(capacity, 0, -1, dtype=torch.int32, device=device)
        return cls(
            htd, torch.full_like(pri, I32_MAX), pri, free,
            torch.full((rows,), capacity, dtype=torch.int32, device=device),
            torch.ones(rows, dtype=torch.int32, device=device),
            torch.zeros((rows, topk), dtype=torch.int32, device=device),
            torch.zeros((rows, topk), dtype=torch.int32, device=device),
            torch.zeros((rows, topk), dtype=torch.int32, device=device),
            torch.zeros((rows, topk), dtype=torch.uint8, device=device),
            torch.empty(elements, dtype=torch.int32, device=device), blocks)


def echo_lru(*, pos: torch.Tensor, stable_prefix_lens: torch.Tensor,
             reset_mask: torch.Tensor, state: EchoLruState, spec_enabled: bool = True) -> None:
    """按 Echo 原语义更新状态。

    调用方契约：有效前 min(topk, capacity) 个 token 行内不重复且在
    [0, max_token) 内；0 是保留 token。spl 在 [0, max_token] 内。
    持久状态须由 create、合法转换或上一轮 kernel 产生；fifo 不能溢出 int32。
    spec_enabled=False 仅用于所有 spl==max_token 的调用；默认 True 无需 Host 读回。
    Echo 不实现 SIMT Plan 的 visible_seq_lens/block_table/去重检查。
    """
    return torch.ops.sparse_kv_plan_op.echo_lru(
        pos, stable_prefix_lens, reset_mask, state.htd, state.dth, state.pri,
        state.free_slots, state.avail, state.fifo, state.current_slots,
        state.miss_host_pos, state.miss_alloc_flat, state.miss_mask, state.workspace,
        state.blocks, spec_enabled)


def echo_lru_skip_sort(*, pos: torch.Tensor, stable_prefix_lens: torch.Tensor,
             reset_mask: torch.Tensor, state: EchoLruState, spec_enabled: bool = True) -> None:
    """Echo 实验版：freeSize=0 时跳过排序，输入与输出契约同 echo_lru。"""
    return torch.ops.sparse_kv_plan_op.echo_lru_skip_sort(
        pos, stable_prefix_lens, reset_mask, state.htd, state.dth, state.pri,
        state.free_slots, state.avail, state.fifo, state.current_slots,
        state.miss_host_pos, state.miss_alloc_flat, state.miss_mask, state.workspace,
        state.blocks, spec_enabled)
