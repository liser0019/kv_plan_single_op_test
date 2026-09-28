# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""旧 AIV LRU 单算子与新版 Plan 的共同语义、生产尺寸正确性测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from golden import golden_sparse_kv_plan  # noqa: E402
from golden_old_lru import golden_old_lru  # noqa: E402
from old_lru_cases import make_common_case  # noqa: E402

OUTPUT_KEYS = ("last_req_ids", "slot_to_token", "lru_slots", "current_slots",
               "miss_count", "miss_tokens", "miss_slots")

try:
    import torch
    import torch_npu  # noqa: F401

    NPU_AVAILABLE = torch.npu.is_available()
except ImportError:
    NPU_AVAILABLE = False

requires_npu = pytest.mark.skipif(not NPU_AVAILABLE, reason="需要 torch_npu 和 NPU")


@pytest.mark.parametrize("shape", [(2, 8, 16, 128), (32, 2048, 4096, 256 * 1024)])
@pytest.mark.parametrize("resident_fraction", [0.0, 0.75, 1.0])
def test_common_case_expected_misses(shape, resident_fraction):
    case = make_common_case(rows=shape[0], topk=shape[1], capacity=shape[2],
                            max_token=shape[3], resident_fraction=resident_fraction)
    assert np.array_equal(case["token_to_req"], np.arange(shape[0]))
    assert np.all(case["topk_indices"] < case["visible_seq_lens"][:, None])
    assert np.all(case["block_table"] >= 0)
    expected = golden_sparse_kv_plan(case)
    np.testing.assert_array_equal(expected["miss_count"],
                                  np.full(shape[0], shape[1] - int(shape[1] * resident_fraction)))
    assert np.all(expected["current_slots"] >= 0)
    old_expected = golden_old_lru(case)
    for key in OUTPUT_KEYS:
        np.testing.assert_array_equal(old_expected[key], expected[key], err_msg=key)


def _old_edge_case():
    """小 shape 覆盖重复/非法 token、分配不足、stable 失效、请求重置。"""
    return {
        "req_ids": np.array([11, 22], dtype=np.int64),
        "topk_indices": np.array([[1, 1, 2, 99], [3, 4, 5, 6]], dtype=np.int32),
        "stable_prefix_lens": np.array([2, 4], dtype=np.int32),
        "last_req_ids": np.array([11, -1], dtype=np.int64),
        "slot_to_token": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "lru_slots": np.array([[0, 1], [1, 0]], dtype=np.int32),
        "max_token": 16,
    }


def test_old_golden_edge_anchor():
    out = golden_old_lru(_old_edge_case())
    np.testing.assert_array_equal(out["current_slots"], [[0, 1, -1, -1], [0, 1, -1, -1]])
    np.testing.assert_array_equal(out["miss_tokens"], [[1, 2, -1, -1], [3, 4, 5, 6]])
    np.testing.assert_array_equal(out["miss_slots"], [[1, -1, -1, -1], [0, 1, -1, -1]])
    np.testing.assert_array_equal(out["miss_count"], [1, 2])
    np.testing.assert_array_equal(out["last_req_ids"], [11, 22])


def _to_npu(value):
    return torch.from_numpy(np.ascontiguousarray(value)).npu()


def _run_both(case):
    import sparse_kv_plan_op as op

    rows, topk = case["topk_indices"].shape
    capacity, max_token = case["capacity"], case["max_token"]
    req_ids = _to_npu(case["req_ids"])
    topk_indices = _to_npu(case["topk_indices"])
    stable_prefix = _to_npu(case["stable_prefix_lens"])
    old = op.OldLruState.create(rows, topk, capacity, max_token, "npu")
    new = op.SimtLruState.create(rows, topk, capacity, "npu")
    for state in (old, new):
        state.last_req_ids.copy_(_to_npu(case["last_req_ids"]))
        state.slot_to_token.copy_(_to_npu(case["slot_to_token"]))
        state.lru_slots.copy_(_to_npu(case["lru_slots"]))

    op.old_lru_compact(req_ids=req_ids, topk_indices=topk_indices,
                       stable_prefix_lens=stable_prefix, state=old, max_token=max_token)
    new.active_rows.copy_(_to_npu(np.atleast_1d(case["active_rows"])))
    op.sparse_kv_plan(
        req_ids=req_ids, topk_indices=topk_indices, stable_prefix_lens=stable_prefix,
        visible_seq_lens=_to_npu(case["visible_seq_lens"]),
        token_to_req=_to_npu(case["token_to_req"]), block_table=_to_npu(case["block_table"]),
        active_rows=new.active_rows, state=new, max_token=max_token,
        block_size=case["block_size"], host_num_blocks=case["host_num_blocks"],
    )
    torch.npu.synchronize()
    return ({key: getattr(old, key).cpu().numpy() for key in OUTPUT_KEYS},
            {key: getattr(new, key).cpu().numpy() for key in OUTPUT_KEYS})


@requires_npu
@pytest.mark.parametrize("shape", [(2, 8, 16, 128), (32, 2048, 4096, 256 * 1024)])
@pytest.mark.parametrize("resident_fraction", [0.0, 0.75, 1.0])
def test_old_and_new_match_golden(shape, resident_fraction):
    import sparse_kv_plan_op  # noqa: F401

    assert hasattr(torch.ops.sparse_kv_plan_op, "old_lru_compact"), (
        "请先 bash build.sh --soc=ascend950 --install 重新编译扩展")
    case = make_common_case(rows=shape[0], topk=shape[1], capacity=shape[2],
                            max_token=shape[3], resident_fraction=resident_fraction)
    expected = golden_sparse_kv_plan(case)
    old, new = _run_both(case)
    for key in OUTPUT_KEYS:
        np.testing.assert_array_equal(old[key], expected[key], err_msg=f"old:{key}")
        np.testing.assert_array_equal(new[key], expected[key], err_msg=f"new:{key}")


@requires_npu
def test_old_edge_matches_own_golden():
    import sparse_kv_plan_op as op

    case = _old_edge_case()
    expected = golden_old_lru(case)
    state = op.OldLruState.create(2, 4, 2, 16, "npu")
    state.last_req_ids.copy_(_to_npu(case["last_req_ids"]))
    state.slot_to_token.copy_(_to_npu(case["slot_to_token"]))
    state.lru_slots.copy_(_to_npu(case["lru_slots"]))
    op.old_lru_compact(req_ids=_to_npu(case["req_ids"]),
                       topk_indices=_to_npu(case["topk_indices"]),
                       stable_prefix_lens=_to_npu(case["stable_prefix_lens"]),
                       state=state, max_token=16)
    torch.npu.synchronize()
    for key in OUTPUT_KEYS:
        np.testing.assert_array_equal(getattr(state, key).cpu().numpy(), expected[key], err_msg=key)
