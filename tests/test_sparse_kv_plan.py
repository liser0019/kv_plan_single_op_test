# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""SparseKvPlan 单算子测试。

分三层,前两层无 NPU 也能跑(当前交付机器即如此),第三层在 NPU 上跑:

1. 锚点用例:三步手工推导的精确期望值(算法真值,golden 的基准);
2. golden 自检:全用例结构不变量 + 确定性(仅 CPU);
3. NPU 精确比对:int32 全等比较(kernel 为确定性算法,不允许任何偏差)。

分支覆盖说明(白盒):
- reset / no-reset          → l0 锚点三步、l2_req_change_reset
- 空行(activeRows<=0)      → l2_active_rows_zero
- activeRows > maxRows 截断 → l2_active_rows_overflow
- TopK 重复 token            → l0 锚点 call1(row0)、l2_dup_heavy
- 同 token 多 slot 驻留       → l0 锚点 call2(row0:slot0/1 同持 token1)
- stablePrefix 失效投机后缀   → l0 锚点 call3(row1:token9 被失效)、MTP 用例
- 非法 token(visible/block) → l1/l2 all_invalid、block_table 空洞
- miss > evictable           → l2_miss_exceed_evictable
- UB / GM workspace 路径      → l2_ub_boundary_exact / l2_gm_workspace
- 越界 LRU 项                 → l2_lru_oob_entries
- capacity=1 / 单行           → l2_capacity_one / l0_single_row
- stablePrefix 全失效         → l2_stable_prefix_full
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from gen_data import ALL_CASE_NAMES, build_case  # noqa: E402
from golden import golden_sparse_kv_plan  # noqa: E402

try:
    import torch
    import torch_npu  # noqa: F401

    NPU_AVAILABLE = torch.npu.is_available() if hasattr(torch, "npu") else False
except ImportError:
    NPU_AVAILABLE = False

requires_npu = pytest.mark.skipif(not NPU_AVAILABLE, reason="无 NPU 环境(安装 torch_npu 并在 NPU 机器上运行)")


# ===========================================================================
# 1. 锚点用例:手工推导的精确期望值
# ===========================================================================
def _anchor_case(topk_rows, stable):
    """构造锚点场景:1 个请求、2 行、T=2、C=4。block_table 全合法。"""
    return {
        "req_ids": np.array([100], dtype=np.int64),
        "topk_indices": np.array(topk_rows, dtype=np.int32),
        "stable_prefix_lens": np.array(stable, dtype=np.int32),
        "visible_seq_lens": np.array([16, 16], dtype=np.int32),
        "token_to_req": np.array([0, 0], dtype=np.int32),
        "block_table": np.array([[0, 1, 2, 3]], dtype=np.int32),
        "active_rows": np.int32(2),
        "last_req_ids": np.full(2, -1, dtype=np.int64),
        "slot_to_token": np.full((2, 4), -1, dtype=np.int32),
        "lru_slots": np.tile(np.arange(4, dtype=np.int32), (2, 1)),
        "topk": 2,
        "capacity": 4,
        "max_token": 16,
        "block_size": 4,
        "host_num_blocks": 8,
    }


def test_golden_anchor_three_steps():
    """三步手工推导:reset+重复token → 命中+多slot驻留 → stable前移+投机失效。"""
    # ---- call 1:冷启动。row0 topk=[1,1](重复 token),row1 topk=[2,0] ----
    case = _anchor_case([[1, 1], [2, 0]], stable=[3, 3])
    out = golden_sparse_kv_plan(case)
    # row0:全 miss,重复 token 各占一个 slot(0 和 1)
    np.testing.assert_array_equal(out["current_slots"][0], [0, 1])
    np.testing.assert_array_equal(out["miss_tokens"][0], [1, 1])
    np.testing.assert_array_equal(out["miss_slots"][0], [0, 1])
    np.testing.assert_array_equal(out["slot_to_token"][0], [1, 1, -1, -1])
    np.testing.assert_array_equal(out["lru_slots"][0], [2, 3, 0, 1])
    assert out["miss_count"][0] == 2
    # row1:token 2 与 0 各分一个 slot
    np.testing.assert_array_equal(out["current_slots"][1], [0, 1])
    np.testing.assert_array_equal(out["miss_tokens"][1], [2, 0])
    np.testing.assert_array_equal(out["miss_slots"][1], [0, 1])
    np.testing.assert_array_equal(out["slot_to_token"][1], [2, 0, -1, -1])
    np.testing.assert_array_equal(out["lru_slots"][1], [2, 3, 0, 1])
    assert out["miss_count"][1] == 2
    np.testing.assert_array_equal(out["last_req_ids"], [100, 100])

    # ---- call 2:同请求续用。row0 topk=[1,2],row1 topk=[2,9] ----
    state = {
        "last_req_ids": out["last_req_ids"],
        "slot_to_token": out["slot_to_token"],
        "lru_slots": out["lru_slots"],
    }
    case2 = _anchor_case([[1, 2], [2, 9]], stable=[3, 3])
    case2.update(state)
    out2 = golden_sparse_kv_plan(case2)
    # row0:token1 双 slot 驻留(slot0/slot1),旧 LRU 最后 owner(slot1)提供命中;
    # token2 走 miss 分到 slot2。
    np.testing.assert_array_equal(out2["current_slots"][0], [1, 2])
    np.testing.assert_array_equal(out2["miss_tokens"][0], [2, -1])
    np.testing.assert_array_equal(out2["miss_slots"][0], [2, -1])
    np.testing.assert_array_equal(out2["slot_to_token"][0], [1, 1, 2, -1])
    np.testing.assert_array_equal(out2["lru_slots"][0], [3, 2, 0, 1])
    assert out2["miss_count"][0] == 1
    # row1:token2 命中(旧 LRU 中 slot0,owner order=2 → slot0);token9 miss → slot2。
    np.testing.assert_array_equal(out2["current_slots"][1], [0, 2])
    np.testing.assert_array_equal(out2["miss_tokens"][1], [9, -1])
    np.testing.assert_array_equal(out2["miss_slots"][1], [2, -1])
    np.testing.assert_array_equal(out2["slot_to_token"][1], [2, 0, 9, -1])
    np.testing.assert_array_equal(out2["lru_slots"][1], [3, 1, 2, 0])
    assert out2["miss_count"][1] == 1

    # ---- call 3:stable 前移到 5(等价于投机 token 被接受)。
    #      row0 topk=[2,1] 全命中;row1 topk=[2,9],token9 >= 5 被失效后重走 miss。
    state2 = {
        "last_req_ids": out2["last_req_ids"],
        "slot_to_token": out2["slot_to_token"],
        "lru_slots": out2["lru_slots"],
    }
    case3 = _anchor_case([[2, 1], [2, 9]], stable=[5, 5])
    case3.update(state2)
    out3 = golden_sparse_kv_plan(case3)
    # row0:双 token 全命中(token2→slot2,token1→slot1),无 miss。
    np.testing.assert_array_equal(out3["current_slots"][0], [2, 1])
    np.testing.assert_array_equal(out3["miss_tokens"][0], [-1, -1])
    np.testing.assert_array_equal(out3["miss_slots"][0], [-1, -1])
    assert out3["miss_count"][0] == 0
    np.testing.assert_array_equal(out3["lru_slots"][0], [3, 2, 0, 1])
    # row1:token2 命中(旧 LRU order3 → slot0);token9 因 >= stable(5) 被失效,
    # 重新 miss 分到 slot3。
    np.testing.assert_array_equal(out3["current_slots"][1], [0, 3])
    np.testing.assert_array_equal(out3["miss_tokens"][1], [9, -1])
    np.testing.assert_array_equal(out3["miss_slots"][1], [3, -1])
    np.testing.assert_array_equal(out3["slot_to_token"][1], [2, 0, -1, 9])
    np.testing.assert_array_equal(out3["lru_slots"][1], [1, 2, 3, 0])
    assert out3["miss_count"][1] == 1


# ===========================================================================
# 2. golden 自检(CPU):结构不变量 + 确定性
# ===========================================================================
def _assert_row_invariants(case, out):
    topk = case["topk"]
    capacity = case["capacity"]
    num_rows = out["num_rows"]
    for r in range(num_rows):
        cs = out["current_slots"][r]
        mc = int(out["miss_count"][r])
        mt = out["miss_tokens"][r]
        ms = out["miss_slots"][r]
        stt = out["slot_to_token"][r]
        lru = out["lru_slots"][r]
        # 输出值域
        assert ((cs == -1) | ((cs >= 0) & (cs < capacity))).all()
        assert ((ms[:mc] >= 0) & (ms[:mc] < capacity)).all()
        assert (ms[mc:] == -1).all() and (mt[mc:] == -1).all()
        # miss 分配的 slot 互不相同
        assert len(set(ms[:mc].tolist())) == mc
        # 命中/分配的 slot 内容与 TopK token 一致
        for p in range(topk):
            if cs[p] >= 0:
                assert stt[cs[p]] == case["topk_indices"][r][p], (r, p)
        # LRU 前缀值域
        assert ((lru < 0) | (lru < capacity)).all()


@pytest.mark.parametrize("case_name", ALL_CASE_NAMES)
def test_golden_invariants(case_name):
    case = build_case(case_name)
    out = golden_sparse_kv_plan(case)
    _assert_row_invariants(case, out)


@pytest.mark.parametrize("case_name", ALL_CASE_NAMES)
def test_golden_deterministic(case_name):
    case1 = build_case(case_name)
    case2 = build_case(case_name)
    out1 = golden_sparse_kv_plan(case1)
    out2 = golden_sparse_kv_plan(case2)
    for key in ("current_slots", "miss_count", "miss_tokens", "miss_slots", "slot_to_token", "lru_slots",
                "last_req_ids"):
        np.testing.assert_array_equal(out1[key], out2[key], err_msg=f"{case_name}:{key}")


def test_golden_active_rows_zero_touches_nothing():
    """active_rows=0:所有输出保持 -1/初始状态(kernel 语义:0 行处理)。"""
    case = build_case("l2_active_rows_zero")
    out = golden_sparse_kv_plan(case)
    np.testing.assert_array_equal(out["current_slots"], np.full_like(out["current_slots"], -1))
    np.testing.assert_array_equal(out["miss_count"], np.zeros_like(out["miss_count"]))
    np.testing.assert_array_equal(out["miss_tokens"], np.full_like(out["miss_tokens"], -1))
    np.testing.assert_array_equal(out["miss_slots"], np.full_like(out["miss_slots"], -1))
    assert out["num_rows"] == 0


# ===========================================================================
# 3. NPU 精确比对
# ===========================================================================
def _to_npu(arr):
    import torch

    t = torch.from_numpy(np.ascontiguousarray(arr))
    if t.dtype == np.int32:
        t = t.to(torch.int32)
    return t.npu()


def _npu_run(case):
    """把 case 搬上 NPU 执行一次 sparse_kv_plan,返回 numpy 输出。"""
    import sparse_kv_plan_op  # 安装后的直调包

    import torch

    max_rows = case["topk_indices"].shape[0]
    topk = case["topk"]
    capacity = case["capacity"]
    device = "npu"
    state = sparse_kv_plan_op.SimtLruState.create(max_rows, topk, capacity, device)
    state.last_req_ids.copy_(_to_npu(case["last_req_ids"]))
    state.slot_to_token.copy_(_to_npu(case["slot_to_token"]))
    state.lru_slots.copy_(_to_npu(case["lru_slots"]))

    inputs = {k: _to_npu(case[k]) for k in
              ("req_ids", "topk_indices", "stable_prefix_lens", "visible_seq_lens", "token_to_req",
               "block_table")}
    state.active_rows.copy_(_to_npu(np.atleast_1d(case["active_rows"])))

    sparse_kv_plan_op.sparse_kv_plan(
        req_ids=inputs["req_ids"],
        topk_indices=inputs["topk_indices"],
        stable_prefix_lens=inputs["stable_prefix_lens"],
        visible_seq_lens=inputs["visible_seq_lens"],
        token_to_req=inputs["token_to_req"],
        block_table=inputs["block_table"],
        active_rows=state.active_rows,
        state=state,
        max_token=case["max_token"],
        block_size=case["block_size"],
        host_num_blocks=case["host_num_blocks"],
    )
    torch.npu.synchronize()
    return {k: v.cpu().numpy() for k, v in {
        "current_slots": state.current_slots,
        "miss_count": state.miss_count,
        "miss_tokens": state.miss_tokens,
        "miss_slots": state.miss_slots,
        "slot_to_token": state.slot_to_token,
        "lru_slots": state.lru_slots,
        "last_req_ids": state.last_req_ids,
    }.items()}


@requires_npu
def test_npu_symbol_probe():
    """符号探针:导入的包含本次构建的算子,不通过则先 build --install。"""
    import sparse_kv_plan_op
    import torch

    assert hasattr(torch.ops.sparse_kv_plan_op, "sparse_kv_plan"), (
        "torch.ops.sparse_kv_plan_op.sparse_kv_plan 不存在:请先执行 "
        "`bash build.sh --soc=ascend950 --install` 后重试,失败结果不得作为结论。"
    )
    # 锚点场景在 NPU 上跑一遍并比对(三步,含状态传递)
    case = _anchor_case([[1, 1], [2, 0]], stable=[3, 3])
    out = golden_sparse_kv_plan(case)
    npu_out = _npu_run(case)
    for key in ("current_slots", "miss_count", "miss_tokens", "miss_slots", "slot_to_token", "lru_slots",
                "last_req_ids"):
        np.testing.assert_array_equal(npu_out[key], out[key], err_msg=f"anchor call1:{key}")


@requires_npu
def test_meta_rejects_invalid_direct_launch_inputs():
    """直调按连续内存寻址，且内核属性有 int32 上限。"""
    import sparse_kv_plan_op  # noqa: F401  注册 torch.ops
    import torch

    device = "meta"
    args = [
        torch.empty(1, dtype=torch.int64, device=device),       # req_ids
        torch.empty((2, 2), dtype=torch.int32, device=device),  # topk_indices
        torch.empty(2, dtype=torch.int32, device=device),       # stable_prefix_lens
        torch.empty(2, dtype=torch.int32, device=device),       # visible_seq_lens
        torch.empty(2, dtype=torch.int32, device=device),       # token_to_req
        torch.empty((1, 4), dtype=torch.int32, device=device),  # block_table
        torch.empty(1, dtype=torch.int32, device=device),       # active_rows
        torch.empty(2, dtype=torch.int64, device=device),       # last_req_ids
        torch.empty((2, 4), dtype=torch.int32, device=device),  # slot_to_token
        torch.empty((2, 4), dtype=torch.int32, device=device),  # lru_slots
        torch.empty((2, 2), dtype=torch.int32, device=device),  # current_slots
        torch.empty(2, dtype=torch.int32, device=device),       # miss_count
        torch.empty((2, 2), dtype=torch.int32, device=device),  # miss_tokens
        torch.empty((2, 2), dtype=torch.int32, device=device),  # miss_slots
        torch.empty(1, dtype=torch.int32, device=device),       # compact_workspace (UB path)
        2, 4, 16, 4, 8,
    ]
    args[1] = args[1].t()
    with pytest.raises(RuntimeError, match="topk_indices must be contiguous"):
        torch.ops.sparse_kv_plan_op.sparse_kv_plan(*args)

    args[1] = args[1].contiguous()
    args[-3] = 1 << 31
    with pytest.raises(RuntimeError, match="exceeds int32 kernel limits"):
        torch.ops.sparse_kv_plan_op.sparse_kv_plan(*args)


@requires_npu
@pytest.mark.parametrize("case_name", ALL_CASE_NAMES)
def test_npu_matches_golden(case_name):
    """全量用例 NPU vs golden 精确比对(int32 确定性算法,要求全等)。"""
    case = build_case(case_name)
    expected = golden_sparse_kv_plan(case)
    actual = _npu_run(case)
    for key in ("current_slots", "miss_count", "miss_tokens", "miss_slots", "slot_to_token", "lru_slots",
                "last_req_ids"):
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=f"{case_name}:{key}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
