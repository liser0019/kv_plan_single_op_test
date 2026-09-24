# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""开 MTP 投机推理的 SparseKvPlan 单算子测试。

结论(为什么单算子层面就能测 MTP):
SparseKvPlan 的 MTP 语义完全由输入表达,不需要真的跑模型:
- stable_prefix_lens[row] = 该行请求的"已提交前缀长度"
  (真实集成:sfa_kv_offload.py 中 actual_seq_lens_key - query_len);
- visible_seq_lens[row]  = 投机可见长度 = 已提交 + 1 个目标 token + draft_tokens;
- MTP 一步 = 每请求 (1 + draft_tokens) 行(token batch 展开,
  token_to_req 把行映射回请求,与 fused_overlap_mtp 的 flatten 一致);
- kernel 侧 stablePrefix 失效逻辑即投机回滚语义:
  resident slot 持有 >= stablePrefix 的 token 一律先失效再分类。

因此 MTP 单测 = 多步有状态驱动器:propose(带投机后缀的 visible)
→ verify/accept(随机接受 a ∈ [0, draft] 个 draft token,context += 1 + a,
被拒 token 回滚)→ 下一步。逐步比对 golden/NPU,并断言 MTP 不变量:
投机 token 的驻留只能来自"本步 miss 分配",绝不能来自历史 resident 命中。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from golden import golden_sparse_kv_plan  # noqa: E402

try:
    import torch
    import torch_npu  # noqa: F401

    NPU_AVAILABLE = torch.npu.is_available() if hasattr(torch, "npu") else False
except ImportError:
    NPU_AVAILABLE = False

requires_npu = pytest.mark.skipif(not NPU_AVAILABLE, reason="无 NPU 环境")


class MTPScenario:
    """模拟 vLLM MTP 投机解码循环的输入生成器。

    每步:
      - 每请求展开 q_len = 1 + draft_tokens 行;
      - stable[row] = context[req](已提交),visible[row] = context[req] + q_len;
      - topk 混合稳定区与投机区 token;
      - 每 3 步做一次行重排(模拟真实 serving 的行压缩/重排,触发 reset 分支)。
    """

    def __init__(self, seed: int, *, num_reqs: int = 3, draft_tokens: int = 2, topk: int = 8,
                 capacity: int = 32, max_token: int = 512, block_size: int = 32, host_num_blocks: int = 64,
                 max_num_blocks: int = 32, num_steps: int = 6):
        self.rng = np.random.default_rng(seed)
        self.num_reqs = num_reqs
        self.draft_tokens = draft_tokens
        self.topk = topk
        self.capacity = capacity
        self.max_token = max_token
        self.block_size = block_size
        self.host_num_blocks = host_num_blocks
        self.max_num_blocks = max_num_blocks
        self.num_steps = num_steps
        self.max_rows = num_reqs * (1 + draft_tokens)
        # 每请求已提交长度
        self.context = self.rng.integers(16, 64, size=num_reqs).astype(np.int64)
        # 行 → 请求的当前映射(行重排时变化)
        self.row_owner = np.repeat(np.arange(num_reqs), 1 + draft_tokens)
        # 持久状态
        self.last_req_ids = np.full(self.max_rows, -1, dtype=np.int64)
        self.slot_to_token = np.full((self.max_rows, capacity), -1, dtype=np.int32)
        self.lru_slots = np.tile(np.arange(capacity, dtype=np.int32), (self.max_rows, 1))
        # 请求 id:步间保持不变,保证"同请求续用"路径真实成立
        self.req_ids = (5000 + self.rng.integers(0, 1 << 30, size=num_reqs)).astype(np.int64)
        self.block_table = self.rng.integers(0, host_num_blocks,
                                             size=(num_reqs, max_num_blocks)).astype(np.int32)

    def step_case(self, step: int) -> dict:
        q_len = 1 + self.draft_tokens
        visible = self.context[self.row_owner] + q_len
        visible = np.minimum(visible, self.max_token).astype(np.int32)
        stable = np.minimum(self.context[self.row_owner], self.max_token).astype(np.int32)

        topk_indices = np.zeros((self.max_rows, self.topk), dtype=np.int32)
        for row in range(self.max_rows):
            # 70% 取稳定区 [0, stable),30% 取投机区 [stable, visible)
            spec_mask = self.rng.random(self.topk) < 0.3
            stable_pool = np.arange(max(int(stable[row]), 1))
            spec_pool = np.arange(int(stable[row]), int(visible[row]))
            for p in range(self.topk):
                if spec_mask[p] and len(spec_pool) > 0:
                    topk_indices[row, p] = self.rng.choice(spec_pool)
                else:
                    topk_indices[row, p] = self.rng.choice(stable_pool)

        return {
            "req_ids": self.req_ids.copy(),
            "topk_indices": topk_indices,
            "stable_prefix_lens": stable,
            "visible_seq_lens": visible,
            "token_to_req": self.row_owner.astype(np.int32),
            "block_table": self.block_table.copy(),
            "active_rows": np.int32(self.max_rows),
            "last_req_ids": self.last_req_ids.copy(),
            "slot_to_token": self.slot_to_token.copy(),
            "lru_slots": self.lru_slots.copy(),
            "topk": self.topk,
            "capacity": self.capacity,
            "max_token": self.max_token,
            "block_size": self.block_size,
            "host_num_blocks": self.host_num_blocks,
        }

    def commit(self, out: dict, step: int) -> None:
        """推进状态:随机接受 a ∈ [0, draft] 个 draft token(回滚其余)。"""
        self.last_req_ids = out["last_req_ids"].copy()
        self.slot_to_token = out["slot_to_token"].copy()
        self.lru_slots = out["lru_slots"].copy()
        accepted = self.rng.integers(0, self.draft_tokens + 1, size=self.num_reqs)
        for r in range(self.num_reqs):
            # 目标 token 恒定提交;接受的 draft 一并提交,被拒的回滚。
            self.context[r] += 1 + int(accepted[r])
        # 每 3 步行重排:行换请求 → lastReqIds 失配 → kernel reset 分支
        if (step + 1) % 3 == 0:
            self.row_owner = self.rng.permutation(self.row_owner)


def assert_mtp_semantics(case: dict, out: dict) -> None:
    """MTP 不变量:投机 token(>= stablePrefix)不可能命中历史 resident。"""
    max_token = case["max_token"]
    for row in range(out["num_rows"]):
        stable = int(case["stable_prefix_lens"][row])
        stable = 0 if stable < 0 else min(stable, max_token)
        cs = out["current_slots"][row]
        ms = out["miss_slots"][row]
        mc = int(out["miss_count"][row])
        assigned = set(int(x) for x in ms[:mc])
        for p in range(case["topk"]):
            token = int(case["topk_indices"][row, p])
            if token >= stable and cs[p] >= 0:
                # 命中的是投机 token:slot 只能来自本步 miss 分配
                assert int(cs[p]) in assigned, (
                    f"row={row} pos={p} token={token} >= stable={stable} 却命中了历史 resident slot {cs[p]}"
                )


def _run_scenario(seed: int, **kw) -> list[dict]:
    scenario = MTPScenario(seed, **kw)
    outs = []
    for step in range(scenario.num_steps):
        case = scenario.step_case(step)
        out = golden_sparse_kv_plan(case)
        assert_mtp_semantics(case, out)
        outs.append(out)
        scenario.commit(out, step)
    return outs


def test_mtp_scenario_cpu():
    """CPU 侧 MTP 多步驱动:golden 逐步推进 + MTP 语义不变量。"""
    outs = _run_scenario(seed=42)
    assert len(outs) == 6
    # 至少有一步发生过 miss 分配,保证用例真的驱动了驻留逻辑
    total_miss = sum(int(o["miss_count"][: o["num_rows"]].sum()) for o in outs)
    assert total_miss > 0


def test_mtp_visible_length_covers_full_query():
    scenario = MTPScenario(seed=42)
    case = scenario.step_case(0)
    np.testing.assert_array_equal(
        case["visible_seq_lens"] - case["stable_prefix_lens"],
        np.full(scenario.max_rows, 1 + scenario.draft_tokens, dtype=np.int32),
    )


def test_mtp_scenario_accept_and_rollback():
    """接受/回滚交替:接受步之后 stable 前移,历史投机 token 若仍在投机区
    则下一步不能作为 resident 命中(assert_mtp_semantics 逐步把关)。"""
    outs = _run_scenario(seed=7, num_reqs=2, draft_tokens=3, num_steps=5)
    assert all(o["num_rows"] > 0 for o in outs)


def test_mtp_reset_on_row_remap():
    """第 3/6 步行重排后:换了请求的行必须整行 reset——本步命中只能来自
    miss 分配,不允许出现任何历史 resident 命中。"""
    scenario = MTPScenario(seed=99, num_steps=6)
    prev_owner = scenario.row_owner.copy()
    for step in range(scenario.num_steps):
        remapped_rows = np.where(scenario.row_owner != prev_owner)[0] if step > 0 else []
        case = scenario.step_case(step)
        out = golden_sparse_kv_plan(case)
        assert_mtp_semantics(case, out)
        for row in remapped_rows:
            assigned = {int(x) for x in out["miss_slots"][row][: int(out["miss_count"][row])]}
            hit_slots = {int(x) for x in out["current_slots"][row] if x >= 0}
            assert hit_slots <= assigned, (
                f"step={step} row={row} 行重排后仍出现历史 resident 命中: {hit_slots - assigned}"
            )
        prev_owner = scenario.row_owner.copy()
        scenario.commit(out, step)


@requires_npu
def test_mtp_scenario_npu_matches_golden():
    """NPU 侧 MTP 多步:与 golden 逐步全等比对(状态跨步传递)。"""
    import sparse_kv_plan_op

    import torch

    scenario = MTPScenario(seed=42)
    state = sparse_kv_plan_op.SimtLruState.create(
        scenario.max_rows, scenario.topk, scenario.capacity, "npu")

    def to_npu(arr):
        return torch.from_numpy(np.ascontiguousarray(arr)).npu()

    for step in range(scenario.num_steps):
        case = scenario.step_case(step)
        expected = golden_sparse_kv_plan(case)
        # 仅初始化第一步；后续直接使用上一轮 NPU kernel 写出的持久状态。
        if step == 0:
            state.last_req_ids.copy_(to_npu(case["last_req_ids"]))
            state.slot_to_token.copy_(to_npu(case["slot_to_token"]))
            state.lru_slots.copy_(to_npu(case["lru_slots"]))
        state.active_rows.copy_(to_npu(np.atleast_1d(np.int32(case["active_rows"]))))
        sparse_kv_plan_op.sparse_kv_plan(
            req_ids=to_npu(case["req_ids"]),
            topk_indices=to_npu(case["topk_indices"]),
            stable_prefix_lens=to_npu(case["stable_prefix_lens"]),
            visible_seq_lens=to_npu(case["visible_seq_lens"]),
            token_to_req=to_npu(case["token_to_req"]),
            block_table=to_npu(case["block_table"]),
            active_rows=state.active_rows,
            state=state,
            max_token=case["max_token"],
            block_size=case["block_size"],
            host_num_blocks=case["host_num_blocks"],
        )
        torch.npu.synchronize()
        actual = {
            "current_slots": state.current_slots.cpu().numpy(),
            "miss_count": state.miss_count.cpu().numpy(),
            "miss_tokens": state.miss_tokens.cpu().numpy(),
            "miss_slots": state.miss_slots.cpu().numpy(),
            "slot_to_token": state.slot_to_token.cpu().numpy(),
            "lru_slots": state.lru_slots.cpu().numpy(),
            "last_req_ids": state.last_req_ids.cpu().numpy(),
            "num_rows": scenario.max_rows,
        }
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=f"mtp step{step}:{key}")
        assert_mtp_semantics(case, actual)
        scenario.commit(expected, step)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
