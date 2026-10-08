"""三版算法共同输入与 Echo 状态转换；转换不进入计时区间。"""
from __future__ import annotations
import numpy as np
from old_lru_cases import make_common_case
from golden_echo_lru import I32_MAX, empty_echo_state


def make_peer_common_case(*, rows=32, topk=2048, capacity=4096, max_token=256*1024,
                          resident_fraction=0.75, full_cache=False, seed=2026):
    if not (0 < topk <= capacity <= 4096 and max_token > topk):
        raise ValueError("requires topk <= capacity <= 4096 and enough nonzero tokens")
    case = make_common_case(rows=rows, topk=topk, capacity=capacity,
                            max_token=max_token, resident_fraction=resident_fraction, seed=seed)
    rng = np.random.default_rng(seed)
    case["topk_indices"] = np.stack([
        rng.choice(max_token - 1, size=topk, replace=False).astype(np.int32) + 1
        for _ in range(rows)])
    hits = int(topk * resident_fraction)
    case["slot_to_token"].fill(-1)
    case["slot_to_token"][:, :hits] = case["topk_indices"][:, :hits]
    if full_cache:
        if max_token - 1 - topk < capacity - hits:
            raise ValueError("not enough distinct tokens to fill non-hit residents")
        for row in range(rows):
            other = np.setdiff1d(np.arange(1, max_token, dtype=np.int32), case["topk_indices"][row])
            case["slot_to_token"][row, hits:] = rng.choice(other, capacity - hits, replace=False)
        case["last_req_ids"] = case["req_ids"].copy()
    return case


def echo_from_common(case):
    """0 基槽转换为 Echo 1 基槽；LRU 顺序转换成不同的历史时间戳。"""
    rows, topk = case["topk_indices"].shape
    capacity, max_token = case["capacity"], case["max_token"]
    out = empty_echo_state(rows, topk, capacity, max_token)
    for row in range(rows):
        resident = case["slot_to_token"][row]
        occupied = np.flatnonzero(resident >= 0)
        if np.any(resident[occupied] <= 0) or np.any(resident[occupied] >= max_token):
            raise ValueError("common Echo residents must be positive legal tokens")
        if len(np.unique(resident[occupied])) != len(occupied):
            raise ValueError("resident tokens must be unique")
        out["dth"][row, occupied + 1] = resident[occupied]
        out["htd"][row, resident[occupied]] = occupied + 1
        rank = np.empty(capacity, np.int32)
        rank[case["lru_slots"][row]] = np.arange(capacity, dtype=np.int32)
        # lru_slots 按最旧→最新排列；Echo 时间戳越小越先淘汰。
        out["pri"][row, occupied + 1] = rank[occupied] + 1
        empty = np.flatnonzero(resident < 0) + 1
        out["free_slots"][row].fill(0)
        out["free_slots"][row, :len(empty)] = empty[::-1]
        out["avail"][row] = len(empty)
        out["fifo"][row] = capacity + 1
    return out


def assert_echo_invariants(pos, state):
    """检查实际结果的双向映射、空闲槽唯一性和输出地址约定。"""
    rows, topk = pos.shape
    capacity = state["dth"].shape[1] - 1
    eff = min(topk, capacity)
    for row in range(rows):
        assert state["htd"][row, 0] == 0
        used = np.flatnonzero(state["dth"][row, 1:] != I32_MAX) + 1
        tokens = state["dth"][row, used]
        assert len(np.unique(tokens)) == len(tokens)
        np.testing.assert_array_equal(state["htd"][row, tokens], used)
        avail = int(state["avail"][row])
        assert 0 <= avail <= capacity
        free = state["free_slots"][row, :avail]
        assert len(np.unique(free)) == avail
        assert np.all((free > 0) & (free <= capacity))
        assert not np.intersect1d(free, used).size
        assert len(used) + avail == capacity
        np.testing.assert_array_equal(np.sort(np.flatnonzero(state["htd"][row, 1:] != I32_MAX) + 1),
                                      np.sort(tokens))
        np.testing.assert_array_equal(state["current_slots"][row, :eff], state["htd"][row, pos[row, :eff]])
        assert np.all(state["current_slots"][row, :eff] != I32_MAX)
        mask = state["miss_mask"][row].astype(bool)
        np.testing.assert_array_equal(state["miss_host_pos"][row, mask], pos[row, mask])
        np.testing.assert_array_equal(state["miss_alloc_flat"][row, mask],
                                      row * capacity + state["current_slots"][row, mask])
        assert np.all(state["current_slots"][row, eff:] == 0)
        assert np.all(state["miss_mask"][row, eff:] == 0)


def assert_common_eviction_tokens(case, echo_dth, common_slot_to_token):
    """共同单步输入下核对被淘汰的历史 token 集合，不要求分配槽号相同。

    用于显式保留相同初始历史的转换场景；不适用于两种策略多轮演化后的状态。
    """
    for row, before in enumerate(case["slot_to_token"]):
        resident = before[before >= 0]
        echo_after = echo_dth[row, 1:]
        echo_after = echo_after[echo_after != I32_MAX]
        common_after = common_slot_to_token[row]
        common_after = common_after[common_after >= 0]
        np.testing.assert_array_equal(
            np.setdiff1d(resident, echo_after), np.setdiff1d(resident, common_after),
            err_msg=f"row {row}: historical eviction tokens differ")
