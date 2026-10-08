"""独立 NumPy 参考：Echo 的时间戳、稳定排序和 free 栈语义。"""
from __future__ import annotations
import numpy as np

I32_MAX = np.iinfo(np.int32).max
STATE_KEYS = ("htd", "dth", "pri", "free_slots", "avail", "fifo")
OUTPUT_KEYS = ("current_slots", "miss_host_pos", "miss_alloc_flat", "miss_mask")


def empty_echo_state(rows, topk, capacity, max_token):
    htd = np.full((rows, max_token), I32_MAX, np.int32)
    htd[:, 0] = 0
    pri = np.full((rows, capacity + 1), -1, np.int32)
    pri[:, 0] = I32_MAX
    free = np.zeros((rows, capacity + min(topk, capacity)), np.int32)
    free[:, :capacity] = np.arange(capacity, 0, -1, dtype=np.int32)
    return dict(htd=htd, dth=np.full_like(pri, I32_MAX), pri=pri, free_slots=free,
                avail=np.full(rows, capacity, np.int32), fifo=np.ones(rows, np.int32))


def golden_echo_lru(pos, spl, reset_mask, state, *, spec_enabled=True):
    rows, topk = pos.shape
    capacity = state["dth"].shape[1] - 1
    max_token = state["htd"].shape[1]
    eff = min(topk, capacity)
    if np.any(pos[:, :eff] < 0) or np.any(pos[:, :eff] >= max_token):
        raise ValueError("Echo token outside legal range")
    if any(len(np.unique(row[:eff])) != eff for row in pos):
        raise ValueError("Echo requires unique effective tokens per row")
    if np.any(spl < 0) or np.any(spl > max_token):
        raise ValueError("invalid stable prefix")
    if not spec_enabled and np.any(spl != max_token):
        raise ValueError("spec_enabled=False requires full stable prefixes")
    out = {key: state[key].copy() for key in STATE_KEYS}
    out.update({key: np.zeros_like(pos, dtype=np.uint8 if key == "miss_mask" else np.int32)
                for key in OUTPUT_KEYS})
    for row in range(rows):
        if reset_mask[row]:
            initial = empty_echo_state(1, topk, capacity, max_token)
            for key in STATE_KEYS:
                out[key][row] = initial[key][0]
        htd, dth, pri, free = (out[key][row] for key in ("htd", "dth", "pri", "free_slots"))
        avail, fifo = int(out["avail"][row]), int(out["fifo"][row])
        if fifo > I32_MAX - 2:
            raise ValueError("Echo fifo would overflow")
        if spec_enabled:
            invalid = np.flatnonzero((dth != I32_MAX) & (dth >= spl[row]))
            for slot in invalid:
                htd[dth[slot]] = I32_MAX
                dth[slot], pri[slot] = I32_MAX, -1
            free[avail:avail + len(invalid)] = invalid
            avail += len(invalid)
        tokens = pos[row, :eff]
        before = htd[tokens].copy()
        misses = before == I32_MAX
        nm = int(misses.sum())
        pri[before[~misses]] = fifo
        fifo += 1
        release_count = max(nm - avail, 0)
        if release_count:
            candidates = [slot for slot in range(1, capacity + 1) if pri[slot] >= 0]
            released = sorted(candidates, key=lambda slot: (int(pri[slot]), slot))[:release_count]
            assert len(released) == release_count
            for slot in released:
                if dth[slot] != I32_MAX:
                    htd[dth[slot]] = I32_MAX
                dth[slot], pri[slot] = I32_MAX, -1
            free[avail:avail + release_count] = released
            avail += release_count
        allocated = free[avail - nm:avail].copy()
        avail -= nm
        pri[allocated] = fifo
        fifo += 1
        for token, slot in zip(tokens[misses], allocated):
            htd[token] = slot
            dth[slot] = token
        out["current_slots"][row, :eff] = htd[tokens]
        out["miss_host_pos"][row, :eff] = np.where(misses, tokens, 0)
        out["miss_alloc_flat"][row, :eff] = np.where(misses, row * capacity + htd[tokens], 0)
        out["miss_mask"][row, :eff] = misses
        out["avail"][row], out["fifo"][row] = avail, fifo
    return out
