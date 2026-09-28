# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""MemFabric_Hybrid 旧 AIV LRU compact 的独立 CPU 参考实现。

旧核没有来源 token 检查；miss 分配不足时，miss_tokens 的尾部仍保留候选
token，尽管 miss_count 只统计成功分配的前缀。这与新版 Plan 不同。
"""

from __future__ import annotations

import numpy as np


def golden_old_lru(case: dict) -> dict:
    req_ids = np.asarray(case["req_ids"])
    topk_indices = np.asarray(case["topk_indices"])
    stable_prefix = np.asarray(case["stable_prefix_lens"])
    last_req_ids = np.asarray(case["last_req_ids"]).copy()
    slot_to_token = np.asarray(case["slot_to_token"]).copy()
    lru_slots = np.asarray(case["lru_slots"]).copy()
    rows, topk = topk_indices.shape
    capacity = slot_to_token.shape[1]
    max_token = int(case["max_token"])
    current_slots = np.full((rows, topk), -1, dtype=np.int32)
    miss_count = np.zeros(rows, dtype=np.int32)
    miss_tokens = np.full((rows, topk), -1, dtype=np.int32)
    miss_slots = np.full((rows, topk), -1, dtype=np.int32)

    for row in range(rows):
        if last_req_ids[row] != req_ids[row]:
            slot_to_token[row] = -1
            lru_slots[row] = np.arange(capacity, dtype=np.int32)
            last_req_ids[row] = req_ids[row]
        stable = max(0, min(int(stable_prefix[row]), max_token))
        first_pos = {}
        for pos, token in enumerate(topk_indices[row]):
            token = int(token)
            if 0 <= token < max_token:
                first_pos.setdefault(token, pos)

        evictable = []
        hits = []
        for slot in lru_slots[row].copy():
            slot = int(slot)
            if not 0 <= slot < capacity:
                continue
            token = int(slot_to_token[row, slot])
            if 0 <= token < max_token and token >= stable:
                slot_to_token[row, slot] = -1
                token = -1
            if 0 <= token < max_token and token in first_pos:
                current_slots[row, first_pos[token]] = slot  # 最后一个 owner 覆盖
                hits.append(slot)
            else:
                evictable.append(slot)

        misses = [pos for pos, token in enumerate(topk_indices[row])
                  if 0 <= int(token) < max_token and current_slots[row, pos] < 0]
        for idx, pos in enumerate(misses):
            miss_tokens[row, idx] = topk_indices[row, pos]
        assigned = min(len(misses), len(evictable))
        for idx in range(assigned):
            slot = evictable[idx]
            pos = misses[idx]
            slot_to_token[row, slot] = topk_indices[row, pos]
            current_slots[row, pos] = slot
            miss_slots[row, idx] = slot
        prefix = evictable[assigned:] + evictable[:assigned] + hits
        lru_slots[row, :len(prefix)] = prefix
        miss_count[row] = assigned

    return {"last_req_ids": last_req_ids, "slot_to_token": slot_to_token,
            "lru_slots": lru_slots, "current_slots": current_slots,
            "miss_count": miss_count, "miss_tokens": miss_tokens,
            "miss_slots": miss_slots}
