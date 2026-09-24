# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""SparseKvPlan 的 CPU golden 参考实现(纯 numpy,独立于 kernel)。

golden 按 kernel(sparse_kv_plan_kernel.cpp)的八个阶段逐行推导,但刻意不复用
kernel 代码路径;hash 桶内部布局不在可观测输出里,golden 直接以集合与顺序
语义等价实现:

- firstPos(token) = token 在合法 TopK 位置中的最小下标
  (kernel: InsertTopkToken 的 asc_atomic_min)。
- owner(token) = 命中该 token 的旧 LRU 项中的最大 order
  (kernel: hashOwner 的 asc_atomic_max)。
- evictSlots / hitSlots 均按旧 LRU 扫描序稳定排列
  (kernel: 打包 scan 的 evictRank/hitRank)。
- missPositions 按 TopK 原顺序压缩(kernel: 阶段 6 scan)。

必须保留的业务语义(与 kernel 头注释一致):
- LRU 从旧到新排列;miss 优先用最老的可淘汰 slot(空 slot 亦视为可淘汰)。
- 同一 token 重复出现在 TopK:仅第一次位置拿 resident hit,后续重复位置
  仍按 miss 处理(可能各自分到 slot)。
- 同 token 驻留多个 slot:旧 LRU 最后一个 owner 提供命中 slot,这些 slot
  全部保留在 hit 段。
- missCount 只统计实际分到 slot 的 miss;未分配的 missTokens/missSlots 与
  currentSlots 一起保持 -1。
- stablePrefix 之后的 KV(投机后缀)不可靠:resident 侧命中前先失效。
- 行换 request(lastReqIds 不匹配)时整行重置:slotToToken=-1、LRU=identity。
- 越界 LRU 项(slot 不在 [0, C))跳过,不计入 evict/hit;LRU 只覆写前
  evictCount+hitCount 个表项,尾部保留旧值。
- activeRows <= 0 时本行不处理;> maxRows 时按 maxRows 截断。
"""

from __future__ import annotations

import numpy as np


def golden_sparse_kv_plan(case: dict) -> dict:
    """执行一次 SparseKvPlan 的 golden 计算。

    Parameters
    ----------
    case: dict,包含以下键(numpy 数组):
        req_ids            int64 [max_requests]
        topk_indices       int32 [max_rows, topk]
        stable_prefix_lens int32 [max_rows]
        visible_seq_lens   int32 [max_rows]
        token_to_req       int32 [max_rows]
        block_table        int32 [max_requests, max_num_blocks]
        active_rows        int32 标量(数组或 python int)
        last_req_ids       int64 [max_rows]            (调用前状态)
        slot_to_token      int32 [max_rows, capacity]  (调用前状态)
        lru_slots          int32 [max_rows, capacity]  (调用前状态)
      以及标量属性:topk, capacity, max_token, block_size, host_num_blocks

    Returns
    -------
    dict:current_slots/miss_count/miss_tokens/miss_slots(本步输出),
         slot_to_token/lru_slots/last_req_ids(更新后的状态),num_rows。
         未处理的行保持调用前的值(输出张量在副本上初始化为 -1)。
    """
    req_ids = np.asarray(case["req_ids"])
    topk_indices = np.asarray(case["topk_indices"])
    stable_prefix_lens = np.asarray(case["stable_prefix_lens"])
    visible_seq_lens = np.asarray(case["visible_seq_lens"])
    token_to_req = np.asarray(case["token_to_req"])
    block_table = np.asarray(case["block_table"])
    active_rows = int(np.asarray(case["active_rows"]).reshape(-1)[0])
    last_req_ids_in = np.asarray(case["last_req_ids"]).astype(np.int64).copy()
    slot_to_token_in = np.asarray(case["slot_to_token"]).astype(np.int32).copy()
    lru_slots_in = np.asarray(case["lru_slots"]).astype(np.int32).copy()

    topk = int(case["topk"])
    capacity = int(case["capacity"])
    max_token = int(case["max_token"])
    block_size = int(case["block_size"])
    host_num_blocks = int(case["host_num_blocks"])

    max_rows, topk_dim = topk_indices.shape
    assert topk_dim == topk, "topk_indices.size(1) 必须等于 topk 属性"
    max_requests = req_ids.shape[0]
    max_num_blocks = block_table.shape[1]

    # 输出/状态副本:kernel 只写 rows < num_rows,其余行保持 -1 初值。
    current_slots = np.full((max_rows, topk), -1, dtype=np.int32)
    miss_count = np.zeros(max_rows, dtype=np.int32)
    miss_tokens = np.full((max_rows, topk), -1, dtype=np.int32)
    miss_slots = np.full((max_rows, topk), -1, dtype=np.int32)
    slot_to_token = slot_to_token_in
    lru_slots = lru_slots_in
    last_req_ids = last_req_ids_in

    # kernel 外壳:requestedRows <= 0 → 0 行;否则 min(requestedRows, maxRows)。
    if active_rows <= 0:
        num_rows = 0
    else:
        num_rows = min(active_rows, max_rows)

    def valid_source_token(row: int, token: int) -> bool:
        """对应 kernel IsSourceTokenValid。"""
        if token < 0 or token >= max_token or token >= int(visible_seq_lens[row]):
            return False
        request = int(token_to_req[row])
        if request < 0 or request >= max_requests or block_size <= 0:
            return False
        block_id = token // block_size
        if block_id < 0 or block_id >= max_num_blocks:
            return False
        physical_block = int(block_table[request, block_id])
        return 0 <= physical_block < host_num_blocks

    for row in range(num_rows):
        # ---- 阶段 1/2:reset 判定与整行重置 ----
        request = int(token_to_req[row])
        current_req_id = int(req_ids[request]) if 0 <= request < max_requests else -1
        if last_req_ids[row] != current_req_id:
            slot_to_token[row, :] = -1
            lru_slots[row, :] = np.arange(capacity, dtype=np.int32)
            last_req_ids[row] = current_req_id

        stable_prefix = int(stable_prefix_lens[row])
        stable_prefix = 0 if stable_prefix < 0 else min(stable_prefix, max_token)

        # ---- 阶段 3:合法 TopK token → firstPos(最小位置) ----
        first_pos: dict[int, int] = {}
        for pos in range(topk):
            token = int(topk_indices[row, pos])
            if valid_source_token(row, token):
                if token not in first_pos:
                    first_pos[token] = pos

        # ---- 阶段 4:按旧 LRU 扫描序稳定分类 ----
        old_lru = lru_slots[row].copy()
        evictable_slots: list[int] = []
        hit_slots: list[int] = []
        owner_order: dict[int, int] = {}
        for order in range(capacity):
            slot = int(old_lru[order])
            if slot < 0 or slot >= capacity:
                continue  # 越界 LRU 项:跳过
            token = int(slot_to_token[row, slot])
            # 投机后缀(stablePrefix 之后)先失效,再判 hit。
            if 0 <= token < max_token and token >= stable_prefix:
                slot_to_token[row, slot] = -1
                token = -1
            if 0 <= token < max_token and token in first_pos:
                hit_slots.append(slot)
                owner_order[token] = order  # order 递增,最后一次写入即 max
            else:
                evictable_slots.append(slot)

        # ---- 阶段 5:owner(旧 LRU 最后一个命中 order)发布 resident hit ----
        current_slots[row, :] = -1
        for token, pos in first_pos.items():
            if token in owner_order:
                current_slots[row, pos] = old_lru[owner_order[token]]

        # ---- 阶段 6:按 TopK 原顺序压缩 miss 位置 ----
        miss_positions = [
            pos
            for pos in range(topk)
            if int(topk_indices[row, pos]) in first_pos and current_slots[row, pos] < 0
        ]

        # ---- 阶段 7:miss 只分配到可容纳的前缀(最老 evictable 优先) ----
        assign_count = min(len(miss_positions), len(evictable_slots))
        for i in range(assign_count):
            slot = evictable_slots[i]
            pos = miss_positions[i]
            token = int(topk_indices[row, pos])
            slot_to_token[row, slot] = token
            current_slots[row, pos] = slot
            miss_tokens[row, i] = token
            miss_slots[row, i] = slot

        # ---- 阶段 8:LRU = 未分配 evictable + 新 miss + 原 hit ----
        new_lru_prefix = (
            evictable_slots[assign_count:]
            + [int(miss_slots[row, i]) for i in range(assign_count)]
            + hit_slots
        )
        # 只覆写前缀;尾部保留调用前的旧值(与 kernel 一致)。
        lru_slots[row, : len(new_lru_prefix)] = new_lru_prefix
        miss_count[row] = assign_count

    return {
        "current_slots": current_slots,
        "miss_count": miss_count,
        "miss_tokens": miss_tokens,
        "miss_slots": miss_slots,
        "slot_to_token": slot_to_token,
        "lru_slots": lru_slots,
        "last_req_ids": last_req_ids,
        "num_rows": num_rows,
    }
