# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""两版 LRU 规划核的共同合法输入；每行对应同号 request。"""

from __future__ import annotations

import numpy as np


def make_common_case(
    *, rows: int = 32, topk: int = 2048, capacity: int = 4096,
    max_token: int = 256 * 1024, resident_fraction: float = 0.75,
    seed: int = 2026,
) -> dict:
    """构造 cold / warm 场景，所有 token、可见长度和物理 block 均合法。

    两个核语义不同：旧核不检查 visible_seq_lens / block_table，且行号就是
    request 号。共同输入必须满足这些前提，才能要求逐项输出全等。
    """
    if not (rows > 0 and 0 < topk <= capacity and topk <= max_token):
        raise ValueError("requires rows > 0 and 0 < topk <= capacity, max_token")
    if not 0 <= resident_fraction <= 1:
        raise ValueError("resident_fraction must be in [0, 1]")
    block_size = 64
    max_num_blocks = (max_token + block_size - 1) // block_size
    rng = np.random.default_rng(seed)
    topk_indices = np.stack(
        [rng.choice(max_token, size=topk, replace=False).astype(np.int32)
         for _ in range(rows)]
    )
    hits = int(topk * resident_fraction)
    req_ids = np.arange(1000, 1000 + rows, dtype=np.int64)
    slot_to_token = np.full((rows, capacity), -1, dtype=np.int32)
    slot_to_token[:, :hits] = topk_indices[:, :hits]
    return {
        "req_ids": req_ids,
        "topk_indices": topk_indices,
        "stable_prefix_lens": np.full(rows, max_token, dtype=np.int32),
        "visible_seq_lens": np.full(rows, max_token, dtype=np.int32),
        "token_to_req": np.arange(rows, dtype=np.int32),
        "block_table": np.broadcast_to(
            np.arange(max_num_blocks, dtype=np.int32), (rows, max_num_blocks)
        ).copy(),
        "active_rows": np.int32(rows),
        "last_req_ids": req_ids.copy() if hits else np.full(rows, -1, dtype=np.int64),
        "slot_to_token": slot_to_token,
        "lru_slots": np.tile(np.arange(capacity, dtype=np.int32), (rows, 1)),
        "topk": topk,
        "capacity": capacity,
        "max_token": max_token,
        "block_size": block_size,
        "host_num_blocks": max_num_blocks,
    }
