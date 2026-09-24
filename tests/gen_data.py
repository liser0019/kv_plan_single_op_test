# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""SparseKvPlan 单算子测试数据生成(L0/L1/L2 分级用例)。

所有用例均通过固定 seed 生成,保证可复现。warm 用例先用 golden 在随机前置
输入上跑到稳态,再以其输出状态作为被测调用的初始状态(与真实推理中
"上一 decode 步遗留状态"等价)。

用例分级(与仓库测试规范对齐):
- L0 门槛:小 shape 基础功能,执行快;
- L1 功能:典型规模/形态(对齐 vllm-ascend 稀疏 KV 典型配置);
- L2 异常/边界:activeRows 裁剪、重复 token、全非法 TopK、capacity=1、
  UB/GM workspace 路径边界、越界 LRU 项、stablePrefix 全失效、请求切换
  reset、miss 超过 evictable 等。
"""

from __future__ import annotations

import zlib

import numpy as np

from golden import golden_sparse_kv_plan


def _cold_state(max_rows: int, capacity: int, rng: np.random.Generator) -> dict:
    """冷启动状态:lastReqIds=-1 触发整行 reset;LRU 给 identity。"""
    return {
        "last_req_ids": np.full(max_rows, -1, dtype=np.int64),
        "slot_to_token": np.full((max_rows, capacity), -1, dtype=np.int32),
        "lru_slots": np.tile(np.arange(capacity, dtype=np.int32), (max_rows, 1)),
    }


def _base_inputs(
    *,
    num_reqs: int,
    max_rows: int,
    topk: int,
    max_token: int,
    max_num_blocks: int,
    seed: int,
    valid_ratio: float = 0.85,
) -> dict:
    """生成一批合法率约 valid_ratio 的请求级输入。"""
    rng = np.random.default_rng(seed)
    req_ids = (1000 + rng.integers(0, 1 << 30, size=num_reqs)).astype(np.int64)
    visible = rng.integers(1, max_token + 1, size=max_rows).astype(np.int32)
    stable = (rng.random(max_rows) * np.maximum(visible, 1)).astype(np.int32)
    # token_to_req:行 → 请求;留少量 -1 制造"行无归属"分支。
    token_to_req = rng.integers(0, num_reqs, size=max_rows).astype(np.int32)
    token_to_req[rng.random(max_rows) < 0.05] = -1
    # topk_indices:大部分落在 [0, visible),其余落在 [0, max_token) 制造非法。
    tokens = rng.integers(0, max_token, size=(max_rows, topk)).astype(np.int32)
    keep = rng.random((max_rows, topk)) < valid_ratio
    tokens = np.where(keep, np.minimum(tokens, np.maximum(visible - 1, 0)[:, None]), tokens).astype(np.int32)
    # block_table:大部分物理 block 合法,少量 -1(空洞 block)。
    block_table = rng.integers(0, max_num_blocks, size=(num_reqs, max_num_blocks)).astype(np.int32)
    block_table[rng.random(block_table.shape) < 0.05] = -1
    return {
        "req_ids": req_ids,
        "topk_indices": tokens,
        "stable_prefix_lens": stable,
        "visible_seq_lens": visible,
        "token_to_req": token_to_req,
        "block_table": block_table,
        "active_rows": np.int32(max_rows),
    }


def make_case(name: str, seed: int, *, warm_steps: int = 0, **params) -> dict:
    """组装一个完整用例(输入 + 初始状态 + 属性)。

    warm_steps > 0 时,先用随机输入跑 warm_steps 次 golden,以终态作为
    被测调用的初始状态。
    """
    num_reqs = params["num_reqs"]
    max_rows = params["max_rows"]
    topk = params["topk"]
    capacity = params["capacity"]
    max_token = params["max_token"]
    block_size = params["block_size"]
    host_num_blocks = params["host_num_blocks"]
    max_num_blocks = params.get("max_num_blocks", host_num_blocks)

    case = _base_inputs(
        num_reqs=num_reqs,
        max_rows=max_rows,
        topk=topk,
        max_token=max_token,
        max_num_blocks=max_num_blocks,
        seed=seed,
        valid_ratio=params.get("valid_ratio", 0.85),
    )
    if params.get("lru_with_oob"):
        # 初始 LRU 混入越界项(负值 / >= capacity),覆盖 kernel 的跳过分支。
        rng = np.random.default_rng(seed + 7919)
        oob = rng.random((max_rows, capacity)) < 0.1
        values = case.setdefault("_lru_init", np.tile(np.arange(capacity, dtype=np.int32), (max_rows, 1)))
        values[oob] = rng.choice([-1, capacity, capacity + 7], size=oob.sum()).astype(np.int32)

    state = _cold_state(max_rows, capacity, np.random.default_rng(seed))
    if "_lru_init" in case:
        state["lru_slots"] = case.pop("_lru_init").copy()

    case.update(state)
    case.update(
        topk=topk,
        capacity=capacity,
        max_token=max_token,
        block_size=block_size,
        host_num_blocks=host_num_blocks,
    )

    for step in range(warm_steps):
        warm = _base_inputs(
            num_reqs=num_reqs,
            max_rows=max_rows,
            topk=topk,
            max_token=max_token,
            max_num_blocks=max_num_blocks,
            seed=seed * 100 + step + 1,
            valid_ratio=params.get("valid_ratio", 0.85),
        )
        warm.update(state)
        warm.update(
            topk=topk,
            capacity=capacity,
            max_token=max_token,
            block_size=block_size,
            host_num_blocks=host_num_blocks,
        )
        out = golden_sparse_kv_plan(warm)
        state = {
            "last_req_ids": out["last_req_ids"],
            "slot_to_token": out["slot_to_token"],
            "lru_slots": out["lru_slots"],
        }
        case.update(state)

    if "active_rows_override" in params:
        case["active_rows"] = np.int32(params["active_rows_override"])
    if "stable_override" in params:
        case["stable_prefix_lens"] = np.full(max_rows, params["stable_override"], dtype=np.int32)
    if "topk_all_invalid" in params:
        # 全部 token 指向非法物理 block。
        case["block_table"] = np.full((num_reqs, max_num_blocks), -1, dtype=np.int32)
    return case


# ---------------------------------------------------------------------------
# 用例表:名字 → 构造参数
# ---------------------------------------------------------------------------
CASE_PARAMS: dict[str, dict] = {
    # ---- L0 门槛 ----
    "l0_basic": dict(
        num_reqs=1, max_rows=2, topk=2, capacity=4, max_token=16,
        block_size=4, host_num_blocks=8, max_num_blocks=4,
    ),
    "l0_single_row": dict(
        num_reqs=2, max_rows=1, topk=4, capacity=8, max_token=64,
        block_size=8, host_num_blocks=16, max_num_blocks=8,
    ),
    # ---- L1 功能 ----
    "l1_small": dict(
        num_reqs=4, max_rows=8, topk=32, capacity=64, max_token=256,
        block_size=16, host_num_blocks=64, max_num_blocks=32,
    ),
    "l1_typical": dict(
        num_reqs=8, max_rows=16, topk=64, capacity=128, max_token=1024,
        block_size=16, host_num_blocks=256, max_num_blocks=64,
    ),
    "l1_warm_two_steps": dict(
        num_reqs=4, max_rows=8, topk=32, capacity=64, max_token=512,
        block_size=16, host_num_blocks=128, max_num_blocks=32, warm_steps=2,
    ),
    # ---- L2 异常/边界 ----
    "l2_active_rows_zero": dict(
        num_reqs=2, max_rows=4, topk=8, capacity=16, max_token=128,
        block_size=16, host_num_blocks=32, max_num_blocks=8,
        active_rows_override=0,
    ),
    "l2_active_rows_overflow": dict(
        num_reqs=2, max_rows=4, topk=8, capacity=16, max_token=128,
        block_size=16, host_num_blocks=32, max_num_blocks=8,
        active_rows_override=999,  # 超过 max_rows → 截断为 max_rows
    ),
    "l2_dup_heavy": dict(
        num_reqs=3, max_rows=8, topk=16, capacity=32, max_token=128,
        block_size=8, host_num_blocks=32, max_num_blocks=16,
        valid_ratio=0.6,  # 高重复率:同一 token 多次出现在 TopK
    ),
    "l2_all_invalid_topk": dict(
        num_reqs=2, max_rows=4, topk=8, capacity=16, max_token=128,
        block_size=16, host_num_blocks=32, max_num_blocks=8,
        topk_all_invalid=True,
    ),
    "l2_capacity_one": dict(
        num_reqs=1, max_rows=2, topk=1, capacity=1, max_token=32,
        block_size=8, host_num_blocks=8, max_num_blocks=4,
    ),
    "l2_stable_prefix_full": dict(
        num_reqs=2, max_rows=4, topk=8, capacity=16, max_token=128,
        block_size=16, host_num_blocks=32, max_num_blocks=8,
        warm_steps=1, stable_override=10 ** 9,  # clamp 到 max_token → 全部失效
    ),
    "l2_req_change_reset": dict(
        num_reqs=3, max_rows=6, topk=8, capacity=16, max_token=256,
        block_size=16, host_num_blocks=32, max_num_blocks=16,
        warm_steps=1,  # warm 后再换一批 req_ids → 触发整行 reset
        req_ids_change=True,
    ),
    "l2_lru_oob_entries": dict(
        num_reqs=2, max_rows=4, topk=8, capacity=16, max_token=128,
        block_size=16, host_num_blocks=32, max_num_blocks=8,
        lru_with_oob=True,
    ),
    "l2_miss_exceed_evictable": dict(
        num_reqs=1, max_rows=2, topk=16, capacity=2, max_token=64,
        block_size=8, host_num_blocks=8, max_num_blocks=8,
        # topk(16) > capacity(2):miss 一定超过 evictable,验证未分配保持 -1
    ),
    # ---- workspace 路径边界(单位换算见 golden/sparse_kv_plan_launch.h) ----
    # T=2048 → H=4096;rowElements = 3*4096 + 2*7168 + 2048 = 28672,
    # 行字节数 114688 = 112 KiB 上限 → 恰好仍走 UB 快路径。
    "l2_ub_boundary_exact": dict(
        num_reqs=2, max_rows=4, topk=2048, capacity=7168, max_token=8192,
        block_size=64, host_num_blocks=256, max_num_blocks=128,
    ),
    # T=2048, C=7169 → 行字节超过 112 KiB → 走 GM compactWorkspace 路径。
    "l2_gm_workspace": dict(
        num_reqs=2, max_rows=4, topk=2048, capacity=7169, max_token=8192,
        block_size=64, host_num_blocks=256, max_num_blocks=128,
    ),
}


def build_case(name: str) -> dict:
    params = dict(CASE_PARAMS[name])
    # 注:Python 内建 hash(name) 跨进程随机化,用 CRC32 保证用例可复现。
    seed = zlib.crc32(name.encode("utf-8"))
    if params.pop("req_ids_change", False):
        # warm 后替换 req_ids,保证 lastReqIds 全部失配。
        case = make_case(name, seed, **params)
        rng = np.random.default_rng(seed + 1)
        case["req_ids"] = (2000 + rng.integers(0, 1 << 30, size=case["req_ids"].shape[0])).astype(np.int64)
        return case
    return make_case(name, seed, **params)


ALL_CASE_NAMES = list(CASE_PARAMS.keys())


if __name__ == "__main__":
    for case_name in ALL_CASE_NAMES:
        case = build_case(case_name)
        out = golden_sparse_kv_plan(case)
        print(f"{case_name}: rows={out['num_rows']} miss_total={int(out['miss_count'][:out['num_rows']].sum())}")
