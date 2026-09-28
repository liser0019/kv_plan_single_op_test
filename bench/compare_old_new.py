# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Ascend 950 上比较旧 AIV LRU compact 与新版 SIMT SparseKvPlan。

例：python bench/compare_old_new.py --warmup 10 --repeat 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from golden import golden_sparse_kv_plan  # noqa: E402
from old_lru_cases import make_common_case  # noqa: E402

OUTPUT_KEYS = ("last_req_ids", "slot_to_token", "lru_slots", "current_slots",
               "miss_count", "miss_tokens", "miss_slots")


def _to_npu(torch, value):
    return torch.from_numpy(np.ascontiguousarray(value)).npu()


def _setup(torch, op, case):
    rows, topk = case["topk_indices"].shape
    capacity, max_token = case["capacity"], case["max_token"]
    inputs = {key: _to_npu(torch, case[key]) for key in
              ("req_ids", "topk_indices", "stable_prefix_lens", "visible_seq_lens",
               "token_to_req", "block_table")}
    old = op.OldLruState.create(rows, topk, capacity, max_token, "npu")
    new = op.SimtLruState.create(rows, topk, capacity, "npu")
    initial = {key: _to_npu(torch, case[key]) for key in
               ("last_req_ids", "slot_to_token", "lru_slots")}
    new.active_rows.fill_(rows)

    def reset(state):
        for key, source in initial.items():
            getattr(state, key).copy_(source)
        torch.npu.synchronize()

    def call_old():
        op.old_lru_compact(req_ids=inputs["req_ids"],
                           topk_indices=inputs["topk_indices"],
                           stable_prefix_lens=inputs["stable_prefix_lens"],
                           state=old, max_token=max_token)

    def call_new():
        op.sparse_kv_plan(
            req_ids=inputs["req_ids"], topk_indices=inputs["topk_indices"],
            stable_prefix_lens=inputs["stable_prefix_lens"],
            visible_seq_lens=inputs["visible_seq_lens"],
            token_to_req=inputs["token_to_req"], block_table=inputs["block_table"],
            active_rows=new.active_rows, state=new, max_token=max_token,
            block_size=case["block_size"], host_num_blocks=case["host_num_blocks"])

    return {"old_aiv": (old, call_old), "new_simt": (new, call_new)}, reset


def _check_correctness(torch, runners, reset, case):
    expected = golden_sparse_kv_plan(case)
    for name, (state, call) in runners.items():
        reset(state)
        call()
        torch.npu.synchronize()
        for key in OUTPUT_KEYS:
            np.testing.assert_array_equal(getattr(state, key).cpu().numpy(),
                                          expected[key], err_msg=f"{name}:{key}")


def _measure(torch, state, call, reset):
    reset(state)  # 状态复制及同步均在计时区间外
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    before = time.perf_counter_ns()
    start.record()
    call()
    end.record()
    end.synchronize()
    host_us = (time.perf_counter_ns() - before) / 1000.0
    device_us = start.elapsed_time(end) * 1000.0
    return device_us, host_us


def _summary(samples):
    values = np.asarray(samples, dtype=np.float64)
    return {"p50_us": float(np.percentile(values, 50)),
            "p95_us": float(np.percentile(values, 95)),
            "min_us": float(values.min())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--max-token", type=int, default=256 * 1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", type=Path, help="可选：保存结果 JSON")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("warmup must be >= 0 and repeat must be > 0")

    try:
        import torch
        import torch_npu  # noqa: F401
        import sparse_kv_plan_op as op
    except ImportError as exc:
        parser.error(f"需要已安装的 torch、torch_npu、sparse_kv_plan_op: {exc}")
    if not torch.npu.is_available():
        parser.error("未检测到 NPU；请在 Ascend 950 服务器上运行")
    if not hasattr(torch.ops.sparse_kv_plan_op, "old_lru_compact"):
        parser.error("扩展缺少旧核；请重新执行 bash build.sh --soc=ascend950 --install")

    result = {"device": torch.npu.get_device_name(0), "shape": {
        "rows": args.rows, "topk": args.topk, "capacity": args.capacity,
        "max_token": args.max_token}, "warmup": args.warmup,
        "repeat": args.repeat, "scenarios": {}}
    for label, fraction in (("cold", 0.0), ("warm_75pct", 0.75), ("warm_all_hit", 1.0)):
        case = make_common_case(rows=args.rows, topk=args.topk,
                                capacity=args.capacity, max_token=args.max_token,
                                resident_fraction=fraction, seed=args.seed)
        runners, reset = _setup(torch, op, case)
        _check_correctness(torch, runners, reset, case)
        for _ in range(args.warmup):
            for state, call in runners.values():
                _measure(torch, state, call, reset)

        samples = {name: {"device": [], "host": []} for name in runners}
        for iteration in range(args.repeat):
            names = list(runners)
            if iteration % 2:
                names.reverse()
            for name in names:
                state, call = runners[name]
                device_us, host_us = _measure(torch, state, call, reset)
                samples[name]["device"].append(device_us)
                samples[name]["host"].append(host_us)
        stats = {name: {"device": _summary(s["device"]),
                        "host": _summary(s["host"])} for name, s in samples.items()}
        stats["new_vs_old_device_speedup"] = (
            stats["old_aiv"]["device"]["p50_us"] / stats["new_simt"]["device"]["p50_us"])
        result["scenarios"][label] = stats
        print(f"{label}: old {stats['old_aiv']['device']['p50_us']:.2f} us, "
              f"new {stats['new_simt']['device']['p50_us']:.2f} us, "
              f"new speedup {stats['new_vs_old_device_speedup']:.2f}x")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
