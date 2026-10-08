"""Ascend 950: old AIV / new SIMT / peer Echo 三方正确性与延迟对比。"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from compare_old_new import _setup, _check_correctness, _measure, _summary, _to_npu
from golden_echo_lru import golden_echo_lru, STATE_KEYS, OUTPUT_KEYS
from echo_lru_cases import (make_peer_common_case, echo_from_common, assert_echo_invariants,
                            assert_common_eviction_tokens)
from golden import golden_sparse_kv_plan


def setup_echo(torch, op, case, *, skip_zero_free_size=False):
    rows, topk = case["topk_indices"].shape
    state = op.EchoLruState.create(rows, topk, case["capacity"], case["max_token"], "npu")
    initial_cpu = echo_from_common(case)
    initial = {key: _to_npu(torch, value) for key, value in initial_cpu.items()}
    pos = _to_npu(torch, case["topk_indices"])
    spl = _to_npu(torch, case["stable_prefix_lens"])
    reset_cpu = (case["req_ids"] != case["last_req_ids"]).astype(np.uint8)
    reset_mask = _to_npu(torch, reset_cpu)
    spec = bool(np.any(case["stable_prefix_lens"] < case["max_token"]))

    def reset(_state):
        for key, value in initial.items():
            getattr(state, key).copy_(value)
        torch.npu.synchronize()

    operator = op.echo_lru_skip_sort if skip_zero_free_size else op.echo_lru

    def call():
        operator(pos=pos, stable_prefix_lens=spl, reset_mask=reset_mask,
                    state=state, spec_enabled=spec)

    reset(state)
    call()
    torch.npu.synchronize()
    expected = golden_echo_lru(case["topk_indices"], case["stable_prefix_lens"],
                               reset_cpu, initial_cpu, spec_enabled=spec)
    actual = {key: getattr(state, key).cpu().numpy() for key in STATE_KEYS + OUTPUT_KEYS}
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=f"peer_echo:{key}")
    assert_echo_invariants(case["topk_indices"], actual)
    assert_common_eviction_tokens(case, actual["dth"], golden_sparse_kv_plan(case)["slot_to_token"])
    hits = int(topk * case["resident_fraction"])
    np.testing.assert_array_equal(actual["miss_mask"].sum(axis=1), np.full(rows, topk - hits))
    return state, call, reset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--max-token", type=int, default=256*1024)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--include-echo-skip-sort", action="store_true",
                        help="增加第四路 Echo 无需淘汰时跳过排序的实验版")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("warmup must be >= 0 and repeat > 0")
    try:
        import torch
        import torch_npu
        import sparse_kv_plan_op as op
    except ImportError as exc:
        parser.error(f"需要 torch_npu 和本次构建的扩展: {exc}")
    if not torch.npu.is_available():
        parser.error("需要 Ascend NPU；CPU 无法给出真实性能")
    torch.npu.set_device(args.device)
    for name in ("old_lru_compact", "sparse_kv_plan", "echo_lru"):
        if not hasattr(torch.ops.sparse_kv_plan_op, name):
            parser.error(f"扩展缺少 {name}，请重新构建")
    if args.include_echo_skip_sort and not hasattr(torch.ops.sparse_kv_plan_op, "echo_lru_skip_sort"):
        parser.error("扩展缺少 echo_lru_skip_sort，请重新构建实验分支")
    result = dict(device=torch.npu.get_device_name(args.device), device_index=args.device,
                  torch=torch.__version__, torch_npu=torch_npu.__version__,
                  shape=dict(rows=args.rows, topk=args.topk, capacity=args.capacity, max_token=args.max_token),
                  timing="NPU event covers public op API stream interval (including tiling H2D); host includes synchronization; resets excluded",
                  warmup=args.warmup, repeat=args.repeat, seed=args.seed, scenarios={})
    scenarios = (("cold_reset", 0.0, False), ("warm_75pct", 0.75, False),
                 ("warm_all_hit", 1.0, False), ("full_cache_75pct", 0.75, True))
    for label, fraction, full in scenarios:
        case = make_peer_common_case(rows=args.rows, topk=args.topk, capacity=args.capacity,
                                     max_token=args.max_token, resident_fraction=fraction,
                                     full_cache=full, seed=args.seed)
        case["resident_fraction"] = fraction
        base, reset = _setup(torch, op, case)
        _check_correctness(torch, base, reset, case)
        runners = {name: (state, call, reset) for name, (state, call) in base.items()}
        runners["peer_echo"] = setup_echo(torch, op, case)
        if args.include_echo_skip_sort:
            runners["peer_echo_skip_sort"] = setup_echo(torch, op, case, skip_zero_free_size=True)
        for _ in range(args.warmup):
            for state, call, restore in runners.values():
                _measure(torch, state, call, restore)
        samples = {name: dict(device=[], host=[]) for name in runners}
        names = list(runners)
        for iteration in range(args.repeat):
            offset = iteration % len(names)
            order = names[offset:] + names[:offset]
            if iteration % 2:
                order = order[::-1]
            for name in order:
                state, call, restore = runners[name]
                device_us, host_us = _measure(torch, state, call, restore)
                samples[name]["device"].append(device_us)
                samples[name]["host"].append(host_us)
        stats = {name: {kind: _summary(values) for kind, values in sample.items()}
                 for name, sample in samples.items()}
        for name, (state, _, _) in runners.items():
            stats[name]["state_bytes"] = sum(value.numel() * value.element_size()
                                              for value in state if torch.is_tensor(value))
            stats[name]["raw_samples_us"] = samples[name]
        stats["peer_vs_new_event_ratio"] = stats["peer_echo"]["device"]["p50_us"] / stats["new_simt"]["device"]["p50_us"]
        if args.include_echo_skip_sort:
            original = stats["peer_echo"]["device"]["p50_us"]
            skipped = stats["peer_echo_skip_sort"]["device"]["p50_us"]
            stats["echo_sort_skip_event_speedup"] = original / skipped
            stats["echo_sort_skip_event_saved_us"] = original - skipped
        result["scenarios"][label] = stats
        print(label + ": " + ", ".join(f"{name} {stats[name]['device']['p50_us']:.2f} us" for name in names))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
