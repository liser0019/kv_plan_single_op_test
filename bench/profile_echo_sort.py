"""Echo 原版与跳过零淘汰排序版：同输入、独立 torch_npu.profiler traces。"""
from __future__ import annotations
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from compare_three import setup_echo
from echo_lru_cases import make_peer_common_case
from profiler_kernel_summary import read_kernel_samples, summarize_kernel

SCHEDULE_WARMUP = SCHEDULE_ACTIVE = 5
SCENARIOS = {"cold_reset": (0.0, False), "warm_75pct": (0.75, False),
             "warm_all_hit": (1.0, False), "full_cache_75pct": (0.75, True)}
KERNELS = {"original": "echo_lru_kernel", "skip_sort": "echo_lru_skip_sort_kernel"}


def report_text(metadata, results):
    lines = ["# Echo 零淘汰排序实验", "", "## 采集条件", "",
             "```json", json.dumps(metadata, ensure_ascii=False, indent=2), "```", "",
             "统计仅来自目标 kernel 的 device duration；恢复任务、TensorMove 和 tiling 上传不相加。",
             "", "## Kernel duration（μs）", "",
             "| 场景 | 版本 | 样本数 | P50 | P95 | 最小 | 最大 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for scenario, pair in results.items():
        for mode, item in pair.items():
            s = item["stats"]
            lines.append(f"| {scenario} | {mode} | {s['count']} | {s['p50_us']:.2f} | {s['p95_us']:.2f} | {s['min_us']:.2f} | {s['max_us']:.2f} |")
    lines += ["", "## 对照", "", "| 场景 | 原版/实验版 P50 | P50 节省（μs） |",
              "|---|---:|---:|"]
    for scenario, pair in results.items():
        original, skipped = (pair[mode]["stats"]["p50_us"] for mode in ("original", "skip_sort"))
        lines.append(f"| {scenario} | {original/skipped:.3f} | {original-skipped:.2f} |")
    lines += ["", "## 解释", "",
              "- 正确性检查通过后才采集；两版每步恢复相同的初始状态。",
              "- 节省值反映整条被跳过路径的净收益，包含排序、访存、内部同步和编译后控制流影响。",
              "- 满缓存场景仍执行排序，可作为对照；5 个 active 样本仅用于初步定位，建议重复独立采集。",
              "", "## 原始采样", ""]
    for scenario, pair in results.items():
        for mode, item in pair.items():
            lines.append(f"- {scenario}/{mode}: `{item['csv']}`；samples_us={item['samples_us']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS) + ("all",), default="warm_75pct")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--max-token", type=int, default=256*1024)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--trace-root", type=Path, default=Path("results/echo_sort_profiler"))
    parser.add_argument("--original-kernel-name", default=KERNELS["original"],
                        help="CSV 中原版 kernel 的唯一名称子串")
    parser.add_argument("--skip-kernel-name", default=KERNELS["skip_sort"],
                        help="CSV 中实验版 kernel 的唯一名称子串")
    parser.add_argument("--reverse-order", action="store_true", help="先采集实验版，再采集原版")
    args = parser.parse_args()
    try:
        import torch
        import torch_npu
        import sparse_kv_plan_op as op
    except ImportError as exc:
        parser.error(f"需要 torch_npu 和实验分支的扩展: {exc}")
    if not torch.npu.is_available():
        parser.error("需要可用 NPU")
    torch.npu.set_device(args.device)
    for symbol in ("echo_lru", "echo_lru_skip_sort"):
        if not hasattr(torch.ops.sparse_kv_plan_op, symbol):
            parser.error(f"缺少 {symbol}，请重新编译当前分支")
    kernel_names = {"original": args.original_kernel_name, "skip_sort": args.skip_kernel_name}
    if (not all(name.strip() for name in kernel_names.values())
            or kernel_names["original"] in kernel_names["skip_sort"]
            or kernel_names["skip_sort"] in kernel_names["original"]):
        parser.error("两个 kernel 名称筛选项必须非空且可区分")
    args.trace_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=args.trace_root.resolve()))
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    metadata = dict(device=torch.npu.get_device_name(args.device), device_index=args.device,
                    torch=torch.__version__, torch_npu=torch_npu.__version__,
                    commit=commit.stdout.strip() if commit.returncode == 0 else "unknown",
                    shape=dict(rows=args.rows, topk=args.topk, capacity=args.capacity, max_token=args.max_token),
                    scenarios=list(SCENARIOS) if args.scenario == "all" else [args.scenario],
                    spec_enabled=False,
                    seed=args.seed, wait=0, warmup=SCHEDULE_WARMUP, active=SCHEDULE_ACTIVE, repeat=1,
                    order=["skip_sort", "original"] if args.reverse_order else ["original", "skip_sort"],
                    metric="target kernel device duration in microseconds", kernel_names=kernel_names)
    (run_dir / "conditions.md").write_text("```json\n" + json.dumps(metadata, ensure_ascii=False, indent=2) + "\n```\n")
    results = {}
    labels = tuple(SCENARIOS) if args.scenario == "all" else (args.scenario,)
    profiler = torch_npu.profiler
    for label in labels:
        fraction, full = SCENARIOS[label]
        case = make_peer_common_case(rows=args.rows, topk=args.topk, capacity=args.capacity,
                                     max_token=args.max_token, resident_fraction=fraction,
                                     full_cache=full, seed=args.seed)
        case["resident_fraction"] = fraction
        results[label] = {}
        for mode in metadata["order"]:
            state, call, reset = setup_echo(torch, op, case, skip_zero_free_size=mode == "skip_sort")
            trace_dir = run_dir / label / mode
            with profiler.profile(
                activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
                schedule=profiler.schedule(wait=0, warmup=SCHEDULE_WARMUP, active=SCHEDULE_ACTIVE, repeat=1),
                on_trace_ready=profiler.tensorboard_trace_handler(str(trace_dir), async_mode=False),
                record_shapes=False, profile_memory=False, with_stack=False,
                experimental_config=profiler._ExperimentalConfig(
                    profiler_level=profiler.ProfilerLevel.Level0,
                    aic_metrics=profiler.AiCMetrics.AiCoreNone,
                    export_type=profiler.ExportType.Text)) as prof:
                for _ in range(SCHEDULE_WARMUP + SCHEDULE_ACTIVE):
                    with torch.autograd.profiler.record_function("EchoSortExperiment/restore"):
                        reset(state)
                    with torch.autograd.profiler.record_function(f"EchoSortExperiment/{mode}"):
                        call()
                        torch.npu.synchronize()
                    prof.step()
            csv_files = list(trace_dir.rglob("kernel_details.csv"))
            if len(csv_files) != 1:
                raise RuntimeError(f"Expected one kernel_details.csv under {trace_dir}, got {len(csv_files)}; inspect retained traces")
            csv_path = csv_files[0]
            samples = read_kernel_samples(csv_path, kernel_names[mode], expected_count=SCHEDULE_ACTIVE)
            stats = summarize_kernel(samples)
            results[label][mode] = dict(csv=str(csv_path), samples_us=samples, stats=stats)
            print(f"{label}/{mode}: kernel P50 {stats['p50_us']:.2f} us, P95 {stats['p95_us']:.2f} us", flush=True)
    report = report_text(metadata, results)
    report_path = run_dir / "comparison.md"
    report_path.write_text(report)
    print(report)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
