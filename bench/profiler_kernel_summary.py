"""只统计目标 kernel 的设备 duration，排除 TensorMove 等恢复任务。"""
from __future__ import annotations
import csv
import math
from pathlib import Path
import numpy as np

NAME_COLUMNS = ("Name", "Kernel Name", "KernelName", "op_name", "Op Name")
# task_duration is in us in Ascend's export_summary schema.
DURATION_COLUMNS = ("Duration(us)", "Duration (us)", "Task Duration(us)", "task_duration")


def read_kernel_samples(path: Path, kernel_name: str, *, expected_count=None):
    samples = []
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        headers = reader.fieldnames or []
        names = [key for key in NAME_COLUMNS if key in headers]
        duration = next((key for key in DURATION_COLUMNS if key in headers), None)
        if not names or duration is None:
            raise ValueError(f"Unsupported kernel CSV columns in {path}: {headers}")
        for row in reader:
            if not any(kernel_name in (row.get(key) or "") for key in names):
                continue
            value = float(row[duration].replace(",", ""))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid duration for {kernel_name}: {row[duration]}")
            samples.append(value)
    if not samples:
        raise ValueError(f"No {kernel_name} samples in {path}; inspect the kernel names before summarizing")
    if expected_count is not None and len(samples) != expected_count:
        raise ValueError(f"Expected {expected_count} {kernel_name} samples, got {len(samples)}; trace retained at {path}")
    return samples


def summarize_kernel(samples):
    values = np.asarray(samples, dtype=np.float64)
    if not len(values) or np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("kernel samples must be positive and finite")
    return dict(count=len(values), p50_us=float(np.percentile(values, 50)),
                p95_us=float(np.percentile(values, 95)), min_us=float(values.min()),
                max_us=float(values.max()))
