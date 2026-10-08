"""防止 profiler 汇总把状态恢复任务或另一版 kernel 算进去。"""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from profiler_kernel_summary import read_kernel_samples, summarize_kernel


def test_kernel_summary_filters_restores_and_other_variant(tmp_path):
    path = tmp_path / "kernel_details.csv"
    path.write_text("\ufeffName,Duration(us)\nTensorMove,9999\necho_lru_kernel,100\n"
                    "echo_lru_skip_sort_kernel,3\necho_lru_kernel,102\n")
    samples = read_kernel_samples(path, "echo_lru_kernel", expected_count=2)
    assert samples == [100, 102]
    stats = summarize_kernel(samples)
    assert stats["count"] == 2 and stats["p50_us"] == 101
    assert stats["min_us"] == 100 and stats["max_us"] == 102
    assert read_kernel_samples(path, "echo_lru_skip_sort_kernel") == [3]


def test_kernel_summary_detects_incomplete_capture(tmp_path):
    path = tmp_path / "kernel_details.csv"
    path.write_text("Kernel Name,Duration (us)\necho_lru_kernel,100\n")
    with pytest.raises(ValueError, match="Expected 5"):
        read_kernel_samples(path, "echo_lru_kernel", expected_count=5)


def test_kernel_summary_accepts_export_schema(tmp_path):
    path = tmp_path / "kernel_details.csv"
    path.write_text("op_name,task_duration\necho_lru_kernel,100\n")
    assert read_kernel_samples(path, "echo_lru_kernel") == [100]


def test_kernel_summary_rejects_wrong_unit_and_missing_kernel(tmp_path):
    path = tmp_path / "kernel_details.csv"
    path.write_text("Name,Duration(ms)\necho_lru_kernel,100\n")
    with pytest.raises(ValueError, match="columns"):
        read_kernel_samples(path, "echo_lru_kernel")
    path.write_text("Name,Duration(us)\nTensorMove,100\n")
    with pytest.raises(ValueError, match="No echo_lru_kernel"):
        read_kernel_samples(path, "echo_lru_kernel")


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_kernel_summary_rejects_invalid_durations(tmp_path, value):
    path = tmp_path / "kernel_details.csv"
    path.write_text(f"Name,Duration(us)\necho_lru_kernel,{value}\n")
    with pytest.raises(ValueError, match="Invalid duration"):
        read_kernel_samples(path, "echo_lru_kernel")
