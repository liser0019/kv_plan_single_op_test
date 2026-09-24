# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""把 tests/gen_data.py 的用例导出为 npusim harness 可读的裸二进制。

用法:
    python3 sim/make_case.py --case l0_basic [--cases-dir sim/cases]
输出:
    <cases_dir>/<case>/meta.txt          # key=value 元信息
    <cases_dir>/<case>/input_*.bin       # 小端裸数组(int32/int64)
    <cases_dir>/<case>/expected_*.bin    # golden 期望输出(比对基准)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

from gen_data import ALL_CASE_NAMES, build_case  # noqa: E402
from golden import golden_sparse_kv_plan  # noqa: E402


def dump(path: Path, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr)
    arr.tofile(path)


def export_case(name: str, out_root: Path) -> None:
    case = build_case(name)
    expected = golden_sparse_kv_plan(case)
    case_dir = out_root / name
    (case_dir / "output").mkdir(parents=True, exist_ok=True)

    max_requests = case["req_ids"].shape[0]
    max_num_blocks = case["block_table"].shape[1]
    meta_lines = [
        f"max_rows={case['topk_indices'].shape[0]}",
        f"topk={case['topk']}",
        f"capacity={case['capacity']}",
        f"max_requests={max_requests}",
        f"max_num_blocks={max_num_blocks}",
        f"max_token={case['max_token']}",
        f"block_size={case['block_size']}",
        f"host_num_blocks={case['host_num_blocks']}",
    ]
    (case_dir / "meta.txt").write_text("\n".join(meta_lines) + "\n")

    dump(case_dir / "input_req_ids.bin", case["req_ids"].astype(np.int64))
    dump(case_dir / "input_topk_indices.bin", case["topk_indices"].astype(np.int32))
    dump(case_dir / "input_stable_prefix_lens.bin", case["stable_prefix_lens"].astype(np.int32))
    dump(case_dir / "input_visible_seq_lens.bin", case["visible_seq_lens"].astype(np.int32))
    dump(case_dir / "input_token_to_req.bin", case["token_to_req"].astype(np.int32))
    dump(case_dir / "input_block_table.bin", case["block_table"].astype(np.int32))
    dump(case_dir / "input_active_rows.bin", np.atleast_1d(np.int32(case["active_rows"])))
    dump(case_dir / "input_last_req_ids.bin", case["last_req_ids"].astype(np.int64))
    dump(case_dir / "input_slot_to_token.bin", case["slot_to_token"].astype(np.int32))
    dump(case_dir / "input_lru_slots.bin", case["lru_slots"].astype(np.int32))

    dump(case_dir / "expected_current_slots.bin", expected["current_slots"].astype(np.int32))
    dump(case_dir / "expected_miss_count.bin", expected["miss_count"].astype(np.int32))
    dump(case_dir / "expected_miss_tokens.bin", expected["miss_tokens"].astype(np.int32))
    dump(case_dir / "expected_miss_slots.bin", expected["miss_slots"].astype(np.int32))
    dump(case_dir / "expected_slot_to_token.bin", expected["slot_to_token"].astype(np.int32))
    dump(case_dir / "expected_lru_slots.bin", expected["lru_slots"].astype(np.int32))
    dump(case_dir / "expected_last_req_ids.bin", expected["last_req_ids"].astype(np.int64))
    print(f"[make_case] {name} -> {case_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=None, help="用例名;缺省导出全部(不含超大 GM 用例可选 --include-heavy)")
    parser.add_argument("--cases-dir", default=str(Path(__file__).parent / "cases"))
    parser.add_argument("--include-heavy", action="store_true",
                        help="包含 l2_ub_boundary_exact / l2_gm_workspace 等大用例(仿真耗时长)")
    args = parser.parse_args()

    out_root = Path(args.cases_dir)
    heavy = {"l2_ub_boundary_exact", "l2_gm_workspace"}
    if args.case:
        export_case(args.case, out_root)
        return
    for name in ALL_CASE_NAMES:
        if name in heavy and not args.include_heavy:
            print(f"[make_case] 跳过大用例 {name}(--include-heavy 开启)")
            continue
        export_case(name, out_root)


if __name__ == "__main__":
    main()
