# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""npusim 输出与 golden 期望的逐字节比对。

用法:
    python3 sim/compare_sim_output.py --case-dir sim/cases/l0_basic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

OUTPUT_DTYPES = {
    "current_slots": np.int32,
    "miss_count": np.int32,
    "miss_tokens": np.int32,
    "miss_slots": np.int32,
    "slot_to_token": np.int32,
    "lru_slots": np.int32,
    "last_req_ids": np.int64,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    args = parser.parse_args()
    case_dir = Path(args.case_dir)
    out_dir = case_dir / "output"

    failed = False
    for name, dtype in OUTPUT_DTYPES.items():
        actual = np.fromfile(out_dir / f"output_{name}.bin", dtype=dtype)
        expected = np.fromfile(case_dir / f"expected_{name}.bin", dtype=dtype)
        if actual.size != expected.size:
            print(f"[FAIL] {name}: size {actual.size} != {expected.size}")
            failed = True
            continue
        if not np.array_equal(actual, expected):
            diff = int((actual != expected).sum())
            idx = np.flatnonzero(actual != expected)[:5]
            print(f"[FAIL] {name}: {diff}/{actual.size} 个元素不一致,首批索引 {idx.tolist()}")
            for i in idx:
                print(f"        [{i}] actual={actual[i]} expected={expected[i]}")
            failed = True
        else:
            print(f"[OK]   {name}: {actual.size} 个元素全等")

    if failed:
        print("比对失败")
        return 1
    print("比对通过:sim 输出与 golden 逐元素一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
