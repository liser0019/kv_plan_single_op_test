"""Echo 语义锚点、共同输入及 NPU 全状态/连续调用验证。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "sparse_kv_plan_op"))
from echo_layout import echo_partition, echo_workspace_elements
from echo_lru_cases import make_peer_common_case, echo_from_common, assert_echo_invariants
from golden_echo_lru import golden_echo_lru, empty_echo_state, STATE_KEYS, OUTPUT_KEYS, I32_MAX
from golden import golden_sparse_kv_plan
from golden_old_lru import golden_old_lru

try:
    import torch
    import torch_npu
    NPU_AVAILABLE = torch.npu.is_available()
except ImportError:
    NPU_AVAILABLE = False
requires_npu = pytest.mark.skipif(not NPU_AVAILABLE, reason="需要 torch_npu 和 NPU")


@pytest.mark.parametrize("rows,cores", [(8, 6), (32, 24), (32, 32), (33, 8), (1, 64)])
def test_echo_partition_covers_rows_once(rows, cores):
    blocks, chunk, tail = echo_partition(rows, cores)
    assert blocks <= min(rows, cores)
    assert 1 <= tail <= chunk
    covered = [row for block in range(blocks)
               for row in range(block * chunk, block * chunk + (tail if block == blocks - 1 else chunk))]
    assert covered == list(range(rows))


def test_echo_workspace_production_layout():
    rows, topk, capacity, blocks = 32, 2048, 4096, 32
    base = rows * (capacity + 1) + 4 * rows + 3 * rows * topk
    npad = 8192
    expected = ((base + 1) & ~1) + blocks * (4*npad + 64*256 + 64*256 + 256 + 8192)
    assert echo_workspace_elements(rows, topk, capacity, blocks) == expected
    assert echo_workspace_elements(1, 1, 1, 1) > 0
    with pytest.raises(ValueError):
        echo_workspace_elements(32, 2048, 4097, 32)
    with pytest.raises(ValueError):
        echo_partition(0, 4)


def test_echo_cold_anchor_and_reserved_token():
    pos = np.array([[0, 4, 5]], np.int32)
    state = empty_echo_state(1, 3, 4, 16)
    out = golden_echo_lru(pos, np.array([16], np.int32), np.array([1], np.uint8), state)
    np.testing.assert_array_equal(out["current_slots"], [[0, 2, 1]])
    np.testing.assert_array_equal(out["miss_mask"], [[0, 1, 1]])
    np.testing.assert_array_equal(out["miss_host_pos"], [[0, 4, 5]])
    np.testing.assert_array_equal(out["miss_alloc_flat"], [[0, 2, 1]])
    assert out["avail"][0] == 2 and out["fifo"][0] == 3
    assert_echo_invariants(pos, out)


def _sequence():
    # 第一次填满；第二次出现同时间戳平局淘汰；第三次驱逐投机后缀；第四次请求 reset。
    return [(np.array([[1, 2, 3, 4]], np.int32), 16, 1),
            (np.array([[1, 5, 6, 7]], np.int32), 16, 0),
            (np.array([[1, 5, 8, 9]], np.int32), 6, 0),
            (np.array([[10, 11, 12, 13]], np.int32), 16, 1)]


def test_echo_eviction_suffix_and_reset_anchors():
    state = empty_echo_state(1, 4, 4, 16)
    results = []
    for pos, spl, reset in _sequence():
        state = golden_echo_lru(pos, np.array([spl], np.int32), np.array([reset], np.uint8), state)
        assert_echo_invariants(pos, state)
        results.append(state)
    np.testing.assert_array_equal(results[0]["current_slots"], [[4, 3, 2, 1]])
    np.testing.assert_array_equal(results[1]["current_slots"], [[4, 1, 2, 3]])
    np.testing.assert_array_equal(results[1]["miss_mask"], [[0, 1, 1, 1]])
    assert results[2]["htd"][0, 6] == I32_MAX and results[2]["htd"][0, 7] == I32_MAX
    np.testing.assert_array_equal(results[2]["current_slots"], [[4, 1, 2, 3]])
    np.testing.assert_array_equal(results[3]["current_slots"], [[4, 3, 2, 1]])
    assert results[3]["fifo"][0] == 3


def test_echo_padding_after_capacity():
    pos = np.array([[1, 2, 999, -1]], np.int32)
    out = golden_echo_lru(pos, np.array([8], np.int32), np.array([1], np.uint8),
                          empty_echo_state(1, 4, 2, 8))
    np.testing.assert_array_equal(out["current_slots"], [[2, 1, 0, 0]])
    np.testing.assert_array_equal(out["miss_mask"], [[1, 1, 0, 0]])
    assert_echo_invariants(pos, out)


@pytest.mark.parametrize("pos", [[[1, 1]], [[-1, 2]], [[1, 16]]])
def test_echo_golden_rejects_unsupported_inputs(pos):
    with pytest.raises(ValueError):
        golden_echo_lru(np.array(pos, np.int32), np.array([16], np.int32),
                        np.array([1], np.uint8), empty_echo_state(1, 2, 4, 16))


@pytest.mark.parametrize("shape", [(2, 8, 16, 128), (32, 2048, 4096, 256*1024)])
@pytest.mark.parametrize("fraction,full", [(0.0, False), (0.75, False), (1.0, False), (0.75, True)])
def test_echo_common_domain_matches_miss_counts(shape, fraction, full):
    case = make_peer_common_case(rows=shape[0], topk=shape[1], capacity=shape[2],
                                 max_token=shape[3], resident_fraction=fraction, full_cache=full)
    echo = golden_echo_lru(case["topk_indices"], case["stable_prefix_lens"],
                           (case["req_ids"] != case["last_req_ids"]).astype(np.uint8), echo_from_common(case))
    new = golden_sparse_kv_plan(case)
    old = golden_old_lru(case)
    np.testing.assert_array_equal(echo["miss_mask"].sum(axis=1), new["miss_count"])
    np.testing.assert_array_equal(old["miss_count"], new["miss_count"])
    assert_echo_invariants(case["topk_indices"], echo)
    for row in range(shape[0]):
        np.testing.assert_array_equal(new["slot_to_token"][row, new["current_slots"][row]], case["topk_indices"][row])


def _to_npu(value):
    return torch.from_numpy(np.ascontiguousarray(value)).npu()


def _actual(state):
    return {key: getattr(state, key).cpu().numpy() for key in STATE_KEYS + OUTPUT_KEYS}


@requires_npu
@pytest.mark.parametrize("shape", [(8, 8, 16, 128), (32, 2048, 4096, 256*1024)])
@pytest.mark.parametrize("fraction,full", [(0.0, False), (0.75, False), (1.0, False), (0.75, True)])
def test_echo_npu_common_state_and_outputs(shape, fraction, full):
    import sparse_kv_plan_op as op
    assert hasattr(torch.ops.sparse_kv_plan_op, "echo_lru"), "请重新构建扩展"
    case = make_peer_common_case(rows=shape[0], topk=shape[1], capacity=shape[2], max_token=shape[3],
                                 resident_fraction=fraction, full_cache=full)
    # 至多六核，覆盖原 Host 分行公式容易下溢的场景及一核多行。
    cores = min(6, torch.ops.sparse_kv_plan_op.echo_lru_core_count())
    state = op.EchoLruState.create(*shape, device="npu", available_cores=cores)
    initial = echo_from_common(case)
    for key, value in initial.items():
        getattr(state, key).copy_(_to_npu(value))
    reset = (case["req_ids"] != case["last_req_ids"]).astype(np.uint8)
    expected = golden_echo_lru(case["topk_indices"], case["stable_prefix_lens"], reset, initial, spec_enabled=False)
    op.echo_lru(pos=_to_npu(case["topk_indices"]), stable_prefix_lens=_to_npu(case["stable_prefix_lens"]),
                reset_mask=_to_npu(reset), state=state, spec_enabled=False)
    torch.npu.synchronize()
    actual = _actual(state)
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    assert_echo_invariants(case["topk_indices"], actual)


@requires_npu
def test_echo_npu_continues_own_state():
    import sparse_kv_plan_op as op
    state = op.EchoLruState.create(1, 4, 4, 16, "npu")
    expected = empty_echo_state(1, 4, 4, 16)
    for pos, spl, reset in _sequence():
        prefix, mask = np.array([spl], np.int32), np.array([reset], np.uint8)
        expected = golden_echo_lru(pos, prefix, mask, expected)
        op.echo_lru(pos=_to_npu(pos), stable_prefix_lens=_to_npu(prefix), reset_mask=_to_npu(mask), state=state)
        torch.npu.synchronize()
        actual = _actual(state)
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        assert_echo_invariants(pos, actual)
        # 下一轮只更换输入，保留上一轮 NPU 状态，不回灌 golden。


@requires_npu
def test_echo_npu_rejects_short_workspace():
    import sparse_kv_plan_op as op
    state = op.EchoLruState.create(1, 2, 4, 16, "npu")
    state = state._replace(workspace=torch.empty(1, dtype=torch.int32, device="npu"))
    with pytest.raises(RuntimeError, match="workspace"):
        op.echo_lru(pos=_to_npu(np.array([[1, 2]], np.int32)),
                    stable_prefix_lens=_to_npu(np.array([16], np.int32)),
                    reset_mask=_to_npu(np.array([1], np.uint8)), state=state)
