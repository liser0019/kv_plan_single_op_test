/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * Direct-launch adapter for the MemFabric_Hybrid LRU compact kernel (Mulan PSL v2).
 */
#include <cstddef>
#include <cstdint>
#include <limits>

#include <torch/all.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/old_lru_compact_launch.h"

namespace {

void CheckOldLruInputs(const at::Tensor& reqIds, const at::Tensor& topkIndices,
                       const at::Tensor& stablePrefixLens, const at::Tensor& lastReqIds,
                       const at::Tensor& slotToToken, const at::Tensor& lruSlots,
                       const at::Tensor& currentSlots, const at::Tensor& missCount,
                       const at::Tensor& missTokens, const at::Tensor& missSlots,
                       const at::Tensor& tokenMark, const at::Tensor& tokenPos,
                       const at::Tensor& epochs, int64_t maxToken, bool requireNpu) {
  TORCH_CHECK(reqIds.dim() == 1 && reqIds.scalar_type() == at::kLong,
              "old_lru_compact: req_ids must be int64 [rows]");
  const int64_t rows = reqIds.numel();
  TORCH_CHECK(rows > 0 && topkIndices.dim() == 2 && topkIndices.scalar_type() == at::kInt &&
                  topkIndices.size(0) == rows,
              "old_lru_compact: topk_indices must be int32 [rows, topk]");
  const int64_t topk = topkIndices.size(1);
  TORCH_CHECK(slotToToken.dim() == 2 && slotToToken.scalar_type() == at::kInt &&
                  slotToToken.size(0) == rows,
              "old_lru_compact: slot_to_token must be int32 [rows, capacity]");
  const int64_t capacity = slotToToken.size(1);
  TORCH_CHECK(maxToken > 0 && maxToken <= std::numeric_limits<int32_t>::max() &&
                  topk > 0 && topk <= OLD_LRU_UB_LIMIT_BYTES / 24 &&
                  capacity > 0 && capacity <= OLD_LRU_UB_LIMIT_BYTES / 16,
              "old_lru_compact: invalid shape or max_token");
  const uint32_t ubBytes = OldLruCompactUbBytes(static_cast<uint32_t>(topk),
                                                 static_cast<uint32_t>(capacity));
  TORCH_CHECK(ubBytes <= OLD_LRU_UB_LIMIT_BYTES,
              "old_lru_compact: UB buffers require ", ubBytes,
              " bytes; limit is ", OLD_LRU_UB_LIMIT_BYTES);
  TORCH_CHECK(stablePrefixLens.sizes() == reqIds.sizes() && stablePrefixLens.scalar_type() == at::kInt &&
                  lastReqIds.sizes() == reqIds.sizes() && lastReqIds.scalar_type() == at::kLong &&
                  lruSlots.sizes() == slotToToken.sizes() && lruSlots.scalar_type() == at::kInt &&
                  currentSlots.sizes() == topkIndices.sizes() && currentSlots.scalar_type() == at::kInt &&
                  missCount.sizes() == reqIds.sizes() && missCount.scalar_type() == at::kInt &&
                  missTokens.sizes() == topkIndices.sizes() && missTokens.scalar_type() == at::kInt &&
                  missSlots.sizes() == topkIndices.sizes() && missSlots.scalar_type() == at::kInt,
              "old_lru_compact: state tensor shape or dtype mismatch");
  TORCH_CHECK(tokenMark.dim() == 2 && tokenMark.scalar_type() == at::kInt &&
                  tokenMark.size(0) == OLD_LRU_BLOCK_DIM && tokenMark.size(1) == maxToken &&
                  tokenPos.sizes() == tokenMark.sizes() && tokenPos.scalar_type() == at::kInt &&
                  epochs.dim() == 1 && epochs.scalar_type() == at::kInt &&
                  epochs.numel() == OLD_LRU_BLOCK_DIM,
              "old_lru_compact: token_mark/token_pos must be int32 [8, max_token], epochs int32 [8]");

  const at::Tensor* tensors[] = {&reqIds, &topkIndices, &stablePrefixLens, &lastReqIds,
                                 &slotToToken, &lruSlots, &currentSlots, &missCount,
                                 &missTokens, &missSlots, &tokenMark, &tokenPos, &epochs};
  const char* names[] = {"req_ids", "topk_indices", "stable_prefix_lens", "last_req_ids",
                         "slot_to_token", "lru_slots", "current_slots", "miss_count",
                         "miss_tokens", "miss_slots", "token_mark", "token_pos", "epochs"};
  if (requireNpu) {
    TORCH_CHECK(reqIds.device().type() == c10::DeviceType::PrivateUse1,
                "old_lru_compact: req_ids must be on NPU");
  }
  for (std::size_t i = 0; i < sizeof(tensors) / sizeof(tensors[0]); ++i) {
    TORCH_CHECK(tensors[i]->is_contiguous(), "old_lru_compact: ", names[i], " must be contiguous");
    if (requireNpu) {
      TORCH_CHECK(tensors[i]->device() == reqIds.device(),
                  "old_lru_compact: ", names[i], " must be on the same NPU");
    }
  }
}

void OldLruCompactMeta(const at::Tensor& reqIds, const at::Tensor& topkIndices,
                       const at::Tensor& stablePrefixLens, const at::Tensor& lastReqIds,
                       const at::Tensor& slotToToken, const at::Tensor& lruSlots,
                       const at::Tensor& currentSlots, const at::Tensor& missCount,
                       const at::Tensor& missTokens, const at::Tensor& missSlots,
                       const at::Tensor& tokenMark, const at::Tensor& tokenPos,
                       const at::Tensor& epochs, int64_t maxToken) {
  CheckOldLruInputs(reqIds, topkIndices, stablePrefixLens, lastReqIds, slotToToken,
                    lruSlots, currentSlots, missCount, missTokens, missSlots,
                    tokenMark, tokenPos, epochs, maxToken, false);
}

void OldLruCompactNpu(const at::Tensor& reqIds, const at::Tensor& topkIndices,
                      const at::Tensor& stablePrefixLens, const at::Tensor& lastReqIds,
                      const at::Tensor& slotToToken, const at::Tensor& lruSlots,
                      const at::Tensor& currentSlots, const at::Tensor& missCount,
                      const at::Tensor& missTokens, const at::Tensor& missSlots,
                      const at::Tensor& tokenMark, const at::Tensor& tokenPos,
                      const at::Tensor& epochs, int64_t maxToken) {
  CheckOldLruInputs(reqIds, topkIndices, stablePrefixLens, lastReqIds, slotToToken,
                    lruSlots, currentSlots, missCount, missTokens, missSlots,
                    tokenMark, tokenPos, epochs, maxToken, true);
  const c10::OptionalDeviceGuard guard(reqIds.device());
  const uint32_t dynUbBytes = OldLruCompactUbBytes(
      static_cast<uint32_t>(topkIndices.size(1)), static_cast<uint32_t>(slotToToken.size(1)));
  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  auto launch = [=]() -> int {
    OffloadOpsLruCompact(
        reinterpret_cast<uint64_t>(reqIds.data_ptr()),
        reinterpret_cast<uint64_t>(lastReqIds.data_ptr()),
        reinterpret_cast<uint64_t>(topkIndices.data_ptr()),
        reinterpret_cast<uint64_t>(stablePrefixLens.data_ptr()),
        reinterpret_cast<uint64_t>(slotToToken.data_ptr()),
        reinterpret_cast<uint64_t>(lruSlots.data_ptr()),
        reinterpret_cast<uint64_t>(currentSlots.data_ptr()),
        reinterpret_cast<uint64_t>(missCount.data_ptr()),
        reinterpret_cast<uint64_t>(missTokens.data_ptr()),
        reinterpret_cast<uint64_t>(missSlots.data_ptr()),
        reinterpret_cast<uint64_t>(tokenMark.data_ptr()),
        reinterpret_cast<uint64_t>(tokenPos.data_ptr()),
        reinterpret_cast<uint64_t>(epochs.data_ptr()), reqIds.numel(),
        topkIndices.size(1), slotToToken.size(1), maxToken,
        dynUbBytes, reinterpret_cast<void*>(stream));
    return 0;
  };
  at_npu::native::OpCommand::RunOpApi("OldLruCompact", launch);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(sparse_kv_plan_op, m) {
  m.def("old_lru_compact(Tensor req_ids, Tensor topk_indices, Tensor stable_prefix_lens, "
        "Tensor(a!) last_req_ids, Tensor(b!) slot_to_token, Tensor(c!) lru_slots, "
        "Tensor(d!) current_slots, Tensor(e!) miss_count, Tensor(f!) miss_tokens, "
        "Tensor(g!) miss_slots, Tensor(h!) token_mark, Tensor(i!) token_pos, "
        "Tensor(j!) epochs, int max_token) -> ()");
}

TORCH_LIBRARY_IMPL(sparse_kv_plan_op, Meta, m) {
  m.impl("old_lru_compact", OldLruCompactMeta);
}

TORCH_LIBRARY_IMPL(sparse_kv_plan_op, PrivateUse1, m) {
  m.impl("old_lru_compact", OldLruCompactNpu);
}
