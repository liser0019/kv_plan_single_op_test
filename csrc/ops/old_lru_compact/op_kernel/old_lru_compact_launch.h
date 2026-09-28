/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * The kernel implementation is copied from MemFabric_Hybrid (Mulan PSL v2).
 */
#ifndef OLD_LRU_COMPACT_LAUNCH_H
#define OLD_LRU_COMPACT_LAUNCH_H

#include <cstdint>

constexpr uint32_t OLD_LRU_BLOCK_DIM = 8;
constexpr uint32_t OLD_LRU_UB_LIMIT_BYTES = 120U * 1024U;

inline uint32_t OldLruCompactUbBytes(uint32_t topk, uint32_t capacity) {
  const auto align32 = [](uint32_t bytes) { return (bytes + 31U) & ~31U; };
  // Original InitBuffer calls: 6 topk buffers, 4 capacity buffers,
  // a 1024-int clear buffer, and two 32-byte scalar buffers.
  return 6U * align32(topk * sizeof(int32_t)) +
         4U * align32(capacity * sizeof(int32_t)) + 1024U * sizeof(int32_t) + 64U;
}

extern "C" void OffloadOpsLruCompact(uint64_t req_ids, uint64_t last_req_ids,
                                      uint64_t topk_indices, uint64_t stable_prefix_lens,
                                      uint64_t slot_to_token, uint64_t lru_slots,
                                      uint64_t current_slots, uint64_t miss_count,
                                      uint64_t miss_tokens, uint64_t miss_slots,
                                      uint64_t token_mark_workspace, uint64_t token_pos_workspace,
                                      uint64_t epochs, int64_t num_reqs, int64_t topk,
                                      int64_t capacity, int64_t max_token,
                                      uint32_t dyn_ub_bytes, void* stream);

#endif
