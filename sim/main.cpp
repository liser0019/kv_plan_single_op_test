/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * (kernel 计算逻辑源自 MemFabric_Hybrid,Mulan PSL v2)
 *
 * SparseKvPlan 无 NPU 精度仿真 harness(npusim,昇腾 950)。
 *
 * 用途:在没有 NPU 卡、但装有 CANN 工具链(x86/ARM 主机)的机器上,通过
 * npusim 对本算子做 bit 级精度仿真。流程:
 *   1. tests/gen_data.py 生成用例 → sim/make_case.py 导出裸二进制
 *      (sim/case/<name>/ 下 meta.txt + input_*.bin + expected_*.bin);
 *   2. 本程序读取输入,device 侧申请缓冲,launch sparse_kv_plan kernel;
 *   3. npusim 拦截 launch 并仿真执行;
 *   4. sim/compare_sim_output.py 把 output_*.bin 与 expected_*.bin 逐字节比对。
 *
 * 注意:本文件按 CANN Runtime API(aclrt*)编写,npusim 仿真模式下这些调用
 * 会被仿真器接管;若所用 npusim 版本的 harness 约定不同(例如 bbit 专用
 * runtime),仅需替换本文件中的分配/拷贝 API,kernel 启动路径
 * (launch_sparse_kv_plan)不变。
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"

#include "sparse_kv_plan_launch.h"

namespace {

struct Buffer {
  void* host = nullptr;
  void* dev = nullptr;
  size_t bytes = 0;
};

std::vector<int32_t> ReadInt32File(const std::string& path, size_t expect_count) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    std::fprintf(stderr, "无法打开输入文件: %s\n", path.c_str());
    std::exit(1);
  }
  std::vector<int32_t> data(expect_count);
  size_t got = std::fread(data.data(), sizeof(int32_t), expect_count, f);
  std::fclose(f);
  if (got != expect_count) {
    std::fprintf(stderr, "文件 %s 期望 %zu 个 int32,实际 %zu\n", path.c_str(), expect_count, got);
    std::exit(1);
  }
  return data;
}

std::vector<int64_t> ReadInt64File(const std::string& path, size_t expect_count) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    std::fprintf(stderr, "无法打开输入文件: %s\n", path.c_str());
    std::exit(1);
  }
  std::vector<int64_t> data(expect_count);
  size_t got = std::fread(data.data(), sizeof(int64_t), expect_count, f);
  std::fclose(f);
  if (got != expect_count) {
    std::fprintf(stderr, "文件 %s 期望 %zu 个 int64,实际 %zu\n", path.c_str(), expect_count, got);
    std::exit(1);
  }
  return data;
}

Buffer Upload(const void* host_data, size_t bytes) {
  Buffer buf;
  buf.bytes = bytes;
  if (aclrtMalloc(&buf.dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
    std::fprintf(stderr, "aclrtMalloc 失败 (%zu bytes)\n", bytes);
    std::exit(1);
  }
  if (aclrtMemcpy(buf.dev, bytes, host_data, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
    std::fprintf(stderr, "aclrtMemcpy H2D 失败\n");
    std::exit(1);
  }
  return buf;
}

void WriteFile(const std::string& path, const void* host, size_t bytes) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) {
    std::fprintf(stderr, "无法写输出文件: %s\n", path.c_str());
    std::exit(1);
  }
  std::fwrite(host, 1, bytes, f);
  std::fclose(f);
}

int ReadMetaInt(const char* path, const char* key, int64_t* value) {
  FILE* f = std::fopen(path, "r");
  if (f == nullptr) return -1;
  char line[256];
  int found = -1;
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    char name[128];
    int64_t v;
    if (std::sscanf(line, "%127[^=]=%ld", name, &v) == 2) {
      if (std::strncmp(name, key, sizeof(name)) == 0) {
        *value = v;
        found = 0;
        break;
      }
    }
  }
  std::fclose(f);
  return found;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "用法: %s <case_dir> [--block-dim N]\n", argv[0]);
    return 1;
  }
  const std::string case_dir = argv[1];
  int64_t block_dim_override = 0;
  for (int i = 2; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--block-dim") == 0) {
      block_dim_override = std::atoll(argv[i + 1]);
    }
  }

  const std::string meta = case_dir + "/meta.txt";
  int64_t max_rows = 0, topk = 0, capacity = 0, max_requests = 0, max_num_blocks = 0;
  int64_t max_token = 0, block_size = 0, host_num_blocks = 0;
  if (ReadMetaInt(meta.c_str(), "max_rows", &max_rows) != 0 ||
      ReadMetaInt(meta.c_str(), "topk", &topk) != 0 || ReadMetaInt(meta.c_str(), "capacity", &capacity) != 0 ||
      ReadMetaInt(meta.c_str(), "max_requests", &max_requests) != 0 ||
      ReadMetaInt(meta.c_str(), "max_num_blocks", &max_num_blocks) != 0 ||
      ReadMetaInt(meta.c_str(), "max_token", &max_token) != 0 ||
      ReadMetaInt(meta.c_str(), "block_size", &block_size) != 0 ||
      ReadMetaInt(meta.c_str(), "host_num_blocks", &host_num_blocks) != 0) {
    std::fprintf(stderr, "meta.txt 缺少必要字段\n");
    return 1;
  }

  if (aclrtSetDevice(0) != ACL_SUCCESS) {
    std::fprintf(stderr, "aclrtSetDevice(0) 失败\n");
    return 1;
  }
  aclrtStream stream = nullptr;
  if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
    std::fprintf(stderr, "aclrtCreateStream 失败\n");
    return 1;
  }

  // ---- 读入并上传 ----
  auto req_ids = ReadInt64File(case_dir + "/input_req_ids.bin", static_cast<size_t>(max_requests));
  auto topk_indices = ReadInt32File(case_dir + "/input_topk_indices.bin",
                                    static_cast<size_t>(max_rows * topk));
  auto stable = ReadInt32File(case_dir + "/input_stable_prefix_lens.bin", static_cast<size_t>(max_rows));
  auto visible = ReadInt32File(case_dir + "/input_visible_seq_lens.bin", static_cast<size_t>(max_rows));
  auto token_to_req = ReadInt32File(case_dir + "/input_token_to_req.bin", static_cast<size_t>(max_rows));
  auto block_table = ReadInt32File(case_dir + "/input_block_table.bin",
                                   static_cast<size_t>(max_requests * max_num_blocks));
  auto active_rows = ReadInt32File(case_dir + "/input_active_rows.bin", 1);
  auto last_req_ids = ReadInt64File(case_dir + "/input_last_req_ids.bin", static_cast<size_t>(max_rows));
  auto slot_to_token = ReadInt32File(case_dir + "/input_slot_to_token.bin",
                                     static_cast<size_t>(max_rows * capacity));
  auto lru_slots = ReadInt32File(case_dir + "/input_lru_slots.bin", static_cast<size_t>(max_rows * capacity));

  Buffer b_req = Upload(req_ids.data(), req_ids.size() * sizeof(int64_t));
  Buffer b_topk = Upload(topk_indices.data(), topk_indices.size() * sizeof(int32_t));
  Buffer b_stable = Upload(stable.data(), stable.size() * sizeof(int32_t));
  Buffer b_visible = Upload(visible.data(), visible.size() * sizeof(int32_t));
  Buffer b_t2r = Upload(token_to_req.data(), token_to_req.size() * sizeof(int32_t));
  Buffer b_bt = Upload(block_table.data(), block_table.size() * sizeof(int32_t));
  Buffer b_active = Upload(active_rows.data(), sizeof(int32_t));
  Buffer b_last = Upload(last_req_ids.data(), last_req_ids.size() * sizeof(int64_t));
  Buffer b_stt = Upload(slot_to_token.data(), slot_to_token.size() * sizeof(int32_t));
  Buffer b_lru = Upload(lru_slots.data(), lru_slots.size() * sizeof(int32_t));

  // 输出缓冲(初值 -1,与真实调用方一致)
  std::vector<int32_t> current_slots(static_cast<size_t>(max_rows * topk), -1);
  std::vector<int32_t> miss_count(static_cast<size_t>(max_rows), 0);
  std::vector<int32_t> miss_tokens(static_cast<size_t>(max_rows * topk), -1);
  std::vector<int32_t> miss_slots(static_cast<size_t>(max_rows * topk), -1);
  Buffer b_cs = Upload(current_slots.data(), current_slots.size() * sizeof(int32_t));
  Buffer b_mc = Upload(miss_count.data(), miss_count.size() * sizeof(int32_t));
  Buffer b_mt = Upload(miss_tokens.data(), miss_tokens.size() * sizeof(int32_t));
  Buffer b_ms = Upload(miss_slots.data(), miss_slots.size() * sizeof(int32_t));

  // compact workspace:UB 快路径 1 个元素即可,GM 路径需完整尺寸(取足量)。
  const uint64_t row_elements = SparseKvPlanRowElements(topk, capacity);
  std::vector<int32_t> workspace(static_cast<size_t>(max_rows * row_elements), 0);
  Buffer b_ws = Upload(workspace.data(), workspace.size() * sizeof(int32_t));

  // tiling 结构体上传
  const SparseKvPlanTilingData tiling = MakeSparseKvPlanTiling(max_rows, topk, capacity, max_token, max_requests,
                                                               max_num_blocks, host_num_blocks, block_size);
  Buffer b_tiling = Upload(&tiling, sizeof(tiling));

  // ---- 启动 ----
  const uint32_t block_dim =
      block_dim_override > 0 ? static_cast<uint32_t>(block_dim_override) : calc_sparse_kv_plan_block_dim();
  std::printf("[sim] case=%s max_rows=%ld topk=%ld capacity=%ld block_dim=%u workspace=%s\n",
              case_dir.c_str(), static_cast<long>(max_rows), static_cast<long>(topk),
              static_cast<long>(capacity), block_dim,
              row_elements * sizeof(int32_t) <= sparse_kv_plan::PLAN_ROW_UB_LIMIT_BYTES ? "UB" : "GM");
  launch_sparse_kv_plan(b_req.dev, b_topk.dev, b_stable.dev, b_visible.dev, b_t2r.dev, b_bt.dev, b_active.dev,
                        b_last.dev, b_stt.dev, b_lru.dev, b_cs.dev, b_mc.dev, b_mt.dev, b_ms.dev, b_ws.dev,
                        b_tiling.dev, block_dim, stream);
  if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    std::fprintf(stderr, "aclrtSynchronizeStream 失败\n");
    return 1;
  }

  // ---- 回读并落盘 ----
  auto download = [&](const Buffer& buf, void* host, size_t bytes) {
    if (aclrtMemcpy(host, bytes, buf.dev, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
      std::fprintf(stderr, "aclrtMemcpy D2H 失败\n");
      std::exit(1);
    }
  };
  download(b_cs, current_slots.data(), current_slots.size() * sizeof(int32_t));
  download(b_mc, miss_count.data(), miss_count.size() * sizeof(int32_t));
  download(b_mt, miss_tokens.data(), miss_tokens.size() * sizeof(int32_t));
  download(b_ms, miss_slots.data(), miss_slots.size() * sizeof(int32_t));
  download(b_stt, slot_to_token.data(), slot_to_token.size() * sizeof(int32_t));
  download(b_lru, lru_slots.data(), lru_slots.size() * sizeof(int32_t));
  download(b_last, last_req_ids.data(), last_req_ids.size() * sizeof(int64_t));

  const std::string out_dir = case_dir + "/output";
  WriteFile(out_dir + "/output_current_slots.bin", current_slots.data(),
            current_slots.size() * sizeof(int32_t));
  WriteFile(out_dir + "/output_miss_count.bin", miss_count.data(), miss_count.size() * sizeof(int32_t));
  WriteFile(out_dir + "/output_miss_tokens.bin", miss_tokens.data(), miss_tokens.size() * sizeof(int32_t));
  WriteFile(out_dir + "/output_miss_slots.bin", miss_slots.data(), miss_slots.size() * sizeof(int32_t));
  WriteFile(out_dir + "/output_slot_to_token.bin", slot_to_token.data(),
            slot_to_token.size() * sizeof(int32_t));
  WriteFile(out_dir + "/output_lru_slots.bin", lru_slots.data(), lru_slots.size() * sizeof(int32_t));
  WriteFile(out_dir + "/output_last_req_ids.bin", last_req_ids.data(),
            last_req_ids.size() * sizeof(int64_t));
  std::printf("[sim] 输出已写入 %s\n", out_dir.c_str());

  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  return 0;
}
