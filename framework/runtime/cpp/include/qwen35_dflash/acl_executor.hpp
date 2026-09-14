#pragma once

#include "qwen35_dflash/generation.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace qwen35::dflash {

struct ConstantInput {
  std::string name;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::size_t bytes = 0;
  std::string sha256;
  std::filesystem::path path;
};

std::vector<ConstantInput> ReadConstantInputs(
    const std::filesystem::path& table, const std::string& sha256);

// Direct AscendCL executor for the ordered deployment ABI:
//   inputs:  input_ids [1,S], attention_mask [1,S]
//   outputs: target_top1 [1,S], draft_top1 [1,K]
// Every tensor is INT64. The implementation allocates pinned host memory,
// device buffers, datasets, context and stream once, then reuses them.
class AclExecutor final : public GraphExecutor {
 public:
  explicit AclExecutor(
      const std::filesystem::path& model_path,
      int device_id = 0,
      std::vector<ConstantInput> constants = {});
  ~AclExecutor() override;

  AclExecutor(const AclExecutor&) = delete;
  AclExecutor& operator=(const AclExecutor&) = delete;
  AclExecutor(AclExecutor&&) noexcept;
  AclExecutor& operator=(AclExecutor&&) noexcept;

  std::size_t sequence_length() const noexcept override;
  std::size_t draft_width() const noexcept override;
  const std::vector<ConstantInput>& constant_inputs() const noexcept;
  const GraphOutputs& Execute(
      const std::vector<std::int64_t>& committed_prefix,
      std::int64_t pad_token_id) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace qwen35::dflash
