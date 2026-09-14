#include "qwen35_dflash/acl_executor.hpp"
#include "qwen35_dflash/sha256.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace qwen35::dflash;

void Require(bool condition) {
  if (!condition) throw std::runtime_error("constant-input test failed");
}

template <class F> void Reject(F function) {
  bool rejected = false;
  try { function(); } catch (const std::exception&) { rejected = true; }
  Require(rejected);
}

int main(int argc, char** argv) {
  try {
    Require(argc == 2);
    const std::filesystem::path root = std::filesystem::absolute(argv[1]);
    std::filesystem::create_directories(root);
    const auto model = root / "fake.om";
    std::ofstream(model) << "fake-om";
    for (const auto& variant : {std::string("w8a16"), std::string("w4a16")}) {
      setenv("QWEN35_FAKE_DRAFT_CONSTANTS", variant.c_str(), 1);
      const bool w4 = variant == "w4a16";
      const std::size_t cols = w4 ? 64 : 128;
      const auto weights = root / "weights.bin";
      const auto scale = root / "scale.bin";
      const auto table = root / "constant-inputs.tsv";
      std::ofstream(weights, std::ios::binary) << std::string(cols, w4 ? '\x89' : '\x01');
      std::ofstream(scale, std::ios::binary).write("\0\x38", 2);
      std::ofstream(table) << "qwen35-draft-constants-v1\n"
          << "draft_weight_000\t" << (w4 ? "uint8" : "int8") << "\t1," << cols
          << '\t' << cols << '\t' << Sha256File(weights) << "\tweights.bin\n"
          << "draft_weight_001\tfloat16\t1,1\t2\t" << Sha256File(scale) << "\tscale.bin\n";
      const auto hash = Sha256File(table);
      const auto constants = ReadConstantInputs(table, hash);
      Require(constants.size() == 2);
      {
        AclExecutor executor(model, 0, constants);
        for (int i = 0; i < 14; ++i) {
          const auto result = executor.Execute({1, 2}, 0);
          Require(result.target_top1[1] == 3);
        }
      }
      auto wrong = constants;
      wrong[0].shape[1] /= 2;
      Reject([&] { AclExecutor executor(model, 0, wrong); });
      wrong = constants;
      wrong[0].dtype = "float16";
      Reject([&] { AclExecutor executor(model, 0, wrong); });
      Reject([&] { AclExecutor executor(model, 0); });
      Reject([&] { ReadConstantInputs(table, std::string(64, '0')); });
      std::ofstream(weights, std::ios::binary) << std::string(cols, '\0');
      Reject([&] { ReadConstantInputs(table, hash); });
      Reject([&] { AclExecutor executor(model, 0, constants); });
    }
    unsetenv("QWEN35_FAKE_DRAFT_CONSTANTS");
    std::cout << "PASS W8/W4 one-time device upload, repeated execution, ABI and corruption rejection\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
