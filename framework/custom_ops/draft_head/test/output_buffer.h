// Host runner output storage only; does not change any tensor ABI or kernel.
#pragma once
#include "../../draft_quant/test/runner_common.h"
#include <limits>

namespace draft_head_test {
inline size_t NativeTop1PayloadBytes(size_t logicalBytes, size_t modelReportedBytes)
{
    const size_t size = std::max(logicalBytes, modelReportedBytes);
    // Include the two guards in the overflow check for GuardedBuffer::Allocate.
    if (size > std::numeric_limits<size_t>::max() - 31 - 2 * draft_quant_test::kGuard)
        throw std::runtime_error("native Top1 output allocation size overflow");
    return (size + 31) / 32 * 32;
}

inline std::vector<uint8_t> ReadLogicalOutput(const draft_quant_test::GuardedBuffer &buffer, size_t logicalBytes)
{
    if (logicalBytes > buffer.bytes) throw std::runtime_error("logical output exceeds physical payload");
    std::vector<uint8_t> result(logicalBytes);
    draft_quant_test::Check(aclrtMemcpy(result.data(), logicalBytes, buffer.data, logicalBytes,
                                       ACL_MEMCPY_DEVICE_TO_HOST), "copy logical output");
    return result;
}
}  // namespace draft_head_test
