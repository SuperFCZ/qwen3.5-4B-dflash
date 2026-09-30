// CPU-only callback test of the actual runner scheduling; no ACL execution.
#include <cstddef>
extern "C" const char *aclGetRecentErrMsg();
extern "C" int aclrtMemset(void *, std::size_t, int, std::size_t);
#include "runner_common.h"
#include <cassert>
#include <sstream>

using namespace draft_quant_test;

int main()
{
    for (bool continuous : {false, true}) {
        Timing timing;
        timing.warmup = 3; timing.repetitions = 10; timing.continuous = continuous;
        std::vector<std::string> trace, expected;
        RunMeasuredCalls(timing, [&] { trace.push_back("prepare"); }, [&] { trace.push_back("execute"); },
            [&](int) { trace.push_back("poison"); }, [&](const std::string &name) { trace.push_back("verify:" + name); });
        for (int i = 0; i < 2; ++i) {
            expected.insert(expected.end(), {"poison", "prepare", "execute", "verify:actual-" + std::to_string(i) + ".bin"});
        }
        for (int i = 0; i < 13; ++i) {
            if (!continuous) expected.push_back("poison");
            expected.insert(expected.end(), {"prepare", "execute"});
            if (!continuous) expected.push_back("verify:");
        }
        if (continuous) expected.insert(expected.end(), {"verify:benchmark-last.bin", "poison", "prepare", "execute", "verify:postcheck.bin"});
        assert(trace == expected);
        assert(timing.execute.size() == 10 && timing.prepare.size() == 10);
        std::ostringstream report;
        timing.Write(report);
        assert(report.str().find(continuous ? "continuous-v1" : "checked-v1") != std::string::npos);
    }
    for (bool inputCorruption : {false, true}) {
        Timing timing;
        timing.warmup = 3; timing.repetitions = 10; timing.continuous = true;
        int calls = 0, output = -1;
        bool inputIntact = true, rejected = false;
        try {
            RunMeasuredCalls(timing, [] {}, [&] {
                ++calls;
                // If later calls fail to write, a retained valid output must
                // not let the post-poison correctness check pass.
                if (inputCorruption || calls <= 2) output = 42;
                if (inputCorruption && calls == 7) inputIntact = false;
            }, [&](int) { output = -1; }, [&](const std::string &) {
                if (!inputIntact || output != 42) throw std::runtime_error("corrupt benchmark");
            });
        } catch (const std::runtime_error &) { rejected = true; }
        assert(rejected && calls == (inputCorruption ? 15 : 16));
    }
}
