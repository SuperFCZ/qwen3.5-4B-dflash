#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_add_custom.h"

namespace {
constexpr int64_t kRows = 8;
constexpr int64_t kCols = 2048;
constexpr int64_t kCount = kRows * kCols;
constexpr size_t kBytes = kCount * sizeof(aclFloat16);

template <typename T>
void Check(T result, const char *action)
{
    if (result != ACL_SUCCESS) {
        throw std::runtime_error(std::string(action) + " failed: " + std::to_string(result));
    }
}

struct DeviceTensor {
    void *data = nullptr;
    aclTensor *desc = nullptr;

    void Release()
    {
        if (desc != nullptr) aclDestroyTensor(desc);
        if (data != nullptr) aclrtFree(data);
    }
};

struct Resources {
    bool aclReady = false;
    bool deviceReady = false;
    aclrtStream stream = nullptr;
    void *workspace = nullptr;
    DeviceTensor x, y, z;
    int deviceId = 0;

    ~Resources()
    {
        if (workspace != nullptr) aclrtFree(workspace);
        z.Release();
        y.Release();
        x.Release();
        if (stream != nullptr) aclrtDestroyStream(stream);
        if (deviceReady) aclrtResetDevice(deviceId);
        if (aclReady) aclFinalize();
    }
};

void MakeTensor(DeviceTensor &tensor, const std::vector<aclFloat16> *host)
{
    Check(aclrtMalloc(&tensor.data, kBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    if (host != nullptr) {
        Check(aclrtMemcpy(tensor.data, kBytes, host->data(), kBytes, ACL_MEMCPY_HOST_TO_DEVICE),
              "copy input to device");
    }
    const int64_t shape[2] = {kRows, kCols};
    tensor.desc = aclCreateTensor(shape, 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                  shape, 2, tensor.data);
    if (tensor.desc == nullptr) throw std::runtime_error("aclCreateTensor failed");
}
}  // namespace

int main(int argc, char **argv)
{
    try {
        Resources r;
        if (argc > 2) throw std::runtime_error("usage: smoke_add_test [device_id]");
        if (argc == 2) r.deviceId = std::stoi(argv[1]);
        Check(aclInit(nullptr), "aclInit");
        r.aclReady = true;
        Check(aclrtSetDevice(r.deviceId), "aclrtSetDevice");
        r.deviceReady = true;
        Check(aclrtCreateStream(&r.stream), "aclrtCreateStream");

        std::vector<aclFloat16> x(kCount), y(kCount), actual(kCount);
        for (int64_t i = 0; i < kCount; ++i) {
            x[i] = aclFloatToFloat16(static_cast<float>(i % 17 - 8));
            y[i] = aclFloatToFloat16(static_cast<float>(i % 7 - 3));
        }
        MakeTensor(r.x, &x);
        MakeTensor(r.y, &y);
        MakeTensor(r.z, nullptr);

        uint64_t workspaceBytes = 0;
        aclOpExecutor *executor = nullptr;
        Check(aclnnAddCustomGetWorkspaceSize(r.x.desc, r.y.desc, r.z.desc,
                                             &workspaceBytes, &executor),
              "aclnnAddCustomGetWorkspaceSize");
        if (workspaceBytes > 0) {
            Check(aclrtMalloc(&r.workspace, workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "allocate workspace");
        }
        Check(aclnnAddCustom(r.workspace, workspaceBytes, executor, r.stream), "aclnnAddCustom");
        Check(aclrtSynchronizeStream(r.stream), "aclrtSynchronizeStream");
        Check(aclrtMemcpy(actual.data(), kBytes, r.z.data, kBytes, ACL_MEMCPY_DEVICE_TO_HOST),
              "copy output to host");

        for (int64_t i = 0; i < kCount; ++i) {
            const float expected = static_cast<float>(i % 17 - 8 + i % 7 - 3);
            const float got = aclFloat16ToFloat(actual[i]);
            if (got != expected) {
                throw std::runtime_error("mismatch at index " + std::to_string(i) +
                                         ": expected " + std::to_string(expected) +
                                         ", got " + std::to_string(got));
            }
        }
        std::cout << "PASS: AddCustom verified " << kCount << " FP16 elements\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
