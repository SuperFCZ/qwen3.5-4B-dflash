#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_smoke_matmul.h"

namespace {
constexpr int64_t kM = 16;
constexpr int64_t kK = 256;
constexpr int64_t kN = 64;

template <typename T>
void Check(T result, const char *action)
{
    if (result != ACL_SUCCESS) {
        const char *recent = aclGetRecentErrMsg();
        throw std::runtime_error(std::string(action) + " failed: " + std::to_string(result) +
                                 "; aclGetRecentErrMsg: " + (recent != nullptr ? recent : "<none>"));
    }
}

std::vector<aclFloat16> ReadHalf(const std::string &path, size_t count)
{
    std::vector<aclFloat16> values(count);
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char *>(values.data()), count * sizeof(aclFloat16)) ||
        file.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("input file has wrong size: " + path);
    }
    return values;
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
    int deviceId = 0;
    aclrtStream stream = nullptr;
    void *workspace = nullptr;
    DeviceTensor x, w, y;
    ~Resources()
    {
        if (workspace != nullptr) aclrtFree(workspace);
        y.Release();
        w.Release();
        x.Release();
        if (stream != nullptr) aclrtDestroyStream(stream);
        if (deviceReady) aclrtResetDevice(deviceId);
        if (aclReady) aclFinalize();
    }
};

void MakeTensor(DeviceTensor &tensor, const std::vector<int64_t> &shape,
                const std::vector<aclFloat16> *host)
{
    size_t count = 1;
    for (int64_t dim : shape) count *= static_cast<size_t>(dim);
    const size_t bytes = count * sizeof(aclFloat16);
    Check(aclrtMalloc(&tensor.data, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    if (host != nullptr) {
        Check(aclrtMemcpy(tensor.data, bytes, host->data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE),
              "copy input to device");
    }
    tensor.desc = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16, nullptr, 0,
                                  ACL_FORMAT_ND, shape.data(), shape.size(), tensor.data);
    if (tensor.desc == nullptr) throw std::runtime_error("aclCreateTensor failed");
}
}  // namespace

int main(int argc, char **argv)
{
    try {
        if (argc != 3) throw std::runtime_error("usage: smoke_matmul_test DEVICE_ID DATA_DIR");
        const std::string dataDir = argv[2];
        auto x = ReadHalf(dataDir + "/x.bin", kM * kK);
        auto w = ReadHalf(dataDir + "/w.bin", kN * kK);

        Resources r;
        r.deviceId = std::stoi(argv[1]);
        Check(aclInit(nullptr), "aclInit");
        r.aclReady = true;
        Check(aclrtSetDevice(r.deviceId), "aclrtSetDevice");
        r.deviceReady = true;
        Check(aclrtCreateStream(&r.stream), "aclrtCreateStream");
        MakeTensor(r.x, {kM, kK}, &x);
        MakeTensor(r.w, {kN, kK}, &w);
        MakeTensor(r.y, {kM, kN}, nullptr);

        uint64_t workspaceBytes = 0;
        aclOpExecutor *executor = nullptr;
        Check(aclnnSmokeMatmulGetWorkspaceSize(r.x.desc, r.w.desc, r.y.desc,
                                               &workspaceBytes, &executor),
              "aclnnSmokeMatmulGetWorkspaceSize");
        if (workspaceBytes > 0) {
            Check(aclrtMalloc(&r.workspace, workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "allocate workspace");
        }
        Check(aclnnSmokeMatmul(r.workspace, workspaceBytes, executor, r.stream),
              "aclnnSmokeMatmul");
        Check(aclrtSynchronizeStream(r.stream), "aclrtSynchronizeStream");

        std::vector<aclFloat16> actual(kM * kN);
        const size_t outputBytes = actual.size() * sizeof(aclFloat16);
        Check(aclrtMemcpy(actual.data(), outputBytes, r.y.data, outputBytes,
                          ACL_MEMCPY_DEVICE_TO_HOST), "copy output to host");
        std::ofstream output(dataDir + "/actual.bin", std::ios::binary);
        output.write(reinterpret_cast<const char *>(actual.data()), outputBytes);
        if (!output) throw std::runtime_error("failed to write actual.bin");
        std::cout << "ACLNN SmokeMatmul completed; checking PyTorch CPU reference next\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
