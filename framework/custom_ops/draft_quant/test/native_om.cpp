// Isolated static native WeightQuant OM reference; no model-runtime edits.
#include "runner_common.h"
#include "../op_host/d_flash_group_quant_linear_contract.h"
#include <iostream>

namespace {
using namespace draft_quant_test;
struct Session {
    bool initialized = false, deviceSet = false, loaded = false;
    int device = 0;
    uint32_t model = 0;
    aclrtStream stream = nullptr;
    aclmdlDesc *desc = nullptr;
    aclmdlDataset *inputs = nullptr, *outputs = nullptr;
    aclDataBuffer *input = nullptr, *output = nullptr;
    GuardedBuffer x, y, workspace, weights;
    ~Session()
    {
        if (stream) aclrtSynchronizeStream(stream);
        if (inputs) aclmdlDestroyDataset(inputs);
        if (outputs) aclmdlDestroyDataset(outputs);
        if (input) aclDestroyDataBuffer(input);
        if (output) aclDestroyDataBuffer(output);
        if (desc) aclmdlDestroyDesc(desc);
        if (loaded) aclmdlUnload(model);
        x.Release(); y.Release(); workspace.Release(); weights.Release();
        if (stream) aclrtDestroyStream(stream);
        if (deviceSet) aclrtResetDevice(device);
        if (initialized) aclFinalize();
    }
    void Load(const char *path)
    {
        Check(aclInit(nullptr), "aclInit"); initialized = true;
        Check(aclrtSetDevice(device), "aclrtSetDevice"); deviceSet = true;
        Check(aclrtCreateStream(&stream), "aclrtCreateStream");
        size_t workBytes = 0, weightBytes = 0;
        Check(aclmdlQuerySize(path, &workBytes, &weightBytes), "aclmdlQuerySize");
        if (workBytes) workspace.Allocate(workBytes);
        if (weightBytes) weights.Allocate(weightBytes);
        Check(aclmdlLoadFromFileWithMem(path, &model, workspace.data, workBytes,
                                       weights.data, weightBytes), "aclmdlLoadFromFileWithMem");
        loaded = true;
        desc = aclmdlCreateDesc();
        if (!desc) throw std::runtime_error("aclmdlCreateDesc failed");
        Check(aclmdlGetDesc(desc, model), "aclmdlGetDesc");
    }
    void Bind(int64_t m, int64_t k, int64_t n, const std::vector<uint8_t> &values)
    {
        if (aclmdlGetNumInputs(desc) != 1 || aclmdlGetNumOutputs(desc) != 1)
            throw std::runtime_error("native OM must expose only x -> y");
        for (bool out : {false, true}) {
            aclmdlIODims dims{};
            Check(out ? aclmdlGetOutputDims(desc, 0, &dims) : aclmdlGetInputDims(desc, 0, &dims),
                  "aclmdlGetIODims");
            const auto dtype = out ? aclmdlGetOutputDataType(desc, 0) : aclmdlGetInputDataType(desc, 0);
            const auto bytes = out ? aclmdlGetOutputSizeByIndex(desc, 0) : aclmdlGetInputSizeByIndex(desc, 0);
            const auto columns = out ? n : k;
            if (dtype != ACL_FLOAT16 || dims.dimCount != 2 || dims.dims[0] != m ||
                dims.dims[1] != columns || bytes != static_cast<size_t>(m * columns * 2))
                throw std::runtime_error("native OM ABI differs from FP16 [M,K] -> [M,N]");
        }
        x.Allocate(values.size()); x.CopyIn(values);
        y.Allocate(m * n * 2);
        inputs = aclmdlCreateDataset(); outputs = aclmdlCreateDataset();
        input = aclCreateDataBuffer(x.data, x.bytes); output = aclCreateDataBuffer(y.data, y.bytes);
        if (!inputs || !outputs || !input || !output) throw std::runtime_error("create dataset failed");
        Check(aclmdlAddDatasetBuffer(inputs, input), "add input");
        Check(aclmdlAddDatasetBuffer(outputs, output), "add output");
    }
};
int64_t Number(const char *text)
{
    size_t end = 0;
    const std::string value(text);
    auto number = std::stoll(value, &end);
    if (end != value.size()) throw std::runtime_error("invalid integer");
    return number;
}
}  // namespace

int main(int argc, char **argv)
{
    std::string directory;
    try {
        if (argc != 9 && argc != 10) throw std::runtime_error(
            "usage: dflash_native_om_test DEVICE CASE_DIR OM M K N WARMUP REPETITIONS [--continuous]");
        directory = argv[2];
        std::ofstream(directory + "/execution.json") << "{\"status\":\"RUNNING\"}\n";
        const auto m = Number(argv[4]), k = Number(argv[5]), n = Number(argv[6]);
        const auto warmup = Number(argv[7]), repetitions = Number(argv[8]);
        const auto device = Number(argv[1]);
        if (!draft_quant_contract::IsSupportedShape(m, k, n) || device < 0 || device > 65535 ||
            warmup < 3 || warmup > 100 || repetitions < 10 || repetitions > 1000)
            throw std::runtime_error("unsupported shape/device or timing counts");
        if (argc == 10 && std::string(argv[9]) != "--continuous") throw std::runtime_error("unknown timing option");
        Timing timing;
        timing.continuous = argc == 10;
        timing.collectPrepare = false;
        timing.warmup = static_cast<int>(warmup);
        timing.repetitions = static_cast<int>(repetitions);
        const auto x = ReadBytes(directory + "/x.bin", m * k * 2);
        Session session;
        session.device = static_cast<int>(device);
        session.Load(argv[3]); session.Bind(m, k, n, x);
        std::vector<uint8_t> first;
        auto execute = [&]() {
            Check(aclmdlExecuteAsync(session.model, session.inputs, session.outputs, session.stream),
                  "aclmdlExecuteAsync");
            Check(aclrtSynchronizeStream(session.stream), "aclrtSynchronizeStream");
        };
        auto poison = [&](int repeat) {
            Check(aclrtMemset(session.y.data, session.y.bytes, 0x7F + (repeat % 2) * 128,
                              session.y.bytes), "poison output");
        };
        auto verify = [&](const std::string &outputName) {
            session.x.CheckGuards(); session.y.CheckGuards();
            session.workspace.CheckGuards(); session.weights.CheckGuards();
            if (session.x.CopyOut() != x) throw std::runtime_error("native OM modified input");
            const auto actual = session.y.CopyOut();
            if (first.empty()) first = actual;
            if (actual != first) throw std::runtime_error("native OM output repeat drift");
            if (!outputName.empty()) WriteBytes(directory + "/" + outputName, actual);
        };
        RunMeasuredCalls(timing, []() {}, execute, poison, verify);
        std::ofstream report(directory + "/execution.json");
        report << "{\"status\":\"PASS\",\"runtime\":\"AscendCL native OM\",\"cpu_fallback\":false,"
                  "\"input_readonly\":true,\"guards_intact\":true,\"io_validated\":true,\"repetitions\":2,"
                  "\"m\":" << m << ",\"k\":" << k << ",\"n\":" << n << ",\"device_id\":" << device
               << ",\"workspace_bytes\":" << session.workspace.bytes
               << ",\"model_weight_bytes\":" << session.weights.bytes
               << ",\"tracked_device_allocation_bytes\":"
               << (session.x.bytes + session.y.bytes + session.workspace.bytes + session.weights.bytes +
                   (2 + (session.workspace.bytes ? 1 : 0) + (session.weights.bytes ? 1 : 0)) * 2 * kGuard);
        timing.Write(report);
        report << "}\n";
        if (!report) throw std::runtime_error("cannot save native execution report");
        std::cout << "Native OM execution completed; guards/input intact; numerical comparison follows\n";
        return 0;
    } catch (const std::exception &error) {
        if (!directory.empty()) std::ofstream(directory + "/execution.json") << "{\"status\":\"FAIL\"}\n";
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
