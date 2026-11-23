#include "workload.h"
#include "acl/acl.h"
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>

// Same CANN 9 direct-launch helpers as the SDK-generated 310P3 host stub.
extern "C" uint32_t RegisterAscendBinary(const char*, size_t, uint32_t, void**);
extern "C" uint32_t LaunchAscendKernel(void*, uint64_t, uint32_t, void**, uint32_t, const void*);
extern "C" int UnregisterAscendBinary(void*);

namespace {
using namespace vb;
constexpr unsigned Warmup = 3, Samples = 21, Batch = 3;
std::string phase = "startup";

void check(uint32_t rc, const char* what) {
    if (rc == ACL_SUCCESS) return;
    const char* detail = aclGetRecentErrMsg();
    throw std::runtime_error(std::string(what) + " returned " + std::to_string(rc) +
                             (detail && *detail ? std::string("\n") + detail : ""));
}
#define ACL_CHECK(expr) check((expr), #expr)

struct Options { int device = 0; uint32_t blocks = 0; bool vectorOnly = false; };

Options options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--vector-only") { o.vectorOnly = true; continue; }
        if ((arg != "--device" && arg != "--blocks") || i + 1 == argc)
            throw std::runtime_error("usage: run.sh [--device ID] [--blocks N] [--vector-only]");
        char* end = nullptr;
        const char* value = argv[++i];
        errno = 0;
        const long v = std::strtol(value, &end, 10);
        if (errno || end == value || *end || v < 0 || v > std::numeric_limits<int>::max() ||
            (arg == "--blocks" && (v == 0 || v > 40)))
            throw std::runtime_error("invalid " + arg + " " + value);
        if (arg == "--device") o.device = static_cast<int>(v);
        else o.blocks = static_cast<uint32_t>(v);
    }
    return o;
}

struct Session {
    int device = -1;
    bool initialized = false;
    aclrtStream stream = nullptr;
    void open(int id) {
        ACL_CHECK(aclInit(nullptr)); initialized = true;
        ACL_CHECK(aclrtSetDevice(id)); device = id;
        ACL_CHECK(aclrtCreateStream(&stream));
    }
    ~Session() {
        if (stream) aclrtDestroyStream(stream);
        if (device >= 0) aclrtResetDevice(device);
        if (initialized) aclFinalize();
    }
};

class Buffer {
public:
    explicit Buffer(size_t size) : bytes(size) { ACL_CHECK(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_NORMAL_ONLY)); }
    ~Buffer() { if (ptr) aclrtFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const void* data) { ACL_CHECK(aclrtMemcpy(ptr, bytes, data, bytes, ACL_MEMCPY_HOST_TO_DEVICE)); }
    void* ptr = nullptr;
    size_t bytes;
};

struct Kernel {
    std::vector<char> image;
    Buffer overflow{8};  // persistent implicit m200 argument
    void* handle = nullptr;
    Kernel() {
        std::ifstream f(VECTOR_PROBE_IMAGE, std::ios::binary | std::ios::ate);
        auto length = f.tellg();
        if (length < 64 || length > 16 * 1024 * 1024)
            throw std::runtime_error("cannot read device ELF: " VECTOR_PROBE_IMAGE);
        image.resize(static_cast<size_t>(length));
        f.seekg(0);
        if (!f.read(image.data(), length)) throw std::runtime_error("short read of device ELF");
        ACL_CHECK(aclrtMemset(overflow.ptr,overflow.bytes,0,overflow.bytes));
        ACL_CHECK(RegisterAscendBinary(image.data(),image.size(),2,&handle));
        if (!handle) throw std::runtime_error("SDK returned an empty AiCore binary handle");
    }
    ~Kernel() { if (handle) UnregisterAscendBinary(handle); }
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;
};

struct Inputs {
    const Kernel& kernel;
    Shape shape;
    Buffer nz, packed, scales, out;
    Inputs(const Kernel& program, Shape s, uint32_t blocks) : kernel(program), shape(s), nz(s.weights()), packed(s.weights()),
        scales(s.scaleWords()*2), out(std::max(size_t(blocks)*Elements*2, s.tapBytes())) {
        nz.upload(packCodes(s, false).data());
        packed.upload(packCodes(s, true).data());
        std::vector<uint16_t> gn(s.scaleWords());
        for (uint32_t g = 0; g < s.k / Group; ++g)
            for (uint32_t n = 0; n < s.n; ++n) gn[g*s.n+n] = aclFloatToFloat16(scaleAt(n,g));
        scales.upload(gn.data());
    }
};

struct Event {
    aclrtEvent handle = nullptr;
    Event() { ACL_CHECK(aclrtCreateEventWithFlag(&handle, ACL_EVENT_TIME_LINE)); }
    ~Event() { if (handle) aclrtDestroyEvent(handle); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

struct Case {
    Mode mode;
    const char* name;
    uint32_t blocks;
    std::vector<double> us{};
};

struct Args {
    void *codes, *scales, *output;
    uint32_t mode, n, kTiles, iterations;
    void* overflow;
};
static_assert(sizeof(void*) == 8 && offsetof(Args, mode) == 24 && offsetof(Args, overflow) == 40 && sizeof(Args) == 48);

Args arguments(const Case& row, const Inputs& io, uint32_t iterations = UbIterations) {
    return {flat(row.mode) ? io.packed.ptr : io.nz.ptr, io.scales.ptr, io.out.ptr, row.mode,
            io.shape.n, io.shape.kTiles(), iterations, io.kernel.overflow.ptr};
}

void launch(const Kernel& kernel, uint32_t blocks, Args& args, aclrtStream stream) {
    ACL_CHECK(LaunchAscendKernel(kernel.handle, 0, blocks, reinterpret_cast<void**>(&args), sizeof(args), stream));
}

void sanity(const Case& row, Inputs& io, aclrtStream stream) {
    // Exercise both UB banks; poison and verify outside all timing events.
    for (uint32_t iters : {2u, UbIterations}) {
        ACL_CHECK(aclrtMemset(io.out.ptr, io.out.bytes, 0xa5, io.out.bytes));
        auto args = arguments(row, io, iters);
        launch(io.kernel, row.blocks, args, stream);
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<uint8_t> output(streamMode(row.mode) ? io.shape.tapBytes() : size_t(row.blocks)*Elements*2);
        ACL_CHECK(aclrtMemcpy(output.data(), output.size(), io.out.ptr, output.size(), ACL_MEMCPY_DEVICE_TO_HOST));
        checkOutput(io.shape, row.mode, row.blocks, iters, output, aclFloatToFloat16);
        if (streamMode(row.mode)) break;
    }
}

double timeLaunches(const Case& row, Inputs& io, aclrtStream stream) {
    auto args = arguments(row, io);
    Event start, stop;
    ACL_CHECK(aclrtRecordEvent(start.handle, stream));
    for (unsigned i = 0; i < Batch; ++i) launch(io.kernel, row.blocks, args, stream);
    ACL_CHECK(aclrtRecordEvent(stop.handle, stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    float ms = 0;
    ACL_CHECK(aclrtEventElapsedTime(&ms, start.handle, stop.handle));
    if (!(ms > 0) || !std::isfinite(ms)) throw std::runtime_error("invalid elapsed time from NPU events");
    return double(ms) * 1000.0 / Batch;
}

using Rates = std::map<Mode,double>;
Rates measure(const std::string& scope, std::vector<Case> cases, Inputs& io, aclrtStream stream) {
    for (const auto& row : cases) {
        phase = scope + "/" + row.name + " check/warmup";
        sanity(row, io, stream);
        auto args = arguments(row, io);
        for (unsigned i = 0; i < Warmup; ++i) launch(io.kernel, row.blocks, args, stream);
        ACL_CHECK(aclrtSynchronizeStream(stream));
    }
    for (unsigned round = 0; round < Samples; ++round) {
        for (size_t j = 0; j < cases.size(); ++j) {
            auto& row = cases[round % 2 ? cases.size()-1-j : j];
            phase = scope + "/" + row.name + " sample " + std::to_string(round+1);
            row.us.push_back(timeLaunches(row,io,stream));
        }
    }
    Rates rates;
    std::cout << "\n" << scope << "\n"
              << "case                     blocks   p50_us   p95_us   Gelem/s   TFLOP/s   GM_in_GB/s\n";
    for (auto& row : cases) {
        std::sort(row.us.begin(),row.us.end());
        const double med = row.us[Samples/2], p95 = row.us[(95*Samples+99)/100-1];
        const double items = work(row.mode,io.shape,row.blocks,UbIterations);
        double rate = gigaPerSecond(items,med);
        // The all-block FMA row follows its optional one-block reference.
        rates[row.mode] = rate;
        std::cout << std::left << std::setw(25) << row.name << std::right << std::setw(5) << row.blocks
                  << std::fixed << std::setprecision(2) << std::setw(9) << med << std::setw(9) << p95
                  << std::setw(10) << rate;
        if (row.mode == Fp16Fma || row.mode == Fp16Mul)
            std::cout << std::setprecision(3) << std::setw(10) << fmaOrMulTF(row.mode,items,med);
        else std::cout << std::setw(10) << "-";
        if (streamMode(row.mode))
            std::cout << std::setprecision(2) << std::setw(13) << gigaPerSecond(io.shape.inputBytes(),med);
        else std::cout << std::setw(13) << "-";
        std::cout << "\n";
    }
    return rates;
}

void budgets(Shape shape, uint32_t blocks, const Rates& ub, const Rates& gm) {
    std::cout << "M16 projection rate model (ideal copy/vector overlap; excludes Cube and other work)\n"
              << "layout/scale             vector_ms    copy_ms    max_ms   equiv_MM_TF/s\n";
    struct Budget { const char* name; Mode convert, copy; };
    for (auto row : {Budget{"NZ/Brcb",NzBrcb,CopyNz}, Budget{"NZ/B16",NzB16,CopyNz},
                     Budget{"tile-ND/B16",FlatB16,CopyFlat}}) {
        const double vecMs = vectorBudgetMs(shape, blocks, ub.at(row.convert));
        const double copyMs = shape.weights() / (gm.at(row.copy)*1e6);
        const double ms = std::max(vecMs,copyMs);
        std::cout << std::left << std::setw(25) << row.name << std::right << std::fixed << std::setprecision(4)
                  << std::setw(9) << vecMs << std::setw(11) << copyMs << std::setw(10) << ms
                  << std::setw(16) << (2.0 * 16 * shape.weights() / (ms * 1e9)) << "\n";
    }
}

void run(const Options& opt) {
    Session session;
    session.open(opt.device);
    std::ifstream maps("/proc/self/maps");
    for (std::string line; std::getline(maps,line); ) {
        if (line.find(".so") != std::string::npos &&
            (line.find("/simulator/") != std::string::npos || line.find("/tikicpulib/") != std::string::npos))
            throw std::runtime_error("hardware timing requires the device runtime; loaded: " + line);
    }
    const char* soc = aclrtGetSocName();
    if (!soc || std::string(soc).find("Ascend310P") != 0)
        throw std::runtime_error("bench targets Ascend310P3; runtime SoC: " + std::string(soc ? soc : "unknown"));
    int64_t cores = 0;
    auto rc = aclrtGetDeviceInfo(opt.device, ACL_DEV_ATTR_AICORE_CORE_NUM, &cores);
    std::string source = "ACL_DEV_ATTR_AICORE_CORE_NUM";
    if (rc != ACL_SUCCESS || cores <= 0) {
        uint32_t available = 0;
        auto alt = aclrtGetResInCurrentThread(ACL_RT_DEV_RES_CUBE_CORE, &available);
        if (alt == ACL_SUCCESS && available > 0) { cores = available; source = "thread cube-core limit"; }
        else {
            cores = 0; source = "unavailable";
            if (!opt.blocks) throw std::runtime_error("cannot query AI Core count (ACL returns " +
                std::to_string(rc) + ", " + std::to_string(alt) + "); pass --blocks N for the grid you want to measure");
        }
    }
    if (!opt.blocks && cores > 40) throw std::runtime_error("unexpected 310P3 core count; pass --blocks N (1..40)");
    const uint32_t blocks = opt.blocks ? opt.blocks : uint32_t(cores);
    std::cout << std::unitbuf << "SoC=" << soc << " device=" << opt.device << " " << source << "=" << cores
              << " launch_blocks=" << blocks << "  target=dav-m200/AiCore  timer=ACL device events\n"
              << "UB tile=64x256  iterations/block=" << UbIterations << "  warmup=" << Warmup
              << "  samples=" << Samples << "  launches/sample=" << Batch << "\n";
    Rates ub;
    phase = "load AiCore kernel";
    Kernel kernel;
    {
        phase = "UB allocation/upload";
        Inputs io(kernel,{blocks*NTile, 2*KTile},blocks);
        std::vector<Case> cases;
        if (blocks != 1) cases.push_back({Fp16Fma,"FP16 MAC (1 block)",1});
        for (auto row : {Case{Fp16Fma,"FP16 MAC",blocks}, Case{Fp16Mul,"FP16 Mul",blocks},
                         Case{CastFlat,"Cast tile-ND",blocks}, Case{CastNz,"Cast NZ -> ND",blocks},
                         Case{NzBrcb,"NZ + Brcb + Mul",blocks}, Case{NzB16,"NZ + B16 + Mul",blocks},
                         Case{FlatB16,"tile-ND + B16 + Mul",blocks}}) cases.push_back(row);
        ub = measure("UB resident (Gelem/s = lanes or INT8 weights/s, billions)", cases, io, session.stream);
    }
    if (opt.vectorOnly) return;
    for (Shape shape : {Shape{19456,2560}, Shape{2560,9728}}) {
        const std::string label = shape.n == 19456 ? "Gate/Up" : "Down";
        phase = label + " allocation/packing/upload";
        Inputs io(kernel,shape,blocks);
        std::vector<Case> cases{{CopyNz,"copy NZ + GN",blocks},{CopyFlat,"copy tile-ND + GN",blocks},
            {StreamNzBrcb,"NZ + Brcb + Mul",blocks},{StreamNzB16,"NZ + B16 + Mul",blocks},
            {StreamFlatB16,"tile-ND + B16 + Mul",blocks}};
        auto gm = measure(label + " GM stream: K=" + std::to_string(shape.k) + " N=" + std::to_string(shape.n) +
                          "  logical input bytes/launch=" + std::to_string(shape.inputBytes()),cases,io,session.stream);
        budgets(shape,blocks,ub,gm);
    }
}
}  // namespace

int main(int argc, char** argv) {
    try { run(options(argc,argv)); }
    catch (const std::exception& e) {
        std::cerr << "ERROR [" << phase << "] " << e.what() << "\n"; return 1;
    }
    return 0;
}
