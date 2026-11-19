from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from w16_common import DOUBLE, HEADER, OVERLAP, REVIEW, STEM, c, render_kernel, replace_once
from a3_launch import configure_build

sys.path.insert(0, str(REVIEW))
from review_contracts import schedule, check_trace


class CandidateKernelTests(unittest.TestCase):
    def test_exact_shared_loop_all_k_failure_and_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            exe = Path(temp) / "schedule"
            subprocess.run(
                ["c++", "-std=c++17", str(HERE / "test/schedule_test.cpp"), "-o", str(exe)], check=True
            )
            for overlap in (False, True):
                for fail in range(-1, 10):
                    emitted = subprocess.check_output([exe, str(int(overlap)), str(fail)], text=True)
                    steps = (
                        [("reserve", -1)]
                        + [(s, int(k)) for s, k in (l.split() for l in emitted.splitlines())]
                        + [("release", -1), ("end", -1)]
                    )
                    self.assertEqual(steps, list(schedule(overlap=overlap, fail_at=None if fail < 0 else fail)))
                    self.assertEqual(check_trace(steps, 10)[1], fail < 0)

    def test_build_input_exact_and_only_one_schedule_switch(self):
        raw = (c.QUANT / "op_kernel" / (STEM + ".cpp")).read_bytes()
        b, cand = (render_kernel(raw, mode) for mode in DOUBLE)
        self.assertEqual(b.replace(b"w16::Run<false>", b"w16::Run<true>"), cand)
        with self.assertRaises(ValueError):
            render_kernel(raw + b" ", OVERLAP)
        self.assertNotIn(b"w16::", render_kernel(raw, "down-2d"))

    def test_compiled_candidate_nz_group_rounding_and_dma_lifetimes(self):
        # Extend only the historical CPU fixture's missing API surface in a temp dir.
        # Physical offsets here model TPipe; never submit them as actual static evidence.
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            include = c.QUANT / "test/cpu_kernel_include"
            shutil.copytree(include, temp / "inc")
            header = temp / "inc/kernel_operator.h"
            text = header.read_text()
            if platform.system() == "Linux" and platform.machine() == "aarch64":
                text = replace_once(text, "using half = _Float16;", "using half = __fp16;")
            text = replace_once(
                text,
                "using TEventID = int;",
                """using TEventID = int;
constexpr int QUE_MAX_EVENT=8;
inline std::map<uintptr_t, std::pair<uint32_t,uint32_t>> modelRanges;
inline uint32_t highWater=0;""",
            )
            text = replace_once(
                text,
                "    Tensor operator[]",
                """    void SetSize(uint32_t n) { assert(n<=count); count=n; }
    uint64_t GetPhyAddr() const {
        uintptr_t addr=reinterpret_cast<uintptr_t>(data);
        auto it=modelRanges.upper_bound(addr); assert(it!=modelRanges.begin()); --it;
        assert(addr-it->first<it->second.second); return it->second.first+addr-it->first;
    }
    Tensor operator[]""",
            )
            text = replace_once(
                text, "TPipe() { cpuSync = {}; }", "TPipe() { cpuSync = {}; highWater=0;modelRanges.clear(); }"
            )
            text = (
                replace_once(text, "void InitBuffer(TBuf<P>&", "bool InitBuffer(TBuf<P>&")
                if "void InitBuffer(TBuf<P>&" in text
                else replace_once(text, "void InitBuffer(TBuf<P> &", "bool InitBuffer(TBuf<P> &")
            )
            text = replace_once(
                text,
                "        cpuMetrics.ubBytes += bytes;",
                """        cpuMetrics.ubBytes += bytes;
        auto address=buffer.template Get<uint8_t>(bytes).data;
        modelRanges[reinterpret_cast<uintptr_t>(address)]={highWater,static_cast<uint32_t>(bytes)};
        highWater+=bytes;return true;""",
            )
            # Scope the legacy test adapter to the additional validated Down descriptor pair.
            text = text.replace(
                "params.srcStride==19392", "(params.srcStride==19392 || params.srcStride==2496)"
            )
            text = text.replace("params.srcStride==1212", "(params.srcStride==1212 || params.srcStride==156)")
            header.write_text(text)
            mm = temp / "inc/lib/matmul_intf.h"
            mm.write_text(
                replace_once(
                    mm.read_text(),
                    "namespace matmul {",
                    "namespace matmul {\nstruct CpuConfig{bool doNorm,enVecND2NZ,enableMixDualMaster;}; constexpr CpuConfig CFG_NORM{true,false,false};",
                )
            )
            configure_build(
                temp / HEADER,
                temp / "header.json",
                0,
                "batched",
                None,
                "prefetch",
                "a-ub",
                "broadcast",
                256,
                "loop",
                "raw",
            )
            raw = (c.QUANT / "op_kernel" / (STEM + ".cpp")).read_bytes()
            for mode in (*DOUBLE, "down-2d"):
                (temp / "candidate.cpp").write_bytes(render_kernel(raw, mode))
                exe = temp / mode
                subprocess.run(
                    [
                        "c++",
                        "-std=c++17",
                        "-O2",
                        "-ffp-contract=off",
                        "-I" + str(temp),
                        "-I" + str(temp / "inc"),
                        str(HERE / "test/kernel_fixture.cpp"),
                        "-o",
                        str(exe),
                    ],
                    check=True,
                )
                result = subprocess.check_output([exe, "down" if mode == "down-2d" else "gate"], text=True)
                self.assertIn("CPU_CANDIDATE_OWNERS_AND_BITS_MATCH", result)
