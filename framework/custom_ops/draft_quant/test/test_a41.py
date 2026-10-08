"""CPU-only A4.1 metadata and evidence regressions. No device is executed."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from a2_common import record, sha256, write_json
from a3_launch import CONFIG, PREFIX, CUBE_PREFIX, configure_build, load_build_config, launch_evidence, pipeline_plan
from compare_a41 import compatible, match_plans, run as compare_runs
from run_a41 import KV_CASES, run
from profile_a31 import collect_pipe_counters
from test_a3 import launch_record


def cube(shape, mode):
    m,k,n = shape
    staged = mode == "a-ub" and shape == (80,2560,2048)
    return dict(version=1, kv_m80_mode=mode, a_source="VECOUT" if staged else "GM",
                a_stage_bytes=20480 if staged else 0, a_row_stride=128 if staged else k,
                base_m=m, base_n=64, base_k=128, depth_a1=2, depth_b1=2, step_m=1, step_n=1,
                step_ka=1, step_kb=1, db_l0a=2, db_l0b=2, db_l0c=1, trans_length=20480,
                weight_dequantizations_per_n_tile=k//128)


def plan(shape, mode):
    m,k,n = shape
    resource = cube(shape,mode)
    p = dict(launch_record(k,n,available=7), version=5, m=m, tile_m=m, weight_reuse_rows=m,
             dequant_mode="batched", **pipeline_plan(k,n,"serial"))
    p["user_ub_bytes"] += resource["a_stage_bytes"]
    p["matmul_ub_bytes"] = 262144 - p["user_ub_bytes"]
    return dict(p,cube=resource)


def log_plan(path, value):
    value = dict(value); resource = value.pop("cube")
    path.write_text(PREFIX+json.dumps(value)+"\n"+CUBE_PREFIX+json.dumps(resource)+"\n")


class A41Tests(unittest.TestCase):
    def test_profile_keeps_mte2_column_units_and_missing_is_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/"profiler").mkdir()
            self.assertEqual(collect_pipe_counters(root)["status"],"NOT_FOUND")
            (root/"profiler/op_summary.csv").write_text("Op Type,Task Duration(us),aic_mte2_time(us),aic_mte2_ratio,aic_mac_ratio\n"
                "DFlashGroupQuantLinear,3057.8,2283.4,0.893,0.731\nOther,1,1,1,1\n")
            result=collect_pipe_counters(root)
            self.assertEqual(len(result["rows"]),1)
            self.assertEqual(result["rows"][0]["fields"]["aic_mte2_time(us)"],"2283.4")
            self.assertEqual(result["rows"][0]["fields"]["aic_mte2_ratio"],"0.893")

    def test_generated_mode_is_explicit_and_hash_bound(self):
        original = CONFIG.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); host, kernel, config = (root/name for name in ("host.h","kernel.h","build.json"))
            configure_build(host,config,0,"batched",kernel,"serial","a-ub")
            self.assertEqual(load_build_config(config)["kv_m80_mode"], "a-ub")
            self.assertEqual(host.read_bytes(),kernel.read_bytes())
            self.assertEqual(CONFIG.read_bytes(), original)
            kernel.write_text(kernel.read_text().replace("KV_M80_MODE 1U","KV_M80_MODE 0U"))
            with self.assertRaisesRegex(ValueError,"host/kernel"): load_build_config(config)

    def test_a_source_stride_and_reuse_are_shape_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory)/"runner.log"
            for shape in ((80,2560,2048),(32,2560,2048),(16,2560,19456),(16,9728,2560),(64,12800,2560)):
                left, right = [], []
                for mode in ("baseline","a-ub"):
                    p = plan(shape,mode); log_plan(log,p)
                    got = launch_evidence(log,shape,0,2097152,"batched","serial",5,mode)
                    (left if mode=="baseline" else right).append(got)
                    self.assertEqual(got["weight_reuse_rows"],shape[0])
                    self.assertEqual(got["cube"]["a_stage_bytes"],20480 if mode=="a-ub" and shape==(80,2560,2048) else 0)
                match_plans(left[0],right[0],shape)
            shape=(80,2560,2048)
            for key,value in (("a_source","GM"),("a_row_stride",2560),("a_stage_bytes",0),("base_m",32),
                              ("weight_dequantizations_per_n_tile",60)):
                bad=plan(shape,"a-ub"); bad["cube"][key]=value; log_plan(log,bad)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    launch_evidence(log,shape,0,2097152,"batched","serial",5,"a-ub")
            bad=plan(shape,"a-ub"); bad["user_ub_bytes"]-=20480; log_plan(log,bad)
            with self.assertRaises(ValueError): launch_evidence(log,shape,0,2097152,"batched","serial",5,"a-ub")

    def test_pairing_rejects_changed_other_shape_resources_and_protocol(self):
        a=dict(device_id=0,stop_after="full",timing_protocol="continuous-v1",
               build=dict(source_sha256={"kernel":"same"},core_limit=0,dequant_mode="batched",pipeline_mode="serial",
                          tiling_abi="full-m-v2",launch_version=5))
        b=copy.deepcopy(a); compatible(a,b)
        b["build"]["pipeline_mode"]="prefetch"
        with self.assertRaises(ValueError): compatible(a,b)
        shape=(32,2560,2048); p,q=plan(shape,"baseline"),plan(shape,"a-ub")
        q["cube"]["depth_a1"]=4
        with self.assertRaisesRegex(ValueError,"non-target"): match_plans(p,q,shape)

    def test_failed_kv80_stops_before_kv32_and_full_a4(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); cfg=root/"build.json"
            configure_build(root/"host.h",cfg,0,"batched",root/"kernel.h","serial","a-ub")
            args=SimpleNamespace(build_config=cfg,output_dir=root/"run",device_id=0,stop_after="full",
                                 c16_bundle=root/"c16.json",c64_bundle=root/"c64.json",
                                 c16_native_om_manifest=root/"n16.json",c64_native_om_manifest=root/"n64.json")
            with patch("run_a41.load_bundle"),patch("run_a41.a4_contract.paired_gears"), \
                 patch("run_a41.run_a2.run",return_value=1) as child, patch("run_a41.run_a4.run") as full, redirect_stdout(io.StringIO()):
                self.assertEqual(run(args),1)
            self.assertEqual(child.call_count,1)
            self.assertEqual(child.call_args.args[0].case_filter,KV_CASES)
            self.assertEqual(child.call_args.args[0].bundle,args.c64_bundle)
            full.assert_not_called()
            result=json.loads((root/"run/suite.json").read_text())
            self.assertEqual(result["full_a4_regression"],"NOT_RUN")

    def test_comparator_rechecks_postcheck_bytes_and_reports_p95(self):
        # Real dimensions, fake finite byte fixtures, never device execution.
        shape=(80,2560,2048)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root=Path(directory)
            capture=root/"capture.json"; native=root/"native.json"
            capture.write_text("CPU fixture"); native.write_text("CPU fixture")
            summaries=[]
            for mode in ("baseline","a-ub"):
                dest=root/mode; dest.mkdir(); realroot=dest/"kv80"; realroot.mkdir()
                cases=[]
                for name in KV_CASES:
                    case=dict(name=name,status="PASS",executions={},outputs={},bracket_outputs={})
                    for kind in ("custom","native_om"):
                        files=[]; location=realroot/name/kind; location.mkdir(parents=True)
                        for filename in ("actual-0.bin","actual-1.bin","benchmark-last.bin","postcheck.bin"):
                            path=location/filename; path.write_bytes(b"\x00\x3c"*(80*2048)); files.append(record(path,realroot))
                        case["outputs"][kind],case["bracket_outputs"][kind]=files[:2],files[2:]
                        timing=dict(status="MEASURED",protocol="continuous-v1",warmup=5,repetitions=30,
                                    execute_sync=dict(samples_ms=[1.0]*30),prepare=dict(samples_ms=[0.01]*30),
                                    per_timed_call_readback=False,per_timed_call_poison=False,
                                    correctness_before_calls=2,correctness_after_calls=1,timed_tail_checked=True)
                        case["executions"][kind]=dict(status="PASS",runtime="AscendCL ACLNN" if kind=="custom" else "AscendCL native OM",
                            m=80,k=2560,n=2048,device_id=0,op="DFlashGroupQuantLinear",tile_n=64,tile_k=128,
                            cpu_fallback=False,input_readonly=True,guards_intact=True,io_validated=True,repetitions=2,
                            workspace_bytes=2097152,tracked_device_allocation_bytes=3000000,timing=timing)
                        if kind=="custom": log_plan(location/"runner.log",plan(shape,mode))
                    cases.append(case)
                write_json(realroot/"suite.json",dict(abi="dflash-group-quant-linear-a4-real-v1",status="PASS",context_rows=64,
                    cases=cases,timing_protocol="continuous-v1",device_id=0,bundle=str(capture),bundle_sha256=sha256(capture),
                    native_om_manifest=str(native),native_om_manifest_sha256=sha256(native)))
                report=dict(abi="dflash-group-quant-linear-a41-v1",status="PASS",device_id=0,stop_after="kv80",
                    timing_protocol="continuous-v1",full_a4_regression="NOT_RUN",build=dict(kv_m80_mode=mode,
                    launch_version=5,core_limit=0,dequant_mode="batched",pipeline_mode="serial",tiling_abi="full-m-v2",source_sha256={"kernel":"same"}),
                    checks=dict(kv80=dict(status="PASS",summary=record(realroot/"suite.json",dest))))
                write_json(dest/"suite.json",report); summaries.append(dest/"suite.json")
            self.assertEqual(compare_runs(*summaries,root/"ok.json"),0)
            result=json.loads((root/"ok.json").read_text())
            self.assertEqual(result["cases"][0]["candidate"]["p95_ms"],1.0)
            # Consistently rehashing an incorrect postcheck must still fail bits.
            realpath=root/"a-ub/kv80/suite.json"; real=json.loads(realpath.read_text())
            tail=root/"a-ub/kv80/layer-0-kv/custom/postcheck.bin"
            tail.write_bytes(b"\x00\x40"+tail.read_bytes()[2:])
            real["cases"][0]["bracket_outputs"]["custom"][1]=record(tail,realpath.parent); write_json(realpath,real)
            report=json.loads(summaries[1].read_text()); report["checks"]["kv80"]["summary"]=record(realpath,summaries[1].parent)
            write_json(summaries[1],report)
            self.assertEqual(compare_runs(*summaries,root/"bad.json"),1)


if __name__ == "__main__": unittest.main()
