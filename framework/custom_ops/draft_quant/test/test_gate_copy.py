"""CPU-only Gate copy experiment tests. Never hardware performance evidence."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from a2_common import record,sha256,write_json
from a3_launch import configure_build,load_build_config
import gate_copy_sdk
import run_gate_copy as gate
import run_gate_copy_pair as pair

HERE=Path(__file__).resolve().parent
FLAGS=['-DDFLASH_GROUP_QUANT_K_TILE=256','-DDFLASH_GROUP_QUANT_SCALE_MODE=1','-DDFLASH_GROUP_QUANT_KV_M80_MODE=1','-DDFLASH_GROUP_QUANT_PIPELINE_MODE=1','-DDFLASH_GROUP_QUANT_GATE_COPY_MODE=1']


class GateCopyTests(unittest.TestCase):
    def test_real_kernel_and_all_3040_raw_tiles(self):
        compiler=shutil.which('c++')
        if not compiler:self.skipTest('C++ CPU model unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            for source in ('gate_copy_model.cpp','kernel_model.cpp'):
                binary=Path(tmp)/source
                compiled=subprocess.run([compiler,'-std=c++17','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-Wno-unused-parameter',*FLAGS,'-I',str(HERE/'cpu_kernel_include'),str(HERE/source),'-o',str(binary)],capture_output=True,text=True)
                self.assertEqual(compiled.returncode,0,compiled.stdout+compiled.stderr)
                result=subprocess.run([str(binary)],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertIn('NPU NOT_RUN',result.stdout)

    def test_build_default_opt_in_binding_and_invalid_combinations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for mode in ('loop','2d'):
                configure_build(root/'h',root/'config',0,'batched',root/'kh','prefetch','a-ub','broadcast',256,mode)
                c=load_build_config(root/'config');self.assertEqual(c['gate_copy_mode'],mode)
                self.assertEqual((root/'h').read_bytes(),(root/'kh').read_bytes())
            c['gate_copy_mode']='loop';write_json(root/'config',c)
            with self.assertRaisesRegex(ValueError,'Gate copy'):load_build_config(root/'config')
            for kt,scale in ((128,'broadcast'),(256,'scalar')):
                with self.assertRaisesRegex(ValueError,'Gate copy'):configure_build(root/'h',root/'config',0,'batched',root/'kh','serial','a-ub',scale,kt,'2d')
        self.assertIn('#define DFLASH_GROUP_QUANT_GATE_COPY_MODE 0U',(HERE.parent/'op_host/d_flash_group_quant_linear_build_config.h').read_text())

    def test_sdk_evidence_rejects_wrong_arch_and_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'version.info').write_text('Version=9.0.0\n')
            (root/'kernel_struct_data_copy.h').write_text('DataCopyParams uint16_t blockCount, blockLen, srcStride, dstStride;')
            impl=root/'dav_m200/kernel_operator_data_copy_impl.h';impl.parent.mkdir();impl.write_text('copy_gm_to_ubuf(dst,src,blockCount,blockLen,srcStride,dstStride);')
            r=gate_copy_sdk.inspect(root);self.assertEqual(r['status'],'SOURCE_INSPECTED');self.assertEqual(r['npu_correctness'],'NOT_RUN')
            (root/'version.info').write_text('Version=8.5.0\n')
            with self.assertRaisesRegex(ValueError,'9.0.0'):gate_copy_sdk.inspect(root)
            (root/'version.info').write_text('Version=9.0.0\n');impl.rename(root/'kernel_operator_data_copy_impl.h')
            with self.assertRaisesRegex(ValueError,'dav_m200'):gate_copy_sdk.inspect(root)

    def test_copy_log_and_untimed_gate_are_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'log';log.write_text(gate.COPY_PREFIX+'{"version":1,"mode":"2d","selected":"2d"}\n')
            self.assertEqual(gate.copy_evidence(log,'2d')['raw_calls_per_tile'],2)
            with self.assertRaises(ValueError):gate.copy_evidence(log,'loop')
        r=dict(status='PASS',runtime='AscendCL ACLNN',op='DFlashGroupQuantLinear',cpu_fallback=False,input_readonly=True,guards_intact=True,repetitions=2,device_id=0,m=16,k=2560,n=19456,tile_n=64,tile_k=256,tile_m=16,weight_reuse_rows=16,timing=dict(status='NOT_RUN',warmup=0,repetitions=0))
        gate.validate_untimed(r,0)
        for key,value in (('guards_intact',False),('input_readonly',False),('tile_k',128),('cpu_fallback',True),('m',32)):
            with self.assertRaises(ValueError):gate.validate_untimed(dict(r,**{key:value}),0)

    def test_pair_builds_separate_opps_only_gate_capture_and_retains_not_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);files={}
            for name in ('bundle','native','golden','accepted'):
                f=root/name;f.write_text('CPU fixture');files[name]=str(f)
            source=dict(bundle=files['bundle'],native=files['native'],golden=files['golden'],config=dict(core_limit=0,pipeline_mode='prefetch',kv_m80_mode='a-ub'))
            variants=[]
            def launch(command,**kwargs):
                env=kwargs['env'];mode=env['DFLASH_GATE_COPY_MODE'];variants.append(mode)
                self.assertEqual(env['DFLASH_SUITE'],'gate-copy');self.assertEqual(env['DFLASH_K_TILE'],'256')
                self.assertEqual(env['GATE_LAYERS'],'0');self.assertEqual(env['GATE_GOLDEN_SUITE'],files['golden'])
                run=root/mode;data=run/'data';data.mkdir(parents=True);write_json(data/'suite.json',dict(status='PASS'))
                write_json(run/'sdk-copy.json',dict(status='SOURCE_INSPECTED'))
                write_json(run/'opp-preflight.json',dict(status='PASS'));(run/'source-commit.txt').write_text('CPU fixture')
                Path(env['DFLASH_SUMMARY_POINTER']).write_text(str(data/'suite.json'))
            def compare(left,right,out):write_json(out,dict(status='PASS',runtime='CPU fixture'));return dict(status='PASS')
            args=SimpleNamespace(output_dir=root/'pair',a51_evidence=Path(files['accepted']),layers=[0],device_id=0,order='baseline-first')
            with patch.object(pair,'source_evidence',return_value=source),patch.object(pair.subprocess,'run',side_effect=launch),patch.object(pair,'compare_runs',side_effect=compare):self.assertEqual(pair.run(args),0)
            self.assertEqual(variants,['loop','2d'])
            result=json.loads((root/'pair/suite.json').read_text());self.assertEqual(result['full_draft_token_kv'],'NOT_RUN')
            self.assertEqual(result['adoption'],'NOT_DECIDED')

    def test_failed_pre_gate_never_runs_performance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            paths={name:root/name for name in ('bundle','native_om_manifest','golden_suite','build_config','runner','om_runner')}
            for p in paths.values():p.write_text('CPU fixture')
            case=dict(name='layer-0-gate_up',files={})
            for name in ('x.bin','w_nz.bin','s_gn.bin'):
                p=root/name;p.write_bytes(b'fixture');case['files'][name]=record(p,root)
            expected=root/'expected.bin';expected.write_bytes(b'\0\0'*(16*19456));out=record(expected,root)
            old=dict(name=case['name'],status='PASS',executions={'native_om':{}},outputs={'native_om':[out,out]},bracket_outputs={'native_om':[out,out]})
            write_json(paths['golden_suite'],dict(status='PASS',context_rows=16,bundle_sha256=sha256(paths['bundle']),native_om_manifest_sha256=sha256(paths['native_om_manifest']),cases=[old]))
            args=SimpleNamespace(**paths,output_dir=root/'new',layers=[0],device_id=0,warmup=5,repetitions=30,timeout=20)
            cfg=dict(k_tile=256,scale_mode='broadcast',dequant_mode='batched',gate_copy_mode='2d')
            def fail_launch(*a,**k):raise RuntimeError('injected ACLNN pre-gate failure')
            with patch.object(gate,'load_build_config',return_value=cfg),patch.object(gate,'load_bundle',return_value=dict(context_rows=16,cases=[case])),patch.object(gate.run_a2,'validate_om_manifest'),patch.object(gate,'validate_layout'),patch.object(gate,'validate_execution'),patch.object(gate.subprocess,'run',side_effect=fail_launch),patch.object(gate.run_a2,'run') as timing:
                self.assertEqual(gate.run(args),1);timing.assert_not_called()
            self.assertEqual(json.loads((root/'new/suite.json').read_text())['performance'],'NOT_RUN')
