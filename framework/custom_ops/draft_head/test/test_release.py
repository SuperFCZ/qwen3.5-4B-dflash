"""CPU-only release configuration, dispatch, evidence and regression-gate tests."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from common import HERE,ABI
from build_config import configure,load
from head_opp_preflight import check_install,OP_KERNELS
from test_head_opp_preflight import installed
from test_vector_top1 import execution
from release_acceptance import frozen_contract,compare_performance,require_full_suite,ALL_CASES,build_environment


class ReleaseTests(unittest.TestCase):
    def test_frozen_algorithm_and_default_vector_scalar_fallback(self):
        contract=frozen_contract()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for top in ('vector','scalar'):
                configure(root/'head_config.h',root/'config.json',0,top1=top)
                x=load(root/'config.json')
                self.assertEqual(x['build_kind'],'release'); self.assertEqual(x['operators'],list(OP_KERNELS))
                self.assertFalse(x['top1_probe']); self.assertFalse(x['diagnostics'])
                self.assertEqual(x['profile'],'none')
                self.assertIn(f"#define DFLASH_HEAD_VECTOR_TOP1 {int(top=='vector')}",(root/'head_config.h').read_text())
                for name,digest in contract['source_sha256'].items(): self.assertEqual(x['source_sha256'][name],digest)
            configure(root/'head_config.h',root/'config.json',0,top1_probe=True)
            x=load(root/'config.json'); self.assertEqual(x['build_kind'],'experiment')
            self.assertIn('d_flash_top1_probe',x['operators'])
            x['top1_probe']=False; (root/'config.json').write_text(json.dumps(x))
            with self.assertRaises(ValueError): load(root/'config.json')

    def release_install(self,root):
        installed(root)
        include=root/'vendors/customize/op_api/include'; include.mkdir(parents=True)
        for op in OP_KERNELS: (include/f'aclnn_{op}.h').write_text('CPU fixture')
        lib=root/'vendors/customize/op_api/lib'; lib.mkdir(); (lib/'libcust_opapi.so').write_bytes(b'CPU fixture')

    def test_release_opp_requires_exact_dual_operator_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'opp'; self.release_install(root)
            x=check_install(root,release=True)
            self.assertEqual(len(x['operators']),2); self.assertEqual(len(x['headers']),2)
            self.assertEqual(len(x['libraries']),1); self.assertEqual(x['npu_status'],'NOT_RUN')
            with self.assertRaises(ValueError): check_install(root,vector_top1=True,release=True)
            binaries=installed(root,('d_flash_top1_probe',))
            with self.assertRaisesRegex(ValueError,'extra operators'): check_install(root,release=True)
            shutil.rmtree(binaries[0].parent)
            include=root/'vendors/customize/op_api/include'
            (include/'aclnn_d_flash_top1_probe.h').write_text('CPU fixture')
            with self.assertRaisesRegex(ValueError,'inventory'): check_install(root,release=True)
            (include/'aclnn_d_flash_top1_probe.h').unlink()
            (include/f'aclnn_{OP_KERNELS[0]}.h').write_text('')
            with self.assertRaisesRegex(ValueError,'empty release header'): check_install(root,release=True)

    def suite(self):
        contract=frozen_contract(); cases=[]
        for name in sorted(ALL_CASES):
            m=int(name[-2:]) if name.startswith('real-') else 15
            modes={k:execution(k,m) for k in ('native-logits','native-top1','fused')}
            for k,v in modes.items():
                samples=[17.1 if k=='fused' else 23.26]*30
                v['timing']['execute_sync']['samples_ms']=samples
            audit=execution('audit',m); audit['timing']=dict(status='NOT_RUN',protocol='checked-v1',warmup=0,repetitions=0)
            modes['audit']=audit
            cases.append(dict(name=name,m=m,status='PASS',inputs=dict(hidden=dict(sha256=name),weight=dict(sha256='w')),
                executions=modes,launches=dict(fused=dict(base_k=128,single_k=2560,partitions=7,core_limit=0,user_ub_bytes=88448,
                    matmul_ub_bytes=173696,user_workspace_bytes=1344,system_workspace_bytes=2097152))))
        return dict(abi=ABI,status='PASS',full_41_case_acceptance='PASS',custom_correctness='PASS',capture_sha256='capture',native_manifest_sha256='golden',
                    build=dict(conversion='compare_select',top1='vector',profile='none',source_sha256=contract['source_sha256']),cases=cases)

    def test_new_full_acceptance_required_not_historical_pass(self):
        contract=frozen_contract(); data=self.suite(); require_full_suite(data,contract)
        for field,value in (('status','FAIL'),('full_41_case_acceptance','NOT_RUN'),('custom_correctness','NOT_RUN')):
            bad=copy.deepcopy(data); bad[field]=value
            with self.assertRaises(ValueError): compare_performance(bad,data,contract)
        bad=copy.deepcopy(data); bad['cases'].pop()
        with self.assertRaises(ValueError): require_full_suite(bad,contract)
        bad=copy.deepcopy(data); bad['cases'][0]['executions']['fused']['input_readonly']=False
        with self.assertRaises(ValueError): require_full_suite(bad,contract)
        bad=copy.deepcopy(data); bad['build']['top1']='scalar'
        with self.assertRaises(ValueError): require_full_suite(bad,contract)

    def test_exact_baseline_samples_median_p95_and_regression(self):
        old=self.suite(); new=copy.deepcopy(old); contract=frozen_contract()
        x=compare_performance(new,old,contract)
        self.assertEqual(x['status'],'PASS'); self.assertEqual(len(x['cases']),6)
        target=next(c for c in new['cases'] if c['name']=='real-c16-m15')
        target['executions']['fused']['timing']['execute_sync']['samples_ms'][-2:]=[30.,30.]
        x=compare_performance(new,old,contract)
        self.assertEqual(x['status'],'REVIEW_REQUIRED')
        self.assertTrue(any(c.get('metric')=='p95_ms' for c in x['concerns']))
        self.assertEqual(new['status'],'PASS') # Performance flag does not relabel bitwise correctness.
        target['inputs']['weight']['sha256']='other'
        with self.assertRaises(ValueError): compare_performance(new,old,contract)

    def test_release_environment_uses_defaults_not_stale_experiment_flags(self):
        args=SimpleNamespace(capture=Path('/cap'),native_manifest=Path('/native'),device_id=0)
        with patch.dict(os.environ,dict(HEAD_TOP1='scalar',HEAD_DIAGNOSTICS='1',HEAD_TOP1_PROBE='1',HEAD_PROFILE_STAGE='matmul')):
            env=build_environment(args,Path('/state'))
        for key in ('HEAD_TOP1','HEAD_DIAGNOSTICS','HEAD_TOP1_PROBE','HEAD_PROFILE_STAGE','HEAD_CONVERSION'):
            self.assertNotIn(key,env)
        self.assertEqual(env['HEAD_NATIVE_ONLY'],'0')

    def test_server_default_dispatch_without_cann_has_only_two_generated_ops(self):
        # Mock external CANN commands and STOP at build (exit 77). This is a shell
        # dispatch/configuration test, explicitly NOT CANN compilation or NPU.
        for top in (None,'scalar'):
            with self.subTest(top=top),tempfile.TemporaryDirectory(prefix='head-release-') as directory:
                root=Path(directory); head=root/'draft_head'; head.mkdir()
                # macOS sandbox disallows /dev/fd process substitution. Replace
                # only tee logging in this TEMPORARY copy; all build dispatch is unchanged.
                script=(HERE/'run_server.sh').read_text()
                logger='exec > >(tee "$run_root/server.log") 2>&1'
                self.assertEqual(script.count(logger),1)
                (head/'run_server.sh').write_text(script.replace(logger,'exec > "$run_root/server.log" 2>&1'))
                for item in ('op_host','op_kernel','test','DFlashDraftLmHeadTop1.json'):
                    (head/item).symlink_to(HERE/item,target_is_directory=(HERE/item).is_dir())
                (root/'draft_quant').symlink_to(HERE.parent/'draft_quant',target_is_directory=True)
                cann=root/'cann'; cann.mkdir(); (cann/'set_env.sh').write_text(':\n')
                tools=root/'tools'; tools.mkdir()
                for name,body in (('git','echo CPU-dispatch-only'),('cmake','exit 0')):
                    path=tools/name; path.write_text('#!/bin/sh\n'+body+'\n'); path.chmod(0o755)
                generator=tools/'msopgen'
                generator.write_text('#!'+sys.executable+'\n'+'''import sys
from pathlib import Path
a=sys.argv; out=Path(a[a.index('-out')+1]); op=a[a.index('-op')+1]
names={'DFlashDraftLmHeadTop1':'d_flash_draft_lm_head_top1','DFlashDraftLmHeadTop1Audit':'d_flash_draft_lm_head_top1_audit'}
name=names[op] # any probe generation fails this mock immediately
for part in ('op_host','op_kernel'):
    p=out/part; p.mkdir(parents=True,exist_ok=True)
    (p/(name+'.cpp')).write_text('CPU fixture')
    (p/'CMakeLists.txt').write_text('CPU fixture')
(out/'CMakeLists.txt').write_text('CPU fixture')
(out/'build.sh').write_text('exit 77\\n')
'''); generator.chmod(0o755)
                args=SimpleNamespace(capture=root/'capture.json',native_manifest=root/'native.json',device_id=0)
                env=build_environment(args,root/'state.json'); env.update(CANN_ROOT=str(cann),PATH=str(tools)+os.pathsep+os.environ['PATH'])
                if top: env['HEAD_TOP1']=top
                result=subprocess.run(['bash',str(head/'run_server.sh')],env=env,capture_output=True,text=True)
                logs=''.join(p.read_text() for p in (head/'.runs').glob('*/server.log'))
                self.assertEqual(result.returncode,77,result.stdout+result.stderr+logs)
                config=load(next((head/'.runs').glob('*/build-config.json')))
                self.assertEqual(config['top1'],top or 'vector'); self.assertFalse(config['top1_probe'])
                self.assertEqual(config['operators'],list(OP_KERNELS)); self.assertEqual(config['build_kind'],'release')
                generated=next((head/'.build').glob('*/CustomOp'))
                self.assertEqual({p.stem for p in (generated/'op_host').glob('*.cpp')},set(OP_KERNELS))
                self.assertEqual({p.stem for p in (generated/'op_kernel').glob('*.cpp')},set(OP_KERNELS))

if __name__=='__main__': unittest.main()
