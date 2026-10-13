"""CPU tests only; no CANN build or device acceptance claims."""
import copy
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock,Mock,sentinel
from common import K,N,REPO,HERE,ABI,WEIGHT_BYTES,load_capture,write_json,record
from build_config import configure,load


class HeadTests(unittest.TestCase):
    def test_observed_overflow_one_ulp_difference_is_still_failure(self):
        import numpy as np
        from run_suite import compare_case
        logits=np.zeros((15,N),dtype='<u2'); logits[:,[63,64]]=0x7bff
        ids=struct.pack('<15q',*([63]*15))
        outputs={'native-logits':[logits.tobytes()]*2,'native-top1':[ids]*2,
                 'fused':[ids]*2,'audit':[ids]*2,'audit-logits':[logits.tobytes()]*2}
        def compare(): return compare_case(outputs,15,[63]*15,True,'synthetic-overflow_inf')
        self.assertEqual(compare()['status'],'PASS')
        old=logits.copy(); old[:,[63,64]]=0x7c00
        outputs['audit-logits']=[old.tobytes()]*2
        result=compare()
        self.assertEqual(result['status'],'FAIL')
        diff=result['comparisons']['full_fp16_logits']
        self.assertEqual((diff['bit_mismatches'],diff['max_ulp']),(30,1))
        self.assertEqual(diff['first_difference'],dict(index=[0,63],expected_bits='0x7bff',actual_bits='0x7c00'))
        # Even both sides returning Inf with matching IDs must not rewrite the
        # observed Native saturation expectation to force a pass.
        outputs['native-logits']=[old.tobytes()]*2
        self.assertEqual(compare()['status'],'FAIL')

    def test_conversion_launch_reports_ub_cost_without_changing_gm_workspace(self):
        from common import verify_launch
        p=7; user=16*K*2+16*64*6+160+p*32
        launch=dict(version=1,m=15,k=K,n=N,tile_n=64,padded_m=16,audit=True,core_limit=0,batchmode=1,
            available_cores=p,partitions=p,user_workspace_bytes=192*p,system_workspace_bytes=2097152,
            user_ub_bytes=user,matmul_ub_bytes=262144-user,base_k=128,single_k=K,
            output_conversion='fp32-ub-cast-none-v1',fp32_tile_bytes=4096)
        with tempfile.TemporaryDirectory() as directory:
            log=Path(directory)/'runner.log'
            log.write_text('DFLASH_HEAD_LAUNCH '+json.dumps(launch)+'\n')
            self.assertEqual(verify_launch(log,15,True,0)['user_workspace_bytes'],1344)
            launch['output_conversion']='fp32-ub-compare-select-cast-none-v1'
            log.write_text('DFLASH_HEAD_LAUNCH '+json.dumps(launch)+'\n')
            self.assertEqual(verify_launch(log,15,True,0,'compare_select')['user_ub_bytes'],user)
            with self.assertRaisesRegex(ValueError,'output_conversion'): verify_launch(log,15,True,0,'minmax')
            launch['output_conversion']='fp32-ub-cast-none-v1'
            launch['user_ub_bytes']-=4096
            log.write_text('DFLASH_HEAD_LAUNCH '+json.dumps(launch)+'\n')
            with self.assertRaisesRegex(ValueError,'UB accounting'): verify_launch(log,15,True,0)

    def test_matmul_stub_rejects_the_reported_cubeformat_namespace_error(self):
        compiler=shutil.which('c++')
        if not compiler: self.skipTest('C++ compiler required')
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'lookup.cpp'
            for format_name,valid in (('CubeFormat::ND',True),('matmul::CubeFormat::ND',False)):
                source.write_text('#include "lib/matmul_intf.h"\n'
                    f'using A=matmul::MatmulType<AscendC::TPosition::VECOUT,{format_name},half>;\n')
                result=subprocess.run([compiler,'-std=c++17','-fsyntax-only','-I',str(HERE/'test/cpu_include'),
                    str(source)],capture_output=True,text=True)
                if valid: self.assertEqual(result.returncode,0,result.stderr)
                else:
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn('CubeFormat',result.stderr)
                    self.assertIn('AscendC',result.stderr)

    def test_nan_inf_raw_weights_and_all_hidden_rows_are_preserved(self):
        import numpy as np
        from fixtures import set_nan_first_weights,check_nan_first_inputs
        # Same writer as fixtures.prepare, but avoid materializing a 1.27 GB
        # CPU tensor. The input validator still requires the full file length.
        weight=np.zeros((65,K),dtype='<f2'); set_nan_first_weights(weight)
        self.assertEqual(int(weight.view('<u2')[63,0]),0x7e00)
        self.assertEqual(int(weight.view('<u2')[64,0]),0x7c00)
        self.assertTrue(np.isnan(weight[63,0])); self.assertTrue(np.isposinf(weight[64,0]))
        hidden=np.zeros((15,K),dtype='<f2'); hidden[:,0]=1
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'head.bin'
            with path.open('wb') as out:
                out.truncate(WEIGHT_BYTES)
                for token in (63,64):
                    out.seek(token*K*2); out.write(weight[token,:1].tobytes())
            self.assertEqual(check_nan_first_inputs(path,hidden.tobytes(),15)['status'],'PASS')
            for token,bad in ((63,0x0000),(63,0x7e01),(64,0x7bff)):
                with path.open('r+b') as out: out.seek(token*K*2); out.write(struct.pack('<H',bad))
                with self.assertRaisesRegex(ValueError,f'weight\\[{token},0\\].*actual=0x{bad:04x}'):
                    check_nan_first_inputs(path,hidden.tobytes(),15)
                with path.open('r+b') as out: out.seek(token*K*2); out.write(weight[token,:1].tobytes())
            hidden[14,1]=1
            with self.assertRaisesRegex(ValueError,'14, 1'): check_nan_first_inputs(path,hidden.tobytes(),15)

    def nan_first_outputs(self):
        import numpy as np
        logits=np.zeros((15,N),dtype='<u2'); logits[:,64]=0x7bff
        ids=np.full(15,64,dtype='<i8').tobytes()
        return logits,{'native-logits':[logits.tobytes()]*2,'native-top1':[ids]*2,
                       'fused':[ids]*2,'audit':[ids]*2,'audit-logits':[logits.tobytes()]*2}

    def test_nan_inf_native_observation_all_rows_both_repeats(self):
        from run_suite import compare_case
        logits,outputs=self.nan_first_outputs()
        def check(value): return compare_case(value,15,[64]*15,False,'synthetic-nan_first')
        self.assertEqual(check(outputs)['status'],'PASS')
        for row in range(15):
            for token,bad in ((63,0x8000),(63,0x3c00),(64,0x7bfe)):
                changed=logits.copy(); changed[row,token]=bad
                broken=copy.deepcopy(outputs); broken['native-logits']=[changed.tobytes()]*2
                result=check(broken)
                self.assertEqual(result['status'],'FAIL')
                self.assertTrue(result['comparisons']['synthetic_expected_ids']['equal'])
                self.assertTrue(result['comparisons']['native_references_consistent']['equal'])
                for repeat in (0,1):
                    diff=result['comparisons'][f'native_observed_logits_{repeat}']['first_difference']
                    self.assertEqual(diff['index'],[row,token]); self.assertEqual(diff['actual_bits'],f'0x{bad:04x}')
        changed=logits.copy(); changed[14,63]=0x7e00
        broken=copy.deepcopy(outputs); broken['native-logits'][1]=changed.tobytes()
        self.assertFalse(check(broken)['comparisons']['native_observed_logits_1']['equal'])
        broken=copy.deepcopy(outputs); broken['native-top1']=[struct.pack('<15q',*([63]*15))]*2
        self.assertFalse(check(broken)['comparisons']['synthetic_expected_ids']['equal'])
        with self.assertRaises(ValueError): compare_case(outputs,15,[63]*15,False,'synthetic-nan_first')

    def test_nan_inf_custom_logits_never_borrow_native_or_skip_nonfinite(self):
        from run_suite import compare_case
        logits,outputs=self.nan_first_outputs()
        self.assertEqual(compare_case(outputs,15,[64]*15,True,'synthetic-nan_first')['status'],'PASS')
        # Custom ID can remain correct, but any logit bit (including signed
        # zero or NaN payload) must still fail and locate the full vocabulary.
        for row,token,bad in ((7,63,0x7e00),(14,64,0x7c00),(3,1000,0x8000)):
            changed=logits.copy(); changed[row,token]=bad
            broken=copy.deepcopy(outputs); broken['audit-logits']=[changed.tobytes()]*2
            result=compare_case(broken,15,[64]*15,True,'synthetic-nan_first')
            self.assertEqual(result['status'],'FAIL'); self.assertTrue(result['comparisons']['custom_ids']['equal'])
            diff=result['comparisons']['full_fp16_logits']['first_difference']
            self.assertEqual(diff['index'],[row,token]); self.assertEqual(diff['actual_bits'],f'0x{bad:04x}')
            self.assertEqual(diff['expected_bits'],f'0x{int(logits[row,token]):04x}')

    def test_named_guards_and_native_top1_logical_physical_boundaries(self):
        compiler=shutil.which('c++')
        if not compiler: self.skipTest('C++ compiler required')
        with tempfile.TemporaryDirectory() as directory:
            binary=Path(directory)/'guard-model'
            result=subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Werror',
                '-I',str(REPO/'framework/runtime/cpp/tests/fake_acl'),
                str(HERE/'test/guard_model.cpp'),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([str(binary),directory],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('fixed runner NPU acceptance NOT_RUN',result.stdout)
            for m in range(1,16):
                self.assertEqual((Path(directory)/f'm{m}.bin').read_bytes(),struct.pack('<'+'q'*m,*range(100,100+m)))

    def test_kernel_full_vocabulary_and_partition_reduction_model(self):
        compiler=shutil.which('c++')
        if not compiler: self.skipTest('C++17/_Float16 required')
        with tempfile.TemporaryDirectory() as directory:
            for conversion,top1,profile in (('cast','scalar','none'),('minmax','scalar','none'),
                    ('compare_select','scalar','none'),('compare_select','vector','none'),
                    ('compare_select','scalar','matmul'),('compare_select','scalar','convert')):
                configure(Path(directory)/'head_config.h',Path(directory)/'build.json',0,conversion,True,top1,profile)
                binary=Path(directory)/('model-'+conversion)
                result=subprocess.run([compiler,'-std=c++17','-O2','-pthread','-ffp-contract=off','-Wall','-Wextra','-Werror',
                    '-Wno-unused-parameter','-I',directory,'-I',str(HERE/'test/cpu_include'),
                    str(HERE/'test/kernel_model.cpp'),'-o',str(binary)],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=120)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertIn('NPU NOT_RUN',result.stdout)

    def test_build_config_does_not_modify_default_and_rejects_oversubscription_cap(self):
        original=(HERE/'op_host/head_config.h').read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            configure(root/'config.h',root/'build.json',3)
            self.assertIn('CORE_LIMIT 3U',(root/'config.h').read_text())
            self.assertEqual((HERE/'op_host/head_config.h').read_bytes(),original)
            self.assertEqual(load(root/'build.json')['core_limit'],3)
            (root/'config.h').write_text('tampered')
            with self.assertRaises(ValueError): load(root/'build.json')
            for bad in (-1,65,True):
                with self.assertRaises(ValueError): configure(root/'config.h',root/'build.json',bad)
            with self.assertRaises(ValueError): configure(HERE/'op_host/head_config.h',root/'b.json',0)

    def test_capture_takes_only_post_norm_candidate_rows_and_rejects_cpu(self):
        from capture import normalized_rows
        tensor=MagicMock(); tensor.shape=(1,16,K); tensor.dtype=sentinel.fp16; tensor.device.type='npu'
        tensor.__getitem__.return_value.detach.return_value.contiguous.return_value=sentinel.hidden
        torch=SimpleNamespace(float16=sentinel.fp16,isfinite=Mock())
        torch.isfinite.return_value.all.return_value.item.return_value=True
        self.assertIs(normalized_rows(tensor,torch),sentinel.hidden)
        tensor.__getitem__.assert_called_once_with((slice(None),slice(1,None),slice(None)))
        tensor.device.type='cpu'
        with self.assertRaisesRegex(ValueError,'NPU FP16'): normalized_rows(tensor,torch)

    def test_partial_capture_cannot_replace_two_gears_or_full_head(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'manifest.json'
            for value in ({'abi':ABI,'status':'PASS','gears':{}},
                          {'abi':ABI,'status':'PASS','runtime':'synthetic','weight':{'shape':[64,K]}}):
                write_json(path,value)
                with self.assertRaises(ValueError): load_capture(path)

    def test_native_manifest_requires_all_shapes_and_full_weight_input(self):
        from run_suite import validate_native
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); om=root/'fake.om'; air=root/'air.json'
            om.write_text('CPU metadata fixture, never executed'); air.write_text('{}')
            models=[]
            for m in range(1,16):
                for kind in ('logits','top1'):
                    tensor_abi=dict(inputs=[dict(name='hidden',dtype='float16',shape=[m,K]),dict(name='weight',dtype='float16',shape=[N,K])],
                                    outputs=[dict(name='y',dtype='float16' if kind=='logits' else 'int64',shape=[m,N] if kind=='logits' else [m])])
                    deployment=root/f'{m}-{kind}.json'
                    write_json(deployment,dict(status='PASS',graphs=[dict(input_names=['hidden','weight'],output_names=['y'],
                        metadata=dict(tensor_abi=tensor_abi),om=record(om,root))]))
                    models.append(dict(m=m,kind=kind,status='PASS',om=record(om,root),deployment=record(deployment,root),air=record(air,root),
                                       runtime_input_abi=dict(status='PASS',logical_input_names=['hidden','weight'])))
            manifest=dict(abi=ABI,status='PASS',capture_sha256='capture',weight_binding='runtime-readonly-ND',models=models)
            path=root/'native.json'; write_json(path,manifest)
            self.assertEqual(len(validate_native(path,'capture')),30)
            broken=copy.deepcopy(manifest); broken['models'].pop(); write_json(path,broken)
            with self.assertRaises(ValueError): validate_native(path,'capture')
            # Even consistently re-hashed metadata must reject a truncated head.
            graph=json.loads(deployment.read_text()); graph['graphs'][0]['metadata']['tensor_abi']['inputs'][1]['shape']=[64,K]
            write_json(deployment,graph); manifest['models'][-1]['deployment']=record(deployment,root); write_json(path,manifest)
            with self.assertRaises(ValueError): validate_native(path,'capture')

    def test_all_logits_are_a_gate_even_when_top1_matches(self):
        try: import numpy as np
        except ImportError: self.skipTest('numpy needed for byte-comparison tests')
        from run_suite import compare_case
        ids=struct.pack('<q',0); logits=bytes(N*2)
        out={'native-logits':[logits]*2,'native-top1':[ids]*2,'fused':[ids]*2,'audit':[ids]*2,'audit-logits':[logits]*2}
        self.assertEqual(compare_case(out,1)['status'],'PASS')
        bad=copy.deepcopy(out)
        bad['audit-logits']=[logits[:200]+struct.pack('<e',-1)+logits[202:]]*2
        value=compare_case(bad,1)
        self.assertEqual(value['status'],'FAIL')
        self.assertTrue(value['comparisons']['custom_ids']['equal'])
        self.assertEqual(value['comparisons']['full_fp16_logits']['bit_mismatches'],1)
        bad=copy.deepcopy(out); bad['native-top1']=[struct.pack('<q',1)]*2
        self.assertEqual(compare_case(bad,1)['status'],'FAIL')

    def test_native_protocol_does_not_accept_audit_timing_or_changed_m(self):
        from run_suite import validate_execution
        x=dict(status='PASS',mode='fused',runtime='AscendCL NPU',cpu_fallback=False,guards_intact=True,
               input_readonly=True,repeat_equal=True,io_validated=True,row_permutation_checked=True,m=15,k=K,n=N,device_id=0,
               timing=dict(status='MEASURED',protocol='continuous-v1',warmup=5,repetitions=30,per_timed_call_readback=False,
                           per_timed_call_poison=False,correctness_before_calls=2,correctness_after_calls=1,
                           timed_tail_checked=True,execute_sync=dict(samples_ms=[1.]*30),prepare=dict(samples_ms=[0.01]*30)))
        validate_execution(x,'fused',15,0,5,30)
        bad=copy.deepcopy(x); bad['timing']['timed_tail_checked']=False
        with self.assertRaises(ValueError): validate_execution(bad,'fused',15,0,5,30)
        with self.assertRaises(ValueError): validate_execution(x,'fused',1,0,5,30)

    def test_runner_syntax_with_acl_declarations_only(self):
        compiler=shutil.which('c++')
        if not compiler: self.skipTest('C++ compiler required')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'extras.h').write_text('''#include <cstddef>
extern "C" const char* aclGetRecentErrMsg();
extern "C" int aclrtMemset(void*,std::size_t,int,std::size_t);
''')
            declarations='''#pragma once
#include "acl/acl.h"
struct aclTensor; struct aclOpExecutor;
enum aclFormat { ACL_FORMAT_ND };
aclTensor* aclCreateTensor(const int64_t*,uint64_t,aclDataType,const int64_t*,int64_t,aclFormat,const int64_t*,uint64_t,void*);
aclError aclDestroyTensor(const aclTensor*);
aclError aclnnDFlashDraftLmHeadTop1GetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,uint64_t*,aclOpExecutor**);
aclError aclnnDFlashDraftLmHeadTop1(void*,uint64_t,aclOpExecutor*,aclrtStream);
'''
            (root/'aclnn_d_flash_draft_lm_head_top1.h').write_text(declarations)
            (root/'aclnn_d_flash_draft_lm_head_top1_audit.h').write_text('''#pragma once
#include "aclnn_d_flash_draft_lm_head_top1.h"
aclError aclnnDFlashDraftLmHeadTop1AuditGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,uint64_t*,aclOpExecutor**);
aclError aclnnDFlashDraftLmHeadTop1Audit(void*,uint64_t,aclOpExecutor*,aclrtStream);
''')
            for defines in ([],['-DHEAD_NATIVE_ONLY=1'],['-DHEAD_PROFILE_ONLY=1']):
                result=subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Werror','-fsyntax-only',
                    '-I',str(root),'-I',str(REPO/'framework/runtime/cpp/tests/fake_acl'),'-include',str(root/'extras.h'),
                    str(HERE/'test/runner.cpp'),*defines],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)

if __name__=='__main__': unittest.main()
