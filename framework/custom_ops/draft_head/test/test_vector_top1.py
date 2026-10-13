"""Vector Top1 semantics and benchmark safety; CPU results are not NPU validation."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import numpy as np
from common import HERE,REPO
from build_config import configure,load
from vector_top1_probe import winner,rows
from top1_experiment import paired_report,CASES
from run_suite import validate_execution


def execution(mode,m,profile=False):
    timing=dict(status='MEASURED',protocol='continuous-v1',warmup=5,repetitions=30,
                per_timed_call_readback=False,per_timed_call_poison=False,correctness_before_calls=2,
                correctness_after_calls=1,timed_tail_checked=True,execute_sync=dict(samples_ms=[1.]*30,median_ms=1.,p95_ms=1.),
                prepare=dict(samples_ms=[0.]*30))
    return dict(status='MEASURED_NOT_CORRECTNESS' if profile else 'PASS',profile_only=profile,mode=mode,
                runtime='AscendCL NPU',cpu_fallback=False,guards_intact=True,input_readonly=True,repeat_equal=True,
                io_validated=True,m=m,k=2560,n=248320,device_id=0,row_permutation_checked=m>1 and not profile,timing=timing)


class VectorTop1Tests(unittest.TestCase):
    def test_actual_helper_against_better_all_fp16_patterns(self):
        compiler=shutil.which('c++')
        if not compiler: self.skipTest('C++ compiler required')
        with tempfile.TemporaryDirectory() as directory:
            binary=Path(directory)/'model'
            built=subprocess.run([compiler,'-std=c++17','-O2','-Wall','-Wextra','-Werror','-I',str(HERE/'test/cpu_include'),
                str(HERE/'test/vector_top1_model.cpp'),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('NPU NOT_RUN',result.stdout)

    def test_npu_primitive_inputs_include_every_fp16_pattern(self):
        raw=rows()
        self.assertEqual(set(np.unique(raw).tolist()),set(range(65536)))
        self.assertTrue(np.array_equal(raw,rows()))
        x=np.zeros(64,dtype='<u2'); x[63]=1; self.assertEqual(winner(x),63)
        x[:]=0x8002; x[62]=0x8001; self.assertEqual(winner(x),62)
        x[:]=0x7c00; x[31]=0xfc01; x[32]=0x7e00; self.assertEqual(winner(x),31)
        x[:]=0x8000; x[1]=0; self.assertEqual(winner(x),0)

    def test_build_default_is_accepted_scalar_compare_select(self):
        with tempfile.TemporaryDirectory() as directory:
            h=Path(directory)/'head_config.h'; out=Path(directory)/'build.json'
            configure(h,out,0)
            config=load(out)
            self.assertEqual((config['conversion'],config['top1'],config['profile']),('compare_select','scalar','none'))
            configure(h,out,0,top1='vector')
            self.assertIn('DFLASH_HEAD_VECTOR_TOP1 1',h.read_text())
            data=json.loads(out.read_text()); data['profile']='matmul'; out.write_text(json.dumps(data))
            with self.assertRaises(ValueError): load(out)
            with self.assertRaises(ValueError): configure(h,out,0,top1='vector',profile='matmul')
            with self.assertRaises(ValueError): configure(h,out,0,conversion='cast',top1='vector')

    def test_profiling_execution_can_never_pass_correctness_validator(self):
        x=execution('fused',15,profile=True)
        validate_execution(x,'fused',15,0,5,30,profiling=True)
        with self.assertRaises(ValueError): validate_execution(x,'fused',15,0,5,30)
        for key in ('guards_intact','input_readonly','repeat_equal'):
            bad=copy.deepcopy(x); bad[key]=False
            with self.assertRaises(ValueError): validate_execution(bad,'fused',15,0,5,30,profiling=True)
        bad=copy.deepcopy(x); bad['timing']['timed_tail_checked']=False
        with self.assertRaises(ValueError): validate_execution(bad,'fused',15,0,5,30,profiling=True)

    def paired(self):
        data=[]
        for conv,top1 in (('cast','scalar'),('compare_select','scalar'),('compare_select','vector')):
            cases=[]
            for name in CASES:
                m=int(name[-2:]); cases.append(dict(name=name,m=m,inputs=dict(hidden=dict(sha256=name),weight=dict(sha256='same')),
                    launches=dict(fused=dict(base_k=128,single_k=2560,partitions=7,matmul_ub_bytes=173696,user_ub_bytes=88448,user_workspace_bytes=1344)),
                    executions={mode:execution(mode,m) for mode in ('fused','native-logits','native-top1')}))
            data.append(dict(status='PASS',capture_sha256='capture',native_manifest_sha256='native',
                             build=dict(profile='none',conversion=conv,top1=top1),cases=cases))
        return data

    def test_paired_report_rejects_input_tiling_golden_protocol_and_observer_mismatch(self):
        data=self.paired(); self.assertEqual(len(paired_report(*data)),6)
        edits=[lambda d:d[2]['cases'][0]['inputs']['weight'].update(sha256='different'),
               lambda d:d[2]['cases'][0]['launches']['fused'].update(base_k=256),
               lambda d:d[2].update(native_manifest_sha256='other'),
               lambda d:d[2]['build'].update(profile='convert'),
               lambda d:d[2]['cases'][0]['executions']['fused']['timing'].update(repetitions=10)]
        for edit in edits:
            bad=self.paired(); edit(bad)
            with self.assertRaises(ValueError): paired_report(*bad)

    def test_primitive_runner_acl_syntax(self):
        compiler=shutil.which('c++')
        if not compiler: self.skipTest('compiler required')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'api.h').write_text('''#include "acl/acl.h"
struct aclTensor; struct aclOpExecutor;
enum aclFormat { ACL_FORMAT_ND };
aclTensor* aclCreateTensor(const int64_t*,uint64_t,aclDataType,const int64_t*,int64_t,aclFormat,const int64_t*,uint64_t,void*);
aclError aclDestroyTensor(const aclTensor*);
extern "C" const char* aclGetRecentErrMsg();
extern "C" int aclrtMemset(void*,std::size_t,int,std::size_t);
''')
            (root/'aclnn_d_flash_top1_probe.h').write_text('''aclError aclnnDFlashTop1ProbeGetWorkspaceSize(const aclTensor*,const aclTensor*,uint64_t*,aclOpExecutor**);
aclError aclnnDFlashTop1Probe(void*,uint64_t,aclOpExecutor*,aclrtStream);
''')
            result=subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Werror','-fsyntax-only',
                '-I',str(root),'-I',str(REPO/'framework/runtime/cpp/tests/fake_acl'),'-include',str(root/'api.h'),
                str(HERE/'test/top1_probe/runner.cpp')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__': unittest.main()
