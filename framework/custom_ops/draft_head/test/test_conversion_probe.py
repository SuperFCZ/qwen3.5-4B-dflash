"""CPU regressions for diagnostic gates; no emulated result is NPU acceptance."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np
from common import HERE,REPO
from conversion_probe import sample_bits,vector_report,precast_report
from build_config import configure,load


class ConversionProbeTests(unittest.TestCase):
    def cpu_example(self):
        raw=sample_bits(); x=raw.view('<f4')
        # Required semantics, NOT an emulation of CANN Mins/Maxs.
        upper=np.where(x>65504,np.float32(65504),x)
        lower=np.where(upper<-65504,np.float32(-65504),upper)
        stages=np.array([x,upper,lower],dtype='<f4')
        with np.errstate(over='ignore',invalid='ignore'):
            cast=stages.astype('<f2')
        return raw,stages,cast

    def test_boundaries_negative_zero_subnormal_nan_inf_are_included(self):
        raw,x,y=self.cpu_example()
        self.assertEqual(len(raw),64)
        for b in (0,0x80000000,1,0x80000001,0x7fc00000,0xffc12345,0x7f800000,0xff800000):
            self.assertIn(b,raw.tolist())
        for value in (65504,65519,65520,65536,131008):
            for sign in (1,-1): self.assertIn(np.float32(sign*value),raw.view('<f4'))
        self.assertEqual(vector_report(raw.tobytes(),x.tobytes(),y.tobytes())['status'],'PASS')
        for i in np.where(np.abs(raw.view('<f4'))>65504)[0]:
            self.assertIn(int(y.view('<u2')[2,i]),(0x7bff,0xfbff))

    def test_control_overflow_is_observed_not_presumed(self):
        raw,x,y=self.cpu_example()
        i=raw.view('<f4').tolist().index(131008.)
        for output in (0x7bff,0x7c00):
            y.view('<u2')[0,i]=output
            self.assertEqual(vector_report(raw.tobytes(),x.tobytes(),y.tobytes())['status'],'PASS')

    def test_gate_rejects_special_value_rewrites_and_one_bit_rounding_error(self):
        raw,x,y=self.cpu_example()
        for original,replacement in ((0x80000000,0),(0x7fc00000,0x477fe000),(1,0)):
            bad=x.copy(); i=raw.tolist().index(original); bad.view('<u4')[2,i]=replacement
            self.assertEqual(vector_report(raw.tobytes(),bad.tobytes(),y.tobytes())['status'],'FAIL')
        bad=y.copy(); bad.view('<u2')[2,10]=0x7c00
        self.assertEqual(vector_report(raw.tobytes(),x.tobytes(),bad.tobytes())['status'],'FAIL')

    def test_precast_requires_all_30_finite_sums_and_reports_raw_bits(self):
        x=np.zeros((2,16,64),dtype='<f4'); x[0,:15,63]=131008; x[1,:15,0]=131008
        result=precast_report(x.tobytes(),'overflow_inf')
        self.assertEqual(result['status'],'PASS'); self.assertEqual(len(result['rows']),30)
        self.assertTrue(all(r['fp32_bits']=='0x47ffe000' for r in result['rows']))
        x[1,14,0]=np.inf
        result=precast_report(x.tobytes(),'overflow_inf')
        self.assertEqual(result['status'],'FAIL'); self.assertEqual(result['errors'][0]['fp32_bits'],'0x7f800000')
        self.assertEqual(result['errors'][0]['row'],14)

    def test_conversion_build_identity_is_bound_and_default_unchanged(self):
        original=(HERE/'op_host/head_config.h').read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            configure(root/'head_config.h',root/'build.json',0,'minmax',True)
            self.assertEqual(load(root/'build.json')['conversion'],'minmax')
            report=json.loads((root/'build.json').read_text()); report['conversion']='cast'
            (root/'build.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError): load(root/'build.json')
        self.assertEqual((HERE/'op_host/head_config.h').read_bytes(),original)

    def test_probe_runner_cpp_syntax(self):
        compiler=shutil.which('c++')
        if not compiler: self.skipTest('compiler required')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            base='''#pragma once
#include "acl/acl.h"
struct aclTensor; struct aclOpExecutor;
enum aclFormat { ACL_FORMAT_ND };
aclTensor* aclCreateTensor(const int64_t*,uint64_t,aclDataType,const int64_t*,int64_t,aclFormat,const int64_t*,uint64_t,void*);
aclError aclDestroyTensor(const aclTensor*);
extern "C" const char* aclGetRecentErrMsg();
extern "C" int aclrtMemset(void*,std::size_t,int,std::size_t);
'''
            (root/'base.h').write_text(base)
            for op,argc in (('Head',8),('Cast',3)):
                name=f'aclnnDFlash{op}Probe'
                (root/f'aclnn_d_flash_{op.lower()}_probe.h').write_text('#include "base.h"\n'+
                    f'aclError {name}GetWorkspaceSize('+','.join(['const aclTensor*']*argc)+',uint64_t*,aclOpExecutor**);\n'+
                    f'aclError {name}(void*,uint64_t,aclOpExecutor*,aclrtStream);\n')
            result=subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Werror','-fsyntax-only',
                '-I',str(root),'-I',str(REPO/'framework/runtime/cpp/tests/fake_acl'),'-include',str(root/'base.h'),
                str(HERE/'test/probe/runner.cpp')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__': unittest.main()
