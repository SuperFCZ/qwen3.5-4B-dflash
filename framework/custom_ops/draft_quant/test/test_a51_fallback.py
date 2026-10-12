"""Six-shape fallback gate/evidence tests; CPU fixtures, no NPU execution."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import a51_fallback as gate
from a2_common import write_json,sha256
from a3_launch import selected_k_tile,k_tile_reason,K_TILE_POLICY,K_PREFIX,launch_evidence,configure_build
from test_a5 import scale_launch
from run_a51 import run
from compare_a5 import match_plans

class FallbackTests(unittest.TestCase):
    def test_only_six_synthetics_change_and_reason_is_checked(self):
        self.assertEqual(len(gate.TARGETS),6)
        with tempfile.TemporaryDirectory() as directory:
            log=Path(directory)/'runner.log'
            for target in gate.TARGETS:
                shape=(80,target.k,target.n)
                left=scale_launch(log,shape,'broadcast')
                with log.open('a') as f:
                    f.write(K_PREFIX+json.dumps(dict(version=1,requested_k_tile=256,selected_k_tile=128,selection_reason='m80-synthetic-k128'))+'\n')
                def read(): return launch_evidence(log,shape,0,2097152,'batched','serial',5,'a-ub','broadcast',256,K_TILE_POLICY)
                right=read(); match_plans(left,right,shape,True)
                self.assertEqual(right['tile_k'],128); self.assertEqual(right['cube']['base_k'],128)
                text=log.read_text()
                log.write_text(text.replace('"selected_k_tile": 128','"selected_k_tile": 256'))
                with self.assertRaises(ValueError): read()
                log.write_text(text.replace('m80-synthetic-k128','configured'))
                with self.assertRaisesRegex(ValueError,'reason'): read()
                for m in (16,32,64): self.assertEqual(selected_k_tile((m,target.k,target.n),256),256)
            for n in (64,128,256): self.assertEqual(selected_k_tile((80,256,n),256),256)
            self.assertEqual(selected_k_tile((80,2560,2048),256),128)
            self.assertEqual(k_tile_reason((80,2560,2048),256),'kv-m80-k128')

    def test_failed_target_prevents_real_and_full_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); cfg=root/'build.json'
            configure_build(root/'h',cfg,0,'batched',root/'kh','prefetch','a-ub','broadcast',256)
            args=SimpleNamespace(build_config=cfg,runner=root/'runner',output_dir=root/'run',warmup=5,repetitions=30,
                device_id=0,c16_bundle=root/'16',c64_bundle=root/'64')
            def fail(target,*args,**kwargs):
                self.assertEqual(kwargs['shape_filter'],gate.TARGET_NAMES); self.assertTrue(kwargs['continuous'])
                target.mkdir(); write_json(target/'suite.json',dict(status='FAIL')); return False
            with patch('run_a51.load_bundle'),patch('run_a51.a4_contract.paired_gears'),patch('run_a51.run_suite.run',side_effect=fail), \
                 patch('run_a51.run_a2.run') as real,patch('run_a51.run_a4.run') as full:
                self.assertEqual(run(args),1)
            real.assert_not_called(); full.assert_not_called()
            report=json.loads((root/'run/suite.json').read_text())
            self.assertEqual(report['phase'],'fallback-synthetic'); self.assertEqual(report['full_a4_regression'],'NOT_RUN')

    def test_target_verifier_reads_tail_bytes_and_rejects_missing_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cfg=dict(k_tile=256,k_tile_policy=K_TILE_POLICY,core_limit=0,dequant_mode='batched',pipeline_mode='serial',
                     kv_m80_mode='a-ub',scale_mode='broadcast')
            workloads=[]
            for shape in gate.TARGETS:
                loc=root/shape.name; case=loc/'probe'; case.mkdir(parents=True)
                write_json(loc/'manifest.json',dict(cases=[dict(name='probe')]))
                for name in ('native-0.bin','native-1.bin','actual-0.bin','actual-1.bin','benchmark-last.bin','postcheck.bin'):
                    (case/name).write_bytes(b'\x00\x00'*(80*shape.n))
                write_json(loc/'native-eager.json',dict(status='PASS',cpu_fallback=False,mkn=[80,shape.k,shape.n],group_size=128,
                    inner_precise=0,environment=dict(device_id=0),reference_policy=gate.ref.NATIVE_POLICY,
                    manifest_sha256=sha256(loc/'manifest.json'),cases=[dict(name='probe',output_sha256=[sha256(case/f'native-{i}.bin') for i in range(2)])]))
                write_json(loc/'comparison.json',dict(status='PASS',cases=[dict(name='probe',status='PASS')]))
                timing=dict(status='MEASURED',protocol='continuous-v1',warmup=5,repetitions=30,
                    execute_sync=dict(samples_ms=[1.0]*30),prepare=dict(samples_ms=[0.01]*30),per_timed_call_readback=False,
                    per_timed_call_poison=False,correctness_before_calls=2,correctness_after_calls=1,timed_tail_checked=True)
                write_json(case/'execution.json',dict(status='PASS',runtime='AscendCL ACLNN',m=80,k=shape.k,n=shape.n,device_id=0,
                    op='DFlashGroupQuantLinear',tile_n=64,tile_k=128,tile_m=80,weight_reuse_rows=80,requested_k_tile=256,
                    k_tile_selection_reason='m80-synthetic-k128',cpu_fallback=False,input_readonly=True,guards_intact=True,
                    repetitions=2,workspace_bytes=2097152,tracked_device_allocation_bytes=3000000,timing=timing))
                scale_launch(case/'runner.log',(80,shape.k,shape.n),'broadcast')
                with (case/'runner.log').open('a') as f:
                    f.write(K_PREFIX+json.dumps(dict(version=1,requested_k_tile=256,selected_k_tile=128,selection_reason='m80-synthetic-k128'))+'\n')
                workloads.append(dict(name=shape.name,mkn=[80,shape.k,shape.n],status='PASS',custom_cases=[dict(name='probe')]))
            summary=dict(status='PASS',selection='SUBSET',suite='a4',requested_k_tile=256,device_id=0,
                         timing_protocol='continuous-v1',workloads=workloads)
            write_json(root/'suite.json',summary)
            # Only substitute fixture generation/inventory; validator and actual
            # byte comparisons, runtime checks, launch parser are production code.
            with patch('a51_fallback.ref.load_cases',return_value=dict(cases=[dict(name='probe')])), \
                 patch('a51_fallback.ref.cases',return_value=[('probe',)]):
                self.assertEqual(len(gate.collect(root,cfg,0)),6)
                tail=root/gate.TARGET_NAMES[-1]/'probe/postcheck.bin'; saved=tail.read_bytes()
                tail.write_bytes(b'\x00\x3c'+saved[2:])
                with self.assertRaisesRegex(ValueError,'bitwise'): gate.collect(root,cfg,0)
                tail.write_bytes(saved)
                summary['workloads']=workloads[:-1]; write_json(root/'suite.json',summary)
                with self.assertRaises(ValueError): gate.collect(root,cfg,0)

if __name__=='__main__': unittest.main()
