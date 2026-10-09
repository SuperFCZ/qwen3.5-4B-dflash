"""A5 host evidence and pairing tests; fixtures never execute an NPU."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import subprocess

import a4_contract
from a2_common import record,write_json
from a3_launch import CONFIG,configure_build,load_build_config,launch_evidence,scale_plan,SCALE_PREFIX
from compare_a5 import compatible,match_plans,totals,projection,load
from run_a5 import run
from test_a41 import plan,log_plan


def scale_launch(path,shape,mode):
    value=plan(shape,'a-ub')
    scale=scale_plan(shape,mode)
    extra=scale['broadcast_bytes']+scale['sdk_reserved_bytes']
    value['user_ub_bytes']+=extra; value['matmul_ub_bytes']-=extra
    log_plan(path,value)
    with path.open('a') as out: out.write(SCALE_PREFIX+json.dumps(scale)+'\n')
    return launch_evidence(path,shape,0,2097152,'batched','serial',5,'a-ub',mode)


class A5Tests(unittest.TestCase):
    def test_switch_defaults_and_hash_binding(self):
        original=CONFIG.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for mode in ('scalar','broadcast'):
                configure_build(root/'host.h',root/'build.json',0,'batched',root/'kernel.h','prefetch','a-ub',mode)
                got=load_build_config(root/'build.json')
                self.assertEqual(got['scale_mode'],mode)
                self.assertEqual((root/'host.h').read_bytes(),(root/'kernel.h').read_bytes())
                self.assertEqual(CONFIG.read_bytes(),original)
            bad=json.loads((root/'build.json').read_text()); bad['scale_mode']='scalar'; write_json(root/'build.json',bad)
            with self.assertRaisesRegex(ValueError,'scale mode'): load_build_config(root/'build.json')
            with self.assertRaises(ValueError): configure_build(root/'h',root/'b',0,'legacy',None,'serial','a-ub','broadcast')

    def test_ub_reservation_and_kv80_exact_control(self):
        with tempfile.TemporaryDirectory() as directory:
            log=Path(directory)/'runner.log'
            for shape in ((16,2560,19456),(16,9728,2560),(16,2560,4096),(16,4096,2560),
                          (32,2560,2048),(80,2560,2048),(16,12800,2560),(64,12800,2560)):
                a=scale_launch(log,shape,'scalar'); b=scale_launch(log,shape,'broadcast')
                match_plans(a,b,shape)
                extra=0 if shape==(80,2560,2048) else 10496
                self.assertEqual(b['user_ub_bytes']-a['user_ub_bytes'],extra)
                self.assertEqual(a['matmul_ub_bytes']-b['matmul_ub_bytes'],extra)
                if extra: self.assertEqual(b['user_ub_bytes']-8192+b['matmul_ub_bytes'],248*1024)
                text=log.read_text(); log.write_text(text.replace('"sdk_reserved_bytes": 8192','"sdk_reserved_bytes": 0'))
                if extra:
                    with self.assertRaises(ValueError): launch_evidence(log,shape,0,2097152,'batched','serial',5,'a-ub','broadcast')
            shape=(80,2560,2048); a=scale_launch(log,shape,'scalar'); b=scale_launch(log,shape,'broadcast')
            b['cube']['depth_a1']+=1
            with self.assertRaisesRegex(ValueError,'KV M80'): match_plans(a,b,shape)

    def test_totals_include_all_26_and_report_regressions_without_end_to_end_claim(self):
        for gear in (16,64):
            rows=[]
            for case in a4_contract.cases(gear):
                row=dict(name=case['name'],shape=list(a4_contract.case_shape(case)))
                for mode in ('baseline','candidate','native_before','native_after'):
                    # gate/up improves but every other projection regresses.
                    median=(0.5 if case['projection']=='gate_up' else 2.0) if mode=='candidate' else 1.0
                    row[mode]=dict(median_ms=median,p95_ms=median+0.1)
                rows.append(row)
            got=totals(rows)
            self.assertEqual(got['projection_count'],26)
            self.assertEqual(got['baseline']['sum_projection_median_ms'],26)
            self.assertEqual(got['candidate']['sum_projection_median_ms'],44.5)
            self.assertLess(got['median_sum_speedup'],1)
            self.assertEqual(sum(x['projection_count'] for x in got['by_shape']),26)
            self.assertIn('NOT median/p95',got['interpretation'])
            with self.assertRaises(ValueError): totals(rows[:-1])
            with self.assertRaises(ValueError): totals(rows[:25]+[rows[0]])

    def test_pairing_rejects_other_build_changes_and_incomplete_a4(self):
        a=dict(device_id=0,timing_protocol='continuous-v1',inputs={'c16':'same','c64':'same'},
               build=dict(source_sha256={'kernel':'same'},core_limit=0,dequant_mode='batched',pipeline_mode='prefetch',
                          kv_m80_mode='a-ub',tiling_abi='full-m-v2',launch_version=5,scale_mode='scalar'))
        b=copy.deepcopy(a); b['build']['scale_mode']='broadcast'; compatible(a,b)
        b['build']['pipeline_mode']='serial'
        with self.assertRaises(ValueError): compatible(a,b)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'suite.json'; write_json(path,dict(status='PASS',checks={'c16':{'status':'PASS'}}))
            with self.assertRaises(ValueError): load(path,'scalar')

    def test_projection_rechecks_bits_after_consistent_rehash_and_guards(self):
        shape=(16,2560,4096)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); reports=[]; cases=[]; roots=[]
            for mode in ('scalar','broadcast'):
                dest=root/mode; dest.mkdir(); roots.append(dest)
                cfg=dict(core_limit=0,dequant_mode='batched',pipeline_mode='serial',kv_m80_mode='a-ub',scale_mode=mode)
                reports.append(dict(device_id=0,build=cfg))
                case=dict(name='layer-0-q',status='PASS',executions={},outputs={},bracket_outputs={})
                for kind in ('custom','native_om'):
                    location=dest/case['name']/kind; location.mkdir(parents=True); files=[]
                    for filename in ('actual-0.bin','actual-1.bin','benchmark-last.bin','postcheck.bin'):
                        file=location/filename; file.write_bytes(b'\x00\x3c'*(16*4096)); files.append(record(file,dest))
                    case['outputs'][kind],case['bracket_outputs'][kind]=files[:2],files[2:]
                    timing=dict(status='MEASURED',protocol='continuous-v1',warmup=5,repetitions=30,
                        execute_sync=dict(samples_ms=[1.0]*30),prepare=dict(samples_ms=[0.01]*30),
                        per_timed_call_readback=False,per_timed_call_poison=False,correctness_before_calls=2,
                        correctness_after_calls=1,timed_tail_checked=True)
                    case['executions'][kind]=dict(status='PASS',runtime='AscendCL ACLNN' if kind=='custom' else 'AscendCL native OM',
                        m=16,k=2560,n=4096,device_id=0,op='DFlashGroupQuantLinear',tile_n=64,tile_k=128,
                        cpu_fallback=False,input_readonly=True,guards_intact=True,io_validated=True,repetitions=2,
                        workspace_bytes=2097152,tracked_device_allocation_bytes=3000000,timing=timing)
                    if kind=='custom': scale_launch(location/'runner.log',shape,mode)
                cases.append(case)
            def compare(): return projection(*cases,*roots,shape,*reports)
            self.assertEqual(compare()['status'],'PASS')
            tail=roots[1]/'layer-0-q/custom/postcheck.bin'; tail.write_bytes(b'\x00\x40'+tail.read_bytes()[2:])
            cases[1]['bracket_outputs']['custom'][1]=record(tail,roots[1])
            self.assertEqual(compare()['status'],'FAIL')
            cases[1]['executions']['custom']['guards_intact']=False
            with self.assertRaises(ValueError): compare()

    def test_pair_driver_preserves_failure_and_never_profiles_failed_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); inputfile=root/'existing.json'; inputfile.write_text('CPU fixture')
            env={name:str(inputfile) for name in ('A4_C16_BUNDLE','A4_C64_BUNDLE','A4_C16_NATIVE_OM_MANIFEST','A4_C64_NATIVE_OM_MANIFEST')}
            args=SimpleNamespace(output_dir=root/'new',skip_profiling=True,order='baseline-first',pipeline='prefetch',warmup=5,repetitions=30)
            with patch.dict('os.environ',env),patch('run_a5.subprocess.run',side_effect=subprocess.CalledProcessError(1,['fake'])) as child:
                with self.assertRaises(subprocess.CalledProcessError): run(args)
            self.assertEqual(child.call_count,1)
            report=json.loads((root/'new/suite.json').read_text())
            self.assertEqual(report['status'],'FAIL'); self.assertEqual(report['profiling'],'NOT_RUN')
            self.assertEqual(inputfile.read_text(),'CPU fixture')

    def test_pair_driver_keeps_external_build_paths_and_explicit_not_run_profiling(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); existing=root/'existing.json'; existing.write_text('CPU fixture')
            env={name:str(existing) for name in ('A4_C16_BUNDLE','A4_C64_BUNDLE','A4_C16_NATIVE_OM_MANIFEST','A4_C64_NATIVE_OM_MANIFEST')}
            args=SimpleNamespace(output_dir=root/'pair',skip_profiling=True,order='candidate-first',pipeline='prefetch',warmup=5,repetitions=30)
            calls=[]
            def build(command,**kwargs):
                settings=kwargs['env']; calls.append(settings['DFLASH_SCALE_MODE'])
                self.assertEqual(settings['DFLASH_KV_M80_MODE'],'a-ub')
                self.assertEqual(settings['DFLASH_SUITE'],'a4')
                summary=root/(settings['DFLASH_SCALE_MODE']+'-external-suite.json')
                summary.write_text('CPU fixture only')
                Path(settings['DFLASH_SUMMARY_POINTER']).write_text(str(summary)+'\n')
            def compare(a,b,output):
                self.assertIn('scalar',a.name); self.assertIn('broadcast',b.name)
                write_json(output,dict(status='PASS',runtime='CPU fixture')); return 0
            with patch.dict('os.environ',env),patch('run_a5.subprocess.run',side_effect=build),patch('run_a5.compare_runs',side_effect=compare):
                run(args)
            report=json.loads((root/'pair/suite.json').read_text())
            self.assertEqual(calls,['broadcast','scalar'])
            self.assertEqual(report['profiling'],'NOT_RUN')
            self.assertEqual(report['adoption'],'NOT_DECIDED')
            self.assertEqual(report['status'],'PASS')

if __name__=='__main__': unittest.main()
