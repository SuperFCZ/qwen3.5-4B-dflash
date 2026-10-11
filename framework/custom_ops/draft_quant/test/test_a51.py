"""K-only policy/evidence tests. CPU fixtures are not NPU acceptance."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from a2_common import write_json
from a3_launch import configure_build,load_build_config,selected_k_tile,pipeline_plan,K_PREFIX,launch_evidence
from test_a5 import scale_launch
from compare_a5 import match_plans,compatible
from run_a51 import run,REPRESENTATIVES
from compare_a51 import check_protocol
from run_a51_pair import run as pair_run
from a2_common import sha256

class A51Tests(unittest.TestCase):
    def test_exact_protocol_not_relaxed(self):
        timing=dict(warmup=5,repetitions=30,protocol='continuous-v1')
        report=dict(cases=[dict(executions={kind:dict(timing=copy.deepcopy(timing)) for kind in ('custom','native_om')})])
        check_protocol(report)
        for key,value in (('warmup',3),('repetitions',10),('protocol','checked-v1')):
            bad=copy.deepcopy(report); bad['cases'][0]['executions']['native_om']['timing'][key]=value
            with self.assertRaises(ValueError): check_protocol(bad)

    def test_pair_reuses_a5_inputs_and_only_changes_k(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'a5'; source.mkdir()
            write_json(source/'suite.json',dict(status='PASS'))
            inputs={}
            for gear in ('c16','c64'):
                inputs[gear]={}
                for name in ('bundle','native_om_manifest'):
                    file=root/f'{gear}-{name}.json'; file.write_text('CPU fixture only')
                    inputs[gear][name]=dict(path=str(file),sha256=sha256(file))
            old=root/'a4.json'; write_json(old,dict(abi='dflash-group-quant-linear-a4-suite-v1',status='PASS',inputs=inputs))
            (source/'scalar-summary.txt').write_text(str(old)+'\n')
            saved=old.read_bytes(); variants=[]
            def launch(command,**kwargs):
                env=kwargs['env']; variants.append(env['DFLASH_K_TILE'])
                for key,value in dict(DFLASH_SUITE='a51',DFLASH_SCALE_MODE='broadcast',DFLASH_KV_M80_MODE='a-ub',
                                      A2_WARMUP='5',A2_REPETITIONS='30').items(): self.assertEqual(env[key],value)
                self.assertEqual(env['A4_C64_BUNDLE'],inputs['c64']['bundle']['path'])
                output=root/f"k{env['DFLASH_K_TILE']}.json"; write_json(output,dict(status='PASS'))
                Path(env['DFLASH_SUMMARY_POINTER']).write_text(str(output)+'\n')
            args=SimpleNamespace(output_dir=root/'pair',a5_evidence=source,order='baseline-first',pipeline='prefetch')
            def compare(a,b,out):
                write_json(out,dict(status='PASS',runtime='CPU fixture')); return 0
            with patch('run_a51_pair.subprocess.run',side_effect=launch),patch('run_a51_pair.compare',side_effect=compare):
                self.assertEqual(pair_run(args),0)
            self.assertEqual(variants,['128','256']); self.assertEqual(old.read_bytes(),saved)
            report=json.loads((root/'pair/suite.json').read_text()); self.assertEqual(report['adoption'],'NOT_DECIDED')

    def test_build_k_tile_binding_and_unchanged_kv(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for tile in (128,256):
                configure_build(root/'h',root/'b',0,'batched',root/'kh','prefetch','a-ub','broadcast',tile)
                self.assertEqual(load_build_config(root/'b')['k_tile'],tile)
                self.assertEqual(selected_k_tile((80,2560,2048),tile),128)
                self.assertEqual(selected_k_tile((16,256,64),tile),256)
                self.assertEqual(selected_k_tile((32,2560,2048),tile),tile)
            cfg=json.loads((root/'b').read_text()); cfg['k_tile']=128; write_json(root/'b',cfg)
            with self.assertRaisesRegex(ValueError,'K tile'): load_build_config(root/'b')
            for tile in (True,64,512,'256'):
                with self.assertRaises(ValueError): selected_k_tile((16,2560,19456),tile)

    def test_k256_launch_rejects_wrong_base_k_ub_and_resource_overflow(self):
        shape=(16,2560,19456)
        with tempfile.TemporaryDirectory() as directory:
            log=Path(directory)/'runner.log'
            left=scale_launch(log,shape,'broadcast')
            lines=log.read_text().splitlines(); launch=json.loads(lines[0].split(' ',1)[1]); cube=json.loads(lines[1].split(' ',1)[1])
            launch.update(tile_k=256,**pipeline_plan(shape[1],shape[2],'serial',256))
            launch['user_ub_bytes']+=10496; launch['matmul_ub_bytes']=262144-launch['user_ub_bytes']
            cube.update(base_k=256,weight_dequantizations_per_n_tile=10)
            kt=dict(version=1,requested_k_tile=256,selected_k_tile=256,l1_bytes=81920,l1_capacity=1048576,
                    l0a_bytes=16384,l0a_capacity=65536,l0b_bytes=65536,l0b_capacity=65536,l0c_bytes=4096,l0c_capacity=262144)
            def emit():
                log.write_text(lines[0].split(' ',1)[0]+' '+json.dumps(launch)+'\n'+lines[1].split(' ',1)[0]+' '+json.dumps(cube)+'\n'+lines[2]+'\n'+K_PREFIX+json.dumps(kt)+'\n')
            def read(): return launch_evidence(log,shape,0,2097152,'batched','serial',5,'a-ub','broadcast',256)
            emit(); right=read(); match_plans(left,right,shape,True)
            with self.assertRaises(ValueError): match_plans(left,right,shape)
            cube['base_k']=128; emit()
            with self.assertRaises(ValueError): read()
            cube['base_k']=256; kt['l0b_bytes']=65537; emit()
            with self.assertRaisesRegex(ValueError,'resource'): read()
            kt['l0b_bytes']=65536; launch['user_ub_bytes']-=128; emit()
            with self.assertRaises(ValueError): read()

    def test_representative_failure_prevents_full_a4(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); cfg=root/'b.json'
            configure_build(root/'h',cfg,0,'batched',root/'kh','prefetch','a-ub','broadcast',256)
            args=SimpleNamespace(build_config=cfg,output_dir=root/'run',warmup=5,repetitions=30,device_id=0,
                c16_bundle=root/'16',c64_bundle=root/'64',c16_native_om_manifest=root/'n16',c64_native_om_manifest=root/'n64')
            def fail(child):
                child.output_dir.mkdir(); write_json(child.output_dir/'suite.json',dict(status='FAIL')); return 1
            with patch('run_a51.load_bundle'),patch('run_a51.a4_contract.paired_gears'),patch('run_a51.run_a2.run',side_effect=fail) as real,patch('run_a51.run_a4.run') as full:
                self.assertEqual(run(args),1)
            self.assertEqual(real.call_count,1); full.assert_not_called()
            self.assertEqual(real.call_args.args[0].case_filter,REPRESENTATIVES)
            report=json.loads((root/'run/suite.json').read_text()); self.assertEqual(report['full_a4_regression'],'NOT_RUN')

    def test_both_gears_must_pass_before_full(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); cfg=root/'b.json'
            configure_build(root/'h',cfg,0,'batched',root/'kh','prefetch','a-ub','broadcast',256)
            args=SimpleNamespace(build_config=cfg,output_dir=root/'run',warmup=5,repetitions=30,device_id=0,
                c16_bundle=root/'16',c64_bundle=root/'64',c16_native_om_manifest=root/'n16',c64_native_om_manifest=root/'n64')
            calls=[]
            def real(child):
                calls.append(child.output_dir.name); child.output_dir.mkdir(); write_json(child.output_dir/'suite.json',dict(status='PASS')); return 0
            def full(child):
                self.assertEqual(calls,['representative-c16','representative-c64'])
                calls.append('full'); child.output_dir.mkdir(); write_json(child.output_dir/'suite.json',dict(status='PASS')); return 0
            with patch('run_a51.load_bundle'),patch('run_a51.a4_contract.paired_gears'),patch('run_a51.run_a2.run',side_effect=real),patch('run_a51.collect',return_value=[]),patch('run_a51.run_a4.run',side_effect=full):
                self.assertEqual(run(args),0)
            self.assertEqual(calls[-1],'full')

if __name__=='__main__': unittest.main()
