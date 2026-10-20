"""Gate-only prefetch: CPU dependency/control-flow and strict evidence gates."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from a3_launch import configure_build,load_build_config,pipeline_plan,launch_evidence
from a2_common import write_json
from gate_prefetch_contract import check_plan,match_plans
import run_gate_copy_pair as pair

HERE=Path(__file__).resolve().parent
FLAGS=['-DDFLASH_GROUP_QUANT_K_TILE=256','-DDFLASH_GROUP_QUANT_SCALE_MODE=1','-DDFLASH_GROUP_QUANT_KV_M80_MODE=1','-DDFLASH_GROUP_QUANT_GATE_COPY_MODE=0','-DDFLASH_GROUP_QUANT_GATE_PREFETCH_MODE=1']


def plan(mode):
    enabled=mode=='raw'
    return dict(m=16,k=2560,n=19456,tile_m=16,tile_n=64,tile_k=256,weight_reuse_rows=16,
                pipeline_mode='prefetch',raw_banks=2 if enabled else 1,selected_pipeline='raw-prefetch-v1' if enabled else 'serial-v1',
                user_ub_bytes=76544 if enabled else 59904,matmul_ub_bytes=185600 if enabled else 202240,
                gate_prefetch=dict(version=1,mode=mode,enabled=enabled),system_workspace_bytes=2097152,block_dim=7,
                cube=dict(a_source='GM',base_m=16,base_n=64,base_k=256,weight_dequantizations_per_n_tile=10))


class GatePrefetchTests(unittest.TestCase):
    def test_actual_kernel_models_and_hazard_mutations(self):
        compiler=shutil.which('c++')
        if not compiler:self.skipTest('C++ CPU model unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def execute(source,name,extra=()):
                binary=root/name
                c=subprocess.run([compiler,'-std=c++17','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-Wno-unused-parameter',*FLAGS,*extra,'-I',str(HERE/'cpu_kernel_include'),'-I',str(HERE.parent/'op_kernel'),str(source),'-o',str(binary)],capture_output=True,text=True)
                self.assertEqual(c.returncode,0,c.stdout+c.stderr)
                return subprocess.run([str(binary)],cwd=root,capture_output=True,text=True)
            for old_pipeline in (0,1):
                r=execute(HERE/'kernel_model.cpp',f'full-{old_pipeline}',[f'-DDFLASH_GROUP_QUANT_PIPELINE_MODE={old_pipeline}'])
                self.assertEqual(r.returncode,0,r.stdout+r.stderr);self.assertIn('NPU NOT_RUN',r.stdout)
            r=execute(HERE/'gate_prefetch_fault_model.cpp','faults')
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            kernel=(HERE.parent/'op_kernel/d_flash_group_quant_linear.cpp').read_text()
            tail='// All consumers of q/scale/W and MatMul'
            begin=kernel.index(tail);barrier=kernel.index('AscendC::PipeBarrier<PIPE_ALL>();',begin)
            broken={
                'missing_consumer_fence':kernel[:barrier]+kernel[barrier:].replace('AscendC::PipeBarrier<PIPE_ALL>();','/* missing consumer fence */',1),
                'borrowed_event_id':kernel.replace('pipe.AllocEventID<AscendC::HardEvent::MTE2_S>()','pipe.FetchEventID(AscendC::HardEvent::MTE2_S)'),
                'missing_vector_wait':kernel.replace('AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputVector);','/* missing MTE2_V wait */'),
                'overwrite_current_bank':kernel.replace('qBanks[(1 - bank) * kTileN * tileK]','qBanks[bank * kTileN * tileK]'),
            }
            for name,code in broken.items():
                kp=root/(name+'.h');kp.write_text(code)
                source=root/(name+'.cpp');source.write_text((HERE/'gate_prefetch_fault_model.cpp').read_text().replace('"../op_kernel/d_flash_group_quant_linear.cpp"',json.dumps(str(kp))))
                r=execute(source,name)
                self.assertNotEqual(r.returncode,0,'model failed to detect '+name)
                reasons={'missing_consumer_fence':'Cube consumer','borrowed_event_id':'collision','missing_vector_wait':'collision','overwrite_current_bank':'DMA completion'}
                self.assertIn(reasons[name],r.stdout+r.stderr)

    def test_config_independent_shape_and_2d_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for mode in ('off','raw'):
                configure_build(root/'h',root/'c',0,'batched',root/'kh','prefetch','a-ub','broadcast',256,'loop',mode)
                self.assertEqual(load_build_config(root/'c')['gate_prefetch_mode'],mode)
            c=json.loads((root/'c').read_text());c['gate_prefetch_mode']='off';write_json(root/'c',c)
            with self.assertRaisesRegex(ValueError,'Gate prefetch'):load_build_config(root/'c')
            with self.assertRaisesRegex(ValueError,'Gate prefetch'):configure_build(root/'h',root/'c',0,'batched',root/'kh','prefetch','a-ub','broadcast',256,'2d','raw')
        self.assertEqual(pipeline_plan(2560,19456,'serial',256,'raw',16)['raw_banks'],2)
        self.assertEqual(pipeline_plan(2560,19456,'prefetch',256,'off',16)['raw_banks'],1)
        for m,k,n in ((32,2560,19456),(16,9728,2560),(16,2560,4096),(16,4096,2560),(80,2560,2048),(16,12800,2560)):
            self.assertEqual(pipeline_plan(k,n,'prefetch',256,'off',m),pipeline_plan(k,n,'prefetch',256,'raw',m))

    def test_only_raw_bank_budget_can_change(self):
        a,b=plan('off'),plan('raw');match_plans(a,b)
        self.assertEqual(check_plan(b,'raw')['single_w16_bytes'],32768)
        for key,value in (('tile_k',128),('matmul_ub_bytes',202240),('user_ub_bytes',59904),('block_dim',8),('system_workspace_bytes',4194304)):
            with self.assertRaises(ValueError):match_plans(a,dict(b,**{key:value}))
        bad=copy.deepcopy(b);bad['cube']['base_k']=128
        with self.assertRaises(ValueError):match_plans(a,bad)

    def test_pair_sets_only_prefetch_and_forces_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);files={}
            for n in ('bundle','native','golden','accepted'):
                p=root/n;p.write_text('CPU fixture');files[n]=str(p)
            source=dict(bundle=files['bundle'],native=files['native'],golden=files['golden'],config=dict(core_limit=0,pipeline_mode='prefetch',kv_m80_mode='a-ub'))
            calls=[]
            def launch(command,**kwargs):
                env=kwargs['env'];mode=env['DFLASH_GATE_PREFETCH_MODE'];calls.append(mode)
                self.assertEqual(env['DFLASH_GATE_COPY_MODE'],'loop');self.assertEqual(env['DFLASH_SUITE'],'gate-prefetch')
                self.assertEqual(env['DFLASH_PIPELINE_MODE'],'prefetch');self.assertEqual(env['GATE_LAYERS'],'0')
                run=root/mode;data=run/'data';data.mkdir(parents=True);write_json(data/'suite.json',dict(status='PASS',selected_cases=['layer-0-gate_up']))
                for n in ('sdk-copy.json','opp-preflight.json'):write_json(run/n,dict(status='SOURCE_INSPECTED'))
                (run/'source-commit.txt').write_text('fixture');Path(env['DFLASH_SUMMARY_POINTER']).write_text(str(data/'suite.json'))
            def compare(a,b,out,experiment):
                self.assertEqual(experiment,'prefetch');write_json(out,dict(status='PASS'));return dict(status='PASS')
            args=SimpleNamespace(experiment='prefetch',output_dir=root/'out',a51_evidence=Path(files['accepted']),layers=[0],device_id=0,order='baseline-first')
            with patch.object(pair,'source_evidence',return_value=source),patch.object(pair.subprocess,'run',side_effect=launch),patch.object(pair,'compare_runs',side_effect=compare):self.assertEqual(pair.run(args),0)
            self.assertEqual(calls,['off','raw'])
            result=json.loads((root/'out/suite.json').read_text());self.assertEqual(result['adoption'],'NOT_DECIDED');self.assertEqual(result['full_draft_token_kv'],'NOT_RUN')
            args.layers=[0,0]
            with self.assertRaises(ValueError):pair.run(args)

    def test_real_launch_log_binds_gate_selection_and_ub(self):
        from test_a5 import scale_launch
        from a3_launch import K_PREFIX
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'runner.log';shape=(16,2560,19456)
            scale_launch(log,shape,'broadcast')
            lines=log.read_text().splitlines()
            launch=json.loads(lines[0].split(' ',1)[1]);cube=json.loads(lines[1].split(' ',1)[1])
            launch.update(tile_k=256,**pipeline_plan(2560,19456,'serial',256,'raw',16))
            launch['user_ub_bytes']+=10496;launch['matmul_ub_bytes']=262144-launch['user_ub_bytes']
            cube.update(base_k=256,weight_dequantizations_per_n_tile=10)
            kt=dict(version=1,requested_k_tile=256,selected_k_tile=256,selection_reason='configured',l1_bytes=81920,l1_capacity=1048576,l0a_bytes=16384,l0a_capacity=65536,l0b_bytes=65536,l0b_capacity=65536,l0c_bytes=4096,l0c_capacity=262144)
            text=lines[0].split(' ',1)[0]+' '+json.dumps(launch)+'\n'+lines[1].split(' ',1)[0]+' '+json.dumps(cube)+'\n'+lines[2]+'\n'+K_PREFIX+json.dumps(kt)+'\n'
            def read():return launch_evidence(log,shape,0,2097152,'batched','serial',5,'a-ub','broadcast',256,'m80-synthetic-k128-v1','raw')
            log.write_text(text)
            with self.assertRaisesRegex(ValueError,'Gate prefetch'):read()
            tag='DFLASH_GROUP_QUANT_GATE_PREFETCH {"version":1,"mode":"raw","enabled":true}\n'
            log.write_text(text+tag);p=read();self.assertEqual(check_plan(p,'raw')['matmul_ub_bytes'],185600)
            log.write_text(text+tag.replace('true','false'))
            with self.assertRaisesRegex(ValueError,'Gate prefetch'):read()
            launch['matmul_ub_bytes']=202240
            log.write_text(lines[0].split(' ',1)[0]+' '+json.dumps(launch)+'\n'+text.split('\n',1)[1]+tag)
            with self.assertRaises(ValueError):check_plan(read(),'raw')
