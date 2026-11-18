import copy
import json
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import run_flow as flow
from flow_common import a4_contract,c,cases
from flow_query import assignments,compare_pair,cpu_plan,read_query
from raw_protocol import OUTPUTS


def fixtures():
    evidence={'projections':{}}
    for gear in (16,64):
        rows=[]
        for spec in a4_contract.cases(gear):
            m,k,n=(spec[key] for key in ('m','k','n'))
            rows.append(dict(name=spec['name'],shape=[m,k,n],inputs={key:dict(bytes=size,sha256='fixture') for key,size in
                        [('x.bin',m*k*2),('w_nz.bin',k*n),('s_gn.bin',k*n//64)]},golden=dict(bytes=m*n*2)))
        evidence['projections'][str(gear)]=rows
    return evidence


class Contracts(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)

    def test_full_shape_inventory(self):
        evidence=fixtures();inventory=cases(evidence);self.assertEqual(len(inventory),52)
        self.assertEqual(inventory['64/layer-4-kv']['shape'],[80,2560,2048])
        for case in inventory.values():
            tiles=case['shape'][2]//64
            model=cpu_plan(tiles,5,6,6)
            for blocks in (model['baseline'],model['expected_calc']):
                self.assertEqual(sorted(sum(assignments(tiles,blocks),[])),list(range(tiles)))
        for damage in ('shape','scale','bytes','missing','renamed'):
            broken=copy.deepcopy(evidence);row=broken['projections']['64'][1]
            if damage=='shape':row['shape'][0]=16
            if damage=='scale':row['inputs']['s_gn.bin']['sha256']='changed'
            if damage=='bytes':row['golden']['bytes']-=2
            if damage=='missing':broken['projections']['16'].pop()
            if damage=='renamed':row['name']='foreign'
            with self.subTest(damage=damage),self.assertRaises(ValueError):cases(broken)

    def test_gate_callback(self):
        sys.path.insert(0,str(HERE.parent/'draft_a_core_scheduler_diag/test'))
        import test_scheduler
        fixture=test_scheduler.Contracts();fixture.setUp();self.addCleanup(fixture.doCleanups)
        records=[]
        for variant in 'AB':
            p,s=fixture.query(variant)
            p['n']=s['n']=19456;p['k']=s['k']=2560;s['n_tiles']=304
            s.update(baseline_assignments=assignments(304,7),candidate_assignments=assignments(304,8))
            def parse():
                paths=[]
                for name,row in [('platform',p),('schedule',s)]:
                    path=self.root/(name+variant);path.write_text(json.dumps(row)+'\n');paths.append(path)
                return read_query(*paths,fixture.opp,'ACLNN',variant,[16,2560,19456])
            records.append(parse())
            s['candidate_assignments'][0].pop()
            with self.assertRaises(ValueError):parse()
            s['candidate_assignments']=assignments(304,8)
            p['resource_map']['SoCInfo']['core_type_list']='CubeCore,VectorCore'
            with self.assertRaises(ValueError):parse()
        self.assertEqual(compare_pair(*records)['B_block_dim'],8)
        records[1]['schedule']['resources']['workspace']+=1
        with self.assertRaises(ValueError):compare_pair(*records)

    def test_benchmark_counts_and_no_history(self):
        runner=self.root/'runner';runner.touch();rid=c.identity(runner)
        golden={};abi={'inputs':[],'outputs':[]}
        for name in OUTPUTS:
            path=self.root/name;path.write_bytes(bytes(120 if name=='draft_top1' else 16));golden[name]=c.identity(path)
        spec=dict(abi=abi,inputs={},golden=golden)
        prior=dict(current=dict(cann_root=str(self.root),evidence={'H':{'gears':{'16':spec,'64':spec}}}))
        built={'variants':{v:{'om':dict(runner=rid,deps=[])} for v in 'AB'}}
        exported=dict(attribution='ACTUAL_ATC_SCHEDULES_IDENTICAL',models={v+'/'+g:{'om':rid} for v in 'AB' for g in ('16','64')})
        def command(argv,log,env,admit):
            admit()
            _,_,_,folder,_,_,mode=argv;timed=mode=='benchmark'
            for name,item in golden.items():
                for suffix in ['actual-0.bin','actual-1.bin']+(['benchmark-last.bin','postcheck.bin'] if timed else []):
                    (folder/(name+'-'+suffix)).write_bytes(c.verify(item).read_bytes())
            timing=dict(status='MEASURED' if timed else 'NOT_RUN',protocol='continuous-v1' if timed else 'checked-v1',warmup=5 if timed else 0,repetitions=30 if timed else 0,correctness_before_calls=2,correctness_after_calls=int(timed),timed_tail_checked=timed,per_timed_call_readback=not timed,per_timed_call_poison=not timed,prepare={'samples_ms':[]},execute_sync={'samples_ms':([2 if folder.parent.name=='A' else 1]*30 if timed else [])})
            c.write(folder/'execution.json',dict(status='PASS',guards=True,readonly=True,repeat=True,cpu_fallback=False,golden_match=True,poison_verified=True,sync_completed=True,timed_log_stable=True,device=0,verification_calls=4 if timed else 2,row_permutation_checked=False,workspace_bytes=128,model_weight_bytes=256,timing=timing))
        with patch.object(flow,'build_env',return_value={}),patch.object(flow,'mapped'),patch.object(c,'dependencies',return_value=[]),patch.object(flow,'invoke_om',side_effect=command):
            smoke={};bench={}
            flow.full(self.root/'smoke',prior,built,exported,smoke)
            flow.full(self.root/'bench',prior,built,exported,bench,smoke)
            self.assertEqual(len(bench['runs']),12)
            for result in bench['statistics'].values():
                self.assertEqual(result['round_A_over_B'],[2,2,2]);self.assertEqual(len(result['B']['samples_ms']),90)
            self.assertEqual(bench['attribution'],'ACTUAL_ATC_SCHEDULES_IDENTICAL')

    def test_missing_predecessor_blocks(self):
        args=SimpleNamespace(phase='benchmark',root=self.root/'new',host_root=self.root/'old',smoke_root=self.root/'fc')
        with patch.object(flow,'full') as execute:
            self.assertEqual(flow.main(args),1);execute.assert_not_called()
        self.assertEqual(c.read(args.root/'benchmark/report.json')['status'],'BLOCKED')

    def test_projection_mismatch_retains_output_and_stops(self):
        frozen=self.root/'golden';frozen.write_bytes(bytes(128))
        case=dict(shape=[1,128,64],golden=c.identity(frozen),inputs={'x.bin':c.identity(frozen)})
        binary=dict(runner=c.identity(frozen),deps=[])
        plan={'schedule':{'resources':{'workspace':128}},'host_dso':{},'tiling_sha256':'fixture','platform':{}}
        selected=dict(runner=binary,env={},opp={},api={})
        report={};dest=self.root/'runs';dest.mkdir()
        def invoke(exe,folder,repeat,env,workspace,admit):
            admit();(folder/'actual.bin').write_bytes(bytes(126)+b'\x01\x00')
        with patch.object(flow,'build_env',return_value={}),patch.object(c,'dependencies',return_value=[]),patch.object(flow.host,'verify_environment'),patch.object(flow,'read_query',return_value=plan),patch.object(flow,'mapped',return_value=[dict(file={})]),patch.object(flow.fc,'invoke',side_effect=invoke) as runner:
            # Simulated prelaunch maps artifact (not a device run).
            def launch(exe,folder,repeat,env,workspace,admit):
                (folder/'pre-maps.txt').touch();invoke(exe,folder,repeat,env,workspace,admit)
            runner.side_effect=launch
            with self.assertRaisesRegex(ValueError,'Native bits'):
                flow.projections(dest,{'current':{'cann_root':'/unused'}},{'cases':{'16/test':case}},{'variants':{'A':selected}},report,{'A/16/test':plan})
        runner.assert_called_once()
        row=report['runs']['A/16/test'][0];self.assertEqual(row['numerics']['bit_mismatches'],1);c.verify(row['actual'])

    def test_partial_104_gate_rejected(self):
        with self.assertRaisesRegex(ValueError,'partial'):
            flow.inventory('projection-smoke',dict(plans={},runs={}),{'source':{'binding':{'cases':cases(fixtures())}}})

    def test_fixed_source_diff(self):
        old=HERE.parent/'draft_a_core_scheduler_diag/op_host/d_flash_group_quant_linear.cpp'
        self.assertEqual(old.read_text().replace('m!=16 || k!=12800 || n!=2560','n<=0 || n%kTileN!=0').replace('Write(context,scheduling','Write(context,m,k,n,scheduling'),(HERE/'op_host'/old.name).read_text())
        expected=(old.parent/'scheduler_evidence.h').read_text().split('inline bool Append')[0]
        self.assertEqual(expected,(HERE/'op_host/scheduler_evidence.h').read_text().split('inline bool Append')[0])
