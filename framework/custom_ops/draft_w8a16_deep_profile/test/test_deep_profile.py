import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import run_deep as driver
from deep_common import c
import hotspots as h


class DeepProfileTests(unittest.TestCase):
    def test_reused_counter_ids_and_missing_stalls(self):
        def event(i,call,start):
            return dict(idx=i,stream='2',task_id='4',op='gate',kind='G',model='1',call=call,start=start,label='Linear/gate_up')
        def row(line,start,value):
            return dict(line=line,op_name='gate',op_type='G',fields={'Stream ID':'2','Task ID':'4','Task Start Time(us)':start,'mte2_time(us)':value,'stall cycles':'N/A'})
        events=[event(1,1,'1792390000000000.0001'),event(2,7,'1792390000000000.0002')]
        table=dict(file='op_summary.csv',rows=[row(2,events[0]['start'],'99'),row(3,events[1]['start'],'4.5')])
        result=h.counters(events,table)
        self.assertEqual(result['joined'],1);self.assertEqual(result['unique_csv_links'][0]['metrics']['mte2_time(us)'],4.5)
        self.assertEqual(result['stall'],'UNAVAILABLE')
        table['rows'][0]['fields']['Task Start Time(us)']=events[1]['start']
        self.assertEqual(h.counters(events,table)['joined'],0)
        table['rows'].pop();table['rows'][0]['fields'].pop('Task Start Time(us)')
        self.assertEqual(h.counters(events,table)['joined'],0)

    def test_no_benchmark_or_accidental_capture(self):
        self.assertNotIn('benchmark',driver.PHASES)
        self.assertEqual(driver.deps('analyze-raw'),['admit','profile-raw-c16','profile-raw-c64'])
        self.assertIn('profile-off-c64',driver.deps('profile-paired-analysis'))
        with tempfile.TemporaryDirectory() as tmp:
            from types import SimpleNamespace
            t=Path(tmp).resolve();args=SimpleNamespace(phase='profile-raw-c16',root=t/'new',gate8_root=t/'old',historical_off=t/'off',allow_profile=False)
            with patch.object(driver,'collect') as launch,patch('builtins.print'):
                self.assertEqual(driver.main(args),1);launch.assert_not_called()

    def test_manifested_ub_budget(self):
        import hashlib
        r=json.loads((HERE/'w16-ub-budget.json').read_text());a,b=r['queries']
        self.assertEqual((a['get_tiling_rc'],b['get_tiling_rc']),(0,0))
        self.assertEqual(a['cube_hex'],b['cube_hex']);raw=bytes.fromhex(a['cube_hex'])
        self.assertEqual(len(raw),200)
        self.assertEqual(hashlib.sha256(struct.pack('<IIIIQ',16,2560,19456,256,202240)+raw).hexdigest(),
                         'ee2d5b1e3c35105c56aeaeec79c79d0bef64074ee774d3715618e7c2390b751a')
        intervals=[r['accounting'][k] for k in ('raw_codes','raw_scales','W0','W1','broadcast','matmul','brcb_reserve')]
        self.assertEqual([x[1] for x in intervals[:-1]],[x[0] for x in intervals[1:]])
        self.assertEqual((intervals[0][0],intervals[-1][1]),(0,262144));self.assertEqual(b['scratch'],152832)
        self.assertEqual(b['db'][1]*b['base'][1]*b['base'][2]*2,r['L0B'])

    def test_admitted_root_still_requires_capture_opt_in(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();old=root/'history';archive=root/'off'
            new=root/'new';folder=new/'admit';folder.mkdir(parents=True)
            c.write(folder/'profile-report.json',dict(phase='admit',status=driver.STATE['admit'],sources=driver.sources(),folder=str(folder),
                predecessors={},gate8_root=str(old),historical_off=str(archive),binding={}))
            args=SimpleNamespace(root=new,gate8_root=old,historical_off=archive,phase='profile-raw-c16',allow_profile=False)
            with patch.object(driver,'collect') as launch,patch('builtins.print'):
                self.assertEqual(driver.main(args),1);launch.assert_not_called()
            self.assertIn('--allow-profile',c.read(new/args.phase/'profile-report.json')['error'])

    def test_synthetic_end_to_end_strict_and_provisional(self):
        sys.path.insert(0,str(HERE.parent/'draft_a_core_scheduler_profile_v2/test'))
        from test_offline import data
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();cases={};variants={m:{'cases':{}} for m in ('off','raw')}
            results={}
            for gear in ('16','64'):
                fake=root/gear;fake.mkdir();cap_id,_=data(fake,gear,gear=='64');cap=c.read(c.verify(cap_id))
                folder=Path(cap_id['path']).parent;cap.update(folder=str(folder),strict_status=cap['device_assignment'])
                records=[]
                for i in range(26):
                    name='fc' if i==0 else f'layer-{(i-1)//5}-'+('kv','q','o','gate_up','down')[(i-1)%5]
                    cases[gear+'/'+name]=dict(name=name,shape=[16,2560,19456],inputs={'w_nz.bin':{'bytes':1,'sha256':str(i)},'s_gn.bin':{'sha256':str(i)}})
                    records.append(dict(name='g'+str(i),ge_type='DFlashGroupQuantLinear'))
                dep=fake/'deployment.json';c.write(dep,{'graphs':[{'runtime_input_abi':{'draft_ac':{'counts':{'DFlashGroupQuantLinear':26},'nodes':records}}}]})
                src=root/'source.json';c.write(src,dict(binding=dict(cases=cases)))
                for mode in variants:
                    variants[mode]['cases'][gear]=dict(baseline=dict(median_ms=31,p95_ms=32),deployment=c.identity(dep))
                    results[mode,gear]=h.analyze_case(root/(mode+'-c'+gear),cap,dict(variants=variants,flow_source=c.identity(src)),mode,gear)
                report=results['raw',gear]
                self.assertEqual(report['scope'],h.STRICT if gear=='16' else h.PROVISIONAL)
                self.assertEqual(report['projections']['layer-0-gate_up']['status'],'UNRESOLVED')
                self.assertEqual(report['classes']['Unmapped MatMul']['tasks']['count'],30)
                self.assertEqual(report['classes']['Unmapped MatMul']['time_per_execute']['mean_us'],5)
                self.assertFalse(report['classes']['Head Top1']['observed'])
            driver.render(root,results,True)
            self.assertTrue((root/'optimization-next-step.md').is_file())

    def test_named_projection_across_all_measured_calls(self):
        sys.path.insert(0,str(HERE.parent/'draft_a_core_scheduler_profile_v2/test'))
        from test_offline import data
        from attribution import recover
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();cid,_=data(root,'16',False);capture=c.read(c.verify(cid))
            folder=Path(cid['path']).parent;raw=folder/'profiler';timeline=raw/'msprof_0.json'
            events=c.read(timeline);csvrows=['Op Name,OP Type,Task Duration(us),Task Start Time(us),Stream ID,Task ID,mac_time(us)']
            for e in events:
                if e['name']=='mm':
                    e['name']='identified';e['args']['OpType']='DFlashGroupQuantLinear'
                    csvrows.append('identified,DFlashGroupQuantLinear,5,'+e['ts']+',2,4,1')
            c.write(timeline,events);(raw/'op_summary_0.csv').write_text('\n'.join(csvrows)+'\n')
            t,_,_=recover(raw,7);c.write(folder/'windows.json',t)
            g=folder/'om-graph.json';c.write(g,{'op':[{'name':'identified','type':'DFlashGroupQuantLinear'}]})
            capture.update(folder=str(folder),windows=c.identity(folder/'windows.json'),strict_status=t['device_assignment'],graph=c.identity(g),files=[c.identity(p) for p in raw.iterdir()])
            src=root/'src.json';cases={};nodes=[]
            for i in range(26):
                name='layer-2-gate_up' if i==3 else 'fixture-'+str(i)
                cases['16/'+name]=dict(name=name,shape=[16,2560,19456],inputs={'w_nz.bin':{'sha256':str(i),'bytes':49807360},'s_gn.bin':{'sha256':'s'+str(i)}})
                nodes.append(dict(name='identified' if i==3 else 'n'+str(i),ge_type='DFlashGroupQuantLinear',nz_sha256=str(i),scale_gn_sha256='s'+str(i),k=2560,n=19456,nz_bytes=49807360))
            c.write(src,{'binding':{'cases':cases}})
            dep=root/'dep.json';c.write(dep,{'graphs':[{'runtime_input_abi':{'draft_ac':{'counts':{'DFlashGroupQuantLinear':26},'nodes':nodes}}}]})
            dest=root/'analysis'
            result=h.analyze_case(dest,capture,dict(flow_source=c.identity(src),variants={'raw':{'cases':{'16':{'deployment':c.identity(dep),'baseline':{}}}}}),'raw','16')
            actual=result['projections']['layer-2-gate_up'];self.assertEqual(actual['status'],'BOUND_EVERY_CALL')
            self.assertEqual(actual['timing']['counts_per_execute'],[1]*30)
            self.assertEqual(c.read(dest/'counters.json')['coverage']['mac_time(us)'],30)


if __name__=='__main__':unittest.main()
