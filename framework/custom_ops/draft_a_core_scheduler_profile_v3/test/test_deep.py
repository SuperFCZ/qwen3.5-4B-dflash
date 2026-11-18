import sys
from pathlib import Path
from decimal import Decimal
from tempfile import TemporaryDirectory
import unittest
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import analyze_v3 as deep
from signals import Graph,UNKNOWN,notify,projection
from audit import PROVISIONAL


def ev(idx,lo,hi,op,category,signal=None):
    return dict(idx=idx,lo=Decimal(lo),hi=Decimal(hi),start=str(lo),end=str(hi),us=float(hi-lo),op=op,category=category,
                call=7,pid=1,model='0',stream=str(idx),task_id='7',state=PROVISIONAL,compute=signal is None,signal=signal,raw={},
                csv_join=UNKNOWN,csv_lines=[],csv_fields=None,notify_id='9' if signal else None)


def node(name,kind,inputs=()):
    return dict(name=name,type=kind,inputs=list(inputs),input_desc=[],output_desc=[],serialized_descriptors=[])


class DeepContracts(unittest.TestCase):
    def test_two_gear_offline_chain(self):
        sys.path.insert(0,str(HERE.parent/'draft_a_core_scheduler_profile_v2/test'))
        from test_offline import data
        def save(path,obj):
            deep.v2.write(path,obj);return deep.v2.identity(path)
        with TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            cases={g+'/'+str(i):dict(name=str(i),shape=[16,2560,4096]) for g in ('16','64') for i in range(26)}
            source=save(root/'source.json',{'binding':{'cases':cases}})
            smoke=save(root/'smoke.json',dict(status='FLOW_PROJECTION_SMOKE_PASS',plans={'A/'+k:{'schedule':dict(m=16,k=2560,n=4096)} for k in cases}))
            dep=save(root/'deployment.json',{'graphs':[{'runtime_input_abi':{'draft_ac':{'status':'PASS','counts':{'DFlashGroupQuantLinear':26},'nodes':[]}}}]})
            model=dict(om={'sha256':'fixture'},deployment=dep)
            export=save(root/'export.json',dict(status='FLOW_EXPORT_PASS',predecessors={'projection-smoke':smoke},models={'A/16':model,'A/64':model}))
            admit=save(root/'admit.json',dict(status='A_OM_PROFILE_ADMITTED',binding={'cases':{'16':model,'64':model},'history':{'source':source,'projection-smoke':smoke,'export':export}}))
            main=dict(status='A_OM_PROFILE_ANALYZED',predecessors={'admit':admit},gears={})
            for gear in ('16','64'):
                cap,main['gears'][gear]=data(root,gear,gear=='16');path=Path(cap['path'])
                main['predecessors']['profile-c'+gear]=save(path,dict(deep.v2.read(path),predecessors={'admit':admit}))
            save(root/'analyze/profile-report.json',main)
            self.assertEqual(deep.v2.main(root),0);self.assertEqual(deep.main(root),0)
            self.assertEqual(len(list((root/'analysis-v3').glob('*.md'))),5)
            self.assertEqual(deep.v2.read(root/'analysis-v3/c16/notify.json')['wait_count'],0)
            with self.assertRaises(FileExistsError):deep.main(root)

    def test_wait_intersections_not_sums(self):
        events=[ev(0,100,109,'NOTIFY_WAIT','other','wait'),ev(1,101,105,'NOTIFY_WAIT','other','wait'),
                ev(2,102,106,'gate','GroupQuantLinear'),ev(3,103,108,'gate2','GroupQuantLinear'),
                ev(4,107,108,'NOTIFY_RECORD','other','record'),ev(5,109,110,'add','Elementwise')]
        report=notify(events,[dict(call=7,raw_host_start_us='100',raw_sync_end_us='110')]);call=report['calls'][0]
        self.assertEqual((call['sum_us'],call['wait_union_us'],call['intersection_us']),(13,9,6))
        self.assertEqual(report['waits'][0]['groupquant_overlap_us'],6)
        self.assertEqual(report['waits'][0]['matching_notify_id_during'],[4])
        self.assertEqual(report['waits'][0]['compute_neighbors']['following'][0]['idx'],5)
        self.assertEqual(report['waits'][0]['wait_object'],UNKNOWN)

    def test_hash_pair_not_node_suffix(self):
        cases=[dict(name=f'layer-{i}-q',shape=[16,2560,4096],inputs={'w_nz.bin':dict(bytes=2560*4096,sha256=str(i)),'s_gn.bin':dict(bytes=163840,sha256='scale')}) for i in range(5)]
        name='shuffled_25';n=node(name,'DFlashGroupQuantLinear');fields=[{'Input Shapes':'16,2560','Output Shapes':'16,4096'}]
        audit=[dict(name=name,ge_type=n['type'],nz_sha256='1',scale_gn_sha256='scale',n=4096,k=2560,nz_bytes=2560*4096)]
        self.assertEqual(projection(name,n,fields,audit,cases)['projection'],'layer-1-q')
        self.assertEqual(projection(name,n,fields,[],cases)['role'],'q')
        self.assertEqual(projection(name,n,fields,[],cases)['projection'],UNKNOWN)
        audit[0]['nz_sha256']='wrong'
        self.assertEqual(projection(name,n,fields,audit,cases)['role'],UNKNOWN)

    def test_attention_requires_dataflow_and_layer_provenance(self):
        nodes=[node('Q','DFlashGroupQuantLinear'),node('KV','DFlashGroupQuantLinear'),node('mm','BatchMatMul',['Q:0','KV:0']),
               node('scale','Mul',['mm:0']),node('soft','Softmax',['scale:0']),node('pv','BatchMatMul',['soft:0','KV:1'])]
        proj=[dict(name='Q',projection='layer-2-q'),dict(name='KV',projection='layer-2-kv')]
        def analyze():return deep.paths_report(Graph(nodes),{'soft':[]},'attention',{'rows':[]},proj)[0]
        self.assertEqual(analyze()['layer'],2)
        proj[0]['projection']='layer-1-q';self.assertIsNone(analyze()['layer'])
        nodes[-1]['inputs'][0]='soft:-1';self.assertEqual(analyze()['status'],UNKNOWN)

    def test_head_conversion_must_be_on_weight_branch(self):
        net=Graph([node('weight','Const'),node('layout','TransData',['weight:0']),node('head','MatMul',['data:0','layout:0']),node('top','ArgMaxV2',['head:0'])])
        item=deep.paths_report(net,{'head':[]},'head',{'rows':[]},[])[0]
        self.assertEqual(item['weight_conversion_paths'],[['head','layout']]);self.assertEqual(item['top1_paths'],[['head','top']])

    def test_all_waits_including_unassigned_and_raw_csv_only(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'msprof.json';deep.v2.write(p,[dict(ph='M',name='process_name',pid=5,args={'name':'Ascend Hardware'}),
                dict(ph='X',pid=5,tid=2,ts='1.5',dur='2',name='NOTIFY_WAIT',args={'Task Id':4,'Task Type':'NOTIFY_WAIT'})])
            rows=[dict(event_index='1',start_us='1.5',end_us='3.5',call='',state='CROSSING',op_name='NOTIFY_WAIT',category='Other / Unmapped',op_type='')]
            table={'rows':[dict(line=2,op_name='NOTIFY_WAIT',fields={'Task ID':'4','Stream ID':'2','Task Type':'NOTIFY_WAIT','Op State':'running'}),
                           dict(line=3,op_name='NOTIFY_RECORD',fields={'Task ID':'42','Stream ID':'3','Task Type':'NOTIFY_RECORD'})]}
            data,raw=deep.task_data(p,rows,table);self.assertEqual(len(raw),2)
            self.assertIsNone(data[0]['call']);self.assertEqual(data[0]['csv_fields']['Op State'],'running')
