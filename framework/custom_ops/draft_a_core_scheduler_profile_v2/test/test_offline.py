import copy
import csv
from pathlib import Path
import tempfile
import sys
import unittest
from decimal import Decimal

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import analyze as v
from audit import PROVISIONAL,complete,enclose,partition,step_audit,step_csv
from attribution import aggregate,recover


def data(root,gear,damage):
    phase='profile-c'+gear;folder=root/phase;raw=folder/'profiler';raw.mkdir(parents=True)
    origin=Decimal('1792390000000000.0000')
    events=[dict(ph='M',name='process_name',pid=100,args={'name':'Ascend Hardware'}),
            dict(ph='M',name='thread_name',pid=100,tid=99,args={'name':'Step Trace(Model ID:1)'})]
    def event(name,pid,tid,start,dur,args,cat=''):
        events.append(dict(ph='X',name=name,pid=pid,tid=tid,ts=str(origin+start),dur=str(dur),args=args,cat=cat))
    csvrows=[]
    for i in range(38):
        start=i*100
        event('aclmdlExecuteAsync',7,12,start,2,{})
        event('aclrtSynchronizeStream',7,12,start+2,8,{})
        end=str(origin+start+9)
        event('Iteration '+str(i+1),100,99,start+2,'7.1234' if damage and i==12 else '7.0000',{'Iteration ID':i+1,'Iteration End':end},'Iteration Time')
        event('mm',100,21,start+3,5,dict(ModelId=1,TaskId=4,StreamId=2,TaskType='AI_CORE',OpType='MatMul'))
        csvrows.append(dict(zip(('Model ID','Iteration ID','FP Start(us)','Iteration End(us)','Iteration Time(us)'),(1,i+1,'N/A',end,'7.1234' if damage and i==12 else '7.0000'))))
    with (raw/'step_trace_0.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    args=dict(TaskId=3,ModelId=1,StreamId=2,TaskType='AI_CORE',OpType='Add')
    if damage:
        event('cross',100,21,798,4,args);event('outside',100,21,720,3,args)
        event('dup',100,21,703,2,args);event('dup',100,21,705,1,args)
        event('unknown',100,21,704,1,{})
    timeline=raw/'msprof_0.json';v.write(timeline,events)
    (raw/'op_summary_0.csv').write_text('Op Name,OP Type,Task Duration(us)\nmm,MatMul,100000\n')
    g=folder/'om-graph.json';v.write(g,{'op':[{'name':'mm','type':'MatMul'}]})
    t,tables,_=recover(raw,7);w=folder/'windows.json';v.write(w,t)
    cap=dict(status=f'A_C{gear}_RUNNER_CAPTURE_VERIFIED',execute='38_PROFILED_CALLS_GOLDEN_VERIFIED',runner_pid=7,
             graph=v.identity(g),windows=v.identity(w),device_assignment=t['device_assignment'],csv_sources=[{'file':tables[0]['file'],'rows':1}],
             files=[v.identity(p) for p in raw.iterdir()])
    cp=folder/'profile-report.json';v.write(cp,cap)
    analyzed=root/'analyze'/('c'+gear);analyzed.mkdir(parents=True);aggregate(analyzed,t,g,{'median_ms':31,'p95_ms':32})
    return v.identity(cp),dict(report=v.identity(analyzed/'profile-report.json'))


class Offline(unittest.TestCase):
    def test_bound_replay_both_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();main=dict(status='A_OM_PROFILE_ANALYZED',predecessors={},gears={})
            for gear in ('16','64'):
                main['predecessors']['profile-c'+gear],main['gears'][gear]=data(root,gear,gear=='16')
            v.write(root/'analyze/profile-report.json',main);self.assertEqual(v.main(root),0)
            out=root/'analysis-v2';broken=v.read(out/'c16/step-audit.json');report=v.read(out/'c16/provisional.json')
            self.assertEqual(broken['strict_status'],'UNRESOLVED_DEVICE_ASSIGNMENT')
            self.assertEqual(broken['first_blocking_step']['id'],13);self.assertEqual(broken['missing_strict_host_calls'],[12])
            endpoint=broken['first_blocking_step']['csv'];self.assertEqual(endpoint['status'],'UNPROVEN_ENDPOINTS')
            self.assertEqual(endpoint['unsupported_rows'][0]['line'],14)
            self.assertEqual(report['accounting'],{PROVISIONAL:38,'CROSSING':1,'OUTSIDE':1,'DUPLICATE_ID':2,'MISSING_TASK_ID':1})
            self.assertEqual(report['scope'],PROVISIONAL);self.assertEqual(report['timed_tasks'],30)
            self.assertEqual(report['classified_fraction'],0)
            self.assertEqual(report['categories'][0],dict(category='Unmapped MatMul',count=30,sum_us=150.,mean_us_per_execute=5.,max_task_us=5.,mean_task_us=5.))
            self.assertEqual(v.read(out/'c64/strict.json')['status'],'DEVICE_STEPS_VERIFIED')
            self.assertIsNone(v.read(out/'c64/step-audit.json')['first_blocking_step'])
            with self.assertRaises(FileExistsError):v.main(root)

    def test_csv_exact_and_precision_gate_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();data(root,'16',True);folder=root/'profile-c16';t=v.read(folder/'windows.json')
            done,tracks,hw=complete(v.read(folder/'profiler/msprof_0.json'));markers,_,_=partition(done,tracks,hw)
            evidence=step_csv.load(folder/'profiler')
            row=evidence['tables'][0]['rows'][12]['fields'];row['Iteration Start(us)']=markers[12]['raw_timestamp_us']
            result=step_audit(markers,tracks,[],evidence,t)
            item=result['steps'][12];self.assertEqual(item['precision']['status'],'REJECTED')
            self.assertEqual(item['csv']['status'],'EXACT_ENDPOINTS_CORROBORATED')
            row['Iteration Start(us)']=str(Decimal(row['Iteration Start(us)'])+Decimal('.0001'))
            item=step_audit(markers,tracks,[],evidence,t)['steps'][12]
            self.assertEqual(item['csv']['status'],'UNPROVEN_ENDPOINTS')
            self.assertEqual(Decimal(item['csv']['matches'][0]['time_field_comparison']['iterationstartus']['csv_minus_json_us']),Decimal('.0001'))

    def test_sub_microsecond_boundary(self):
        event=dict(event_index=10,name='task',pid=1,tid=1,raw_timestamp_us='1792390000000000.0001',dur='0.0001',args={'Task Id':1})
        start=Decimal('1792390000000000');windows=[(start,start+Decimal('.00015'))]
        self.assertEqual(enclose([event],windows)[0]['state'],'CROSSING')
        other=copy.deepcopy(event);other['raw_timestamp_us']=str(start)
        self.assertEqual(enclose([other],windows)[0]['call'],0)

    def test_missing_and_duplicate_iteration_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();data(root,'16',False);raw=root/'profile-c16/profiler'
            path=raw/'msprof_0.json';events=v.read(path)
            marker=next(e for e in events if e['name']=='Iteration 19')
            marker['name']='Iteration 18';marker['args']['Iteration ID']=18
            v.write(path,events);t,_,_=recover(raw,7)
            complete_events,tracks,hw=complete(events);markers,_,_=partition(complete_events,tracks,hw)
            result=step_audit(markers,tracks,[],step_csv.load(raw),t)
            self.assertEqual(result['missing_id_ranges_inclusive'],[(19,19)])
            self.assertEqual(result['duplicates'],{18:2});self.assertNotEqual(result['strict_status'],'DEVICE_STEPS_VERIFIED')

    def test_tampered_raw_rejected_before_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();cap,gear=data(root,'16',True)
            main=dict(status='A_OM_PROFILE_ANALYZED',predecessors={'profile-c16':cap},gears={'16':gear})
            v.write(root/'analyze/profile-report.json',main)
            with (root/'profile-c16/profiler/step_trace_0.csv').open('a') as f:f.write('\n')
            self.assertEqual(v.main(root),1);self.assertFalse((root/'analysis-v2/c16/provisional.json').exists())
