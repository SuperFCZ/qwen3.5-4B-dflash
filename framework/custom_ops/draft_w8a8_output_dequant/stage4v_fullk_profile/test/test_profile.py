import csv
import json
from pathlib import Path
import sys
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import profile_parse as p
import profile_analysis as a
import profile_evidence as e
import run_profile as run


def table(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return path


def record(task=1,**overrides):
 r={'Op Name':'DFlashW8A8FullKChannel','Task Type':'AI_CORE','Device ID':'0','Stream ID':'3','Task ID':str(task),
 'Task Duration(us)':'1600','scalar_time(us)':'1200','mte1_time(us)':'40','mte2_time(us)':'600','mte3_time(us)':'90',
 'vec_time(us)':'100','mac_time(us)':'20','cube_utilization(%)':'81','scalar_ratio':'0.8'}
 r.update(overrides);return r


def test_two_tasks_explicit_units_no_pipe_sums(tmp_path):
 table(tmp_path/'PROF_test/op_summary_0.csv',[record(1),record(2,**{'Task Duration(us)':'1800'})])
 r=p.collect(tmp_path)
 assert r['task_coverage']=='TWO_IDENTIFIED_AICORE_TASKS'
 assert r['availability']['memory']=='UNAVAILABLE' and r['availability']['cube']=='AVAILABLE_IDENTIFIED_ROWS'
 assert r['groups'][0]['metrics']['Task Duration(us)']['median']==1700
 assert 'scalar_ratio' not in r['groups'][0]['metrics']
 assert r['rows'][0]['metrics']['scalar_ratio']['raw']=='0.8'
 assert r['rows'][0]['metrics']['scalar_ratio']['unit']=='UNAVAILABLE'
 assert r['pipe_sum'].startswith('FORBIDDEN') and not r['benchmark_latency']
 assert all(x.startswith('UNAVAILABLE') for x in r['causal_phase_times'].values())


def test_types_streams_cores_units_and_sources_stay_separate(tmp_path):
 rows=[record(1),record(2,**{'Stream ID':'4'}),record(3,**{'Task Type':'AI_CPU'})]
 table(tmp_path/'op_summary.csv',rows)
 core=record(1,**{'Core ID':'0'});table(tmp_path/'op_summary_core.csv',[core])
 nanos=record(1);nanos['Task Duration(ns)']=nanos.pop('Task Duration(us)');table(tmp_path/'op_summary_ns.csv',[nanos])
 r=p.collect(tmp_path);assert len(r['groups'])==5
 assert next(x for x in r['groups'] if x['task_type']=='AI_CPU')['metrics']=={}
 assert any(g['row_type']=='CORE_TAGGED' for g in r['groups'])
 assert any('Task Duration(ns)' in g['metrics'] for g in r['groups'])


def test_duplicate_csv_records_not_extra_samples_or_cross_type_merging(tmp_path):
 rows=[record(1),record(1),record(2)]
 table(tmp_path/'op_summary.csv',rows);table(tmp_path/'op_summary_copy.csv',rows)
 task={k:v for k,v in record(1).items() if k!='Op Name'}
 table(tmp_path/'task_time.csv',[task])
 r=p.collect(tmp_path)
 g=next(x for x in r['groups'] if x['file']['path'].endswith('/op_summary.csv'))
 assert g['metrics']['Task Duration(us)']['count']==2 and g['duplicate_records']==[2]
 assert next(x for x in r['groups'] if 'copy.csv' in x['file']['path'])['metrics']=={}
 tr=next(x for x in r['rows'] if x['file_kind']=='task_time')
 assert tr['operator_binding']=='UNIQUE_TASK_STREAM_DEVICE_JOIN' and tr['duplicate_of'] is None
 assert r['primary_device_tasks']==2


def test_ambiguous_task_join_missing_ids_and_wrong_operator_stay_unavailable(tmp_path):
 table(tmp_path/'op_summary.csv',[record(1),record(1,**{'Task Duration(us)':'1700'}),record(2,**{'Op Name':'OtherOperator'})])
 task={k:v for k,v in record(1).items() if k!='Op Name'}
 table(tmp_path/'task_time.csv',[task])
 r=p.collect(tmp_path);assert len(r['unbound_task_time'])==1
 assert not any(x['file_kind']=='task_time' for x in r['rows'])
 assert not any('OtherOperator' in x['names'].values() for x in r['rows'])


def test_aggregate_or_unitless_duration_not_summarized(tmp_path):
 table(tmp_path/'op_summary.csv',[record(1,Count='5')])
 r=p.collect(tmp_path);assert r['groups'][0]['row_type']=='AGGREGATE' and not r['groups'][0]['metrics']
 assert not p.metric('Task Duration','1600')['statistic_eligible']
 assert not p.metric('Task Duration(%)','12')['statistic_eligible']
 assert p.metric('MTE2 bandwidth(GB/s)','12')['category']=='memory'


@pytest.mark.parametrize('bad',['N/A','-','NaN','inf','-1'])
def test_unavailable_counter_values_not_fabricated(bad):
 assert not p.metric('mte2_time(us)',bad)['statistic_eligible']


def test_timeline_display_unit_not_assumed_and_raw_identity_preserved(tmp_path):
 events=[dict(name='DFlashW8A8FullKChannel',ph='X',ts=123,dur=1600,args={'Task Type':'AI_CORE','Task ID':1,'Stream ID':3,'Device ID':0})]
 (tmp_path/'msprof_1.json').write_text(json.dumps(dict(displayTimeUnit='ns',traceEvents=events)))
 r=p.collect(tmp_path);assert len(r['timeline'])==1
 assert r['timeline'][0]['timestamp_unit'].startswith('UNAVAILABLE')
 assert r['timeline'][0]['raw_event']['dur']==1600
 assert r['timeline'][0]['file']['sha256']


def test_artifact_symlink_escape_rejected(tmp_path):
 folder=tmp_path/'prof';folder.mkdir();outside=tmp_path/'outside';outside.write_text('private')
 (folder/'op_summary.csv').symlink_to(outside)
 with pytest.raises(ValueError):p.collect(folder)


def test_task_join_does_not_cross_sessions(tmp_path):
 table(tmp_path/'PROF_one/op_summary.csv',[record()])
 table(tmp_path/'PROF_two/task_time.csv',[{k:v for k,v in record().items() if k!='Op Name'}])
 r=p.collect(tmp_path);assert len(r['unbound_task_time'])==1


HELP='''--ai-core Turn on / off.
--aic-mode task-based or sample-based.
--aic-metrics PipeUtilization, Memory.
--task-time on or off.
--task-trace on or off.
--type text or db.
--output path.
--application path.
'''

def test_actual_option_values_required_no_guessing():
 assert len(run.validate_help(HELP))==8
 for text in (HELP.replace('PipeUtilization','Other'),HELP.replace('--task-trace','--missing'),HELP.replace('task-based','unknown')):
  with pytest.raises(ValueError):run.validate_help(text)


def test_collect_requires_prepare_report():
 with pytest.raises(SystemExit):run.parse(['collect','--output-dir','/tmp/new'])
 assert run.parse(['prepare','--output-dir','/tmp/new']).phase=='prepare'


def test_source_count_scope_and_recommendations():
 kernel=e.c.BASE/'stage4v/op_kernel/d_flash_w8_a8_full_k_channel.cpp'
 proof=dict(normal_audit=dict(files=[dict(source=e.identity(kernel))]),policy=dict(selected=7))
 c=e.source_counts(proof)
 assert c['tiles_per_core']==[11]*6+[10] and sorted(x for y in c['tile_owners'] for x in y)==list(range(76))
 assert c['gm_output_calls']==19456 and c['gm_output_payload_bytes']==622592
 assert c['ub_guard_getvalue_words']==87552 and c['arena_getvalue_words']==21248
 m=dict(availability={k:'UNAVAILABLE' for k in ('duration','cube','mac','mte1','mte2','mte3','vec','scalar','memory')})
 result=a.analyze(m,c,{})
 assert result['first_implementation'].startswith('UNRESOLVED')
 assert all(x['implementation_go']=='NOT_GRANTED' for x in result['recommendations'])
 assert result['numeric_status']=='NUMERIC_MISMATCH_UNRESOLVED'


def test_profile_reuses_two_call_receiver_and_never_benchmark_flag():
 text=(HERE/'run_profile.py').read_text()
 assert "str(folder),'0','0','-1']" in text
 assert "expected_device_executions=2" in text
 assert "e.validate(row,proof,a.device,False,False)" in text
 assert "anchor=e.verify(bound['full_anchor']).read_bytes()" in text
 assert not any((HERE/x).exists() for x in ('op_host','op_kernel','operator.json','runner'))


def test_empty_or_malformed_profiler_artifacts_remain_partial(tmp_path):
 (tmp_path/'op_summary_bad.csv').write_text('Task ID,Task ID\n1,2\n')
 r=p.collect(tmp_path)
 assert r['status']=='METRICS_UNAVAILABLE' and r['skipped']
 assert r['task_coverage'].startswith('UNRESOLVED')
 assert all(v=='UNAVAILABLE' for v in r['availability'].values())


def test_profiled_output_drift_stops_before_metric_collection(tmp_path,monkeypatch):
 case=tmp_path/'case.json';case.write_text('{}');anchor=tmp_path/'anchor.bin';anchor.write_bytes(b'frozen')
 runner=tmp_path/'runner';runner.write_bytes(b'accepted executable')
 tool=tmp_path/'msprof';tool.write_bytes(b'profiler')
 proof=dict(case=e.identity(case),policy=dict(selected=7))
 bound=dict(proof=proof,build=dict(artifacts=dict(runner=e.identity(runner))),full_anchor=e.identity(anchor),full_numeric={})
 ready=dict(profiler=dict(tool=e.identity(tool)))
 monkeypatch.setattr(e,'verify_bound',lambda _:None)
 files={}
 for name in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin'):
  f=tmp_path/name;f.write_bytes(b'input');files[name]=e.identity(f)
 case.write_text(json.dumps(dict(files=files)));proof['case']=e.identity(case)
 monkeypatch.setattr(e.c.frozen.ref,'verify_case',lambda _:dict())
 monkeypatch.setattr(e.c.old_protocol,'expected',lambda *args:b'expected')
 monkeypatch.setattr(e.c,'environment',lambda *args:{})
 called=[]
 def fake_command(argv,log,env,r,key,timeout):
  called.append(argv)
  folder=tmp_path/'out/run'
  for name in ('runner.log','execution.json','resource-admission.json','actual-0.bin','actual-1.bin','actual-0.bin.numeric.json','actual-1.bin.numeric.json'):(folder/name).write_bytes(b'changed')
 monkeypatch.setattr(e.c,'command',fake_command)
 monkeypatch.setattr(e,'validate',lambda *args:dict(numeric=[{},{}]))
 monkeypatch.setattr(run,'collect',lambda *_:pytest.fail('cannot analyze failed outputs'))
 class Args:device=0;cann_root=Path('/sdk')
 out=tmp_path/'out';out.mkdir()
 with pytest.raises(ValueError,match='differs from admitted'):run.profile_run(Args(),out,{},ready,bound)
 assert len(called)==1
 cmd=called[0];assert '--aic-metrics=PipeUtilization' in cmd and '--task-time=on' in cmd
 launch=(out/'launch-profile.sh').read_text();assert '--benchmark' not in launch


def test_prior_perf_sources_inventory_excludes_new_profile_files():
 assert not any('/stage4v_fullk_profile/' in f['path'] for f in e.c.sources())


def test_different_kernel_identities_are_not_pooled(tmp_path):
 table(tmp_path/'op_summary.csv',[record(1,**{'Kernel Name':'DFlashW8A8FullKChannel_aaa'}),record(2,**{'Kernel Name':'DFlashW8A8FullKChannel_bbb'})])
 r=p.collect(tmp_path);assert len(r['groups'])==2
 assert all(g['metrics']['Task Duration(us)']['count']==1 for g in r['groups'])


def test_report_keeps_exact_source_units_and_no_implementation_go(tmp_path):
 table(tmp_path/'op_summary.csv',[record(1),record(2)])
 metrics=p.collect(tmp_path);report=a.analyze(metrics,{},dict(execute_sync=dict(median_ms=1.641665)))
 text=a.markdown(report,metrics)
 assert 'scalar_time(us)' in text and 'op_summary.csv' in text and 'SHA256' in text
 assert 'first implementation is **UNRESOLVED**' in text
 assert report['recommendations'][0]['counter_evidence']
 assert 'Scope' not in metrics['groups'][0] # no inferred phase/source scope
