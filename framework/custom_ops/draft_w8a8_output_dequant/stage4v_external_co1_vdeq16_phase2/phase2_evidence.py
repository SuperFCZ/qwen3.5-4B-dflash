"""Read-only verification of the actual historical experiment; no Runner invoked."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4v_external_co1_vdeq16_exec'))
import echo_common as echo
import run_echo as run
from echo_execution_contract import validate_execution
from echo_serialization import compare_callbacks,verify_serialization_record
identity,verify,read,write,fresh=echo.identity,echo.verify,echo.read,echo.write,echo.fresh


def sources():return [identity(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]


def bind(a):
 proof,raw=echo.bind(a)
 e=read(a.echo_root/'echo/report.json');expected=dict(sources=echo.sources(),evidence=proof,device=a.device)
 run.prior(a.echo_root/'echo/report.json','ECHO_NPU_MATCH',expected)
 if e['npu']!='COMPLETED' or e.get('error') or e.get('integrity_error'):raise ValueError('Echo did not complete cleanly')
 stages={}
 for stage,status in run.STATUS.items():
  if stage=='echo':continue
  path=a.echo_root/stage/'report.json';stages[stage]=run.prior(path,status,expected)
  if stages[stage].get('error') or stages[stage].get('integrity_error'):raise ValueError('predecessor carries failure')
  if stage+'_report' in e and e[stage+'_report']!=identity(path):raise ValueError('Echo predecessor identity differs')
 if stages['cpu']['source_report']!=identity(a.echo_root/'source/report.json'):raise ValueError('CPU/source link differs')
 for stage in ('build','host'):
  if any(stages[stage][k]!=e[k] for k in ('source_report','cpu_report')):raise ValueError('stage gate chain mismatch')
 if stages['host']['build_report']!=e['build_report']:raise ValueError('Host/build link differs')
 if any(stages[s]['artifacts']!=e['artifacts'] for s in ('build','host')):raise ValueError('Echo private artifact identity differs')
 if stages['cpu']['commands']['cpu']['returncode']!=0 or stages['host']['commands']['query']['returncode']!=0 or e['commands']['echo']['returncode']!=0:raise ValueError('stage command failed')
 if any(x['returncode']!=0 for x in e['artifacts']['commands'].values()):raise ValueError('private build command failed')
 run.artifacts(e['artifacts'],proof);run.verify_host(stages['host'])
 files={Path(x['path']).name:verify(x) for x in e['files']}
 ex=read(files['execution.json']);validate_execution(ex,a.device,proof['accepted_workspace'],False)
 if identity(files['execution.json'])!=identity(a.echo_root/'echo/run/execution.json'):raise ValueError('unexpected receiver path')
 compare_callbacks(e['callbacks'])
 for row in e['callbacks']:
  if read(verify(row['json']))!=row['data']:raise ValueError('callback JSON changed')
  echo.validate_callback(row['data'],verify(row['op_serialization']).read_bytes(),proof,ex)
  verify_serialization_record(row['serialization'],proof)
 if len(e['outputs'])!=2 or len(e['witnesses'])!=2:raise ValueError('two complete repeats required')
 for name in ('a.bin','b.bin','scale.bin','expected.bin'):
  if identity(a.echo_root/'echo/run'/name)!=identity(files[name]):raise ValueError('run input path identity differs')
  if identity(files[name])!=e['data'][name] and verify(e['data'][name]).read_bytes()!=files[name].read_bytes():raise ValueError('run input differs')
 original=raw['original'];expected_bytes=verify(raw['raw']['outputs'][0]).read_bytes()
 if verify(e['data']['expected.bin']).read_bytes()!=expected_bytes:raise ValueError('Echo Reference differs from historical Raw NPU')
 if verify(e['data']['a.bin']).read_bytes()!=verify(raw['raw']['data']['a_i8.bin']).read_bytes() or verify(e['data']['b.bin']).read_bytes()!=verify(raw['raw']['data']['b_i8.bin']).read_bytes():raise ValueError('Echo A/B differ from Frozen')
 if verify(e['data']['scale.bin']).read_bytes()!=original['words'].astype('<u8').tobytes():raise ValueError('frozen Scale words changed')
 outputs=[]
 for i in range(2):
  path=a.echo_root/f'echo/run/actual-{i}.bin'
  if identity(path)!=e['outputs'][i] or identity(path)!=identity(files[path.name]):raise ValueError('Echo output identity differs')
  outputs.append(path.read_bytes())
  path=a.echo_root/f'echo/run/witness-{i}.bin';item=e['witnesses'][i]
  if identity(path)!=item['file'] or identity(path)!=identity(files[path.name]):raise ValueError('Witness identity differs')
  result=run.witness(path.read_bytes())
  if result!=item['analysis'] or result['status']!='VALID':raise ValueError('invalid real Witness')
 comparison=echo.compare_echo(expected_bytes,outputs)
 comparison_file=a.echo_root/'echo/echo-comparison.json'
 if comparison!=e['comparison'] or comparison!=read(comparison_file) or comparison['status']!='MATCH':raise ValueError('Echo comparison differs')
 return dict(status='HISTORICAL_ECHO_EVIDENCE_REVERIFIED',reports={s:identity(a.echo_root/s/'report.json') for s in run.STATUS},comparison=identity(comparison_file),outputs=e['outputs'],witnesses=[x['file'] for x in e['witnesses']],inputs=e['data'],
  original=raw['identities'],co1_formal=proof['accepted_host'],accepted=proof['accepted'],workspace=proof['accepted_workspace'],
  l0c_guard='RANGE_ONLY_NO_CANARIES',phase2_npu='NOT_RUN',cause='UNRESOLVED'),e
