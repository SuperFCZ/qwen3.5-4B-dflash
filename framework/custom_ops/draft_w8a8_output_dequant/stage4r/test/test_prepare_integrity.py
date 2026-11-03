"""Prepare finalization with real file hashes, mocked heavy data/device admission.

The fixture files are small structural stand-ins, not FP16 numerical evidence.
The actual machine() wrapper and final verify/read/verify_tree are exercised.
"""
import json
import sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import run_stage4r as run

@pytest.mark.parametrize('changed',[None,'version','source','manifest','reference','nk','kn','preflight','tool','log'])
def test_prepare_finally_preserves_all_nested_identities(tmp_path,monkeypatch,changed):
 sdk=tmp_path/'sdk';sdk.mkdir();version=sdk/'version.info';version.write_text('Version=9.0.0\n')
 preflight=tmp_path/'preflight.json';preflight.write_text('{}\n')
 source=tmp_path/'source.cpp';source.write_text('CPU evidence fixture only\n')
 tool=tmp_path/'npu-smi';tool.write_text('not executable; never run\n')
 root=tmp_path/'new';log=root/'npu-smi.log'
 original_identity=run.q.identity
 source_id=original_identity(source);pf_id=original_identity(preflight)
 proof=dict(variants={'N256':dict(preflight=pf_id,opp=str(tmp_path/'unused-opp'),policy=dict(aic=8,aiv=8,all=7,selected=7,arch=2002))})
 monkeypatch.setattr(run,'sources',lambda:[source_id])
 monkeypatch.setattr(run.q,'evidence',lambda _:proof)
 # No historical OPPs or NPU available in this structural integration test.
 monkeypatch.setattr(run.q,'immutable',lambda value:run.q.verify_tree(value))
 monkeypatch.setattr(run.q.rc.shutil,'which',lambda name:str(tool) if name=='npu-smi' else None)
 def fake_smi(argv,path,*args):
  assert argv==[str(tool),'info'];path.write_text('CPU fixture; no device query\n')
 monkeypatch.setattr(run.q.rc,'command',fake_smi)
 generated={}
 def fixture_prepare(folder,evidence):
  assert evidence==proof;folder.mkdir();(folder/'weights').mkdir()
  for key,path in {'reference':folder/'reference_f64.bin','nk':folder/'weights/w_fp16_nk.bin','kn':folder/'weights/w_fp16_kn.bin','x':folder/'x.bin'}.items():
   path.write_bytes(('fixture:'+key).encode());generated[key]=path
  ids={key:original_identity(path) for key,path in generated.items()}
  return dict(status='STAGE4R_CPU_REFERENCE_PREPARED',policy=run.POLICY,
   sources=[source_id],weights=dict(files={'w_fp16_nk.bin':ids['nk'],'w_fp16_kn.bin':ids['kn']}),
   files={'x.bin':ids['x'],'w_fp16_kn.bin':ids['kn'],'reference_f64.bin':ids['reference']})
 monkeypatch.setattr(run,'prepare',fixture_prepare)
 snapshots={};triggered=[]
 def capture_then_tamper(path):
  item=original_identity(path)
  if Path(path)==root/'data/reference.json':
   snapshots['manifest']=item
   snapshots['data']=json.loads(Path(path).read_text())
   if changed:
    targets=dict(version=version,source=source,manifest=Path(path),preflight=preflight,tool=tool,log=log,**generated)
    target=targets[changed];target.write_bytes(target.read_bytes()+b'changed')
    triggered.append(changed)
  return item
 monkeypatch.setattr(run.q,'identity',capture_then_tamper)
 # Any accidental advance to compilation/device submission must fail this test.
 monkeypatch.setattr(run,'build_native',lambda *_:pytest.fail('Prepare must not compile'))
 monkeypatch.setattr(run,'invoke_native',lambda *_:pytest.fail('Prepare must not launch'))
 args=run.parse(['prepare','--output-dir',str(root),'--cann-root',str(sdk),'--candidate','N256'])
 rc=run.main(args)
 result=json.loads((root/'prepare-report.json').read_text())
 assert result['phase']=='complete' and 'error' not in result
 assert result['npu']==result['benchmark']==result['profile']=='NOT_RUN'
 assert result['reference']==snapshots['manifest']
 assert result['sources']==[source_id]
 assert result['machine']['cann_identity']==pf_id
 assert result['machine']['npu_smi_tool']['path']==str(tool)
 assert result['machine']['npu_smi']['path']==str(log)
 wrapped=next(v for v in result['machine']['version_files'] if v['file']['path']==str(version))
 assert wrapped['text']=='Version=9.0.0\n' and set(wrapped)=={'file','text'}
 if changed:
  assert rc==1 and triggered==[changed] and result['status']=='BLOCKED'
  assert 'identity changed:' in result['integrity_error']
  assert json.loads((root/'comparison.json').read_text())['status']=='NOT_RUN'
 else:
  assert rc==0 and result['status']=='STAGE4R_CPU_PREPARED'
  assert 'integrity_error' not in result
  data=run.q.read(run.q.verify(result['reference']))
  assert data==snapshots['data']
  for key,record in [('reference',data['files']['reference_f64.bin']),('nk',data['weights']['files']['w_fp16_nk.bin']),('kn',data['weights']['files']['w_fp16_kn.bin'])]:
   assert record==original_identity(generated[key])
   run.q.verify(record)
