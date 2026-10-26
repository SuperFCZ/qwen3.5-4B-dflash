"""Generator regression with synthetic SDK data; never hardware evidence."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import w8_sdk
import w8_common
TOOL=ROOT/'tools/generate_raw_review.py'


@pytest.fixture
def setup(tmp_path,monkeypatch):
 spec=importlib.util.spec_from_file_location('raw_review_writer_test',TOOL)
 g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
 assert g.EXPECTED_PREFLIGHT=='312b1a0962ac6dce5a848243031a32f1898351b043faf60ba91aaabf54eab386'
 root=tmp_path/'run';(root/'preflight').mkdir(parents=True)
 sdk=tmp_path/'sdk';src=sdk/g.ASC/'fixture.h';src.parent.mkdir(parents=True)
 src.write_text('  fixture line one\nfixture line two\n',encoding='utf-8')
 snap=root/'preflight/source.snapshot';snap.write_bytes(src.read_bytes())
 # Model the source collection before adding tools/ and this new regression file.
 # Existing candidate and validator files are real unchanged repository files.
 ids=[item for item in w8_common.sources() if 'tools' not in Path(item['path']).relative_to(ROOT).parts and Path(item['path'])!=Path(__file__).resolve()]
 p=dict(status='UNRESOLVED',raw_only=True,soc='Ascend310P3',collection='FIXED_FILES_COLLECTED',
   cann_root=str(sdk),sources=ids,files=[dict(source=g.identity(src),snapshot=g.identity(snap))],tools={})
 pf=root/'preflight/preflight.json';template=root/'preflight/review-template.json'
 r=dict(preflight={},status='UNRESOLVED',raw_only=True,reviewer='',
   common={name:dict(status='UNRESOLVED',finding='CPU fixture',citations=[]) for name in w8_sdk.RAW_CLAIMS},settings=w8_sdk.RAW_SETTINGS.copy())
 def sync():
  pf.write_text(json.dumps(p),encoding='utf-8');r['preflight']=g.identity(pf);template.write_text(json.dumps(r),encoding='utf-8')
  monkeypatch.setattr(g,'EXPECTED_PREFLIGHT',g.identity(pf)['sha256'])
 sync()
 monkeypatch.setattr(g,'SPECS',{'fixture':('fixture.h',g.identity(src)['sha256'])})
 monkeypatch.setattr(g,'CITATIONS',{name:[('fixture',1,2)] for name in w8_sdk.RAW_CLAIMS})
 return g,root,p,r,sync,src,snap


def test_new_tool_does_not_invalidate_existing_collection_and_calls_formal_gate(setup,monkeypatch):
 g,root,p,r,_,src,snap=setup
 assert g.DEFAULT_REPO==ROOT.parents[2]
 assert str(TOOL) not in {item['path'] for item in p['sources']}
 original={f:f.read_bytes() for f in (root/'preflight/preflight.json',root/'preflight/review-template.json',src,snap)}
 calls=[];formal=w8_sdk.reviewed
 def checked(*args,**kwargs):
  calls.append((args,kwargs));return formal(*args,**kwargs)
 monkeypatch.setattr(w8_sdk,'reviewed',checked)
 out=g.generate(root,g.DEFAULT_REPO)
 assert len(calls)==1 and calls[0][1]=={'raw_only':True}
 data=json.loads(out.read_text())
 assert data['status']=='SOURCE_REVIEWED' and data['preflight']==g.identity(root/'preflight/preflight.json')
 assert data['generator']==g.identity(TOOL)
 assert data['settings']==w8_sdk.RAW_SETTINGS
 for claim in data['common'].values():
  c=claim['citations'][0]
  assert c==dict(path=str(src),sha256=g.identity(src)['sha256'],start=1,end=2,quote='  fixture line one\nfixture line two')
 for key in ('host_tiling_runtime','installation','npu','guards','readonly','repeat','performance'):
  assert data[key]=='NOT_RUN'
 assert out.stat().st_mode&0o777==0o600
 assert all(path.read_bytes()==value for path,value in original.items())
 assert not list(root.glob('.raw-review-check-*'))


@pytest.mark.parametrize('fault',['missing_reference','mismatched_hash','snapshot_tamper','live_tamper','range',
 'template_binding','template_settings','missing_validator','wrong_candidate','wrong_prefight_hash'])
def test_missing_or_mismatched_evidence_never_publishes_review(setup,monkeypatch,fault):
 g,root,p,r,sync,src,snap=setup
 if fault=='missing_reference':p['files']=[];sync()
 elif fault=='mismatched_hash':monkeypatch.setattr(g,'SPECS',{'fixture':('fixture.h','0'*64)})
 elif fault=='snapshot_tamper':snap.write_text('changed')
 elif fault=='live_tamper':src.write_text('changed')
 elif fault=='range':g.CITATIONS['integer_cube']=[('fixture',1,999)]
 elif fault=='template_binding':
  r['preflight']['sha256']='0'*64;(root/'preflight/review-template.json').write_text(json.dumps(r))
 elif fault=='template_settings':r['settings']={**r['settings'],'input_count':3};sync()
 elif fault=='missing_validator':p['sources']=[x for x in p['sources'] if not x['path'].endswith('/w8_sdk.py')];sync()
 elif fault=='wrong_candidate':monkeypatch.setattr(g,'RAW',{**g.RAW,'raw/host_common.h':'0'*64})
 else:monkeypatch.setattr(g,'EXPECTED_PREFLIGHT','0'*64)
 with pytest.raises((ValueError,FileNotFoundError)):
  g.generate(root,g.DEFAULT_REPO)
 assert not (root/'review-raw.json').exists()
 assert not list(root.glob('.raw-review-check-*'))


def test_formal_quote_rejection_never_publishes_review(setup,monkeypatch):
 g,root,*_=setup;formal=w8_sdk.reviewed
 def reject_quote(pf,path,*args,**kwargs):
  data=json.loads(path.read_text());data['common']['integer_cube']['citations'][0]['quote']='not the actual source'
  path.write_text(json.dumps(data));return formal(pf,path,*args,**kwargs)
 monkeypatch.setattr(w8_sdk,'reviewed',reject_quote)
 with pytest.raises(ValueError,match='quote differs'):g.generate(root,g.DEFAULT_REPO)
 assert not (root/'review-raw.json').exists() and not list(root.glob('.raw-review-check-*'))


def test_existing_review_is_never_overwritten(setup):
 g,root,*_=setup;p=root/'review-raw.json';p.write_bytes(b'existing human review')
 with pytest.raises(FileExistsError):g.generate(root,g.DEFAULT_REPO)
 assert p.read_bytes()==b'existing human review'


def test_cli_from_other_cwd_reports_unresolved_without_build(setup,tmp_path):
 _,root,*_=setup
 # Production SHA stays fixed in the actual CLI, so this synthetic preflight must fail.
 result=subprocess.run([sys.executable,'-B',str(TOOL),'--root',str(root)],cwd=tmp_path,capture_output=True,text=True)
 assert result.returncode==1 and 'wrong current Preflight SHA256' in result.stderr
 files=list(root.glob('review-unresolved-*.json'));assert len(files)==1
 report=json.loads(files[0].read_text());assert report['status']=='UNRESOLVED' and report['build_invoked'] is False
 assert not (root/'review-raw.json').exists() and not (root/'build').exists()
