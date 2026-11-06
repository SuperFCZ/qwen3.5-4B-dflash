import hashlib,json,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import review as u


def test_resource_no_free_reserve_and_alias():
 r=u.resource_model()
 assert r['total']==r['ub_limit']==262144
 assert r['l0b']==r['l0b_limit']==65536
 assert r['views_overlap'] and r['coefficient_view']==(34816,36864)
 assert r['extra_ub_authorized']==r['extra_l0b_authorized']==0


def test_no_computation_kernel_or_opp_and_no_false_go():
 r=json.loads((HERE/'sdk-review.json').read_text())
 assert r['status']=='BLOCKED_OVERLAP_LIFECYCLE_UNPROVEN'
 assert r['npu']==r['benchmark']==r['profile']=='NOT_RUN' and not r['kernel_created']
 assert not (HERE/'op_kernel').exists() and not (HERE/'operator.json').exists()
 code=(HERE/'compile_probe/interfaces.cpp').read_text()
 assert '__global__ void' not in code and 'int main(' not in code
 assert all(x in code for x in ('Iterate<true>','Iterate<false>','GetTensorC<true>','GetTensorC<false>'))
 assert '-fsyntax-only' in (HERE/'compile_probe/check.sh').read_text()


def test_manifest_paths_hashes_and_source_identity_labels():
 r=json.loads((HERE/'sdk-review.json').read_text());seen=set()
 for row in r['sources']:
  assert row['id'] not in seen;seen.add(row['id'])
  assert len(row['sha256'])==64 and not Path(row['relative_path']).is_absolute() and '..' not in Path(row['relative_path']).parts
  if row['identity_status']=='MATCHES_PREVIOUS_SERVER_REVIEWED_SHA256':assert row['sha256']==row['server_reviewed_sha256']
  else:assert row['identity_status']=='LOCAL_ONLY_NOT_SERVER_VERIFIED'
 assert len(seen)==17


def test_sdk_binding_rejects_tampering_missing_and_escape(tmp_path,monkeypatch):
 home=tmp_path/'review';home.mkdir();root=tmp_path/'sdk';file=root/'aarch64-linux/asc/f.h';file.parent.mkdir(parents=True);file.write_bytes(b'one\ntwo\n')
 row=dict(id='fixture',relative_path='f.h',sha256=hashlib.sha256(file.read_bytes()).hexdigest(),lines=[[1,2]],identity_status='LOCAL_ONLY_NOT_SERVER_VERIFIED')
 (home/'sdk-review.json').write_text(json.dumps(dict(sources=[row])));monkeypatch.setattr(u,'HERE',home)
 assert u.verify_sdk(root)['decision']=='BLOCKED_OVERLAP_LIFECYCLE_UNPROVEN'
 file.write_bytes(b'bad\n')
 with pytest.raises(ValueError):u.verify_sdk(root)
 file.unlink()
 with pytest.raises(FileNotFoundError):u.verify_sdk(root)
 outside=tmp_path/'outside.h';outside.write_bytes(b'one\ntwo\n');file.symlink_to(outside)
 with pytest.raises(ValueError):u.verify_sdk(root)
