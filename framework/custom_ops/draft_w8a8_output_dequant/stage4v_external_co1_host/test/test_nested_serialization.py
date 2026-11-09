"""Nested wire bytes fixtures, not TilingContext or hardware evidence."""
import copy
from pathlib import Path
import pytest
from test_formal_co1 import fixture,binary,HERE
import co1_host_common as c
import co1_serialization as s


@pytest.fixture(autouse=True)
def mock_sdk_files_only(monkeypatch):
 # The production source gate and native Docker self-check verify the real SDK.
 # Pure wire tests use symbolic SDK identities; all other file checks stay live.
 monkeypatch.setattr(s,'verify_sdk_sources',lambda entries:None)


def capture(tmp_path,row,proof):
 p=tmp_path/'callback-0-op.bin';p.write_bytes(binary(row,proof))
 payload=tmp_path/'callback-0-cube-payload.bin';payload.write_bytes(p.read_bytes()[48:])
 child=tmp_path/'callback-0-cube.bin';child.write_bytes(bytes(200))
 j=tmp_path/'callback-0.json';c.write(j,row)
 return dict(data=row,json=c.identity(j),op_serialization=c.identity(p),cube_payload=c.identity(payload),cube=c.identity(child))


def test_zero_child_and_correct_parent_are_accepted(tmp_path):
 r,p,e=fixture();record=capture(tmp_path,r,p)
 resource,binding=s.bind_capture(record,p,e)
 assert resource['status']=='CONDITIONAL_ARITHMETIC_CHECKS_PASS'
 assert binding['standalone_cube']['all_zero'] and binding['standalone_cube']['role']=='DIAGNOSTIC_ONLY'
 assert binding['cube_payload_offset']==48 and binding['cube_payload_bytes']==200
 assert binding['cube_payload_sha256']==record['cube_payload']['sha256']
 assert binding['op_serialization_sha256']==record['op_serialization']['sha256']
 assert binding['schema_sha256']==s.SCHEMA_SHA
 assert record['cube']['sha256']=='6d9c54dee5660c46886f32d80e57e9dd0ffa57ee0cd2a762b036d9c8e0c3a33a'


@pytest.mark.parametrize('index',range(50))
def test_any_nested_field_change_rejected(index):
 r,p,e=fixture();raw=bytearray(binary(r,p));raw[48+4*index]^=1
 with pytest.raises(ValueError):c.validate_callback(r,bytes(raw),p,e)


@pytest.mark.parametrize('index',range(48))
def test_every_outer_byte_including_padding_is_checked(index):
 r,p,e=fixture();raw=bytearray(binary(r,p));raw[index]^=1
 with pytest.raises(ValueError,match='outer48B'):c.validate_callback(r,bytes(raw),p,e)


@pytest.mark.parametrize('offset',[0,44,47,49,52,48.0,True])
def test_wrong_offset_rejected(offset):
 r,p,e=fixture();r['cube_payload_offset']=offset
 with pytest.raises(ValueError):c.validate_callback(r,binary(r,p),p,e)


@pytest.mark.parametrize('length',[0,200,247,249,256])
def test_parent_length_rejected(length):
 r,p,e=fixture();raw=binary(r,p)[:length].ljust(length,b'\0')
 with pytest.raises(ValueError):c.validate_callback(r,raw,p,e)


def test_schema_sha_binding_and_file_tamper(tmp_path):
 r,p,e=fixture();p['schema_identity']=dict(p['schema_identity'],sha256='0'*64)
 with pytest.raises(ValueError):c.validate_callback(r,binary(r,p),p,e)
 r,p,e=fixture();r['schema_sha256']='0'*64
 with pytest.raises(ValueError):c.validate_callback(r,binary(r,p),p,e)
 r,p,e=fixture();f=tmp_path/'schema.json';f.write_bytes((HERE/'test/fixtures/tcube-schema.json').read_bytes());item=c.identity(f)
 p['schema_identity']=item;p['serialization_binding']['schema_identity']=item;f.write_bytes(f.read_bytes()+b' ')
 with pytest.raises(ValueError):c.validate_callback(r,binary(r,p),p,e)


def test_payload_file_tamper_rejected_even_if_its_identity_is_recomputed(tmp_path):
 r,p,e=fixture();rec=capture(tmp_path,r,p);f=Path(rec['cube_payload']['path']);raw=bytearray(f.read_bytes());raw[16]^=1;f.write_bytes(raw)
 with pytest.raises(ValueError):s.bind_capture(rec,p,e)
 rec['cube_payload']=c.identity(f)
 with pytest.raises(ValueError,match='payload differs'):s.bind_capture(rec,p,e)


def test_getter_changed_and_repeated_callbacks_rejected(tmp_path):
 r,p,e=fixture();raw=binary(r,p);r['fields']['shareUbSize']=2048
 with pytest.raises(ValueError):c.validate_callback(r,raw,p,e)
 r,p,e=fixture();first=capture(tmp_path,r,p);second=copy.deepcopy(first);second['data']['fields']['shareUbSize']=2048
 with pytest.raises(ValueError,match='Getter'):s.compare_callbacks([first,second])
 second=copy.deepcopy(first);second['op_serialization']['sha256']='f'*64
 with pytest.raises(ValueError,match='serialization drift'):s.compare_callbacks([first,second])
 s.compare_callbacks([first,copy.deepcopy(first)])


def test_sdk_and_callback_source_identities_cannot_change(tmp_path):
 r,p,e=fixture();p['codec_and_schema_sdk']['files'][1]['file']=dict(p['codec_and_schema_sdk']['files'][1]['file'],sha256='b'*64)
 with pytest.raises(ValueError):c.validate_callback(r,binary(r,p),p,e)
 r,p,e=fixture();f=tmp_path/'host.h';f.write_text('bound');p['serialization_binding']['callback_sources']=[c.identity(f)];f.write_text('changed')
 with pytest.raises(ValueError):c.validate_callback(r,binary(r,p),p,e)


def test_parent_capture_is_after_real_save_and_slice_has_no_getter_repacking():
 text=(HERE/'op_host/host_common.h').read_text()
 assert text.index('td.SaveToBuffer(raw->GetData()')<text.index('Co1Write(r.stem+"-cube-payload.bin"')
 assert 'static_cast<const uint8_t*>(raw->GetData())+r.layout.offset' in text
 assert 'Co1ProveLayout<Tiling>()' in text
 assert 'DIAGNOSTIC_ONLY' in text


@pytest.mark.parametrize('field',['cube_payload_sha256','op_serialization_sha256','schema_sha256'])
def test_report_hash_field_drift_is_rejected(tmp_path,field):
 r,p,e=fixture();row=capture(tmp_path,r,p);_,record=s.bind_capture(row,p,e)
 s.verify_serialization_record(record,p);record[field]='0'*64
 with pytest.raises(ValueError):s.verify_serialization_record(record,p)
