"""Pure CPU design tests; no device, OPP, SDK emulation or conversion Kernel."""
import json,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import phase2_contract as c
import phase2_evidence as e
from audit_phase2 import decision,parse


def f(event,slot=0):return [('alloc',event,slot),('signal',event,slot),('wait_complete',event,slot),('release',event,slot)]
def trace():
 return ['arena_init_write']+f('MTE3_MTE2')+['arena_init_read']+f('MTE2_S')+['scale_load']+f('MTE2_V')+['mmad']+f('M_V')+['echo']+f('V_S')+f('V_MTE3')+['raw_gm_issue']+f('MTE3_S')+['migrate_witness']+f('S_V')+['vdeq']+f('V_S')+f('V_M')+['end','arena_post']+f('MTE2_S')+f('V_MTE3')+['fp16_gm_issue']+f('MTE3_S')+['finalize_witness']+f('S_MTE3')+['publish','release_co1']


def test_geometry_every_element():
 rows=c.geometry();assert c.validate_geometry(rows)['fp16_bytes']==8192
 seen=set();raw_seen=set();scale_seen=[]
 for r in rows:
  for k in ('l0c','raw_echo','fp16','scale'):assert r[k][0]%32==0 and r[k][1]%32==0
  scale_seen.extend(range(*r['channels']))
  for m in range(16):
   for n in range(16):
    j=r['j'];nz=j*256+m*16+n;nd=m*256+j*16+n
    assert 2*nz+222464>=r['fp16'][0] and 2*nz+222466<=r['fp16'][1]
    assert 4*nz>=r['l0c'][0] and 4*nz+4<=r['l0c'][1]
    seen.add(nd);raw_seen.add(nz)
 assert seen==raw_seen==set(range(4096));assert scale_seen==list(range(256))


@pytest.mark.parametrize('key,value',[('blockLen',0),('blockLen',.5),('blockLen',512),('blockLen',True),('blockCount',16),('srcStride',1),('dstStride',1)])
def test_wrong_copy_units_rejected(key,value):
 rows=c.geometry();rows[0][key]=value
 with pytest.raises(ValueError):c.validate_geometry(rows)


def test_scale_bytes_are_never_reencoded(tmp_path):
 raw=b''.join((i|(1<<46)).to_bytes(8,'little') for i in range(256));before=bytes(raw)
 parts=c.scale_offsets(raw);assert b''.join(parts)==before and all(len(x)==128 for x in parts)
 p=tmp_path/'scale.bin';p.write_bytes(raw);identity=e.identity(p);p.write_bytes(raw[:-1]+b'!')
 with pytest.raises(ValueError):e.verify(identity)
 with pytest.raises(ValueError):c.scale_offsets(raw[:-8])


def test_layout_guards_aliases_and_reserve():
 r=c.layout();assert r['total_budget']==239104 and r['explicit_ub_bytes']==230912
 assert sum(b['bytes']+512 for b in r['buffers'].values())==230912
 previous=0
 for b in r['buffers'].values():assert b['raw']==previous and b['payload']==b['raw']+256 and b['end']==b['payload']+b['bytes']+256;previous=b['end']
 for lo,hi in r['aliases'].values():
  assert hi-lo in (256,512) and lo%32==0
  assert not any(lo<g1 and g0<hi for g0,g1 in r['guards'])
  assert hi<=230656
 assert r['reserve_physical_top']==[253952,262144]


def test_all_512_guard_words_single_bits():
 expected=[0x5a39c671]*512
 def delta(words):
  out=0
  for value in words:out|=value^0x5a39c671
  return out
 assert delta(expected)==0
 for i in range(512):
  for bit in range(32):
   expected[i]^=1<<bit;assert delta(expected)!=0;expected[i]^=1<<bit
 assert delta(expected)==0


def test_nominal_abstract_lifetime():assert c.lifetime(trace())=='CPU_DESIGN_CONTRACT_ONLY_NOT_HARDWARE_EVIDENCE'


@pytest.mark.parametrize('removed',['M_V','MTE2_V','MTE3_MTE2','MTE3_S','S_V','V_S','V_M','V_MTE3','MTE2_S','S_MTE3'])
def test_missing_completion_edge_rejected(removed):
 actions=[x for x in trace() if not isinstance(x,tuple) or x[1]!=removed]
 with pytest.raises(ValueError):c.lifetime(actions)


@pytest.mark.parametrize('action,before',[('migrate_witness','raw_gm_issue'),('vdeq','migrate_witness'),('end','vdeq'),('publish','finalize_witness'),('release_co1','vdeq')])
def test_premature_reuse_rejected(action,before):
 actions=trace();actions.remove(action);actions.insert(actions.index(before),action)
 with pytest.raises(ValueError):c.lifetime(actions)


def test_event_ids_and_pending_release():
 for slot in (-1,8,True):
  with pytest.raises(ValueError):c.lifetime(f('M_V',slot))
 with pytest.raises(ValueError):c.lifetime([('alloc','M_V',0),('signal','M_V',0),('release','M_V',0)])
 with pytest.raises(ValueError):c.lifetime([('alloc','S_V',0),('alloc','S_V',0)])


def test_new_vdeq_write_invalidates_old_completion():
 actions=trace();start=actions.index('vdeq');actions=[x for i,x in enumerate(actions) if not(i>start and isinstance(x,tuple) and x[1]=='V_S')]
 with pytest.raises(ValueError):c.lifetime(actions)


def test_abi_and_error_statuses():
 a=json.loads((HERE/'abi-design.json').read_text());assert a['status']=='DESIGN_ONLY_NOT_REGISTERED'
 assert [x['bytes'] for x in a['outputs']]==[16384,8192,256]
 assert a['historical_fp16_row_stride']==19456 and a['compact_row_stride']==256
 assert a['witness']['event_ids']==[8,18] and len(a['witness']['events'])==10
 for raw,repeat,safe in [(False,True,True),(True,False,True),(True,True,False)]:assert c.observation_status(raw,repeat,safe,True,True)['status']=='BLOCKED'
 assert c.observation_status(True,True,True,False,False)['relation']=='MATCHES_NEITHER'
 with pytest.raises(ValueError):c.observation_status('PASS',True,True,True,True)
 assert decision(True)['status']=='BLOCKED' and decision(True)['kernel_implementation']=='NOT_AUTHORIZED'


def test_no_execution_assets():
 assert not any((HERE/n).exists() for n in ('op_kernel','op_host','runner','operator.json','build.sh'))
 with pytest.raises(SystemExit):parse(['--output-dir','/tmp/new','--echo-root','/tmp/partial'])


def test_historical_full_stride_is_not_flat_prefix():
 # Index proof without constructing or changing any frozen output.
 correct=[m*19456+n for m in range(16) for n in range(256)]
 assert len(correct)==4096 and correct!=list(range(4096)) and correct[256]==19456
