"""Conditional CPU contracts only: no simulated hardware completion is evidence."""
import json
import sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import phase21_contract as c
from audit_phase2 import decision


def test_fetch_reads_without_reserving_and_alloc_excludes_held_ids():
 p=c.Pools();before=p.snapshot('S_V')
 assert p.fetch('S_V')==p.fetch('S_V')==0 and before==p.snapshot('S_V')
 slot=p.alloc('S_V');assert slot==0 and p.fetch('S_V')==1
 p.release('S_V',slot);assert p.fetch('S_V')==0


def test_same_type_collision_and_early_release_are_rejected():
 p=c.Pools();slot=p.alloc('S_V');p.use('S_V',slot);p.issue_wait('S_V',slot)
 for fn in (lambda:p.use('S_V',slot),lambda:p.release('S_V',slot),lambda:p.use('S_V',slot,'sdk')):
  with pytest.raises(ValueError):fn()
 assert p.snapshot('S_V')['wait_completed']==[]
 p.observe_completed_wait('S_V',slot);p.release('S_V',slot)


def test_sdk_fetch_use_is_unreserved_and_can_conflict_with_next_alloc():
 p=c.Pools();slot=p.fetch('S_V');p.use('S_V',slot,'sdk')
 assert slot==p.fetch('S_V') and p.snapshot('S_V')['allocated']==[]
 with pytest.raises(ValueError):p.alloc('S_V')
 with pytest.raises(ValueError):p.use('S_V',p.fetch('S_V'),'sdk')
 with pytest.raises(ValueError):p.release('S_V',slot)
 p.issue_wait('S_V',slot)
 with pytest.raises(ValueError):p.alloc('S_V')
 p.observe_completed_wait('S_V',slot);assert p.alloc('S_V')==slot


def test_different_hard_events_have_independent_software_pools():
 p=c.Pools();a=p.alloc('S_V');b=p.alloc('V_S');assert a==b
 p.use('S_V',a);p.use('V_S',b)
 p.issue_wait('V_S',b);p.observe_completed_wait('V_S',b);p.release('V_S',b)
 assert p.snapshot('S_V')['in_use']==[a] and p.fetch('V_S')==0


def test_exhaustion_blocks_before_raw_sdk_alloc_no_invalid_mutation():
 p=c.Pools()
 assert [p.alloc('S_V') for _ in range(8)]==list(range(8))
 before=p.snapshot('S_V');assert p.fetch('S_V')>=8
 with pytest.raises(ValueError,match='preflight exhausted'):p.alloc('S_V')
 assert before==p.snapshot('S_V') and p.fetch('M_V')==0


@pytest.mark.parametrize('plan',['A','B'])
def test_all_sixteen_sdk_intervals_require_completed_waits(plan):
 rows=c.sv_trace(plan)
 issued=[r for r in rows if 'SDK Set/Wait' in r['time']]
 completed=[r for r in rows if 'V_S join confirms' in r['time']]
 assert len(issued)==len(completed)==16
 assert all(r['in_use']==[0] and r['allocated']==[] and r['sdk_fetch_candidate']==0 for r in issued)
 assert all(r['in_use']==[] and r['wait_completed']==[0] for r in completed)
 if plan=='A':
  i=next(i for i,r in enumerate(rows) if 'manual release' in r['time'])
  assert 'witnessed completion' in rows[i-1]['time']
 else:assert not any('manual' in r['time'] for r in rows)


def test_every_manual_event_ledger_has_separate_ownership_and_release():
 report=c.manual_pool_trace();rows=report['rows']
 for event in c.MANUAL_B:
  one=[r for r in rows if r['hard_event']==event]
  assert len(one)==4 and one[0]['allocated']==[0]
  assert one[1]['in_use']==[0] and one[1]['releasable']==[]
  assert one[2]['wait_completed']==[0] and one[2]['releasable']==[0]
  assert one[3]['allocated']==[] and one[3]['in_use']==[]
 assert 'S_V' not in c.MANUAL_B


@pytest.mark.parametrize('op,event',[('mmad','M_V'),('vector','V_S'),('load','MTE2_S'),('store','MTE3_S')])
def test_no_late_cleanup_id_allocation(op,event):
 d=c.Drain(tickets=set(c.MANUAL_B)-{event})
 with pytest.raises(ValueError,match='BEFORE issue'):d.issue(op)
 assert d.pending==set()


def test_mmad_wait_requires_scalar_join_and_sdk_cleanup_proof():
 d=c.Drain();d.issue('mmad');d.join('M_V')
 with pytest.raises(ValueError):d.end()
 d.join('V_S');assert d.pending==set()
 with pytest.raises(ValueError):d.end() # app-pipe completion is not SDK ownership proof
 d.prove_sdk_cleanup();d.end()
 d.publish()
 with pytest.raises(ValueError):d.release_buffers() # Witness DMA is itself pending
 d.join('MTE3_S');d.release_buffers()
 assert not d.success


def test_no_stale_m_completion_can_drain_new_mmad():
 d=c.Drain();d.issue('mmad');d.join('M_V');d.issue('mmad')
 with pytest.raises(ValueError):d.join('V_S')
 assert d.pending=={'mmad'}
 d.join('M_V');d.join('V_S');assert not d.pending


@pytest.mark.parametrize('event',['M_V','V_S','MTE2_S','MTE3_S'])
def test_no_wait_for_unissued_producer(event):
 with pytest.raises(ValueError):c.Drain().join(event)


@pytest.mark.parametrize('stage,ops,joins',[
 ('A',[],[]),('A_init',['vector','load','store'],['V_S','MTE2_S','MTE3_S']),
 ('B',['mmad','load'],['M_V','V_S','MTE2_S']),
 ('C',['store'],['MTE3_S']),('D',[],[]),
 ('E',['vector'],['V_S']),('F',['vector'],['V_S']),
 ('G',['store'],['MTE3_S']),('H',['store'],['MTE3_S'])])
def test_ordinary_error_A_through_H_requires_drains_and_separate_sdk_proof(stage,ops,joins):
 d=c.Drain()
 for op in ops:d.issue(op)
 d.barrier_all_or_return()
 assert d.pending==set(ops)
 with pytest.raises(ValueError):d.end()
 with pytest.raises(ValueError):d.release_buffers()
 for event in joins:d.join(event)
 assert not d.pending
 with pytest.raises(ValueError):d.end()
 # This explicit token is the theorem STILL MISSING in SDK review, not generated evidence.
 d.prove_sdk_cleanup();d.end()
 with pytest.raises(ValueError):d.publish(success=True)
 d.publish();assert d.pending=={'store'}
 with pytest.raises(ValueError):d.release_buffers()
 d.join('MTE3_S');d.release_buffers()
 with pytest.raises(ValueError):d.publish()


@pytest.mark.parametrize('op',['mmad','vector','load','store'])
def test_hardware_fault_or_timeout_cannot_manufacture_completion(op):
 d=c.Drain();d.issue(op);d.fault()
 d.barrier_all_or_return();assert d.pending=={op}
 for fn in (lambda:d.join('M_V'),d.prove_sdk_cleanup,d.end,d.release_buffers,d.publish):
  with pytest.raises(ValueError):fn()
 assert not d.success


def test_b_memory_is_disjoint_no_reserve_or_numerical_alias():
 old=c.a.layout();r=c.layout_b()
 assert r['total_budget']==240384 and r['explicit_ub_bytes']==232192
 assert r['additional_allocation']['total']==1280
 assert r['total_budget']==r['explicit_ub_bytes']+8192
 assert r['unallocated_margin']==262144-r['total_budget']==21760
 assert r['reserve_physical_top']==[253952,262144]
 assert r['aliases']=={} and old['explicit_ub_bytes']==230912
 end=0
 for name,b in r['buffers'].items():
  assert b['raw']==end and b['payload']==b['raw']+256
  assert b['end']==b['payload']+b['bytes']+256 and b['end']<=253952
  assert all(b[k]%32==0 for k in ('raw','payload','end'))
  end=b['end']
  if name!='diagnostic':assert b==old['buffers'][name]
 assert end==232192 and len(r['guards'])==10
 assert sum(hi-lo for lo,hi in r['guards'])==2560
 assert r['diagnostic_views']=={'arena':[231168,231680],'witness':[231680,231936]}
 assert r['l0c']==old['l0c'] and r['workspace_reference_bytes']==4195328


def test_b_all_640_guard_words_all_bits_mathematical_only():
 expected=0x5a39c671;words=[expected]*640
 def delta():
  result=0
  for w in words:result|=w^expected
  return result
 assert delta()==0
 for pos in range(640):
  for bit in range(32):
   words[pos]^=1<<bit;assert delta()==1<<bit;words[pos]^=1<<bit
 assert delta()==0


def test_b_chosen_but_no_kernel_admission_or_hardware_claim():
 r=decision(True);assert r['status']=='BLOCKED' and r['revision']=='PHASE2_1'
 assert r['selected_design']=='B' and len(r['blockers'])==3
 assert r['kernel_implementation']=='NOT_AUTHORIZED' and r['npu']=='NOT_RUN'
 a=json.loads((HERE/'abi-design.json').read_text())
 assert a['selected_design']=='B' and a['ub_accounted_bytes']==240384
 assert a['witness']['sdk_sv_evidence']=='PRECHECKED_SOURCE_INFERRED_NOT_DIRECTLY_INSTRUMENTED'


def test_focused_manifest_cannot_silently_substitute_sdk_identity():
 baseline=json.loads((HERE/'sdk-review.json').read_text())
 focused=json.loads((HERE/'phase21-sdk-review.json').read_text())
 index={s['relative_path']:s for s in baseline['sources']}
 assert len(focused['sources'])==4
 for s in focused['sources']:
  assert all(s[k]==index[s['relative_path']][k] for k in ('id','sha256','bytes'))
  assert all(1<=lo<=hi for lo,hi in s['lines'])


def test_sdk_can_fetch_other_id_while_manual_sv_still_reserved():
 p=c.Pools();app=p.alloc('S_V');p.use('S_V',app);p.issue_wait('S_V',app)
 sdk=p.fetch('S_V');assert sdk!=app
 p.use('S_V',sdk,'sdk');p.issue_wait('S_V',sdk)
 assert p.snapshot('S_V')['in_use']==[app,sdk]
 # No collision does NOT prove either wait completed or grant early app release.
 with pytest.raises(ValueError):p.release('S_V',app)
 p.observe_completed_wait('S_V',sdk)
 assert p.snapshot('S_V')['in_use']==[app]


@pytest.mark.parametrize('operation,event',[('raw_gm_issue','MTE3_S'),('migrate_witness','S_V')])
def test_a_pending_raw_dma_or_scalar_handoff_forbids_next_consumer(operation,event):
 from test_contract import trace
 actions=trace();start=actions.index(operation)
 # Remove the completion observation, while retaining the issued signal.
 i=next(i for i,x in enumerate(actions) if i>start and x==('wait_complete',event,0))
 del actions[i]
 with pytest.raises(ValueError):c.a.lifetime(actions)
