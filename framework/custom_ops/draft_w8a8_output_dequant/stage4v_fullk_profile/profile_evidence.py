"""Read-only binding of all six admitted exploration phases."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4v_fullk_perf_explore'))
import explore_common as c
import run_explore as old
from explore_protocol import validate,summary,OUTPUTS
__all__=["validate","OUTPUTS"]
identity,verify,read,write,fresh,verify_tree=c.identity,c.verify,c.read,c.write,c.fresh,c.verify_tree


def sources():return [identity(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]


def bind(root,cann,device,machine_root,report):
 root=root.resolve(strict=True);path=root/'benchmark/benchmark-report.json';bench=read(path);verify_tree(bench)
 if bench!=read(root/'benchmark/report.json'):raise ValueError('benchmark report copies differ')
 if bench.get('npu')!='COMPLETED' or bench.get('benchmark')!='COMPLETED' or bench.get('numeric_status')!='NUMERIC_MISMATCH_UNRESOLVED':raise ValueError('benchmark state differs')
 original=Path(verify(bench['evidence']['identities']['bounded'])).parent.parent
 proof=c.bind(original,device)
 current=dict(sources=c.sources(),evidence=proof,device=device,cann_root=str(cann))
 current['machine']=c.machine(machine_root,proof,cann,report)
 records={};states={}
 for phase,key in [('source','source_report'),('build','build_report'),('host','host_report'),('tile0','tile_report'),('full','full_report')]:
  fp=verify(bench[key])
  if root not in fp.parents:raise ValueError('history predecessor outside selected experiment')
  row=old.prior(fp,phase,current);records[phase]=bench[key];states[phase]=row
 for phase in ('build','host','tile0','full'):
  if states[phase]['source_report']!=records['source']:raise ValueError('Source chain differs')
 for phase in ('host','tile0','full'):
  if states[phase]['build_report']!=records['build']:raise ValueError('Runner Build chain differs')
 for phase in ('tile0','full'):
  if states[phase]['host_report']!=records['host']:raise ValueError('Host chain differs')
 if states['full']['tile_report']!=records['tile0']:raise ValueError('Tile0 chain differs')
 old.prior(path,'benchmark',current)
 if summary(bench['runs'])!=bench['comparison'] or bench['comparison']!=read(root/'benchmark/comparison.json'):raise ValueError('benchmark summary changed')
 anchor=states['full']['runs'][0]['outputs'][0]
 for row in bench['runs']:
  if row['repeat_anchor']!=anchor or any(verify(x).read_bytes()!=verify(anchor).read_bytes() for x in row['outputs']):raise ValueError('benchmark/full repeat binding differs')
 for phase,expected_count in [('tile0',6),('full',201)]:
  results=states[phase]['runs'][0]['validation']['numeric']
  if len(results)!=2 or any(x['mismatch_count']!=expected_count or x['max_ulp']!=1 for x in results):raise ValueError('reported frozen mismatch scope changed')
 records['benchmark']=identity(path)
 return dict(status='ALL_SIX_EXPLORATION_PHASES_REVERIFIED',reports=records,build=states['build'],proof=proof,
  full_anchor=anchor,full_numeric=states['full']['runs'][0]['validation']['numeric'][0],machine=current['machine'],
  benchmark_statistics=bench['comparison']['statistics'],numeric_status='NUMERIC_MISMATCH_UNRESOLVED')


def verify_bound(bound):
 verify_tree(bound);c.verify_artifacts(bound['build']);c.unchanged(bound['proof'])


def source_counts(proof):
 # Source identity and fixed scope checks, never disassembly or dynamic instruction counts.
 records=proof['normal_audit']['files']
 ks=[x['source'] for x in records if Path(x['source']['path']).parent.name=='op_kernel']
 if len(ks)!=1:raise ValueError('ambiguous normal Kernel source')
 path=verify(ks[0]);text=path.read_text()
 checks=['for(uint32_t tile=core;tile<76;tile+=td.launchBlocks)',
  'for(uint32_t nb=0;nb<16;++nb)for(uint32_t m=0;m<16;++m)DataCopy(out[m*19456+tile*256+nb*16],partial[nb*256+m*16],16);',
  'mm.GetTensorC(partial,0,true);PipeBarrier<PIPE_ALL>();mm.End();',
  'for(uint32_t side=0;side<2;++side)','for(uint32_t i=0;i<128;++i)if(t.GetValue(i)!=guardWord)']
 if any(x not in text for x in checks):raise ValueError('frozen source operation-count derivation no longer applies')
 blocks=proof['policy']['selected'];owners=[list(range(core,76,blocks)) for core in range(blocks)]
 return dict(source=ks[0],scope='SOURCE_COUNTS_NOT_MACHINE_INSTRUCTIONS_OR_WALL_TIME',shape=[16,2560,19456],
  tiles=76,block_dim=blocks,tile_owners=owners,tiles_per_core=[len(x) for x in owners],
  matmul_objects=blocks,iterate_calls=76,get_tensor_c_calls=76,end_calls=76,
  gm_output_calls_per_tile=16*16,gm_output_calls=76*16*16,bytes_per_call=32,gm_output_payload_bytes=622592,
  ub_guard_words_per_boundary=384,ub_guard_boundaries_per_tile=3,ub_guard_getvalue_words=76*3*384,
  arena_checks=blocks+76,arena_getvalue_words=(blocks+76)*2*128,
  measured_per_core_tail='UNAVAILABLE_WITHOUT_PER_CORE_PROFILE',barrier_wait_time='UNAVAILABLE_WITHOUT_INSTRUCTION_OR_STALL_EVIDENCE')
