"""Read-only Stage4N freeze and 76×20 frozen packing checks; no new Golden."""
import sys,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4n'))
import n_common as n
import run_stage4n as nr
old=n.k.bc.full
identity,verify,verify_tree=n.identity,n.verify,n.k.verify_tree

def source_files():
 rows={x['path']:x for x in n.sources()}
 for p in HERE.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:rows[str(p.resolve())]=identity(p)
 return [rows[k] for k in sorted(rows)]

def freeze(root):
 report=n.read(root/'benchmark/benchmark-report.json')
 if report.get('sources')!=n.sources():raise ValueError('Stage4N sources changed')
 build=n.read(verify(report['build_report']))
 if report['active']!=n.active(report['evidence'],build):raise ValueError('Stage4N active identities changed')
 nr.prior(verify(report['build_report']),'build',report)
 nr.prior(verify(report['smoke_report']),'smoke',report)
 nr.prior(root/'benchmark/benchmark-report.json','benchmark',report) # includes fault witness replay
 fault=n.read(verify(report['fault_report']))
 wanted=dict(words=640,bits=32,group_boundaries=20,injections_per_sweep=409600,repeats=2,abort_groups=list(range(20)),performance_eligible=False)
 if any(fault.get('coverage',{}).get(k)!=v for k,v in wanted.items()):raise ValueError('Stage4N fault coverage metadata differs')
 if report['comparison']['decision']!='GO_FURTHER_REVIEW':raise ValueError('Stage4N not an accepted performance candidate')
 profile=n.read(root/'profile/profile-report.json')
 if profile.get('status')!='STAGE4N_PROFILE_COMPLETED' or profile.get('benchmark_report')!=identity(root/'benchmark/benchmark-report.json') or any(profile.get(k)!=report[k] for k in ('sources','evidence','active','device','build_report')):raise ValueError('Stage4N Profile binding mismatch')
 if [r['variant'] for r in profile['runs']]!=list(n.NAMES):raise ValueError('incomplete Stage4N Profile')
 for row in profile['runs']:n.validate_run(row,profile['active'],report['device'],True,True)
 xor=report['active']['variants']['Xor'];n.immutable(report['active'])
 case=n.read(verify(xor['case']))
 return dict(status='STAGE4N_FROZEN_BEST_FULL_GROUP_GUARD_CANDIDATE',adoption='GO_FURTHER_REVIEW_NOT_PRODUCTION',
  fault_coverage=wanted,reports={key:identity(root/sub) for key,sub in [('benchmark','benchmark/benchmark-report.json'),('profile','profile/profile-report.json')]},
  paired_result=report['comparison'],build=report['build_report'],smoke=report['smoke_report'],fault=report['fault_report'],candidate=xor,case=xor['case'],
  sdk=report['evidence']['upstream']['sdk'],formal_core_policy=xor['policy'],device=report['device'],frozen_inputs=case['stage4b_case'],
  new_performance='NOT_RUN',scope=n.SCOPE)

def symbolic():
 seen=np.zeros(19456,dtype=np.uint8)
 for tile in range(76):seen[tile*256:(tile+1)*256]+=1
 m=np.arange(16)[:,None];ch=np.arange(256)[None,:];idx=(ch//16)*256+m*16+ch%16
 if not np.all(seen==1) or not np.array_equal(np.sort(idx.ravel()),np.arange(4096)):raise ValueError('layout coverage failure')
 return dict(tile_count=76,groups=20,validated_tile_groups=1520,B_shape=[76,20,128,256],B_bytes=49807360,scale_shape=[76,20,256],scale_bytes=3112960,
  A_shape=[20,16,128],A_bytes=40960,output_shape=[16,19456],output_bytes=622592,
  local_nz_shape=[16,16,16],local_nz_formula='(n//16)*256 + row*16+n%16',gm_formula='row*19456+tile*256+n',gate_up_boundary=dict(channel=9728,tile=38,aligned=True),
  ordered_groups=list(range(20)),activation_requantized=False,new_golden=False)

def verify_tile(src,b128,s128,tile):
 b=np.empty((20,128,256),'i1');words=np.empty((20,256),'<u8')
 for g in range(20):
  nk=src['nz'][g*4:g*4+4,tile*16:tile*16+16].transpose(1,2,0,3).reshape(256,128)
  if not np.array_equal(nk,src['q'][tile*256:(tile+1)*256,g*128:(g+1)*128]):raise ValueError(f'NZ mismatch tile={tile} group={g}')
  b[g]=nk.T
  if b[g].tobytes()!=np.concatenate([b128[tile*2,g],b128[tile*2+1,g]],axis=-1).tobytes():raise ValueError(f'N128 B mapping differs tile={tile} group={g}')
  sw=src['sw'][g,tile*256:(tile+1)*256]
  if not np.isfinite(sw).all() or np.any(sw<0):raise ValueError('invalid frozen GN scale')
  combined=(np.float64(src['sa'])*sw.astype(np.float64)).astype('<f4')
  expected=np.concatenate([n.common.stage2a.encode_doc_candidate(combined[i:i+64])[0] for i in range(0,256,64)])
  words[g]=np.concatenate([s128[tile*2,g],s128[tile*2+1,g]])
  if words[g].tobytes()!=expected.tobytes():raise ValueError(f'Scale mapping differs tile={tile} group={g}')
 return b,words

def real(root):
 proof=freeze(root);case=n.read(verify(proof['case']));maps=n.k.jref.verify_case(case)
 historical=n.read(verify(case['stage4b_case']));src=n.k.bc.reference.load_source(historical['source'])
 if src['a'].tobytes()!=maps['a_i8.bin'].tobytes():raise ValueError('saved full-X activation differs')
 hb,hs=hashlib.sha256(),hashlib.sha256()
 for t in range(76):
  b,s=verify_tile(src,maps['b_i8.bin'],maps['quant_u64_doc_candidate.bin'],t);hb.update(b.tobytes());hs.update(s.tobytes())
 r=symbolic();r.update(status='CPU_FROZEN_LAYOUT_MATCH',stage4n=proof,source_report=proof['reports']['benchmark'],saved_a=case['files']['a_i8.bin'],
  golden=case['files']['golden_fp16.bin'],packed_B_sha256=hb.hexdigest(),packed_scale_sha256=hs.hexdigest(),packed_files_written=False)
 verify_tree(proof);return r


def core_consistency(result,policy):
 return (result.get('arch')==policy['arch'] and result.get('aic')==policy['aic'] and result.get('aiv')==policy['aiv'] and type(result.get('all')) is int and policy['selected']<=result['all']<=result['aic'])
