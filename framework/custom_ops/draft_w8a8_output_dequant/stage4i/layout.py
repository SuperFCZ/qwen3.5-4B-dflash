"""Read-only N64->N128 layout/index check. No new activation quantization or Golden."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4b'))
import full_common as old
import full_reference as reference
identity,verify,verify_tree=old.identity,old.verify,old.verify_tree

def local_nz_indices(width):
 m=np.arange(16)[:,None];n=np.arange(width)[None,:]
 return (n//16)*256+m*16+n%16

def symbolic():
 width=128;n=19456;seen=np.zeros(n,dtype=np.uint8)
 for t in range(152):seen[t*width:(t+1)*width]+=1
 idx=local_nz_indices(width)
 if not np.all(seen==1) or not np.array_equal(np.sort(idx.ravel()),np.arange(2048)):raise ValueError('coverage/NZ mapping failure')
 for sub in range(2):
  if not np.array_equal(idx[:,sub*64:(sub+1)*64],local_nz_indices(64)+sub*1024):raise ValueError('T2 base placement differs')
 return dict(status='CPU_INDEX_MAPPING_PASS',tile_count=152,groups=20,group_matmuls=3040,
  B_shape=[152,20,128,128],B_bytes=49807360,scale_shape=[152,20,128],scale_bytes=3112960,
  activation_shape=[20,16,128],activation_bytes=40960,output_shape=[16,19456],output_bytes=622592,
  local_nz_shape=[8,16,16],local_nz_formula='(n//16)*256 + row*16 + n%16',
  gm_formula='row*19456 + tile*128 + n',gate_up_boundary=dict(column=9728,tile=76,aligned=True),
  ordered_groups=list(range(20)),activation_requantized=False,new_golden_created=False)

def verify_tile(src,b64,words64,t):
 packed=np.empty((20,128,128),'i1');words=np.empty((20,128),'<u8')
 for g in range(20):
  nk=src['nz'][g*4:g*4+4,t*8:t*8+8].transpose(1,2,0,3).reshape(128,128)
  expected=src['q'][t*128:(t+1)*128,g*128:(g+1)*128]
  if not np.array_equal(nk,expected):raise ValueError(f'NZ/ND mismatch tile={t} group={g}')
  packed[g]=nk.T
  if not np.array_equal(packed[g],np.concatenate([b64[2*t,g],b64[2*t+1,g]],axis=1)):raise ValueError(f'old B mapping mismatch tile={t} group={g}')
  sw=src['sw'][g,t*128:(t+1)*128]
  if not np.isfinite(sw).all() or np.any(sw<0):raise ValueError(f'invalid Scale tile={t} group={g}')
  combined=(np.float64(src['sa'])*sw.astype(np.float64)).astype('<f4')
  encoded=[old.common.stage2a.encode_doc_candidate(combined[i:i+64]) for i in (0,64)]
  expected_words=np.concatenate([e[0] for e in encoded]);effective=np.concatenate([e[1] for e in encoded])
  if not np.isfinite(combined).all() or not np.isfinite(effective).all():raise ValueError(f'nonfinite encoded Scale tile={t} group={g}')
  words[g]=np.concatenate([words64[2*t,g],words64[2*t+1,g]])
  if not np.array_equal(words[g],expected_words):raise ValueError(f'old Scale mapping mismatch tile={t} group={g}')
 return packed,words

def real(report_path):
 r=json.loads(report_path.read_text());verify_tree(r)
 if r.get('status')!='STAGE4B_FULL_GATE_MATCH' or r.get('phase')!='complete' or r.get('settings')!=old.SETTINGS or r.get('sources')!=old.sources():raise ValueError('accepted unchanged Stage4B report required')
 case=json.loads(verify(r['case']).read_text());verify_tree(case);reference.verify_case(case,numerical=False);src=reference.load_source(case['source'])
 files=case['files'];b64=np.memmap(verify(files['b_i8.bin']),mode='r',dtype='i1',shape=(304,20,128,64))
 words64=np.memmap(verify(files['quant_u64_doc_candidate.bin']),mode='r',dtype='<u8',shape=(304,20,64))
 if src['a'].tobytes()!=verify(files['a_i8.bin']).read_bytes():raise ValueError('saved A identity mismatch')
 hb,hs=hashlib.sha256(),hashlib.sha256();count=0
 for t in range(152):
  packed,words=verify_tile(src,b64,words64,t);count+=20
  hb.update(packed.tobytes());hs.update(words.tobytes())
 result=symbolic();result.update(status='CPU_FROZEN_LAYOUT_MATCH',validated_tile_groups=count,
  source_report=identity(report_path),case=r['case'],source_files=case['source']['source']['frozen'],
  saved_a=files['a_i8.bin'],saved_B=files['b_i8.bin'],activation_contract=case['source']['source']['activation'],saved_scale=files['quant_u64_doc_candidate.bin'],
  packed_B_sha256=hb.hexdigest(),packed_scale_sha256=hs.hexdigest(),packed_files_written=False,
  numerical_scope='Index/layout and frozen encoding identity only; no N128 dot product or device result')
 verify_tree(r);return result
