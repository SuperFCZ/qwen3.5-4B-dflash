"""New per-channel scheme, never accepts the original Group128 output as its Golden."""
import hashlib
import numpy as np
from v_common import p,identity,verify,read,write,POLICY
from reference_r import reference_block
SPECS={'a_i8.bin':('i1',(16,2560)),'b_i8.bin':('i1',(76,2560,256)),'quant_u64_doc_candidate.bin':('<u8',(76,256)),
 'weight_scale_f64.bin':('<f8',(19456,)),'combined_f32.bin':('<f4',(19456,)),'effective_f32.bin':('<f4',(19456,)),
 'dot_i32.bin':('<i4',(16,19456)),'golden_fp16.bin':('<f2',(16,19456)), 'reference_f64.bin':('<f8',(16,19456))}

def stats(ref,y):
 x=np.asarray(ref,np.float64);v=np.asarray(y,np.float64)
 if x.shape!=v.shape or not np.isfinite(x).all() or not np.isfinite(v).all():raise ValueError('nonfinite/shape metric input')
 d=v-x;norm=float(np.linalg.norm(x));yn=float(np.linalg.norm(v))
 return dict(rmse=float(np.sqrt(np.mean(d*d))),max_abs=float(np.max(np.abs(d))),relative_l2=float(np.linalg.norm(d)/norm) if norm else (0. if not d.any() else None),cosine=float(np.sum(x*v)/(norm*yn)) if norm and yn else None)

def quantize_weight(w):
 if w.dtype!=np.float16 or w.ndim!=2 or not np.isfinite(w).all():raise ValueError('finite effective FP16 NK required')
 v=w.astype(np.float64);peak=np.max(np.abs(v),axis=1);scale=np.where(peak==0,1.,peak/127.)
 raw=np.rint(v/scale[:,None]);quant=np.clip(raw,-127,127).astype('i1')
 return quant,scale,dict(zero_channels=np.flatnonzero(peak==0).tolist(),rounding='FP64 divide; ties-to-even; clip[-127,127]',clipped=int(np.count_nonzero(np.abs(raw)>127)),endpoint_count=int(np.count_nonzero(np.abs(quant.astype(np.int16))==127)))

def encode(scale):
 s=np.asarray(scale,'<f4')
 if not np.isfinite(s).all() or np.any(s<0):raise ValueError('invalid combined Scale')
 bits=s.view('<u4') & np.uint32(0xffffe000);words=bits.astype('<u8')|np.uint64(1<<46);eff=bits.view('<f4')
 if np.any((s>0)&(eff==0)):raise ValueError('positive Scale encoded as zero')
 return words,eff

def compute(a,b,words):
 if a.dtype!=np.int8 or b.dtype!=np.int8 or a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0] or words.shape!=(b.shape[1],):raise ValueError('dot layout mismatch')
 if np.any(a==-128) or np.any(b==-128):raise ValueError('outside symmetric quantization range')
 if np.any(words & np.uint64(0x1fff)) or np.any(words>>32!=np.uint64(1<<14)):raise ValueError('codec bits differ')
 eff=(words & np.uint64(0xffffffff)).astype('<u4').view('<f4')
 if not np.isfinite(eff).all() or np.any(eff<0):raise ValueError('invalid effective Scale')
 bound=a.shape[1]*127*127
 if bound>np.iinfo(np.int32).max:raise ValueError('INT32 overflow bound')
 dot=a.astype(np.int64)@b.astype(np.int64)
 if np.any(np.abs(dot)>bound):raise ValueError('dot exceeds proven bound')
 with np.errstate(over='ignore',invalid='ignore'):y=(dot.astype('<f4')*eff).astype('<f2')
 if not np.isfinite(y).all():raise ValueError('nonfinite new FP16 reference at row/local_channel='+str(np.argwhere(~np.isfinite(y))[0].tolist()))
 return dot.astype('<i4'),y

def prepare(root,proof):
 root.mkdir();oldcase=read(verify(proof['baseline']['case']));bcase=read(verify(oldcase['stage4b_case']));src=p.reference.load_source(bcase['source'])
 olda=np.fromfile(verify(oldcase['files']['a_i8.bin']),'i1').reshape(20,16,128)
 if not np.array_equal(olda,src['a']):raise ValueError('Activation identity mismatch')
 a=olda.transpose(1,0,2).reshape(16,2560).copy()
 if not np.array_equal(a.reshape(16,20,128).transpose(1,0,2),olda):raise ValueError('Activation inverse mapping differs')
 native=read(verify(proof['native_reference']));wref=np.memmap(verify(native['files']['w_fp16_kn.bin']),mode='r',dtype='<f2',shape=(2560,19456));fref=np.memmap(verify(native['files']['reference_f64.bin']),mode='r',dtype='<f8',shape=(16,19456))
 if native['files']['x.bin']!=bcase['source']['source']['frozen']['x.bin']:raise ValueError('FP16 original X differs')
 old=np.memmap(verify(oldcase['files']['golden_fp16.bin']),mode='r',dtype='<f2',shape=(16,19456))
 maps={n:np.memmap(root/n,mode='w+',dtype=dt,shape=sh) for n,(dt,sh) in SPECS.items()};maps['a_i8.bin'][:]=a
 channels=[];tiles=[];ideal=np.empty((16,19456),'<f2');activation_only=np.empty((16,19456),'<f8');ideal_quant=np.empty((16,19456),'<f8')
 for t in range(76):
  lo=t*256;hi=lo+256
  # Exact original NZ -> logical NK check for all20 original groups.
  for g in range(20):
   nk=src['nz'][g*4:g*4+4,t*16:t*16+16].transpose(1,2,0,3).reshape(256,128)
   if not np.array_equal(nk,src['q'][lo:hi,g*128:(g+1)*128]):raise ValueError(f'original NZ mapping tile={t} group={g}')
  sw=np.repeat(src['sw'][:,lo:hi].T,128,axis=1).astype(np.float64)
  if not np.isfinite(sw).all() or np.any(sw<0):raise ValueError(f'old scale invalid tile={t}')
  w=(src['q'][lo:hi].astype(np.float64)*sw).astype('<f2')
  if w.tobytes()!=wref[:,lo:hi].T.copy().tobytes():raise ValueError('RN16 effective weights differ from Stage4R')
  quant,scale,detail=quantize_weight(w);packed=quant.T.copy()
  unrounded=np.float64(src['sa'])*scale
  combined=unrounded.astype('<f4')
  if np.any((unrounded>0)&(combined==0)):raise ValueError(f'combined Scale underflow tile={t}')
  words,eff=encode(combined)
  try:dot,y=compute(a,packed,words)
  except ValueError as error:raise ValueError(f'tile={t}: {error}') from error
  ref=reference_block(src['x'],w.T.copy())
  if not np.array_equal(ref,fref[:,lo:hi]):raise ValueError('independent FP64 recomputation differs from Stage4R')
  maps['b_i8.bin'][t]=packed;maps['quant_u64_doc_candidate.bin'][t]=words
  for name,v in [('weight_scale_f64.bin',scale),('combined_f32.bin',combined),('effective_f32.bin',eff)]:maps[name][lo:hi]=v
  activation_only[:,lo:hi]=(a.astype(np.float64)*src['sa'])@w.T.astype(np.float64)
  ideal_quant[:,lo:hi]=dot.astype(np.float64)*(np.float64(src['sa'])*scale)
  maps['dot_i32.bin'][:,lo:hi]=dot;maps['golden_fp16.bin'][:,lo:hi]=y;maps['reference_f64.bin'][:,lo:hi]=ref
  with np.errstate(over='ignore'):ideal[:,lo:hi]=(dot.astype('<f4')*combined).astype('<f2')
  if not np.isfinite(ideal[:,lo:hi]).all():raise ValueError(f'ideal FP32 scale output nonfinite tile={t}')
  for i in range(256):
   err=stats(w[i],quant[i].astype(np.float64)*scale[i]);rms=float(np.sqrt(np.mean(w[i].astype(np.float64)**2)));peak=float(np.max(np.abs(w[i])))
   channels.append(dict(channel=lo+i,scale=float(scale[i]),peak_to_rms=peak/rms if rms else None,old_scale_span=float(np.max(sw[i])/np.min(sw[i])) if np.min(sw[i])>0 else None,zero=bool(not np.any(w[i])),scale_min=float(np.min(sw[i])),scale_max=float(np.max(sw[i])),endpoint_ratio=float(np.mean(np.abs(quant[i].astype(np.int16))==127)),**err))
  tiles.append(dict(tile=t,n_range=[lo,hi],B_sha256=hashlib.sha256(packed.tobytes()).hexdigest(),scale_sha256=hashlib.sha256(words.tobytes()).hexdigest(),**detail))
  write(root/'progress.json',dict(status='INCOMPLETE',tiles=t+1))
 for m in maps.values():m.flush()
 quality=dict(status='QUALITY_GO_UNRESOLVED',acceptance_threshold=None,scope='One frozen X and historical quantized W_eff; not original checkpoint/model quality; no calibration or tuning',
  new_vs_fp64=stats(maps['reference_f64.bin'],maps['golden_fp16.bin']),old_vs_fp64=stats(maps['reference_f64.bin'],old),new_vs_old=stats(old,maps['golden_fp16.bin']),
  encoding_output_error=stats(ideal,maps['golden_fp16.bin']),combined_scale_error=stats(maps['combined_f32.bin'],maps['effective_f32.bin']),
  activation_reconstruction_error=stats(src['x'],a.astype(np.float64)*src['sa']),activation_only_vs_fp64=stats(maps['reference_f64.bin'],activation_only),weight_requant_effect_given_quantized_A=stats(activation_only,ideal_quant),capture_count=1,original_checkpoint_available=False,
  regions={name:dict(new_vs_fp64=stats(maps['reference_f64.bin'][:,sl],maps['golden_fp16.bin'][:,sl]),old_vs_fp64=stats(maps['reference_f64.bin'][:,sl],old[:,sl])) for name,sl in [('Gate',slice(0,9728)),('Up',slice(9728,19456))]},
  channels=channels,worst_weight_channels=sorted(channels,key=lambda c:c['rmse'],reverse=True)[:32],worst_relative_weight_channels=sorted(channels,key=lambda c:c['relative_l2'] or 0,reverse=True)[:32])
 write(root/'quality.json',quality)
 case=dict(status='STAGE4V_CPU_READY',policy=POLICY,files={n:identity(root/n) for n in SPECS},tiles=tiles,quality=identity(root/'quality.json'),
  original_stage4b=oldcase['files']['golden_fp16.bin'],baseline_case=proof['baseline']['case'],native_reference=proof['native_reference'],activation_scale=src['sa'],activation_requantized=False,
  original_activation_identity=oldcase['files']['a_i8.bin'],int32_bound=2560*127*127,matmul_calls=76,MMAD_instruction_count='NOT_INFERRED',gate_up_boundary_tile=38)
 write(root/'case.json',case);return case

def verify_case(case,numerical=False):
 if case.get('status')!='STAGE4V_CPU_READY' or case['policy']!=POLICY or case['activation_requantized'] is not False or set(case['files'])!=set(SPECS):raise ValueError('case contract differs')
 maps={}
 for name,(dt,sh) in SPECS.items():
  f=verify(case['files'][name])
  if f.stat().st_size!=int(np.prod(sh))*np.dtype(dt).itemsize:raise ValueError('bytes mismatch '+name)
  maps[name]=np.memmap(f,mode='r',dtype=dt,shape=sh)
 oldcase=read(verify(case['baseline_case']));bcase=read(verify(oldcase['stage4b_case']))
 if case['original_activation_identity']!=oldcase['files']['a_i8.bin'] or case['original_stage4b']!=oldcase['files']['golden_fp16.bin'] or case['activation_scale']!=bcase['source']['source']['activation']['scale']:raise ValueError('historical data binding drift')
 if case['files']['golden_fp16.bin']['path']==case['original_stage4b']['path']:raise ValueError('old Golden cannot be substituted for independent C')
 if not np.array_equal((np.float64(case['activation_scale'])*maps['weight_scale_f64.bin']).astype('<f4'),maps['combined_f32.bin']):raise ValueError('combined Scale equation drift')
 native=read(verify(case['native_reference']))
 if not np.array_equal(maps['reference_f64.bin'],np.memmap(verify(native['files']['reference_f64.bin']),mode='r',dtype='<f8',shape=(16,19456))):raise ValueError('FP64 reference binding drift')
 weights=np.memmap(verify(native['files']['w_fp16_kn.bin']),mode='r',dtype='<f2',shape=(2560,19456)) if numerical else None
 old=np.fromfile(verify(case['original_activation_identity']),'i1').reshape(20,16,128)
 if not np.array_equal(maps['a_i8.bin'],old.transpose(1,0,2).reshape(16,2560)):raise ValueError('Activation changed')
 if len(case['tiles'])!=76 or case['int32_bound']!=41290240:raise ValueError('coverage/overflow contract differs')
 for t,row in enumerate(case['tiles']):
  if row['tile']!=t or row['n_range']!=[t*256,(t+1)*256]:raise ValueError('tile mapping')
  for key,name in [('B_sha256','b_i8.bin'),('scale_sha256','quant_u64_doc_candidate.bin')]:
   if row[key]!=hashlib.sha256(maps[name][t].tobytes()).hexdigest():raise ValueError('tile hash')
  w,e=encode(maps['combined_f32.bin'][t*256:(t+1)*256])
  if not np.array_equal(w,maps['quant_u64_doc_candidate.bin'][t]) or not np.array_equal(e,maps['effective_f32.bin'][t*256:(t+1)*256]):raise ValueError('codec drift')
  if numerical:
   quant,scale,_=quantize_weight(weights[:,t*256:(t+1)*256].T.copy())
   if quant.T.copy().tobytes()!=maps['b_i8.bin'][t].tobytes() or scale.tobytes()!=maps['weight_scale_f64.bin'][t*256:(t+1)*256].tobytes():raise ValueError('per-channel requantization rule drift')
   dot,y=compute(maps['a_i8.bin'],maps['b_i8.bin'][t],w)
   if dot.tobytes()!=maps['dot_i32.bin'][:,t*256:(t+1)*256].copy().tobytes() or y.tobytes()!=maps['golden_fp16.bin'][:,t*256:(t+1)*256].copy().tobytes():raise ValueError('new numerical reference changed')
 return maps
