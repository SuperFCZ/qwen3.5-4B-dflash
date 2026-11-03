"""Independent float64 dot of original X and RN16 dequantized weights."""
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4q'))
import q_common as q
sys.path.insert(0,str(q.common.QUANT/'fp16_compare'))
import fp16_gate_common as fc
POLICY=dict(element_atol=0.002,element_rtol=0.002,relative_l2_max=0.001,finite_required=True,adaptive_tolerance=False)

def check(reference,actual):
 x=np.asarray(reference,dtype=np.float64);y=np.asarray(actual,dtype=np.float64)
 if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('shape/nonfinite numerical failure')
 err=y-x;bad=np.abs(err)>POLICY['element_atol']+POLICY['element_rtol']*np.abs(x)
 l2=float(np.linalg.norm(err)/np.linalg.norm(x)) if np.linalg.norm(x)>0 else (0.0 if not np.any(err) else float('inf'))
 if bad.any() or l2>POLICY['relative_l2_max']:raise ValueError('fixed independent FP64 acceptance failed')
 return dict(status='INDEPENDENT_CPU_NUMERICS_PASS',max_abs=float(np.max(np.abs(err))),rmse=float(np.sqrt(np.mean(err*err))),relative_l2=l2)

def reference_block(x,w):
 if x.dtype!=np.float16 or w.dtype!=np.float16 or x.shape[1]!=w.shape[0]:raise ValueError("FP16 ND operands required")
 return x.astype(np.float64) @ w.astype(np.float64)

def prepare(root,proof):
 root.mkdir();case=q.read(q.verify(proof['variants']['N128']['case']));bcase=q.read(q.verify(case['stage4b_case']));files=bcase['source']['source']['frozen']
 weights=fc.generate_weights(files,root/'weights') # new files; no historical FP16 report/Profile required
 x=np.fromfile(q.verify(files['x.bin']),'<f2').reshape(16,2560)
 w=np.memmap(q.verify(weights['files']['w_fp16_kn.bin']),mode='r',dtype='<f2',shape=(2560,19456))
 ref=np.empty((16,19456),'<f8')
 for n in range(0,19456,256):ref[:,n:n+256]=reference_block(x,w[:,n:n+256])
 if not np.isfinite(ref).all() or np.any(np.abs(ref)>65504):raise ValueError('reference not finite representable FP16')
 ref.tofile(root/'reference_f64.bin')
 return dict(status='STAGE4R_CPU_REFERENCE_PREPARED',policy=POLICY,weights=weights,files={'x.bin':files['x.bin'],'w_fp16_kn.bin':weights['files']['w_fp16_kn.bin'],'reference_f64.bin':q.identity(root/'reference_f64.bin')},
  arithmetic='FP64 dot(X FP16, RN16(INT8 q*original FP16 GN scale)); no bias/post-op; no W8A8 Golden substitution',sources=[q.identity(HERE/'reference_r.py'),q.identity(HERE/'numeric_contract.h')])
