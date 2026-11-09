"""Strict read-only accepted Host and input binding."""
import sys,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4v_diagnostics'))
import evidence as diag
import raw_resources
from sdk_review import bind
from raw_host_dependency_identity import verify_dependency_binding
v=diag.v;common=v.common
identity,verify,read,write,fresh=diag.identity,diag.verify,diag.read,diag.write,diag.fresh
NAME='DFlashFullKInt32Probe';SNAKE='d_flash_full_k_int32_probe'
KEYS=[(0,84),(1,53),(1,201),(5,138),(7,235),(15,210)]

def sources():return [identity(f) for f in sorted(HERE.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and '.pytest_cache' not in f.parts]

def evidence(args):
 frozen=diag.load(args.frozen_root);h=read(args.raw_host_report)
 if h['status']!='RAW_HOST_CANDIDATE_FEASIBLE_CONTEXT_PENDING' or h['evidence']!=frozen['identities'] or h['source_review']!='SOURCE_REVIEWED':raise ValueError('accepted Raw Host/frozen binding required')
 if any(h['commands'][k]['returncode']!=0 for k in ('configure','compile','ldd','query')):raise ValueError('Raw Host command failed')
 if bind(args.cann_root)!=h['sdk']:raise ValueError('Raw SDK identity changed')
 verify_dependency_binding(h['dependency_binding'])
 raw=read(verify(h['host_query']));row=raw['cases'][0]
 if raw['device']!=args.device or frozen['frozen']['device']!=args.device:raise ValueError('device mismatch')
 if raw_resources.validate(raw,row)!=h['resources'] or h['resources']['status']!='RAW_RESOURCE_CHECKS_PASS':raise ValueError('Raw resources not accepted')
 fields=row['fields'];t1=dict(fields={k:fields[k] for k in v.p.FIELDS},capacity=raw['capacity'],system_workspace_bytes=raw['system_workspace_bytes'],scratch=202240)
 required=dict(singleCoreM=16,singleCoreN=256,singleCoreK=2560,baseM=16,baseN=256,baseK=128,depthA1=20,depthB1=10,stepKa=20,stepKb=5,dbL0A=2,dbL0B=2,dbL0C=1,shareL1Size=368640,shareL0CSize=16384,shareUbSize=0,transLength=174080)
 if any(fields[k]!=value for k,value in required.items()):raise ValueError('not the accepted Raw T1')
 case=frozen['case'];a_id=case['files']['a_i8.bin'];b_id=case['files']['b_i8.bin']
 a=np.fromfile(verify(a_id),'i1').reshape(16,2560)
 with verify(b_id).open('rb') as f:braw=f.read(655360)
 if h['input_binding']!=dict(a=a_id,b_source=b_id,b_offset_bytes=0,b_bytes=655360,b_slice_sha256=hashlib.sha256(braw).hexdigest(),tile_id=0,requantization=False,scale_input='NONE'):raise ValueError('input slice changed')
 oracle=a.astype(np.int64)@np.frombuffer(braw,'i1').reshape(2560,256).astype(np.int64)
 if not np.array_equal(oracle,frozen['oracle']) or np.any(np.abs(oracle)>41290240):raise ValueError('oracle/overflow mismatch')
 policy=frozen['frozen']['evidence']['formal_core_policy']
 if policy!=h['formal_policy_reference']:raise ValueError('formal policy reference changed')
 proof=dict(reviewed_platform=h['release']['files'][0]['file'],raw_host=identity(args.raw_host_report),frozen=frozen['identities'],t1=t1,policy=policy,preflight=frozen['frozen']['evidence']['preflight'],a=a_id,b=b_id,b_slice_sha256=hashlib.sha256(braw).hexdigest())
 return proof,a.tobytes(),braw,oracle

def contract_header(t1):return v.p.contract_header(t1).replace('S4P','S4RAW').replace('STAGE4P','STAGE4RAW')

def compare(oracle,raws):
 if not 1<=len(raws)<=2 or any(len(r)!=16384 for r in raws):raise ValueError('one or two complete INT32 outputs required')
 reports=[]
 for raw in raws:
  actual=np.frombuffer(raw,'<i4').reshape(16,256).astype(np.int64);d=actual-oracle;bad=np.argwhere(d!=0);values,counts=np.unique(d,return_counts=True)
  reports.append(dict(bitwise_matches=int(np.count_nonzero(d==0)),mismatches=len(bad),first=None if not len(bad) else dict(row=int(bad[0,0]),channel=int(bad[0,1]),cpu=int(oracle[tuple(bad[0])]),npu=int(actual[tuple(bad[0])])),max_abs_integer_difference=int(np.abs(d).max()),difference_histogram={str(int(x)):int(n) for x,n in zip(values,counts)},key_positions=[dict(row=m,channel=n,cpu=int(oracle[m,n]),npu=int(actual[m,n]),delta=int(d[m,n])) for m,n in KEYS],distribution={s+'_'+p:int(np.count_nonzero(mask&(actual%2==parity))) for s,mask in [('negative',actual<0),('zero',actual==0),('positive',actual>0)] for p,parity in [('even',0),('odd',1)]}))
 return dict(status='MATCH' if len(raws)==2 and all(r['mismatches']==0 for r in reports) and raws[0]==raws[1] else 'MISMATCH',repeats=reports,repeat_equal=len(raws)==2 and raws[0]==raws[1],cpu_distribution={s+'_'+p:int(np.count_nonzero(mask&(oracle%2==parity))) for s,mask in [('negative',oracle<0),('zero',oracle==0),('positive',oracle>0)] for p,parity in [('even',0),('odd',1)]},interpretation='Independent Raw path only; not direct proof of original FP16 Kernel L0C or silicon correctness')


def audit():
 """Mechanical scope checks, not a device correctness claim."""
 kernel=(HERE/'op_kernel/d_flash_full_k_int32_probe.cpp').read_text()
 host=(HERE/'op_host/host_common.h').read_text()
 forbidden=('SetQuantVector','VDEQ','Cast(', 'float','half','Add(')
 if any(x in kernel for x in forbidden) or 'SetDequantType' in host:raise ValueError('Raw path contains floating/dequant operation')
 for x in ('mm.SetLocalWorkspace(scratch);mm.SetTensorA(ag);mm.SetTensorB(bg,false);','mm.Iterate(false)','mm.GetTensorC(partial,0,true);PipeBarrier<PIPE_ALL>();mm.End();PipeBarrier<PIPE_ALL>();','dst[m*256+nb*16],partial[nb*256+m*16],16','__NPU_ARCH__==2002','!IsBasic(CFG_NORM)'):
  if x not in kernel:raise ValueError('Raw lifecycle/layout source changed')
 if kernel.count('REGIST_MATMUL_OBJ')!=1 or kernel.count('pipe.InitBuffer')!=3:raise ValueError('unexpected Matmul or UB allocation')
 return dict(status='RAW_SOURCE_SCOPE_REVIEWED',kernel=identity(HERE/'op_kernel/d_flash_full_k_int32_probe.cpp'),
   sdk_contract=identity(HERE.parent/'stage4v_diagnostics/sdk-review.json'),ub=dict(scratch=202240,partial=16384,temporary=512,guards=1536,reserve=8192,total=228864),
   operation='INT8_FULL_K_TO_INT32_NO_DEQUANT',npu='NOT_RUN',source_review_scope='Identity-bound selected SDK contracts plus this independent lifecycle/layout; runtime pending')
