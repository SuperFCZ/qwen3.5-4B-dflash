"""Bounded revalidation of reviewed identities, not a scanner or an optimization gate."""
import argparse,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def resource_model():
 # Exact Stage4O/P accepted bounds, not a new platform query.
 parts=dict(scratch=202240,partial_buffers=2*8192,cast=16384,sum=16384,guards=2560,reserve=8192)
 coeff=(34816,34816+256*8);b_conversion=(0,2*128*256)
 return dict(origin='USER_ACCEPTED_STAGE4O_P_BOUNDS_NOT_NEW_MEASUREMENT',ub=parts,total=sum(parts.values()),ub_limit=262144,
  l1=2048+32768,l0a=2*16*128,l0b=2*128*256,l0c=16*256*4,l0b_limit=65536,
  coefficient_view=coeff,b_conversion_view=b_conversion,views_overlap=max(coeff[0],b_conversion[0])<min(coeff[1],b_conversion[1]),
  extra_ub_authorized=0,extra_l0b_authorized=0)

def verify_sdk(root):
 root=Path(root).resolve(strict=True);asc=root/'aarch64-linux/asc';r=json.loads((HERE/'sdk-review.json').read_text());rows=[]
 for item in r['sources']:
  path=(asc/item['relative_path']).resolve(strict=True)
  if root not in path.parents or not path.is_file():raise ValueError('SDK file escaped root or absent')
  data=path.read_bytes()
  if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('reviewed SDK identity differs: '+item['id'])
  if any(not (1<=lo<=hi<=len(data.splitlines())) for lo,hi in item['lines']):raise ValueError('invalid evidence range')
  rows.append(dict(id=item['id'],path=str(path),sha256=item['sha256'],bytes=len(data),identity_status=item['identity_status']))
 return dict(status='LOCAL_REVIEWED_FILE_IDENTITIES_MATCH',decision='BLOCKED_OVERLAP_LIFECYCLE_UNPROVEN',files=rows,resources=resource_model(),
  npu='NOT_RUN',new_kernel='NOT_CREATED',benchmark='NOT_RUN',profile='NOT_RUN')

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--cann-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise SystemExit('refusing to overwrite report')
 try:result=verify_sdk(a.cann_root)
 except Exception as e:result=dict(status='BLOCKED_SDK_IDENTITY',error=f'{type(e).__name__}: {e}',npu='NOT_RUN')
 with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(result['status']);raise SystemExit(0 if result['status']=='LOCAL_REVIEWED_FILE_IDENTITIES_MATCH' else 1)
