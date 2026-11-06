"""Exact one-line removal, independently re-derived at each admission."""
import sys,difflib
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4p'))
import p_common as p
BASE=HERE.parent/'stage4p'
COVERAGE='WEAKER_GUARD_DIAGNOSTIC_ONLY'
OLD='        if(GroupDelta(scratchRaw.Get<uint32_t>(),td.localMemSize,p1Raw.Get<uint32_t>(),p0FinalRaw.Get<uint32_t>(),castRaw.Get<uint32_t>(),sumRaw.Get<uint32_t>())!=0)return;\n'
NEW=''

def transform(raw):
 text=raw.decode()
 if text.count(OLD)!=1:raise ValueError('unique GroupDelta check anchor missing')
 return text.replace(OLD,NEW,1).encode()

def originals():
 return [BASE/'operator.json']+[f for sub in ('op_host','op_kernel') for f in sorted((BASE/sub).iterdir()) if f.is_file()]

def materialize(root):
 root.mkdir();rows=[]
 for f in originals():
  out=root/f.relative_to(BASE);out.parent.mkdir(parents=True,exist_ok=True)
  expected=transform(f.read_bytes()) if f.parent.name=='op_kernel' else f.read_bytes()
  # Committed candidate must equal the one-line transform, not a fresh unreviewed copy.
  candidate=HERE/f.relative_to(BASE)
  if candidate.read_bytes()!=expected:raise ValueError('committed source delta drift')
  out.write_bytes(expected);rows.append(dict(original=p.identity(f),candidate=p.identity(out)))
 original=(BASE/'op_kernel/d_flash_w8_a8_n256_gate.cpp').read_text()
 changed=(root/'op_kernel/d_flash_w8_a8_n256_gate.cpp').read_text()
 (root/'source.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),fromfile='Stage4P',tofile='Stage4T_DIAGNOSTIC_ONLY')))
 result=dict(status='ONLY_GROUP_DELTA_CHECK_REMOVED',coverage=COVERAGE,files=rows,diff=p.identity(root/'source.diff'))
 p.write(root/'audit.json',result);return result

def verify_audit(a):
 expected={str(f.resolve()) for f in originals()}
 if a.get('status')!='ONLY_GROUP_DELTA_CHECK_REMOVED' or a.get('coverage')!=COVERAGE:raise ValueError('missing diagnostic coverage label')
 if len(a['files'])!=len(expected) or {x['original']['path'] for x in a['files']}!=expected:raise ValueError('source inventory differs')
 for row in a['files']:
  f=p.verify(row['original']);data=f.read_bytes()
  if f.parent.name=='op_kernel':data=transform(data)
  if p.verify(row['candidate']).read_bytes()!=data:raise ValueError('unexpected source delta')
 p.verify(a['diff'])
