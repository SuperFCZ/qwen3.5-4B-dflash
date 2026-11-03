"""Only final NZ->ND store-loop changes; no math/Guard/sync edits."""
import sys,difflib
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4p'))
import p_common as p
BASE=HERE.parent/'stage4p'
OLD='''    for(uint32_t nb=0;nb<16;++nb)
        for(uint32_t m=0;m<16;++m)DataCopy(out[m*19456+slot*256+nb*16],result[nb*256+m*16],16);
'''
NEW='''    const AscendC::DataCopyParams rowCopy{16,1,15,0};
    for(uint32_t m=0;m<16;++m)
        DataCopy(out[m*19456+slot*256],result[m*16],rowCopy);
'''
def transform(raw):
 text=raw.decode()
 if text.count(OLD)!=1:raise ValueError('final store anchor changed')
 return text.replace(OLD,NEW,1).encode()

def materialize(root,diagnostic=False):
 from p_transform import kernel,J
 root.mkdir();rows=[]
 for f in [BASE/'operator.json']+[f for sub in ('op_host','op_kernel') for f in (BASE/sub).iterdir() if f.is_file()]:
  out=root/f.relative_to(BASE);out.parent.mkdir(parents=True,exist_ok=True)
  data=f.read_bytes()
  if f.parent.name=='op_kernel':data=transform(kernel((J/'op_kernel/d_flash_w8_a8_n128_gate.cpp').read_bytes(),fault=True) if diagnostic else data)
  out.write_bytes(data);rows.append(dict(original=p.identity(f),candidate=p.identity(out)))
 original=(BASE/'op_kernel/d_flash_w8_a8_n256_gate.cpp').read_text();candidate=(root/'op_kernel/d_flash_w8_a8_n256_gate.cpp').read_text()
 (root/'source.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),candidate.splitlines(True),fromfile='Stage4P',tofile='Stage4S_fault' if diagnostic else 'Stage4S')))
 result=dict(status='FAULT_INSTRUMENTED' if diagnostic else 'ONLY_FINAL_STORE_LOOP_CHANGED',diagnostic=diagnostic,files=rows,diff=p.identity(root/'source.diff'))
 p.write(root/'audit.json',result);return result

def verify_audit(a):
 from p_transform import kernel,J
 expected={str(f.resolve()) for f in [BASE/'operator.json']+[f for sub in ('op_host','op_kernel') for f in (BASE/sub).iterdir() if f.is_file()]}
 if len(a['files'])!=len(expected) or {x['original']['path'] for x in a['files']}!=expected:raise ValueError('source inventory differs')
 for row in a['files']:
  f=p.verify(row['original']);data=f.read_bytes()
  if f.parent.name=='op_kernel':data=transform(kernel((J/'op_kernel/d_flash_w8_a8_n128_gate.cpp').read_bytes(),fault=True) if a['diagnostic'] else data)
  if p.verify(row['candidate']).read_bytes()!=data:raise ValueError('unexpected source delta')
 p.verify(a['diff'])
