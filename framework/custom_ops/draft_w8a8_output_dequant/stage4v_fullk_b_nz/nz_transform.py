from pathlib import Path
import difflib
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'stage4v_fullk_scalar_guard_ab'
NAMES={'candidate':('DFlashFullKBNz','d_flash_full_kb_nz'),'fault':('DFlashFullKBNzFault','d_flash_full_kb_nz_fault')}
OLD={'candidate':('DFlashFullKGuardAggregate','d_flash_full_k_guard_aggregate'),'fault':('DFlashFullKGuardFault','d_flash_full_k_guard_fault')}

def once(s,a,b):
 if s.count(a)!=1:raise ValueError('B-only transform anchor drift: '+a)
 return s.replace(a,b,1)

def files(kind):
 name,snake=NAMES[kind];old,olds=OLD[kind];out={}
 for f in (BASE/kind).rglob('*'):
  if not f.is_file():continue
  rel=str(f.relative_to(BASE/kind)).replace(olds,snake);s=f.read_text().replace(old,name).replace(olds,snake)
  if f.parent.name=='op_kernel':
   s=once(s,'using BType=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;','using BType=MatmulType<TPosition::GM,CubeFormat::NZ,int8_t,true>;')
   s=once(s,'mm.SetTensorB(bg[tile*2560*256],false)','mm.SetTensorB(bg[tile*2560*256],true)')
  elif f.name=='host_common.h':
   s=s.replace('{76,2560,256}','{76,80,16,16,32}')
   s=once(s,'mm.SetBType(TPosition::GM,CubeFormat::ND,DataType::DT_INT8,false)','mm.SetBType(TPosition::GM,CubeFormat::NZ,DataType::DT_INT8,true)')
   s=s.replace('SetBType INT8 ND non-transposed','SetBType INT8 NZ transposed storage')
   s=s.replace('// CFG_NORM: non-transposed B ND uses two layout-conversion tiles; C workspace is half.', '// Retain the baseline conservative scratch envelope; selected B NZ/trans uses direct GM-to-L1.')
  elif f.name=='runner.cpp':
   s=once(s,'r.w.Make({76,2560,256},{76,2560,256},','r.w.Make({76,80,16,16,32},{76,80,16,16,32},')
  out[rel]=s.encode()
 return out

def verify_repository():
 for kind in NAMES:
  for rel,data in files(kind).items():
   if (HERE/kind/rel).read_bytes()!=data:raise ValueError('B-only source audit differs '+kind+'/'+rel)

def materialize(dest,kind,identity,write):
 verify_repository();dest.mkdir(parents=True);rows=[]
 for rel,data in files(kind).items():
  if not(rel=='operator.json' or rel.startswith(('op_host/','op_kernel/'))):continue
  p=dest/rel;p.parent.mkdir(exist_ok=True);p.write_bytes(data);rows.append(identity(p))
 old=(BASE/kind/'op_kernel'/(OLD[kind][1]+'.cpp')).read_text()
 new=(dest/'op_kernel'/(NAMES[kind][1]+'.cpp')).read_text()
 diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='ArenaBaseline',tofile='B_NZ_TransposeStorage'))
 (dest/'source.diff').write_text(diff)
 r=dict(kind=kind,files=rows,diff=identity(dest/'source.diff'),status='B_REPRESENTATION_ONLY_WITH_MATCHING_TRANSPOSE_METADATA')
 write(dest/'audit.json',r);return r

def verify_audit(a,verify):
 expected={k:v for k,v in files(a['kind']).items() if k=='operator.json' or k.startswith(('op_host/','op_kernel/'))}
 got={('operator.json' if Path(f['path']).name=='operator.json' else Path(f['path']).parent.name+'/'+Path(f['path']).name):verify(f).read_bytes() for f in a['files']}
 if expected!=got:raise ValueError('compiled input transform audit differs')
 verify(a['diff'])

if __name__=='__main__':
 import sys
 from nz_common import identity,write
 materialize(Path(sys.argv[1]),sys.argv[2],identity,write)
