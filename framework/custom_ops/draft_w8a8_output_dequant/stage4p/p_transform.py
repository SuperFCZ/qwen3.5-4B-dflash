"""N128 accepted XOR source -> N256 only; fault instrumentation stays separate."""
from pathlib import Path
import sys,difflib,hashlib
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4n'))
import guard_transform as ng
J=HERE.parent/'stage4j'

def names(s):return s.replace('DFlashW8A8N128Gate','DFlashW8A8N256Gate').replace('d_flash_w8_a8_n128_gate','d_flash_w8_a8_n256_gate').replace('STAGE4J','STAGE4P').replace('Stage4J','Stage4P').replace('S4J','S4P').replace('stage4j','stage4p')

def kernel(raw,fault=False):
 s=(ng.diagnostic(raw) if fault else ng.transform(raw)).decode()
 s=names(s).replace('152','76').replace('239104','202240').replace('226816','202240').replace('N128','N256')
 for obj in ('p0FinalRaw','p1Raw'):
  s=s.replace(f'InitBuffer({obj},4096+',f'InitBuffer({obj},8192+').replace(f'{obj}.Get<uint32_t>(),4096',f'{obj}.Get<uint32_t>(),8192')
 for obj in ('castRaw','sumRaw'):
  s=s.replace(f'InitBuffer({obj},8192+',f'InitBuffer({obj},16384+').replace(f'{obj}.Get<uint32_t>(),8192',f'{obj}.Get<uint32_t>(),16384')
 s=s.replace('GuardDelta(p1,4096)','GuardDelta(p1,8192)').replace('GuardDelta(p0,4096)','GuardDelta(p0,8192)').replace('GuardDelta(cast,8192)','GuardDelta(cast,16384)').replace('GuardDelta(sum,8192)','GuardDelta(sum,16384)')
 s=s.replace('SetSize(2048)','SetSize(4096)').replace('0.0f,2048)','0.0f,4096)').replace('CAST_NONE,2048)','CAST_NONE,4096)').replace('sum,converted,2048)','sum,converted,4096)').replace('slot/16.0f),2048)','slot/16.0f),4096)')
 s=s.replace('76*20*16384','76*20*32768').replace('76*20*128','76*20*256').replace('(slot*20+g)*16384','(slot*20+g)*32768').replace('(slot*20+g)*128','(slot*20+g)*256').replace('nb<8','nb<16').replace('slot*128+','slot*256+')
 s=s.replace('buffer<3?4096:8192','buffer<3?8192:16384').replace('(256+8192)/4+63','(256+16384)/4+63')
 return s.encode()

def host(text):
 s=names(text).replace('152','76').replace('{76,20,128,128}','{76,20,128,256}').replace('{76,20,128}','{76,20,256}')
 s=s.replace('24576','49152').replace('226816','202240').replace('24 KiB','48 KiB')
 s=s.replace('16,128,128','16,256,128').replace('get_singleCoreN()!=128','get_singleCoreN()!=256').replace('get_baseN()!=128','get_baseN()!=256')
 s=s.replace('bbytes=128*128','bbytes=256*128').replace('16*128*','16*256*').replace('+128*8','+256*8').replace('tileN=128','tileN=256')
 return s

def runner(text):
 return names(text).replace('>152','>76').replace('>=152','>=76').replace('uint64_t(152)','uint64_t(76)').replace('{152,20,128,128}','{76,20,128,256}').replace('{152,20,128}','{76,20,256}').replace('tile_count\\\":152','tile_count\\\":76').replace('col/128','col/256').replace('col>=128','col>=256')

def derived():
 result={}
 for sub in ('op_host','op_kernel'):
  for p in (J/sub).iterdir():
   if not p.is_file():continue
   text=kernel(p.read_bytes()).decode() if sub=='op_kernel' else host(p.read_text()) if p.name=='host_common.h' else names(p.read_text())
   result[sub+'/'+names(p.name)]=(p,text)
 result['operator.json']=(J/'operator.json',names((J/'operator.json').read_text()))
 result['test/runner.cpp']=(J/'test/runner.cpp',runner((J/'test/runner.cpp').read_text()))
 result['test/CMakeLists.txt']=(J/'test/CMakeLists.txt',names((J/'test/CMakeLists.txt').read_text()))
 return result

def audit():
 rows=[];diff=[]
 for rel,(base,expected) in derived().items():
  actual=HERE/rel
  if actual.read_text()!=expected:raise ValueError('unapproved source delta '+rel)
  original=ng.transform(base.read_bytes()).decode() if rel.startswith('op_kernel/') else base.read_text()
  rows.append(dict(original_stage4j=ng.j.identity(base),derived_baseline_sha256=hashlib.sha256(original.encode()).hexdigest(),candidate=ng.j.identity(actual)))
  diff+=list(difflib.unified_diff(original.splitlines(True),expected.splitlines(True),fromfile='Stage4N/Xor/'+str(base.relative_to(J)),tofile='Stage4P/'+rel))
 return dict(status='N256_DIMENSION_ONLY_FROM_ACCEPTED_XOR',files=rows),''.join(diff)
