"""Exact single-variable Arena scan transformation; separate diagnostic-only additions."""
from pathlib import Path
import difflib
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'stage4v'
OLD='DFlashW8A8FullKChannel';OLD_S='d_flash_w8_a8_full_k_channel'
NAMES={'candidate':('DFlashFullKGuardAggregate','d_flash_full_k_guard_aggregate'),
       'fault':('DFlashFullKGuardFault','d_flash_full_k_guard_fault')}
SCAN='  for(uint32_t i=0;i<128;++i)if(t.GetValue(i)!=guardWord)return false;'
AGG='  uint32_t delta=0;for(uint32_t i=0;i<128;++i)delta|=t.GetValue(i)^guardWord;\n  if(delta!=0)return false;'

def once(text,old,new):
 if text.count(old)!=1:raise ValueError('source anchor must appear exactly once: '+old[:80])
 return text.replace(old,new,1)

def aggregate(text):return once(text,SCAN,AGG)

ARENA_FAULT=r'''
// Diagnostic OPP only. Modify actual GM Arena redzones, exercise the normal
// CheckArena path, restore after each sweep. No new UB; existing512B temp only.
__aicore__ inline void ArenaStore(TPipe&pipe,LocalTensor<uint32_t>t,GlobalTensor<uint32_t>a,
 uint64_t off,uint32_t word,uint32_t bits){
 Duplicate(t,guardWord,128);PipeBarrier<PIPE_ALL>();
 auto vs=pipe.FetchEventID(HardEvent::V_S);SetFlag<HardEvent::V_S>(vs);WaitFlag<HardEvent::V_S>(vs);
 t.SetValue(word,guardWord^bits);
 auto sm=pipe.FetchEventID(HardEvent::S_MTE3);SetFlag<HardEvent::S_MTE3>(sm);WaitFlag<HardEvent::S_MTE3>(sm);
 DataCopy(a[off],t,128);PipeBarrier<PIPE_ALL>();
 auto mm=pipe.FetchEventID(HardEvent::MTE3_MTE2);SetFlag<HardEvent::MTE3_MTE2>(mm);WaitFlag<HardEvent::MTE3_MTE2>(mm);
 auto ms=pipe.FetchEventID(HardEvent::MTE3_S);SetFlag<HardEvent::MTE3_S>(ms);WaitFlag<HardEvent::MTE3_S>(ms);
}
__aicore__ inline bool ArenaFault(TPipe&pipe,LocalTensor<uint32_t>t,GlobalTensor<uint32_t>a,
 uint64_t bytes,uint32_t checkpoint,uint32_t mode,GM_ADDR out,uint32_t& count){
 if(!CheckArena(pipe,t,a,bytes))return false;
 if(mode==2){
  for(uint32_t side=0;side<2;++side)for(uint32_t word=0;word<128;++word)for(uint32_t bit=0;bit<32;++bit){
   uint64_t off=side?(512+bytes)/4:0;
   ArenaStore(pipe,t,a,off,word,uint32_t(1)<<bit);
   bool detected=!CheckArena(pipe,t,a,bytes);
   ArenaStore(pipe,t,a,off,word,0);
   if(!detected||!CheckArena(pipe,t,a,bytes))return false;
   ++count;
  }
 }else if(mode==6+checkpoint){
  ArenaStore(pipe,t,a,(512+bytes)/4,127,0x80000001U);
  bool detected=!CheckArena(pipe,t,a,bytes);
  if(!detected)return false;
  // Restore redzone after detection; the wrapper still returns false so no
  // later numerical output can hide the abort. Witness is not a normal output.
  ArenaStore(pipe,t,a,(512+bytes)/4,127,0);
  t.SetValue(0,0x53474146U);t.SetValue(1,checkpoint);t.SetValue(2,0x80000001U);t.SetValue(3,256);
  t.SetValue(4,1);t.SetValue(5,127);t.SetValue(6,checkpoint);t.SetValue(7,1);
  auto sm=pipe.FetchEventID(HardEvent::S_MTE3);SetFlag<HardEvent::S_MTE3>(sm);WaitFlag<HardEvent::S_MTE3>(sm);
  GlobalTensor<uint32_t>w;w.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(out),8);DataCopy(w,t,8);PipeBarrier<PIPE_ALL>();
  auto ms=pipe.FetchEventID(HardEvent::MTE3_S);SetFlag<HardEvent::MTE3_S>(ms);WaitFlag<HardEvent::MTE3_S>(ms);
  return false;
 }
 return CheckArena(pipe,t,a,bytes);
}
'''

def fault(text):
 text=aggregate(text)
 text=once(text,'extern "C" __global__',ARENA_FAULT+'\nextern "C" __global__')
 text=once(text,'td.outputMode>5','td.outputMode>7')
 text=once(text,'PipeBarrier<PIPE_ALL>();if(!CheckArena(pipe,temp,arena,td.arenaBytes))return;',
  'uint32_t arenaSweepCount=0;\n PipeBarrier<PIPE_ALL>();if(!ArenaFault(pipe,temp,arena,td.arenaBytes,0,td.outputMode,o,arenaSweepCount))return;')
 text=once(text,'if(!CheckArena(pipe,temp,arena,td.arenaBytes))return;',
  'if(!ArenaFault(pipe,temp,arena,td.arenaBytes,1,td.outputMode,o,arenaSweepCount))return;')
 text=once(text,'if(sweepCount!=36864)return;','if(sweepCount!=36864||arenaSweepCount!=16384)return;')
 text=once(text,'temp.SetValue(0,0x53565357U);','temp.SetValue(0,0x53474753U);')
 text=once(text,'temp.SetValue(5,0);temp.SetValue(6,1);temp.SetValue(7,1);','temp.SetValue(5,arenaSweepCount);temp.SetValue(6,2);temp.SetValue(7,1);')
 return text


def expected_files(kind):
 name,snake=NAMES[kind];result={}
 for p in [BASE/'operator.json']+[x for sub in ('op_host','op_kernel') for x in (BASE/sub).iterdir() if x.is_file()]:
  rel=str(p.relative_to(BASE)).replace(OLD_S,snake);text=p.read_text()
  if p.parent.name=='op_kernel':text='#define STAGE4V_FAULT '+('1' if kind=='fault' else '0')+'\n'+(fault(text) if kind=='fault' else aggregate(text))
  if kind=='fault' and p.name=='host_common.h':text=once(text,'*mode>5','*mode>7').replace('output_mode must be 0..5','output_mode must be 0..7')
  result[rel]=text.replace(OLD,name).replace(OLD_S,snake).encode()
 return result



def expected_runners(kind):
 name,snake=NAMES[kind]
 src=BASE.parent/('stage4v_fullk_perf_explore' if kind=='candidate' else 'stage4v')/'runner'
 files={}
 for f in src.iterdir():
  if not f.is_file():continue
  text=f.read_text().replace(OLD,name).replace(OLD_S,snake)
  if f.name=='CMakeLists.txt':text=text.replace('fullk_explore_runner','guard_candidate_runner').replace('stage4v_runner','guard_fault_runner')
  if f.name=='runner.cpp':
   if kind=='candidate':text=text.replace('\\"kernel_guards_unchanged\\":true','\\"kernel_guards_unchanged\\":false,\\"guard_coverage_unchanged\\":true,\\"arena_detector\\":\\"XOR_OR_U32\\"')
   else:
    text=text.replace('if(mode<0||mode>5||limit<0||limit>76||tile < -1||tile>=76)','if(!hostOnly&&(mode<2||mode>7||limit!=1||tile!=0))')
    text=text.replace('if(benchmark&&(mode!=0||limit!=0||tile!=-1))','if(benchmark)')
    text=text.replace('EXECUTION_COMPLETED_GOLDEN_MATCH','FAULT_PROTOCOL_MATCH')
    text=text.replace('\\"strict_golden_match\\":true','\\"strict_golden_match\\":false,\\"fault_witness_match\\":true')
    text=text.replace('\\"cpu_fallback\\":false','\\"cpu_fallback\\":false,\\"sync_completed\\":true')
  files['runner/'+f.name]=text.encode()
 return files


def materialize(dest,kind,identity,write):
 dest.mkdir(parents=True,exist_ok=False);records=[]
 for rel,raw in expected_files(kind).items():
  f=dest/rel;f.parent.mkdir(exist_ok=True);f.write_bytes(raw);records.append(identity(f))
 old=(BASE/'op_kernel'/ (OLD_S+'.cpp')).read_text();new=(dest/'op_kernel'/(NAMES[kind][1]+'.cpp')).read_text()
 diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='frozenStage4V',tofile=kind))
 (dest/'source.diff').write_text(diff)
 audit=dict(kind=kind,files=records,diff=identity(dest/'source.diff'),status='EXACT_ARENA_XOR_OR_ONLY_PLUS_OP_RENAME' if kind=='candidate' else 'FAULT_ONLY_NOT_PERFORMANCE_ELIGIBLE')
 write(dest/'audit.json',audit);return audit


def verify_audit(a,verify):
 expected=expected_files(a['kind']);paths={}
 for f in a['files']:
  path=verify(f);rel='operator.json' if path.name=='operator.json' else path.parent.name+'/'+path.name
  if rel in paths:raise ValueError('duplicate generated source')
  paths[rel]=path.read_bytes()
 if paths!=expected:raise ValueError('generated source audit differs')
 old=(BASE/'op_kernel'/(OLD_S+'.cpp')).read_text()
 new=expected['op_kernel/'+NAMES[a['kind']][1]+'.cpp'].decode()
 diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='frozenStage4V',tofile=a['kind']))
 if verify(a['diff']).read_text()!=diff:raise ValueError('source diff document differs')

if __name__=='__main__':
 import sys
 # CLI used only in local compile check; no history or device admission.
 from sg_common import identity,write
 materialize(Path(sys.argv[1]),sys.argv[2],identity,write)
