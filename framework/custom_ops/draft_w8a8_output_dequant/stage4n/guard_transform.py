"""Exact N128 group-only scalar transformation and separate fault instrumentation."""
import sys,difflib
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4j'))
import j_common as j
BASE=HERE.parent/'stage4j'
SCAN='''        if(!CheckGuard(scratchRaw.Get<uint32_t>(),td.localMemSize)||
           !CheckGuard(p1Raw.Get<uint32_t>(),4096)||!CheckGuard(p0FinalRaw.Get<uint32_t>(),4096)||
           !CheckGuard(castRaw.Get<uint32_t>(),8192)||!CheckGuard(sumRaw.Get<uint32_t>(),8192))return;
'''
HELPER='''#if defined(__NPU_ARCH__)
static_assert(g_coreType == AscendC::MIX, "Stage4N requires non-split coupled scalar UB access");
#endif
__aicore__ inline uint32_t GuardDelta(LocalTensor<uint32_t> raw,uint32_t bytes) {
    uint32_t delta=0;
    for(uint32_t i=0;i<64;++i) {
        delta |= raw.GetValue(i)^guardWord;
        delta |= raw.GetValue((guardBytes+bytes)/4+i)^guardWord;
    }
    return delta;
}
__aicore__ inline uint32_t GroupDelta(LocalTensor<uint32_t> scratch,uint32_t bytes,
    LocalTensor<uint32_t> p1,LocalTensor<uint32_t> p0,LocalTensor<uint32_t> cast,LocalTensor<uint32_t> sum) {
    uint32_t delta=GuardDelta(scratch,bytes);
    delta |= GuardDelta(p1,4096);
    delta |= GuardDelta(p0,4096);
    delta |= GuardDelta(cast,8192);
    delta |= GuardDelta(sum,8192);
    return delta;
}
'''
CALL='GroupDelta(scratchRaw.Get<uint32_t>(),td.localMemSize,p1Raw.Get<uint32_t>(),p0FinalRaw.Get<uint32_t>(),castRaw.Get<uint32_t>(),sumRaw.Get<uint32_t>())'
CHECK='        if('+CALL+'!=0)return;\n'

def transform(raw):
 text=raw.decode();start=text.find(SCAN)
 if text.count(SCAN)!=1 or not text[:start].endswith('SetFlag<HardEvent::V_S>(groupReady);WaitFlag<HardEvent::V_S>(groupReady);\n'):raise ValueError('group guard anchor differs')
 text=text.replace(SCAN,CHECK)
 return text.replace('__aicore__ inline bool CheckArena(',HELPER+'__aicore__ inline bool CheckArena(',1).encode()

# Diagnostic OPP only. Mode0 exhausts all words/bits at all20 actual group boundaries.
# Modes1..20 inject a persistent error at that group; no output should be written.
FAULT='''        if(td.tileId!=0 || core!=0 || td.outputMode>20)return;
        if(GROUP_CALL!=0)return;
        if(td.outputMode==0) {
            for(uint32_t buffer=0;buffer<5;++buffer) {
                auto raw=buffer==0?scratchRaw.Get<uint32_t>():buffer==1?p1Raw.Get<uint32_t>():buffer==2?p0FinalRaw.Get<uint32_t>():buffer==3?castRaw.Get<uint32_t>():sumRaw.Get<uint32_t>();
                const uint32_t bytes=buffer==0?td.localMemSize:buffer<3?4096:8192;
                for(uint32_t side=0;side<2;++side)for(uint32_t word=0;word<64;++word)for(uint32_t bit=0;bit<32;++bit) {
                    const uint32_t index=side?(256+bytes)/4+word:word;
                    raw.SetValue(index,guardWord^(uint32_t(1)<<bit));
                    const uint32_t observed=GROUP_CALL;
                    raw.SetValue(index,guardWord);
                    if(observed!=(uint32_t(1)<<bit))return;
                }
            }
        } else if(g+1==td.outputMode) {
            // Integration abort test, including Group19: corruption must stop this group.
            sumRaw.Get<uint32_t>().SetValue((256+8192)/4+63,guardWord^0x80000001U);
        }
'''

def diagnostic(raw):
 text=transform(raw).decode()
 witness='''        const uint32_t faultDelta=GROUP_CALL;
        if(faultDelta!=0) {
            if(td.outputMode==g+1 && faultDelta==0x80000001U) {
                auto meta=converted.ReinterpretCast<uint32_t>();
                meta.SetValue(0,0x534e4641U);meta.SetValue(1,g);meta.SetValue(2,faultDelta);meta.SetValue(3,640);
                meta.SetValue(4,4);meta.SetValue(5,1);meta.SetValue(6,63);meta.SetValue(7,g+1);
                const auto faultReady=pipe.FetchEventID(HardEvent::S_MTE3);
                SetFlag<HardEvent::S_MTE3>(faultReady);WaitFlag<HardEvent::S_MTE3>(faultReady);
                GlobalTensor<uint32_t> witness;witness.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(o),8);
                DataCopy(witness,meta,8);PipeBarrier<PIPE_ALL>();
            }
            return;
        }
'''.replace('GROUP_CALL',CALL)
 return text.replace(CHECK,FAULT.replace('GROUP_CALL',CALL)+witness,1).encode()

def materialize(root,kind):
 if kind not in ('xor','fault'):raise ValueError('invalid kind')
 root.mkdir();rows=[]
 for p in [BASE/'operator.json']+[p for sub in ('op_host','op_kernel') for p in (BASE/sub).iterdir() if p.is_file()]:
  out=root/p.relative_to(BASE);out.parent.mkdir(parents=True,exist_ok=True);data=p.read_bytes()
  if p.parent.name=='op_kernel':data=(transform if kind=='xor' else diagnostic)(data)
  out.write_bytes(data);rows.append(dict(original=j.identity(p),generated=j.identity(out)))
 a=(BASE/'op_kernel/d_flash_w8_a8_n128_gate.cpp').read_text();b=(root/'op_kernel/d_flash_w8_a8_n128_gate.cpp').read_text()
 (root/'source.diff').write_text(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='Stage4J',tofile=kind)))
 r=dict(kind=kind,files=rows,diff=j.identity(root/'source.diff'),status='EXACT_GROUP_XOR_OR' if kind=='xor' else 'FAULT_INSTRUMENTED_NOT_TIMING_ELIGIBLE')
 j.write(root/'audit.json',r);return r

def verify_audit(a):
 expected=[BASE/'operator.json']+[p for sub in ('op_host','op_kernel') for p in (BASE/sub).iterdir() if p.is_file()]
 if a['kind'] not in ('xor','fault') or len(a['files'])!=len(expected) or {r['original']['path'] for r in a['files']}!={str(p.resolve()) for p in expected}:raise ValueError('audit inventory mismatch')
 for row in a['files']:
  src=j.verify(row['original']);out=j.verify(row['generated']);data=src.read_bytes()
  if src.parent.name=='op_kernel':data=(transform if a['kind']=='xor' else diagnostic)(data)
  if out.read_bytes()!=data:raise ValueError('source audit mismatch')
 original=(BASE/'op_kernel/d_flash_w8_a8_n128_gate.cpp').read_text()
 target=(transform if a['kind']=='xor' else diagnostic)(original.encode()).decode()
 expected=''.join(difflib.unified_diff(original.splitlines(True),target.splitlines(True),fromfile='Stage4J',tofile=a['kind']))
 if j.verify(a['diff']).read_text()!=expected:raise ValueError('audit diff changed')
