"""Exact, bounded derivation from the immutable Stage4B full-Guard sources.

No SDK source is embedded. Regeneration is a developer operation, not a build
fallback. audit() rejects any additional Kernel/Host/Runner change.
"""
from pathlib import Path
import difflib
import hashlib

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'stage4b'

def once(s, old, new):
    if s.count(old) != 1:
        raise ValueError('baseline source anchor drift: ' + old[:100])
    return s.replace(old, new, 1)

def rename(s):
    return s.replace('DFlashW8A8FullGate', 'DFlashW8A8N128Gate').replace('d_flash_w8_a8_full_gate', 'd_flash_w8_a8_n128_gate').replace('STAGE4B', 'STAGE4J').replace('Stage4B', 'Stage4J').replace('S4B', 'S4J').replace('stage4b', 'stage4j')

def kernel(s):
    s = rename(s)
    s = s.replace('Full 304 N64', 'Full 152 N128 T1').replace('239104', '226816').replace('>304', '>152')
    # Only output/UB dimensions change; A and Arena's 1024B redzones stay fixed.
    for name in ('p0FinalRaw', 'p1Raw'):
        s = s.replace(f'InitBuffer({name},2048+2*guardBytes)', f'InitBuffer({name},4096+2*guardBytes)')
        s = s.replace(f'{name}.Get<uint32_t>(),2048', f'{name}.Get<uint32_t>(),4096')
    for name in ('castRaw', 'sumRaw'):
        s = s.replace(f'InitBuffer({name},4096+2*guardBytes)', f'InitBuffer({name},8192+2*guardBytes)')
        s = s.replace(f'{name}.Get<uint32_t>(),4096', f'{name}.Get<uint32_t>(),8192')
    s = s.replace('SetSize(1024)', 'SetSize(2048)').replace('0.0f,1024)', '0.0f,2048)').replace('CAST_NONE,1024)', 'CAST_NONE,2048)').replace('sum,converted,1024)', 'sum,converted,2048)').replace('slot/16.0f),1024)', 'slot/16.0f),2048)')
    s = s.replace('304*20*8192', '152*20*16384').replace('304*20*64', '152*20*128').replace('(slot*20+g)*8192', '(slot*20+g)*16384').replace('(slot*20+g)*64', '(slot*20+g)*128')
    s = s.replace('slot<304', 'slot<152').replace('nb<4', 'nb<8').replace('slot*64+', 'slot*128+')
    s = once(s, 'if(td.outputMode>21||', 'if(td.tileId < -1 || td.tileId >= 152 || td.outputMode>21||')
    s = once(s, 'for(uint32_t slot=core;slot<152;slot+=td.launchBlocks) {', 'for(uint32_t slot=core;slot<152;slot+=td.launchBlocks) {\n    if(td.tileId>=0 && slot!=static_cast<uint32_t>(td.tileId))continue; // bounded: other columns remain Poison')
    return s

def host(s):
    s = rename(s).replace('304','152').replace('{152,20,128,64}', '{152,20,128,128}').replace('{152,20,64}', '{152,20,128}')
    s = s.replace('16,64,128', '16,128,128').replace('()!=64', '()!=128').replace('64*128', '128*128').replace('16*64*', '16*128*').replace('+64*8', '+128*8')
    s = s.replace('12288', '24576').replace('12 KiB', '24 KiB').replace('scratch>241664', 'scratch!=226816').replace('tileN=64', 'tileN=128')
    s = once(s, '#include "core_probe.h"', '#include "core_probe.h"\n#include "t1_contract.h"')
    s = once(s, 'const auto y=c->GetOutputShape(0);', 'const auto tile=attrs->GetAttrPointer<int64_t>(2);\n    if(!tile||*tile < -1||*tile>=152)return S4JReject("tile_id must be -1 or 0..151");\n    std::fprintf(stderr,"STAGE4J_SCOPE tile=%lld\\n",(long long)*tile);\n    const auto y=c->GetOutputShape(0);')
    s = once(s, 'auto &p=td.cubeTilingData;', 'auto &p=td.cubeTilingData;\n    if(!S4JAcceptedT1(p,ub,l1,l0a,l0b,l0c,system,scratch))return S4JReject("actual TilingContext differs from accepted T1 resource contract");')
    s = once(s, 'td.set_outputMode(*mode);', 'td.set_tileId(*tile);td.set_outputMode(*mode);')
    return s

def runner(s):
    s = rename(s).replace('>304', '>152').replace('uint64_t(304)', 'uint64_t(152)').replace('limit>304', 'limit>152')
    s = s.replace('{304,20,128,64}', '{152,20,128,128}').replace('{304,20,64}', '{152,20,128}').replace('tile_count\\\":304','tile_count\\\":152')
    s = s.replace('argc!=5&&argc!=6','argc!=6&&argc!=7').replace('hostOnly=argc==6','hostOnly=argc==7').replace('std::string(argv[5])','std::string(argv[6])').replace('CORE_LIMIT [--host-only]', 'CORE_LIMIT TILE_ID [--host-only]')
    s = once(s, 'Session r;size_t end=0;', 'size_t tileEnd=0;std::string tileArg=argv[5];int64_t tile=std::stoll(tileArg,&tileEnd);\n        if(tileEnd!=tileArg.size()||tile < -1||tile>=152)throw std::runtime_error("invalid tile_id");\n        if(hostOnly&&tile!=-1)throw std::runtime_error("Host-only requires full scope");\n        Session r;size_t end=0;')
    s = s.replace('mode,limit,r.y.desc', 'mode,limit,tile,r.y.desc')
    s = once(s, 'std::vector<uint8_t> first;', 'std::vector<uint8_t> first;int repeatIndex=-1;')
    s = once(s, 'uint8_t pattern=i?0xff:0x7f;', 'repeatIndex=i;uint8_t pattern=i?0xff:0x7f;')
    s = once(s, 'if(first.empty())first=out;else if(out!=first)throw std::runtime_error("repeat output drift");', '''// Bounded repeats have different untouched Poison; compare only written columns.
            for(size_t row=0;row<16;++row)for(size_t col=0;col<19456;++col) {
                const size_t off=2*(row*19456+col);const bool active=tile<0||col/128==size_t(tile);
                if(!active) {const uint8_t pattern=repeatIndex?0xff:0x7f;
                    if(out[off]!=pattern||out[off+1]!=pattern)throw std::runtime_error("bounded inactive column changed");}
                else if(!first.empty()&&(out[off]!=first[off]||out[off+1]!=first[off+1]))throw std::runtime_error("repeat output drift");
            }
            if(first.empty())first=out;''')
    s = once(s, 'report<<",\\\"output_mode\\\":"<<mode<<",\\\"core_limit\\\":"<<limit;', 'report<<",\\\"output_mode\\\":"<<mode<<",\\\"core_limit\\\":"<<limit<<",\\\"tile_id\\\":"<<tile;')
    return s

def derived():
    result={}
    for sub in ('op_host','op_kernel'):
        for p in (BASE/sub).iterdir():
            text=p.read_text()
            if sub=='op_kernel': text=kernel(text)
            elif p.name=='host_common.h': text=host(text)
            else:
                text=rename(text)
                if p.name.endswith('_tiling.h'):
                    text=once(text,'TILING_DATA_FIELD_DEF(uint32_t, outputMode);','TILING_DATA_FIELD_DEF(uint32_t, outputMode);\nTILING_DATA_FIELD_DEF(int32_t, tileId);')
                elif p.suffix=='.cpp':
                    text=once(text,'Attr("core_limit").AttrType(OPTIONAL).Int(1);','Attr("core_limit").AttrType(OPTIONAL).Int(1);\n  Attr("tile_id").AttrType(OPTIONAL).Int(-1);')
            result[sub+'/'+rename(p.name)]=(p,text)
    for name in ('CMakeLists.txt','runner.cpp'):
        p=BASE/'test'/name
        result['test/'+name]=(p,runner(p.read_text()) if name=='runner.cpp' else rename(p.read_text()))
    return result

def audit():
    rows=[];diff=[]
    for rel,(base,expected) in derived().items():
        path=HERE/rel;actual=path.read_text()
        if actual!=expected:raise ValueError('unapproved source change: '+rel)
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append(dict(baseline=str(base),baseline_sha256=sha(base),candidate=str(path),candidate_sha256=sha(path)))
        diff.extend(difflib.unified_diff(base.read_text().splitlines(True),actual.splitlines(True),fromfile='stage4b/'+str(base.relative_to(BASE)),tofile='stage4j/'+rel))
    return dict(status='EXACT_ALLOWED_TRANSFORM',files=rows,unchanged='Matmul lifecycle, per-group workspace binding, synchronization, five full Guard scans, Arena formula, ordered FP32 additions'), ''.join(diff)

if __name__=='__main__':
    print(audit()[0]['status'])
