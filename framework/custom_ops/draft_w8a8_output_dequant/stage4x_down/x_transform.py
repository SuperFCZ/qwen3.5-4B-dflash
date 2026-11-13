"""Exact shape/ABI/name transform from accepted Full-K B-NZ; no math/sync edits."""
from pathlib import Path
import re,difflib
HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'stage4v_fullk_b_nz'
NAMES={'candidate':('DFlashDownFullKBNz','d_flash_down_full_kb_nz'),'fault':('DFlashDownFullKBNzFault','d_flash_down_full_kb_nz_fault')}
# Simultaneous source-token mapping: Gate A40960 -> Down A155648, while
# Gate Scale155648 -> Down Scale20480. Never cascade .replace() operations:
# the newly produced A byte count must not be converted again as a Scale.
NUMBERS={'2560':'9728','19456':'2560','76':'10','75':'9','80':'304','40960':'155648','155648':'20480','49807360':'24903680','311296':'40960','622592':'81920'}


def files(kind):
    out={}
    for f in (BASE/kind).rglob('*'):
        if not f.is_file():continue
        rel=str(f.relative_to(BASE/kind)).replace('d_flash_full_kb_nz','d_flash_down_full_kb_nz')
        s=f.read_text().replace('DFlashFullKBNz','DFlashDownFullKBNz').replace('d_flash_full_kb_nz','d_flash_down_full_kb_nz')
        s=s.replace('STAGE4V','STAGE4X').replace('S4V','S4X')
        s=re.sub(r'\b(?:'+'|'.join(NUMBERS)+r')\b',lambda m:NUMBERS[m[0]],s)
        if f.name=='host_common.h':s=s.replace('{10,256}','{2560}')
        if f.name=='runner.cpp':
            s=s.replace('r.s.Make({10,256},{10,256}','r.s.Make({2560},{2560}')
            if kind=='candidate':s=s.replace('limit!=0||(tile!=0&&tile!=-1)','(limit!=0&&limit!=2)||(tile!=0&&tile!=-1)')
            s=s.replace('GuardedBuffer buffer; aclTensor *desc=nullptr;', 'GuardedBuffer buffer; aclTensor *desc=nullptr; double upload_ms=0;')
            s=s.replace('buffer.Allocate(bytes); if(data)buffer.CopyIn(*data);',
                'buffer.Allocate(bytes); if(data){auto start=std::chrono::steady_clock::now();buffer.CopyIn(*data);upload_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();}')
            s=s.replace('std::vector<uint8_t> first;', 'std::vector<uint8_t> first;std::vector<double> output_copyout_ms;')
            s=s.replace('auto out=r.y.buffer.CopyOut();WriteBytes',
                'auto start=std::chrono::steady_clock::now();auto out=r.y.buffer.CopyOut();output_copyout_ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());WriteBytes')
            s=s.replace('timing.Write(report);report<<"}\\n";if(!report)',
                'report<<",\\"initial_copyin_wall_ms\\":"<<(r.x.upload_ms+r.w.upload_ms+r.s.upload_ms)<<",\\"initial_input_payload_bytes\\":25079808,\\"output_copyout_wall_ms\\":[";for(size_t i=0;i<output_copyout_ms.size();++i)report<<(i?",":"")<<output_copyout_ms[i];report<<"]";timing.Write(report);report<<"}\\n";if(!report)')
        out[rel]=s.encode()
    return out


def verify_repository():
    for kind in NAMES:
        actual={str(f.relative_to(HERE/kind)):f.read_bytes() for f in (HERE/kind).rglob('*') if f.is_file()}
        if actual!=files(kind):raise ValueError('Down shape-only source audit failed: '+kind)


def materialize(root,kind,identity,write):
    verify_repository();root.mkdir();records=[]
    for name,data in files(kind).items():
        if name!='operator.json' and not name.startswith(('op_kernel/','op_host/')):continue
        f=root/name;f.parent.mkdir(exist_ok=True);f.write_bytes(data);records.append(identity(f))
    old=next((BASE/kind/'op_kernel').glob('*.cpp')).read_text();new=next((root/'op_kernel').glob('*.cpp')).read_text()
    diff=root/'source.diff';diff.write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='AcceptedGateFullKBNz',tofile='IndependentDownFullKBNz')))
    audit=dict(status='EXACT_SHAPE_ABI_NAME_TRANSFORM_SYNCHRONIZATION_GUARDS_UNCHANGED',files=records,diff=identity(diff),kind=kind)
    write(root/'audit.json',audit);return audit
