"""Pure helpers for the isolated Simulator adapter. No device execution."""
import csv
import json
from pathlib import Path
import re
import struct


def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('runner source anchor changed: '+old)
    return text.replace(old,new,1)


def runner_sources(here,out):
    out.mkdir(parents=True,exist_ok=False)
    main=(here/'test/main.cpp').read_text();common=(here/'test/runner_common.h').read_text()
    # Private copies only: one ACLNN invocation, no warmup/benchmark. Keep the
    # original guards, NZ tensor ABI, input-readonly and workspace allocation.
    main=replace_once(main,'constexpr int kRepetitions = 2;','constexpr int kRepetitions = 1;')
    main=replace_once(main,'int main(int argc, char **argv)','int SimulatorApplication(int argc, char **argv)')
    for name in ('d_flash_group_quant_linear_contract.h','d_flash_group_quant_linear_build_config.h'):
        main=replace_once(main,'"../op_host/'+name+'"',json.dumps(str(here/'op_host'/name)))
    main=replace_once(main,r'\"status\":\"PASS\",\"runtime\":\"AscendCL ACLNN\"',r'\"status\":\"SIMULATOR_COMPLETED\",\"runtime\":\"AscendCL camodel\"')
    main=replace_once(main,r'\"cpu_fallback\":false',r'\"cpu_fallback\":false,\"simulator\":true,\"npu_execution\":\"NOT_RUN\"')
    common=replace_once(common,'for (int repeat = 0; repeat < 2; ++repeat)','for (int repeat = 0; repeat < 1; ++repeat)')
    common=replace_once(common,r'\"correctness_before_calls\":2',r'\"correctness_before_calls\":1')
    common=replace_once(common,'isolated host launch plus stream synchronization; transfers/checks excluded','Simulator only; host duration is not hardware wall-time')
    main+='''
#include "runtime_guard.h"
int main(int argc,char **argv) {
    try {
        if(argc==3 && std::string(argv[1])=="--link-check") {
            VerifySimulatorRuntime(argv[2]); return 0;
        }
        if(argc!=6) throw std::runtime_error("Simulator accepts exactly DEVICE DIR M K N; timing disabled");
        VerifySimulatorRuntime(argv[2]);
        if(!freopen((std::string(argv[2])+"/application.log").c_str(),"w",stdout)) throw std::runtime_error("cannot save application log");
        return SimulatorApplication(argc,argv);
    } catch(const std::exception &e) {
        std::cerr<<"SIMULATOR BLOCKED: "<<e.what()<<"\\n"; return 1;
    }
}
'''
    (out/'main.cpp').write_text(main);(out/'runner_common.h').write_text(common)
    return out/'main.cpp'


def debug_cmake(path):
    # APIs documented by msopprof's custom-op quick start. This modifies only
    # the freshly generated project, never the accepted OPP or tracked CMake.
    original=path.read_text()
    prefix='''# Isolated Simulator debug symbols; keep existing optimization flags.
if(COMMAND add_ops_compile_options)
  add_ops_compile_options(ALL OPTIONS -g)
elseif(COMMAND npu_op_kernel_options)
  npu_op_kernel_options(ascendc_kernels ALL OPTIONS -g)
else()
  message(FATAL_ERROR "No documented Kernel debug option hook; do not build without line info")
endif()
'''
    path.write_text(prefix+original)


def validate_help(text):
    required=('--soc-version','--output','--launch-count','--core-id','--timeout','--aic-metrics','--kernel-name')
    missing=[x for x in required if x not in text]
    if missing:raise ValueError('unsupported installed simulator CLI: '+str(missing))


def profile_command(tool,runner,out,case,shape,timeout_minutes):
    if type(timeout_minutes) is not int or not 1<=timeout_minutes<=2880:raise ValueError('invalid simulator timeout')
    return [str(tool),'simulator','--soc-version=Ascend310P3','--aic-metrics=PipeUtilization',
            '--kernel-name=DFlashGroupQuantLinear|d_flash_group_quant_linear','--launch-count=1','--core-id=0','--timeout='+str(timeout_minutes),'--output='+str(out),
            str(runner),'0',str(case),*map(str,shape)]


def smoke_inputs(root):
    """New M16/K256/N64 basis fixture. Never slices or rewrites Gate weights."""
    import numpy as np
    root.mkdir(parents=True,exist_ok=False)
    x=np.zeros((16,256),dtype='<f2');q=(((np.arange(64)[:,None]*17+np.arange(256)[None,:]*7)%127)*2-127).astype('i1')
    s=np.array([[2.**(-3+(n%3)+g) for n in range(64)] for g in range(2)],dtype='<f2')
    positions=(0,31,32,127,128,159,160,255)*2
    for row,k in enumerate(positions):x[row,k]=1 if row%2==0 else -1
    nz=q.reshape(4,16,8,32).transpose(2,0,1,3).copy()
    w=(q.reshape(64,2,128).astype('<f2')*s.T[:,:,None]).reshape(64,256)
    expected=np.stack([(w[:,k]*x[row,k]).astype('<f2') for row,k in enumerate(positions)])
    for name,a in (('x.bin',x),('w_nz.bin',nz),('s_gn.bin',s),('expected.bin',expected)):a.tofile(root/name)


def elf_machine(path):
    with Path(path).open('rb') as f:h=f.read(20)
    if len(h)<20 or h[:4]!=b'\x7fELF' or h[4]!=2 or h[5]!=1:raise ValueError('need 64-bit little-endian ELF: '+str(path))
    return struct.unpack('<H',h[18:20])[0]


def artifacts(root,record):
    found={key:[] for key in ('trace','code_hotspots','instructions')}
    for path in sorted(root.rglob('*')):
        if not path.is_file():continue
        if path.name=='trace.json':
            data=json.loads(path.read_text());events=data.get('traceEvents',[]) if isinstance(data,dict) else data
            if not isinstance(events,list) or not any(isinstance(e,dict) and e.get('ph')=='X' and 'dur' in e for e in events):raise ValueError('empty/malformed simulator trace: '+str(path))
            found['trace'].append(record(path,root))
        elif path.suffix=='.csv' and ('code_exe' in path.name or 'instr_exe' in path.name):
            with path.open(newline='',encoding='utf-8-sig') as f:
                reader=csv.reader(f);headers=next(reader,None);first=next(reader,None)
            if not headers or not first:raise ValueError('empty simulator CSV: '+str(path))
            found['code_hotspots' if 'code_exe' in path.name else 'instructions'].append(dict(**record(path,root),headers=headers))
    return dict(status='COLLECTED' if all(found.values()) else 'ARTIFACTS_INCOMPLETE',files=found,
                cycles_are_wall_time=False,tile_mapping='UNRESOLVED_TILE_MAPPING',
                limitation='core-id selects parsed cores, not executed N/K tiles; no cropped or synthetic trace is generated')


def focus_lines(kernel):
    groups={'DataCopy':'DataCopy(', 'Cast':'AscendC::Cast(', 'Brcb':'AscendC::Brcb(', 'Mul':'AscendC::Mul(',
            'Matmul':'mm.', 'PIPE_ALL':'PipeBarrier<PIPE_ALL>', 'Event':'Flag<'}
    lines=kernel.read_text().splitlines()
    return {k:[i for i,s in enumerate(lines,1) if token in s] for k,token in groups.items()}
