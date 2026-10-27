"""Host diagnostics only; does not select a fallback Core source."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import tile_common as c
import run_stage4a as run
from build_tiles import registration_evidence


def test_actual_core_probe_and_conservative_coupled_policy(tmp_path):
 compiler=shutil.which('c++')
 if not compiler:pytest.skip('no local compiler')
 host=(HERE/'op_host/host_common.h').read_text()
 guard=host[host.index('    const auto coreProbe='):host.index('    const uint32_t blocks=')]
 assert 'S4ASelectSchedulingLimit(coreProbe)' in guard and 'if(!cores)' in guard
 src=tmp_path/'probe.cpp';src.write_text('#include <cstdlib>\n#include "core_probe.h"\n'+r'''
struct Basic { uint32_t a,b,c;uint32_t GetCoreNumAic(){return a;}uint32_t GetCoreNumAiv(){return b;}uint32_t GetCoreNum(){return c;} };
struct Full:Basic { int GetSocVersion(){return 17;}long long arch=2002;long long GetCurNpuArch(){return arch;} };
int S4AReject(const char *s){std::fprintf(stderr,"STAGE4A_TILING_REJECT: %s\n",s);return 7;}
template<class P>int Gate(P& platform){
'''+guard+r'''
return 0;}
int main(int argc,char**argv){
 Basic p{uint32_t(std::atoi(argv[1])),uint32_t(std::atoi(argv[2])),uint32_t(std::atoi(argv[3]))};
 if(argc==5){Full full;full.a=p.a;full.b=p.b;full.c=p.c;full.arch=std::atoll(argv[4]);return Gate(full);}return Gate(p);
}
''')
 exe=tmp_path/'probe';r=subprocess.run([compiler,'-std=c++14','-I'+str(HERE/'op_host'),str(src),'-o',str(exe)],capture_output=True,text=True)
 assert r.returncode==0,r.stderr
 for a,b,all_ in [(0,0,8),(8,0,8),(8,16,16),(1,0,1),(0,8,0)]:
  p=subprocess.run([str(exe),str(a),str(b),str(all_)],capture_output=True,text=True)
  assert p.returncode==7 # missing architecture never selects a fallback
  assert f'STAGE4A_CORE_PROBE aic={a} aiv={b} all={all_}\n' in p.stderr
  assert 'soc=API_UNAVAILABLE' in p.stderr and 'arch=API_UNAVAILABLE' in p.stderr
  if p.returncode:assert p.stderr.index('STAGE4A_CORE_PROBE')<p.stderr.index('STAGE4A_TILING_REJECT')
 for a,b,all_,arch in [(8,8,7,2002),(8,8,8,2002),(4,4,3,2002),(8,0,7,2002),(8,7,7,2002),(8,8,9,2002),(0,0,7,2002),(8,8,0,2002),(1,1,1,2002),(8,8,7,2201)]:
  p=subprocess.run([str(exe),str(a),str(b),str(all_),str(arch)],capture_output=True,text=True)
  valid=arch==2002 and a>0 and b>0 and all_>0 and a==b and all_<=a
  assert p.returncode==(0 if valid else 7)
  assert f'STAGE4A_CORE_POLICY aic={a} aiv={b} all={all_} selected={all_ if valid else 0}' in p.stderr


def test_nested_macro_logs_echoes_and_absence(tmp_path):
 project=tmp_path/'CustomOp';(project/'build_out/op_kernel').mkdir(parents=True)
 log=tmp_path/'build.log';log.write_text('build successful\n#pragma message("STAGE4A_REGIST_EXPANSION=" S4A_TEXT(REGIST_MATMUL_OBJ(...)))\n')
 report=registration_evidence(project,log)
 assert report['status']=='EXPANSION_NOT_EMITTED' and len(report['source_context'])==1 and not report['evidence']
 nested=project/'build_out/op_kernel/compiler.log';nested.write_text('warning: STAGE4A_REGIST_EXPANSION=mm.Init(tiling, pipe); [-W#pragma-messages]\n')
 report=registration_evidence(project,log)
 assert report['status']=='COMPILER_EXCERPTS_COLLECTED' and report['evidence'][0]['log']==c.identity(nested)
 nested.unlink();large=project/'build_out/large.log'
 with large.open('wb') as f:f.truncate(8*1024*1024+1)
 report=registration_evidence(project,log)
 assert report['status']=='EXPANSION_SEARCH_INCOMPLETE' and report['skipped_logs'][0]['reason']=='BYTE_LIMIT'


def test_host_only_driver_never_enters_numerical_or_parallel_stage(tmp_path):
 runner=tmp_path/'runner';runner.write_text('fixture only');bp=tmp_path/'build.json';out=tmp_path/'smoke';out.mkdir()
 files={}
 for n in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin'):
  p=tmp_path/n;p.write_bytes(b'fixture');files[n]=c.identity(p)
 r=dict(settings={},sources=[],evidence={},device=0,npu='NOT_RUN')
 c.write(bp,dict(status='STAGE4A_BUILT_NPU_NOT_RUN',settings={},sources=[],evidence={},device=0,runner=c.identity(runner),opp=str(tmp_path/'opp'),installed={},cann_compile='fixture',installation='fixture'))
 calls=[]
 def command(argv,log,*rest):
  calls.append(argv)
  assert argv[-1]=='--host-only' and argv[-3:-1]==[21,1]
  ex=dict(status='HOST_TILING_ONLY',host_only=True,npu='NOT_RUN',device=0,output_mode=21,core_limit=1,workspace_bytes=10000)
  c.write(log.parent/'execution.json',ex)
  log.write_text('STAGE4A_CORE_PROBE aic=8 aiv=8 all=7\nSTAGE4A_ARCH_PROBE arch=2002\nSTAGE4A_CORE_POLICY aic=8 aiv=8 all=7 selected=7\nSTAGE4A_SCHEDULE coreLimit=1 blocks=1 cores=7 system=4096 arena=4096 stride=5120 user=5120 total=9216 mode=21\nscratch=239104 userUB=12288 ubGuards=2560 reserve=8192 single=16/64/128 base=16/64/128')
  plan=run.plan_from_log(log.read_text(),21,1)
  c.write(log.parent/'resource-admission.json',run.admission_expected(ex,plan,True))
 args=SimpleNamespace(multi_build_report=bp,device=0,cann_root=tmp_path/'sdk',diagnostic_groups=[],host_only=True)
 with patch.object(run,'installed',return_value={}),patch.object(run,'verify_case',return_value={}),patch.object(run,'command',side_effect=command),patch.object(run,'compare',side_effect=AssertionError('must not compare numerical outputs')):
  run.smoke(args,out,r,dict(files=files))
 assert len(calls)==1 and r['status']=='STAGE4A_HOST_TILING_ONLY' and r['npu']=='NOT_RUN'
 assert 'outputs' not in r['runs'][0]


def policy_log(mode=21):
 return f'STAGE4A_CORE_PROBE aic=8 aiv=8 all=7\nSTAGE4A_ARCH_PROBE arch=2002\nSTAGE4A_CORE_POLICY aic=8 aiv=8 all=7 selected=7\nSTAGE4A_SCHEDULE coreLimit=1 blocks=1 cores=7 system=4096 arena=4096 stride=5120 user=5120 total=9216 mode={mode}\nscratch=239104 userUB=12288 ubGuards=2560 reserve=8192 single=16/64/128 base=16/64/128'


def test_host_only_proof_is_required_bound_and_not_numerical_pass(tmp_path):
 with pytest.raises(ValueError,match='required'):run.validate_host_probe(None,{}, {})
 folder=tmp_path/'probe';folder.mkdir();log=folder/'runner.log';log.write_text(policy_log())
 plan=c.plan_from_log(log.read_text(),21,1)
 ex=dict(status='HOST_TILING_ONLY',host_only=True,npu='NOT_RUN',device=0,output_mode=21,core_limit=1,workspace_bytes=10000)
 c.write(folder/'execution.json',ex);c.write(folder/'resource-admission.json',run.admission_expected(ex,plan,True));c.write(folder/'case.json',{})
 build=tmp_path/'build.json';c.write(build,{})
 base=dict(settings={},sources=[],evidence={},device=0)
 hp=dict(status='STAGE4A_HOST_TILING_ONLY',phase='complete',host_only=True,npu='NOT_RUN',multi_build_report=c.identity(build),**base,
  case=c.identity(folder/'case.json'),core_policy=plan['core_policy'],
  commands={'l1_m21':{'log_identity':c.identity(log)}},runs=[dict(status='HOST_TILING_ONLY',mode=21,core_limit=1,plan=plan,
  execution=c.identity(folder/'execution.json'),resource_admission=c.identity(folder/'resource-admission.json'))])
 path=folder/'report.json';c.write(path,hp)
 with patch.object(run,'verify_case'):
  assert run.validate_host_probe(path,c.identity(build),base)==dict(aic=8,aiv=8,all=7,selected=7,arch=2002)
  with pytest.raises(ValueError,match='bind'):run.validate_host_probe(path,{},base)
  hp['core_policy']['selected']=8;c.write(path,hp)
  with pytest.raises(ValueError,match='proof differs'):run.validate_host_probe(path,c.identity(build),base)


def test_runtime_core_policy_cannot_be_missing_or_inconsistent():
 text=policy_log()
 assert c.plan_from_log(text,21,1)['core_policy']['selected']==7
 for bad in (text.replace('arch=2002','arch=2201'),text.replace('aiv=8','aiv=0'),text.replace('aiv=8','aiv=4'),
             text.replace('selected=7','selected=8'),text.replace('all=7','all=9'),text.replace('cores=7','cores=8'),
             '\n'.join(x for x in text.splitlines() if 'CORE_POLICY' not in x)):
  with pytest.raises(ValueError):c.plan_from_log(bad,21,1)


def test_installed_header_binding_uses_only_compiler_dependencies(tmp_path):
 from build_tiles import core_header_binding
 project=tmp_path/'CustomOp';deps=project/'build_out/op_host';deps.mkdir(parents=True)
 cann=tmp_path/'sdk';cann.mkdir();header=cann/'platform_ascendc.h'
 header.write_text('unsigned GetCoreNum() const;\nunsigned GetCoreNumAic() const;\n')
 (deps/'host.cpp.o.d').write_text('host.o: '+str(header)+'\n')
 result=core_header_binding(project,cann,tmp_path)
 assert result['status']=='INSTALLED_HEADER_BOUND' and result['headers'][0]['source']==c.identity(header)
 assert result['headers'][0]['api_lines'][0]['line']==1
