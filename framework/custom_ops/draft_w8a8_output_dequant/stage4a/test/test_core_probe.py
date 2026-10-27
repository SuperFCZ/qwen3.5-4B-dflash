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


def test_actual_core_probe_and_unchanged_reject_condition(tmp_path):
 compiler=shutil.which('c++')
 if not compiler:pytest.skip('no local compiler')
 host=(HERE/'op_host/host_common.h').read_text()
 guard=host[host.index('    const auto coreProbe='):host.index('    const uint32_t blocks=')]
 assert 'if(!cores || cores!=coreProbe.all)' in guard
 src=tmp_path/'probe.cpp';src.write_text('#include <cstdlib>\n#include "core_probe.h"\n'+r'''
struct Basic { uint32_t a,b,c;uint32_t GetCoreNumAic(){return a;}uint32_t GetCoreNumAiv(){return b;}uint32_t GetCoreNum(){return c;} };
struct Full:Basic { int GetSocVersion(){return 17;}int GetCurNpuArch(){return 2002;} };
int S4AReject(const char *s){std::fprintf(stderr,"STAGE4A_TILING_REJECT: %s\n",s);return 7;}
template<class P>int Gate(P& platform){
'''+guard+r'''
return 0;}
int main(int argc,char**argv){
 Basic p{uint32_t(std::atoi(argv[1])),uint32_t(std::atoi(argv[2])),uint32_t(std::atoi(argv[3]))};
 if(argc==5){Full full;full.a=p.a;full.b=p.b;full.c=p.c;return Gate(full);}return Gate(p);
}
''')
 exe=tmp_path/'probe';r=subprocess.run([compiler,'-std=c++14','-I'+str(HERE/'op_host'),str(src),'-o',str(exe)],capture_output=True,text=True)
 assert r.returncode==0,r.stderr
 for a,b,all_ in [(0,0,8),(8,0,8),(8,16,16),(1,0,1),(0,8,0)]:
  p=subprocess.run([str(exe),str(a),str(b),str(all_)],capture_output=True,text=True)
  assert p.returncode==(0 if a and a==all_ else 7)
  assert f'STAGE4A_CORE_PROBE aic={a} aiv={b} all={all_}\n' in p.stderr
  assert 'soc=API_UNAVAILABLE' in p.stderr and 'arch=API_UNAVAILABLE' in p.stderr
  if p.returncode:assert p.stderr.index('STAGE4A_CORE_PROBE')<p.stderr.index('STAGE4A_TILING_REJECT')
 p=subprocess.run([str(exe),'0','0','8','full'],capture_output=True,text=True)
 assert p.returncode==7 and 'soc=17' in p.stderr and 'arch=2002' in p.stderr


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
  c.write(log.parent/'execution.json',dict(status='HOST_TILING_ONLY',host_only=True,npu='NOT_RUN',device=0,output_mode=21,core_limit=1))
 args=SimpleNamespace(multi_build_report=bp,device=0,cann_root=tmp_path/'sdk',diagnostic_groups=[],host_only=True)
 with patch.object(run,'installed',return_value={}),patch.object(run,'verify_case',return_value={}),patch.object(run,'command',side_effect=command),patch.object(run,'compare',side_effect=AssertionError('must not compare numerical outputs')):
  run.smoke(args,out,r,dict(files=files))
 assert len(calls)==1 and r['status']=='STAGE4A_HOST_TILING_ONLY' and r['npu']=='NOT_RUN'
 assert 'outputs' not in r['runs'][0]
