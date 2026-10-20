"""CPU/build-plan tests only. No SDK, simulator or NPU is executed."""
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from a2_common import record,sha256

SIM=Path(__file__).resolve().parents[1]/'simulator'
sys.path.insert(0,str(SIM))
import simulator_prepare as prep
spec=importlib.util.spec_from_file_location('gate_simulator_entry',SIM/'run.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


def test_private_runner_keeps_abi_guards_one_call_and_no_timing(tmp_path):
    here=SIM.parent
    before={str(p):sha256(p) for p in (here/'test/main.cpp',here/'test/runner_common.h',*[(here/n) for n in entry.FROZEN])}
    path=prep.runner_sources(here,tmp_path/'private')
    text=path.read_text();common=(path.parent/'runner_common.h').read_text()
    assert 'constexpr int kRepetitions = 1;' in text
    assert 'repeat < 1' in common and 'correctness_before_calls\\\":1' in common
    assert 'SIMULATOR_COMPLETED' in text and 'AscendCL camodel' in text
    assert 'if(argc!=6)' in text and 'VerifySimulatorRuntime(argv[2])' in text
    assert text.index('VerifySimulatorRuntime(argv[2]);') < text.index('return SimulatorApplication(argc,argv);')
    assert 'application.log' in text
    for needle in ('aclCreateTensor','ACL_FORMAT_FRACTAL_NZ','GetWorkspaceSize','CheckGuards','read-only input changed'):
        assert needle in text
    assert before=={p:sha256(p) for p in before}
    with pytest.raises(ValueError):prep.replace_once('ambiguous ambiguous','ambiguous','x')


def test_observed_cli_and_arch_are_not_substituted():
    help=' '.join(('--soc-version','--output','--launch-count','--core-id','--timeout','--aic-metrics','--kernel-name'))
    prep.validate_help(help)
    with pytest.raises(ValueError):prep.validate_help(help.replace('--core-id',''))
    cmd=prep.profile_command('/verified/msopprof','/new/runner','/new/profile','/new/input',(16,256,64),5)
    assert cmd[:2]==['/verified/msopprof','simulator']
    assert '--soc-version=Ascend310P3' in cmd and '--core-id=0' in cmd and '--launch-count=1' in cmd
    assert '--kernel-name=DFlashGroupQuantLinear|d_flash_group_quant_linear' in cmd
    assert '--aic-metrics=PipeUtilization' in cmd and not any(x.startswith('--dump') for x in cmd)
    assert cmd[-5:]==['0','/new/input','16','256','64']
    with pytest.raises(ValueError):prep.profile_command('p','r','o','i',(16,256,64),0)


def test_smoke_nz_basis_and_expected_are_exact_without_touching_gate(tmp_path):
    prep.smoke_inputs(tmp_path/'smoke');root=tmp_path/'smoke'
    x=np.fromfile(root/'x.bin','<f2').reshape(16,256)
    nz=np.fromfile(root/'w_nz.bin','i1').reshape(8,4,16,32)
    q=nz.transpose(1,2,0,3).reshape(64,256)
    s=np.fromfile(root/'s_gn.bin','<f2').reshape(2,64)
    w=(q.reshape(64,2,128).astype('<f2')*s.T[:,:,None]).reshape(64,256)
    expected=(x.astype(np.float64)@w.astype(np.float64).T).astype('<f2')
    assert expected.tobytes()==(root/'expected.bin').read_bytes()
    assert np.all(q!=0) and np.all(np.isfinite(expected)) and set(np.sign(q).ravel())=={-1,1}
    assert len(list(root.iterdir()))==4


def test_debug_hook_is_mandatory_and_private(tmp_path):
    f=tmp_path/'CMakeLists.txt';f.write_text('original optimized kernel rules\n')
    prep.debug_cmake(f);s=f.read_text()
    assert 'add_ops_compile_options(ALL OPTIONS -g)' in s
    assert 'npu_op_kernel_options(ascendc_kernels ALL OPTIONS -g)' in s
    assert 'FATAL_ERROR' in s and '-O0' not in s
    assert s.endswith('original optimized kernel rules\n')
    cmake=(SIM/'CMakeLists.txt').read_text()
    assert 'runtime_camodel' in cmake and 'npu_drv_camodel' in cmake and 'NO_DEFAULT_PATH' in cmake and 'Missing required simulator/ACLNN library' in cmake
    assert '--no-as-needed' in cmake


def test_elf_arch_and_artifacts_cannot_fake_completed_collection(tmp_path):
    binary=tmp_path/'lib.so';header=bytearray(20);header[:6]=b'\x7fELF\x02\x01';header[18:20]=struct.pack('<H',183);binary.write_bytes(header)
    assert prep.elf_machine(binary)==183
    binary.write_bytes(b'not elf')
    with pytest.raises(ValueError):prep.elf_machine(binary)
    assert prep.artifacts(tmp_path,record)['status']=='ARTIFACTS_INCOMPLETE'
    (tmp_path/'trace.json').write_text(json.dumps([dict(ph='X',ts=0,dur=1,name='instruction')]))
    for name in ('core0_code_exe.csv','core0_instr_exe.csv'):(tmp_path/name).write_text('line,cycles\n1,100\n')
    r=prep.artifacts(tmp_path,record)
    assert r['status']=='COLLECTED' and r['cycles_are_wall_time'] is False
    assert r['tile_mapping']=='UNRESOLVED_TILE_MAPPING'
    (tmp_path/'core0_code_exe.csv').write_text('line,cycles\n')
    with pytest.raises(ValueError):prep.artifacts(tmp_path,record)


def test_failed_preflight_records_blocked_and_never_builds(tmp_path):
    for f in ('a51','raw'):(tmp_path/f).write_text('fixture')
    args=SimpleNamespace(output_dir=tmp_path/'new',a51_evidence=tmp_path/'a51',prefetch_evidence=tmp_path/'raw',stage='smoke')
    with patch.object(entry,'preflight',side_effect=ValueError('unsupported runtime')),patch.object(entry,'build') as build:
        assert entry.run(args)==1;build.assert_not_called()
    r=json.loads((args.output_dir/'simulator-report.json').read_text())
    assert r['status']=='BLOCKED' and r['npu_correctness']=='NOT_RUN' and r['runs']=={}


@pytest.mark.parametrize('stage',['prepare','smoke','gate'])
def test_stage_order_no_gate_before_smoke_and_no_hardware_pass(tmp_path,stage):
    for f in ('a51','raw'):(tmp_path/f).write_text('fixture')
    args=SimpleNamespace(output_dir=tmp_path/'new',a51_evidence=tmp_path/'a51',prefetch_evidence=tmp_path/'raw',stage=stage)
    calls=[]
    def build(mode,*a):calls.append(('build',mode));return Path('/fake/runner'),{},dict(status='CPU_FIXTURE')
    def simulate(mode,part,*a):calls.append((part,mode));return dict(status='SIMULATOR_DIAGNOSTIC_ONLY',artifacts={'files':{}})
    prepared={'accepted':{'modes':{}}}
    with patch.object(entry,'preflight',return_value=({}, {}, {'off':{},'raw':{}},prepared)),patch.object(entry,'build',side_effect=build),patch.object(entry,'simulate',side_effect=simulate):
        assert entry.run(args)==0
    expected=[] if stage=='prepare' else [('build','off'),('smoke','off')]
    if stage=='gate':expected += [('gate','off'),('build','raw'),('smoke','raw'),('gate','raw')]
    assert calls==expected
    r=json.loads((args.output_dir/'simulator-report.json').read_text())
    assert r['status']!='PASS' and r['hardware_performance']=='NOT_RUN' and r['full_draft_token_kv']=='NOT_RUN'


def test_smoke_failure_never_launches_gate_or_fallback(tmp_path):
    for f in ('a51','raw'):(tmp_path/f).write_text('fixture')
    args=SimpleNamespace(output_dir=tmp_path/'new',a51_evidence=tmp_path/'a51',prefetch_evidence=tmp_path/'raw',stage='gate')
    calls=[]
    def fail(mode,part,*a):calls.append((mode,part));raise RuntimeError('camodel unsupported')
    with patch.object(entry,'preflight',return_value=({}, {}, {'off':{},'raw':{}},{'accepted':{'modes':{}}})),patch.object(entry,'build',return_value=(Path('/fake/runner'),{},{})),patch.object(entry,'simulate',side_effect=fail):
        assert entry.run(args)==1
    assert calls==[('off','smoke')]


def test_runtime_guard_cpp_syntax_only(tmp_path):
    import shutil,subprocess
    compiler=shutil.which('c++')
    if not compiler:pytest.skip('no C++ compiler')
    source=tmp_path/'guard.cpp';source.write_text('#include '+json.dumps(str(SIM/'runtime_guard.h'))+'\n')
    result=subprocess.run([compiler,'-std=c++17','-fsyntax-only',str(source)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
