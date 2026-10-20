"""CPU tests of source transformation, partial evidence and diagnostic limits."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

SIM=Path(__file__).resolve().parents[1]/'simulator'
sys.path.insert(0,str(SIM))
import bounded
from bounded_prepare import bounded_source,patch_kernel,schedule,compare_tile,inspect_partial
from bounded_budget import BudgetCommands,limits


@pytest.mark.parametrize('tile',[0,1,6,7,303])
def test_patch_only_scopes_outer_n_loop_and_owner(tmp_path,tile):
    path=SIM.parent/'op_kernel/d_flash_group_quant_linear.cpp'
    original=path.read_text();modified,owner,select=bounded_source(original,tile)
    assert modified.replace(owner,'',1).replace(select,'',1)==original
    assert modified.index(owner)<modified.index('AscendC::TPipe pipe;')
    assert modified.index(select)<modified.index('const uint32_t nBegin = nTile * kTileN;')
    tail='        const uint32_t nBegin = nTile * kTileN;'
    assert modified[modified.index(tail):]==original[original.index(tail):]
    assert original==path.read_text()
    copy=tmp_path/'copy.cpp';copy.write_text(original)
    audit=patch_kernel(copy,tile)
    assert audit['k_iterations']==10 and audit['group128_count']==20
    assert audit['original_sha256']!=audit['bounded_sha256']
    assert audit['full_gate_correctness']=='NOT_RUN'
    assert all(audit['focus_lines'].values())
    with pytest.raises(ValueError):bounded_source(original.replace('nTile += blockNum','nTile++'),tile)


@pytest.mark.parametrize('blocks',[1,7,8])
def test_all_tile_owners_full_k_banks_and_no_cross_tile_prefetch(blocks):
    for tile in range(304):
        for raw in (False,True):
            steps=schedule(tile,blocks,raw)
            assert len(steps)==10 and {s['core'] for s in steps}=={tile%blocks}
            assert {s['n_tile'] for s in steps}=={tile}
            assert [s['k_begin'] for s in steps]==list(range(0,2560,256))
            assert [s['bank'] for s in steps]==([0,1]*5 if raw else [0]*10)
            assert steps[-1]['prefetch_k'] is None
            for s in steps[:-1]:assert s['prefetch_k']==(s['k_begin']+256 if raw else None)
    for bad in (-1,304,True,0.5):
        with pytest.raises(ValueError):schedule(bad,blocks,True)


@pytest.mark.parametrize('tile',[0,303])
def test_only_selected_columns_compare_and_skipped_bytes_stay_poison(tile):
    golden=np.arange(16*19456,dtype=np.uint16).reshape(16,19456)
    actual=np.full_like(golden,0x7f7f);lo=tile*64
    actual[:,lo:lo+64]=golden[:,lo:lo+64]
    result=compare_tile(actual.tobytes(),golden.tobytes(),tile)
    assert result['selected_tile_bitwise_equal'] and result['skipped_output_poison_intact']
    assert result['elements']==1024 and result['full_gate_correctness']=='NOT_RUN'
    actual[15,lo+63]^=1
    result=compare_tile(actual.tobytes(),golden.tobytes(),tile)
    assert result['bit_mismatches']==1 and result['first_difference']['column']==lo+63
    actual[0,19455 if tile==0 else 0]=0
    assert not compare_tile(actual.tobytes(),golden.tobytes(),tile)['skipped_output_poison_intact']
    with pytest.raises(ValueError):compare_tile(b'',golden.tobytes(),tile)


def partial_fixture(root):
    folder=root/'off/gate';profile=folder/'profile';profile.mkdir(parents=True)
    (folder/'input').mkdir();(folder/'input/execution.json').write_text('{"status":"RUNNING"}')
    (profile/'trace.json').write_text('{"traceEvents": [')  # valid partial diagnostic, not complete
    (profile/'code_exe.csv').write_text('Line,Cycles\n133,100\n')
    report=dict(status='BLOCKED',phase='gate-off',runs={'off':{'smoke':dict(simulator_output_bitwise_match=True,profile_kernel_identity={'status':'MATCHED'},artifacts={'status':'COLLECTED'})}})
    path=root/'simulator-report.json';path.write_text(json.dumps(report))
    source=SIM.parent/'op_kernel/d_flash_group_quant_linear.cpp'
    (root/'off/source-map.json').write_text(json.dumps({'kernel':bounded.base.ident(source)}))
    return path


def test_partial_is_read_only_and_invalid_json_not_promoted(tmp_path):
    path=partial_fixture(tmp_path/'old')
    before={p:bounded.base.ident(p) for p in path.parent.rglob('*') if p.is_file()}
    result=inspect_partial(path,tmp_path/'inspection.json',bounded.base.ident)
    assert result['status']=='INCOMPLETE_INTERRUPTED_READ_ONLY'
    assert result['correctness']=='NOT_RUN' and result['completion']=='NOT_COMPLETED'
    assert {r['parse_status'] for r in result['artifacts']}=={'UNPARSED_PARTIAL','PARSED_PARTIAL'}
    assert before=={p:bounded.base.ident(p) for p in before}
    (path.parent/'off/gate/input/execution.json').write_text('{"status":"PASS"}')
    with pytest.raises(ValueError):inspect_partial(path,tmp_path/'bad.json',bounded.base.ident)


def args_for(tmp_path,inspect_only=False):
    return SimpleNamespace(partial_report=partial_fixture(tmp_path/'old'),output_dir=tmp_path/'new',
                          n_tile=0,max_output_mib=16,max_file_mib=8,inspect_only=inspect_only)


def test_inspect_only_does_not_build_or_simulate(tmp_path):
    args=args_for(tmp_path,True)
    with patch.object(bounded.base,'preflight') as preflight,patch.object(bounded.base,'build') as build:
        assert bounded.run(args)==0;preflight.assert_not_called();build.assert_not_called()
    report=json.loads((args.output_dir/'bounded-report.json').read_text())
    assert report['status']=='PARTIAL_INSPECTED_NOT_EXECUTED' and report['runs']=={}


def test_incomplete_evidence_or_off_failure_cannot_run_raw(tmp_path):
    args=args_for(tmp_path)
    modes=[]
    def build(mode,*a,**kw):
        modes.append(mode)
        assert kw['kernel_adapter'] and isinstance(kw['command_runner'],BudgetCommands)
        return 'runner',{},{}
    with patch.object(bounded.base,'preflight',return_value=({}, {}, {'off':{},'raw':{}},{})),patch.object(bounded.base,'build',side_effect=build),patch.object(bounded,'simulate',side_effect=RuntimeError('selected bits mismatch')):
        assert bounded.run(args)==1
    assert modes==['off']
    report=json.loads((args.output_dir/'bounded-report.json').read_text())
    assert report['status']=='BLOCKED' and report['full_gate_correctness']=='NOT_RUN'


def test_budget_rejects_oversize_without_deleting_evidence(tmp_path):
    f=tmp_path/'large';f.write_bytes(b'x'*5000)
    with pytest.raises(RuntimeError):limits(tmp_path,4096,4096)
    with pytest.raises(RuntimeError):limits(tmp_path,1024*1024,4096)
    assert f.stat().st_size==5000


def test_child_file_limit_stops_cpu_writer_and_preserves_failure_log(tmp_path):
    cmd=BudgetCommands(tmp_path,2*1024*1024,32768)
    with pytest.raises(Exception):
        cmd([sys.executable,'-c',"open('large','wb').write(b'x'*65536)"],tmp_path/'writer.log',os.environ,tmp_path,10)
    assert (tmp_path/'large').stat().st_size<=32768
    audit=json.loads((tmp_path/'writer.budget.json').read_text())
    assert audit['status']=='STOPPED' and audit['truncated_output_accepted'] is False


def test_paired_flow_uses_same_tile_and_never_full_gate_pass(tmp_path):
    args=args_for(tmp_path)
    modes=[]
    def build(mode,*a,**kw):modes.append(('build',mode));return 'runner',{},{}
    def sim(mode,*a):
        modes.append(('simulate',mode))
        return {'scope':'partial only','full_gate_correctness':'NOT_RUN'}, {'copies':{'x.bin':{'sha256':'same'},'expected.bin':{'sha256':'same'}}}
    with patch.object(bounded.base,'preflight',return_value=({}, {}, {'off':{},'raw':{}},{})),patch.object(bounded.base,'build',side_effect=build),patch.object(bounded,'simulate',side_effect=sim):
        assert bounded.run(args)==0
    assert modes==[('build','off'),('simulate','off'),('build','raw'),('simulate','raw')]
    report=json.loads((args.output_dir/'bounded-report.json').read_text())
    assert report['status']=='BOUNDED_SIMULATOR_DIAGNOSTIC_ONLY'
    assert report['full_gate_correctness']==report['full_draft_token_kv']==report['npu_correctness']=='NOT_RUN'


def test_pair_identity_mismatch_is_blocked(tmp_path):
    args=args_for(tmp_path)
    def sim(mode,*a):return {},{'copies':{'x.bin':{'sha256':mode}}}
    with patch.object(bounded.base,'preflight',return_value=({}, {}, {'off':{},'raw':{}},{})),patch.object(bounded.base,'build',return_value=('r',{},{})),patch.object(bounded,'simulate',side_effect=sim):
        assert bounded.run(args)==1
    assert json.loads((args.output_dir/'bounded-report.json').read_text())['status']=='BLOCKED'
