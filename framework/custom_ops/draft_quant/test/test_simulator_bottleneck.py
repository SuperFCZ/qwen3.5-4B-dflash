"""Offline schema/PC/source/identity regressions; no simulator/device execution."""
from decimal import Decimal
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import csv
import pytest

SIM=Path(__file__).resolve().parents[1]/'simulator'
sys.path.insert(0,str(SIM))
import bottleneck as entry
import bottleneck_parse as parse


def row(address='333243192',name='mov_out_to_ub',pipe='MTE2',count='1',cycles='17'):
    return dict(instr=name,addr=address,pipe=pipe,call_count=count,cycles=cycles,**{'running_time(us)':'0.01'},detail='registers')


def trace_file(tmp_path,events):
    path=tmp_path/'trace.json';path.write_text(json.dumps(dict(displayTimeUnit='ns',profilingType='fixture-not-receiver-value',schemaVersion=1,traceEvents=events)))
    return path


def event(address,name,pipe_pid,start,duration):
    return dict(name=name,ph='X',pid=pipe_pid,tid=1,ts=start,dur=duration,args=dict(pc_addr=hex(address),code='/unverified/source.cpp:1',detail='operand'))


def meta(pid,name):return dict(name='process_name',ph='M',pid=pid,tid=0,args={'name':name})


def test_observed_csv_shape_and_decimal_pc_no_rebase(tmp_path):
    p=tmp_path/'core0_instr_exe.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=parse.INSTR);w.writeheader();w.writerow(row())
    schema,rows=parse.csv_table(p,'instructions')
    assert schema['status']=='DOCUMENTED_CSV_SCHEMA' and parse.pc(rows[0]['addr'])==333243192
    assert parse.pc(hex(333243192),trace=True)==333243192
    assert parse.pc('333243192',trace=True) is None and parse.pc('13dcd000') is None
    p.write_text('pc,cycles\n42,17\n')
    assert parse.csv_table(p,'instructions')[0]['status']=='UNSUPPORTED_CSV_SCHEMA'
    p.write_text('code,code\na,b\n')
    with pytest.raises(ValueError):parse.csv_table(p,'code')


def test_decimal_intervals_overlap_not_sum_and_zero_width_not_zero_cycles(tmp_path):
    source=tmp_path/'kernel.cpp';source.write_text('code')
    rows=[row('100','copy','MTE2',cycles='100'),row('104','mmad','CUBE',cycles='200'),row('108','scalar_mov','SCALAR',cycles='3')]
    events=[meta(1,'MTE2'),meta(2,'CUBE'),meta(3,'SCALAR'),event(100,'copy',1,'1000000000000000.001','10'),event(104,'mmad',2,'1000000000000005.001','10'),event(108,'scalar_mov',3,'1000000000000000.001','0.0')]
    path=trace_file(tmp_path,events)
    result=parse.trace_analysis(path,rows,{},source,{})
    assert result['status']=='PC_INSTRUCTION_COUNTS_MATCHED'
    assert result['interval_union']==15 and result['trace_instruction_span']==15
    assert result['mte2_pairwise_intersections']['MTE2 & CUBE']==5
    assert result['zero_duration_events']==1 and result['kernel_elapsed_cycles'] is None
    assert result['synchronization_wait_fraction'].startswith('UNRESOLVED')
    summary=parse.instruction_summary(rows,{},source,{})
    assert summary['by_source_operation']['UNMAPPED']['instruction_cycles']==303
    assert summary['cpp_api_call_count']=='UNRESOLVED'


def test_pc_opcode_pipe_and_count_mismatches_never_verified(tmp_path):
    source=tmp_path/'kernel.cpp';source.write_text('code')
    events=[meta(1,'MTE2'),event(100,'copy',1,'1','2'),event(100,'wrong_opcode',1,'4','2')]
    p=trace_file(tmp_path,events)
    result=parse.trace_analysis(p,[row('100','copy','MTE2',count='2')],{},source,{})
    assert result['status']=='UNRESOLVED_TRACE_CORRELATION'
    assert len(result['unresolved_events'])==1 and result['count_mismatches'][0]['trace']==1
    events[0]=meta(1,'UNKNOWN')
    result=parse.trace_analysis(trace_file(tmp_path,events),[row('100','copy','MTE2')],{},source,{})
    assert len(result['unresolved_events'])==2


def test_dwarf_mapping_requires_exact_source_and_unambiguous_category(tmp_path):
    src=tmp_path/'kernel.cpp';src.write_text('x\ny\nz')
    text=f'0x00000064\ncopy\n{src}:2\n0x00000068\n??\n??:0\n'
    mapping=parse.symbolizer_frames(text,{100,104})
    assert parse.source_category(mapping[100],src,{'DataCopy':[2]})[0]=='DataCopy'
    assert parse.source_category(mapping[100],src,{'DataCopy':[2],'Event':[2]})[0]=='UNMAPPED'
    assert parse.source_category([dict(file='kernel.cpp',line=2,function='x')],src,{'DataCopy':[2]})[0]=='UNMAPPED'
    assert parse.source_category(mapping[104],src,{'DataCopy':[2]})[0]=='UNMAPPED'
    with pytest.raises(ValueError):parse.symbolizer_frames(text,{100,108})
    with pytest.raises(ValueError):parse.symbolizer_frames(text+text,{100,104})


def test_trace_format_preserves_full_args_metadata_and_precision(tmp_path):
    e=event(100,'copy',1,'0.29100000858306885','0.0')
    e['args']['unknown_semantics']='unchanged'
    path=trace_file(tmp_path,[meta(1,'SCALAR'),e])
    schema=parse.trace_schema(path)
    assert schema['samples'][0]==e and schema['metadata'][0]['args']['name']=='SCALAR'
    assert schema['status']=='TRACE_SEMANTICS_UNRESOLVED'


def test_evidence_hash_and_path_refuse_tampering_and_escape(tmp_path):
    root=tmp_path/'old';root.mkdir();f=root/'trace.json';f.write_text('{}')
    evidence=entry.Evidence(root);evidence.read(f);saved=evidence.files[str(f)]
    f.write_text('changed')
    with pytest.raises(ValueError):evidence.read(f,saved)
    with pytest.raises(ValueError):evidence.verify()
    with pytest.raises(ValueError):evidence.relative(root,dict(path='../elsewhere',bytes=1,sha256='x'))


def test_extract_only_never_launches_symbolizer_or_simulator(tmp_path):
    root=tmp_path/'old';root.mkdir();report=root/'bounded-report.json'
    report.write_text(json.dumps(dict(status='BOUNDED_SIMULATOR_DIAGNOSTIC_ONLY',phase='complete',paired_input_identity='MATCHED',runs={'off':{},'raw':{}})))
    code=root/'code.csv';code.write_text('code,call_count,cycles,running_time(us)\n/x:1,1,1,0.1\n')
    instr=root/'instr.csv'
    with instr.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=parse.INSTR);w.writeheader();w.writerow(row())
    trace=trace_file(root,[meta(1,'MTE2'),event(333243192,'mov_out_to_ub',1,'1','1')])
    spec=dict(code=code,instr=instr,trace=trace,inputs={'x':'same'},config={})
    args=SimpleNamespace(report=report,output_dir=tmp_path/'new',inspect_only=True,addr2line=None)
    with patch.object(entry,'load_mode',return_value=spec),patch.object(entry,'symbolize') as symbolizer,patch.object(entry.subprocess,'run') as command:
        assert entry.analyze(args)==0;symbolizer.assert_not_called();command.assert_not_called()
    output=json.loads((args.output_dir/'bottleneck-analysis.json').read_text())
    assert output['status']=='FORMAT_EVIDENCE_ONLY' and output['full_draft_token_kv']=='NOT_RUN'


def complete_fixture(root):
    """Small CPU-only evidence graph; binary payload is explicitly a mock."""
    import shutil
    from bounded_prepare import patch_kernel
    from simulator_prepare import focus_lines
    def identity(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=entry.digest(p))
    def relative(p,base):return dict(identity(p),path=str(p.relative_to(base)))
    root.mkdir();runs={}
    for mode in ('off','raw'):
        folder=root/mode;folder.mkdir();bound=folder/'bounded';bound.mkdir();profile=bound/'profile';profile.mkdir();core=profile/'core0';core.mkdir();inputs=bound/'input';inputs.mkdir()
        source=folder/'kernel.cpp';shutil.copyfile(SIM.parent/'op_kernel/d_flash_group_quant_linear.cpp',source)
        adaptation=patch_kernel(source,0)
        entry.dump(folder/'bounded-source.json',adaptation);(folder/'bounded-source.diff').write_text(adaptation['diff'])
        entry.dump(folder/'source-map.json',dict(kernel=identity(source),focus_lines=focus_lines(source)))
        cfg=dict(gate_prefetch_mode=mode,gate_copy_mode='loop',k_tile=256,scale_mode='broadcast',source_sha256={'op_kernel/d_flash_group_quant_linear.cpp':adaptation['original_sha256']})
        entry.dump(folder/'build-config.json',cfg)
        binary=folder/'kernel.o';binary.write_bytes(b'CPU ONLY MOCK '+mode.encode());profile_binary=profile/'kernel.o';shutil.copyfile(binary,profile_binary)
        elf_log=folder/'elf.log';elf_log.write_text('.debug_info .debug_line CPU mock\n')
        built=dict(config=identity(folder/'build-config.json'),source_map=relative(folder/'source-map.json',folder),simulator_source_adaptation=adaptation,
                   kernels=[dict(identity(binary),debug_info=True,debug_line=True,readelf=relative(elf_log,folder))])
        (inputs/'x.bin').write_bytes(b'CPU identity fixture')
        golden=bytes(16*19456*2);(inputs/'expected.bin').write_bytes(golden)
        actual=(bytes(128)+b'\x7f'*(19456*2-128))*16;(inputs/'actual-0.bin').write_bytes(actual)
        copies={p.name:identity(p) for p in (inputs/'x.bin',inputs/'expected.bin')}
        entry.dump(bound/'inputs.json',dict(originals=[],copies=copies))
        entry.dump(inputs/'execution.json',dict(status='SIMULATOR_COMPLETED',guards_intact=True,input_readonly=True))
        entry.dump(bound/'simulator.budget.json',dict(status='COMMAND_COMPLETED',final_budget_error=None))
        code=core/'core0_code_exe.csv';code.write_text('code,call_count,cycles,running_time(us)\n/x:1,1,17,0.01\n')
        instr=core/'core0_instr_exe.csv'
        with instr.open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=parse.INSTR);w.writeheader();w.writerow(row('100','copy','MTE2'))
        trace=trace_file(core,[meta(1,'MTE2'),event(100,'copy',1,'0','1')])
        result=dict(status='BOUNDED_SIMULATOR_DIAGNOSTIC_ONLY',owner=0,
            checks=dict(n_tile=0,elements=1024,bit_mismatches=0,selected_tile_bitwise_equal=True,skipped_output_poison_intact=True),
            artifacts=dict(status='COLLECTED',files={'trace':[relative(trace,profile)],'instructions':[relative(instr,profile)],'code_hotspots':[relative(code,profile)]}),
            kernel_identity=dict(status='MATCHED',binaries=[identity(profile_binary)]),execution=identity(inputs/'execution.json'),actual=identity(inputs/'actual-0.bin'),source_adaptation=adaptation)
        entry.dump(folder/'build-report.json',built);entry.dump(bound/'result.json',result);runs[mode]=dict(build=built,bounded=result)
    report=root/'bounded-report.json';entry.dump(report,dict(status='BOUNDED_SIMULATOR_DIAGNOSTIC_ONLY',phase='complete',paired_input_identity='MATCHED',runs=runs,n_tile=0))
    return report


def test_complete_offline_flow_with_mock_dwarf_and_no_original_writes(tmp_path):
    report=complete_fixture(tmp_path/'old');before={p:entry.digest(p) for p in report.parent.rglob('*') if p.is_file()}
    def symbolize(tool,binary,addresses,out):
        src=out.parent.parent/'old'/out.name/'kernel.cpp'
        line=parse.copy_sites(src)[0]['first_line']
        from simulator_prepare import focus_lines
        return {100:[dict(file=str(src),line=focus_lines(src)['DataCopy'][0],function='copy'),dict(file=str(src),line=line,function='caller')]},dict(status='DWARF_DECODED')
    args=SimpleNamespace(report=report,output_dir=tmp_path/'new',inspect_only=False,addr2line=None)
    with patch.object(entry,'symbolize',side_effect=symbolize),patch.object(entry.subprocess,'run') as runner:
        assert entry.analyze(args)==0;runner.assert_not_called()
    result=json.loads((args.output_dir/'bottleneck-analysis.json').read_text())
    assert result['status']=='OFFLINE_ANALYSIS_PARTIAL'
    assert result['modes']['off']['instructions']['by_source_operation']['DataCopy']['call_count']==1
    assert result['comparison']['kernel_elapsed_cycles'].startswith('UNRESOLVED')
    assert before=={p:entry.digest(p) for p in before}


@pytest.mark.parametrize('changed',['off/bounded/profile/kernel.o','off/bounded-source.diff','off/bounded/input/actual-0.bin'])
def test_bound_evidence_changes_block_before_symbolization(tmp_path,changed):
    report=complete_fixture(tmp_path/'old');path=report.parent/changed;path.write_bytes(path.read_bytes()+b'changed')
    args=SimpleNamespace(report=report,output_dir=tmp_path/'new',inspect_only=False,addr2line=None)
    with patch.object(entry,'symbolize') as tool:
        assert entry.analyze(args)==1;tool.assert_not_called()
    assert json.loads((args.output_dir/'bottleneck-analysis.json').read_text())['status']=='BLOCKED'


def test_next_copy_role_requires_source_pc_frame_not_occurrence_order(tmp_path):
    source=tmp_path/'source.cpp';source.write_text('CopyRawTile(qBanks[(1 - bank) * kTileN * tileK],\n sBanks, qGm, sGm, n, k, b, kBegin + tileK);\n')
    frames=[dict(file=str(source),line=2,function='next_copy')]
    assert parse.copy_role(frames,parse.copy_sites(source))=='RAW_NEXT_K_COPY'
    events=[meta(1,'MTE2'),meta(2,'CUBE'),event(100,'copy',1,'0','10'),event(104,'mmad',2,'5','10')]
    path=trace_file(tmp_path,events);rows=[row('100','copy','MTE2'),row('104','mmad','CUBE')]
    mapped=parse.trace_analysis(path,rows,{100:frames},source,{})
    unverified=parse.trace_analysis(path,rows,{},source,{})
    assert mapped['prefetch_callsite_overlap']['next_copy_compute_intersection']==5
    assert unverified['prefetch_callsite_overlap']['status']=='UNRESOLVED'
    assert mapped['k_tile_assignment'].startswith('UNRESOLVED')
