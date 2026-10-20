"""CACHEMISS/source-only/ELF diagnostics. No Simulator or Kernel execution."""
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from unittest.mock import patch
import pytest

SIM=Path(__file__).resolve().parents[1]/'simulator';sys.path.insert(0,str(SIM))
import bottleneck as entry
import bottleneck_parse as parse
from bottleneck_elf import elf_layout,address_space,llvm_frames


def elf_fixture(path,text_address=0):
    data=bytearray(4096)
    data[:16]=b'\x7fELF\x02\x01\x01'+b'\0'*9
    # ELF64 ET_EXEC, actual observed AI Core machine; dummy DWARF payload.
    struct.pack_into('<HHIQQQIHHHHHH',data,16,2,0x1029,1,0,64,0x500,0,64,56,1,64,5,4)
    struct.pack_into('<IIQQQQQQ',data,64,1,5,0xe8,text_address,text_address,0x40,0x40,0x1000)
    names=b'\0.text\0.debug_info\0.debug_line\0.shstrtab\0';data[0x400:0x400+len(names)]=names
    for index,name,kind,flags,addr,offset,size in [(1,b'.text',1,6,text_address,0xe8,0x40),(2,b'.debug_info',1,0,0,0x300,8),(3,b'.debug_line',1,0,0,0x308,8),(4,b'.shstrtab',3,0,0,0x400,len(names))]:
        struct.pack_into('<IIQQQQIIQQ',data,0x500+index*64,names.index(name),kind,flags,addr,offset,size,0,0,1,0)
    path.write_bytes(data);return path


@pytest.mark.parametrize('count',[106,118])
def test_cachemiss_separate_channel_preserves_args_and_intervals(tmp_path,count):
    source=tmp_path/'kernel.cpp';source.write_text('source')
    args={'pc_addr':'0x64','uninterpreted_dependency':{'id':99}}
    cache=[dict(name='icache_miss',ph='X',pid=9,tid=1,ts=str(i),dur='2',args=args) for i in range(count)]
    events=[dict(name='process_name',ph='M',pid=9,args={'name':'CACHEMISS'}),dict(name='process_name',ph='M',pid=1,args={'name':'SCALAR'}),
        dict(name='mov',ph='X',pid=1,tid=1,ts='0',dur='1',args={'pc_addr':'0x64'}),*cache]
    path=tmp_path/'trace.json';path.write_text(json.dumps(dict(displayTimeUnit='ns',profilingType='fixture',schemaVersion=1,traceEvents=events)))
    rows=[dict(instr='mov',addr='100',pipe='SCALAR',call_count='1')]
    result=parse.trace_analysis(path,rows,{},source,{})
    assert result['status']=='PC_INSTRUCTION_COUNTS_MATCHED'
    assert result['count_mismatches']==result['unresolved_events']==[]
    assert result['interval_union']==1 and result['cachemiss']['event_count']==count
    assert result['cachemiss']['interval_union']==count+1
    assert result['cachemiss']['events'][0]['raw_event']==cache[0]
    assert result['synchronization_wait_fraction'].startswith('UNRESOLVED')
    # A suspicious name in a normal channel is not enough to exclude a task.
    events[-1]['pid']=1;events[-1]['name']='CACHEMISS';path.write_text(json.dumps(dict(displayTimeUnit='ns',profilingType='fixture',schemaVersion=1,traceEvents=events)))
    result=parse.trace_analysis(path,rows,{},source,{})
    assert result['cachemiss']['event_count']==count-1 and len(result['unresolved_events'])==1


def test_source_rows_use_frozen_path_hash_line_and_never_sum(tmp_path):
    src=tmp_path/'kernel.cpp';src.write_text('RunGroupQuantLinear<false>(x);\nmm.Iterate(false);\nother();\n')
    def row(code,cycles):return dict(code=code,call_count='12',cycles=str(cycles),**{'running_time(us)':'1'})
    rows=[row(str(src)+':1',270856),row(str(src)+':2',167538),row('/unbound/matmul.h:644',167538)]
    result=parse.source_hotspots(rows,src,{'Matmul':[2]},entry.digest(src))
    assert result['verified_rows']==2 and result['unmapped_rows']==1
    assert result['by_category']['KernelDispatch'][0]['cycles']=='270856'
    assert result['by_category']['Matmul'][0]['source_text']=='mm.Iterate(false);'
    assert result['instruction_pc_source_mapping'].startswith('UNRESOLVED')
    assert 'total_cycles' not in result and 'cycle_sum' not in result
    raw=parse.source_hotspots([row(str(src)+':1',271567),row(str(src)+':2',168556)],src,{'Matmul':[2]},entry.digest(src))
    paired=parse.compare_source_hotspots(result,raw)
    assert [r['cycles_delta'] for r in paired['rows']]==[711,1018]
    with pytest.raises(ValueError,match='SHA256'):parse.source_hotspots(rows,src,{},'changed')
    with pytest.raises(ValueError,match='out of range'):parse.source_hotspots([row(str(src)+':99',1)],src,{},entry.digest(src))
    unrelated=tmp_path/'other/kernel.cpp';unrelated.parent.mkdir();unrelated.write_text(src.read_text())
    assert parse.source_hotspots([row(str(unrelated)+':2',1)],src,{'Matmul':[2]},entry.digest(src))['verified_rows']==0


def test_elf_real_machine_text_load_dwarf_and_no_guessed_base(tmp_path):
    binary=elf_fixture(tmp_path/'aicore_binary.o');layout=elf_layout(binary)
    assert layout['machine_hex']=='0x1029' and layout['elf_type']==2
    assert layout['text']['address']==0 and {s['name'] for s in layout['debug_sections']}=={'.debug_info','.debug_line'}
    relation=address_space(layout,{0x13dcd000,0x13dcd004})
    assert relation['status']=='UNRESOLVED_ADDRESS_SPACE' and relation['outside_count']==2
    assert address_space(layout,{0,4})['status']=='DIRECT_ELF_VIRTUAL_ADDRESSES'
    binary.write_bytes(binary.read_bytes()[:100])
    with pytest.raises(ValueError):elf_layout(binary)


def test_llvm_json_matches_explicit_addresses_not_output_order():
    text='\n'.join(json.dumps(dict(Address=hex(a),Symbol=[dict(FileName='/kernel.cpp',Line=a+1,FunctionName='fn')])) for a in (4,0))
    frames=llvm_frames(text,{0,4})
    assert frames[4][0]['line']==5 and frames[0][0]['line']==1
    with pytest.raises(ValueError):llvm_frames(text,{0,8})
    with pytest.raises(ValueError):llvm_frames(text+'\n'+text,{0,4})


def test_llvm_probe_success_does_not_authorize_runtime_pc_translation(tmp_path):
    binary=elf_fixture(tmp_path/'aicore_binary.o');out=tmp_path/'out';out.mkdir()
    mapping={0:[dict(file='/source.cpp',line=1,function='fn')]}
    with patch.object(entry,'llvm_decode',return_value=(mapping,{'status':'ELF_DWARF_DECODED'})) as llvm,patch.object(entry,'symbolize') as old,patch.object(entry.subprocess,'run',return_value=SimpleNamespace(stdout='LLVM version fixture',stderr='',returncode=0)):
        result,symbols,audit=entry.elf_diagnostic({'binary':binary},{0x13dcd000},out,Path('/fixture/llvm-symbolizer'),'/fixture/addr2line')
    assert result=={} and symbols['status']=='UNRESOLVED'
    old.assert_not_called();llvm.assert_called_once()
    assert llvm.call_args.args[2]=={0,4}
    assert audit['llvm_elf_relative_probe']['status']=='ELF_DWARF_DECODED'
    assert audit['trace_pc_correspondence']['status']=='UNRESOLVED_ADDRESS_SPACE'


def test_symbolizer_search_is_scoped_to_bound_installation(tmp_path):
    root=tmp_path/'evidence';root.mkdir();installed=tmp_path/'tools/msopprof';(installed/'bin').mkdir(parents=True)
    tool=installed/'bin/msopprof';tool.write_text('fixture');tool.chmod(0o700)
    llvm=installed/'lib/llvm-symbolizer';llvm.parent.mkdir();llvm.write_text('fixture');llvm.chmod(0o700)
    record=dict(path=str(tool),bytes=tool.stat().st_size,sha256=entry.digest(tool));evidence=entry.Evidence(root)
    with patch.object(entry.shutil,'which',return_value=None):
        chosen,audit=entry.discover_llvm(None,{'preflight':{'tool':record}},evidence)
    assert chosen==llvm and audit['searched']==[str(installed)]
    assert str(llvm) in evidence.files
    tool.write_text('tampered')
    with patch.object(entry.shutil,'which',return_value=None),pytest.raises(ValueError):entry.discover_llvm(None,{'preflight':{'tool':record}},evidence)
