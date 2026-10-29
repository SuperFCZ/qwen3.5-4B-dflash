import json,struct,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import inspect_binary as g


def elf(path,code=b'\x01\x02\x03\x04',debug=b'old',reloc=b'\0'*24):
    names=b'\0.text\0.debug_info\0.rela.text\0.shstrtab\0'
    items=[('',0,0,b''),('.text',1,6,code),('.debug_info',1,0,debug),('.rela.text',4,0,reloc),('.shstrtab',3,0,names)]
    data=bytearray(64);rows=[]
    for name,kind,flags,payload in items:
        offset=len(data);data.extend(payload)
        rows.append((names.index(name.encode()+b'\0') if name else 0,kind,flags,0,offset,len(payload),0,0,1,0))
    off=len(data)
    for row in rows:data.extend(struct.pack('<IIQQQQIIQQ',*row))
    data[:16]=b'\x7fELF\x02\x01\x01'+bytes(9)
    struct.pack_into('<HHIQQQIHHHHHH',data,16,1,183,1,0,0,off,0,64,0,0,64,len(rows),4)
    path.write_bytes(data);return path


def test_debug_difference_not_misreported_as_instruction_change(tmp_path):
    a=elf(tmp_path/'a.o');b=elf(tmp_path/'b.o',debug=b'new_debug_path')
    result=g.compare_sections(g.sections(a),g.sections(b))['section_comparisons']
    assert g.digest(a.read_bytes())!=g.digest(b.read_bytes())
    assert result['executable']['equal'] and result['allocated']['equal']
    assert not result['debug']['equal']


def test_relocation_change_kept_separate(tmp_path):
    a=elf(tmp_path/'a.o');b=elf(tmp_path/'b.o',reloc=b'\1'*24)
    result=g.compare_sections(g.sections(a),g.sections(b))['section_comparisons']
    assert result['executable']['equal'] and not result['relocation']['equal']


def test_actual_code_change_and_truncation(tmp_path):
    a=elf(tmp_path/'a.o');b=elf(tmp_path/'b.o',code=b'\1\2\3\5')
    assert not g.compare_sections(g.sections(a),g.sections(b))['section_comparisons']['executable']['equal']
    b.write_bytes(b.read_bytes()[:-10])
    with pytest.raises(ValueError):g.sections(b)


def test_ambiguous_pair_not_matched_by_order(tmp_path):
    path=elf(tmp_path/'a.o');item=dict(binary=g.f.identity(path),core_type='UNRESOLVED',elf=g.sections(path))
    assert g.pair_artifacts([item,item],[item,item])[0]['status']=='UNRESOLVED_AMBIGUOUS_ARTIFACT_PAIRING'
    one=g.pair_artifacts([item],[item])[0]
    assert one['core_type']=='UNRESOLVED' and one['whole_elf_equal']

@pytest.mark.parametrize('text,rc,expected',[
    ('unknown architecture',1,'UNRESOLVED_TOOL_OUTPUT'),
    ('   0: 00 00 00 00 <unknown>\n',0,'UNRESOLVED_TOOL_OUTPUT'),
    ('ELF file only\n',0,'UNRESOLVED_TOOL_OUTPUT'),
    ('   0: 01 02 03 04 scalar_fixture r0, r1\n',0,'DISASSEMBLY_COLLECTED_REQUIRES_REVIEW'),
])
def test_disassembler_success_is_not_source_or_cycles_proof(tmp_path,monkeypatch,text,rc,expected):
    def run(argv,stdout,stderr,timeout):
        assert '--source' not in argv and '--disassemble' in argv
        stdout.write(text.encode())
        class Result:returncode=rc
        return Result()
    monkeypatch.setattr(g.subprocess,'run',run)
    r=g.disassemble('fixture-tool','fixture-object',tmp_path/'out.txt')
    assert r['status']==expected and r['dynamic_execution_cost']=='UNRESOLVED'


def test_hash_tampering_rejected(tmp_path):
    p=elf(tmp_path/'a.o');item=g.f.identity(p);p.write_bytes(p.read_bytes()+b'tamper')
    with pytest.raises(ValueError):g.f.verify(item)


def test_fixed_tool_discovery_does_not_scan(tmp_path,monkeypatch):
    monkeypatch.setattr(g.shutil,'which',lambda _:None)
    p=tmp_path/'aarch64-linux/ccec_compiler/bin/llvm-objdump';p.parent.mkdir(parents=True);p.write_text('fixture')
    assert g.tool_candidates(tmp_path)==[p]
    other=tmp_path/'unrelated/toolchain/llvm-objdump';other.parent.mkdir(parents=True);other.write_text('ignored')
    assert g.tool_candidates(tmp_path)==[p]


def test_readonly_main_reports_partial_without_tools(tmp_path,monkeypatch):
    from types import SimpleNamespace
    build=tmp_path/'build';build.mkdir();variants={}
    for v in ('A0','A1','B0','B1'):
        folder=build/v;folder.mkdir();opp=folder/'opp';opp.mkdir()
        p=elf(opp/'fixture.o',debug=v.encode());m=opp/'fixture.json';m.write_text('{}')
        variants[v]=dict(opp=str(opp),installed=dict(kernels=[dict(binary=g.f.identity(p),metadata=g.f.identity(m))]))
    report=dict(status='STAGE4F_BUILT_NPU_NOT_RUN',sources=[],source_audit={},variants=variants,binary_comparison=g.f.binary_summary(variants))
    path=build/'build-report.json';path.write_text(json.dumps(report));before=g.f.identity(path)
    monkeypatch.setattr(g.f,'sources',lambda:[])
    monkeypatch.setattr(g.f,'verify_audit',lambda *a:None) # fixture-only source provenance; actual file hashes still checked
    monkeypatch.setattr(g,'tool_candidates',lambda _:[])
    args=SimpleNamespace(build_report=path,output_dir=tmp_path/'new',cann_root=tmp_path/'cann',objdump=None,benchmark_report=None)
    assert g.main(args)==0
    out=json.loads((args.output_dir/'binary-analysis.json').read_text())
    assert out['status']=='STAGE4G_OFFLINE_ELF_REVIEW_PARTIAL' and out['new_kernel']=='NOT_CREATED'
    assert out['pairs']['A'][0]['section_comparisons']['executable']['equal']
    assert out['variants']['A0'][0]['core_type']=='UNRESOLVED'
    assert out['variants']['A0'][0]['disassembly']['status']=='NOT_RUN_TOOL_UNRESOLVED'
    assert g.f.identity(path)==before and out['npu']=='NOT_RUN'
    p=Path(variants['B1']['installed']['kernels'][0]['binary']['path']);p.write_bytes(b'tamper')
    args.output_dir=tmp_path/'reject'
    assert g.main(args)==1
    assert json.loads((args.output_dir/'binary-analysis.json').read_text())['status']=='BLOCKED'
