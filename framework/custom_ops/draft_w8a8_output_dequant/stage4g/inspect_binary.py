#!/usr/bin/env python3
"""Read existing Stage4F ELF artifacts only. Never compile, install, or execute an OM/kernel."""
import argparse,hashlib,json,re,shutil,struct,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4f'))
import f_common as f
import run_stage4f as f_run
sys.path.insert(0,str(HERE.parents[1]/'draft_quant/simulator'))
from bottleneck_elf import elf_layout


def digest(data):return hashlib.sha256(data).hexdigest()

def sections(path):
    if path.stat().st_size>64*1024*1024:raise ValueError('ELF exceeds 64 MiB inspection bound')
    layout=elf_layout(path);data=path.read_bytes()
    if not layout['executable_sections'] or any(s['type']==8 or not s['bytes'] for s in layout['executable_sections']):raise ValueError('no supported file-backed executable sections')
    hdr=struct.unpack_from('<HHIQQQIHHHHHH',data,16)
    shoff,shsize,shnum,strings=hdr[5],hdr[10],hdr[11],hdr[12]
    rows=[struct.unpack_from('<IIQQQQIIQQ',data,shoff+i*shsize) for i in range(shnum)]
    st=rows[strings];names=data[st[4]:st[4]+st[5]];out=[]
    for row in rows:
        name=names[row[0]:names.index(b'\0',row[0])].decode()
        kind,flags,address,offset,size=row[1:6]
        target=None
        if kind in (4,9):
            if row[7]>=len(rows):raise ValueError('invalid relocation target section')
            target_offset=rows[row[7]][0]
            target=names[target_offset:names.index(b'\0',target_offset)].decode()
        out.append(dict(name=name,type=kind,flags=flags,address=address,bytes=size,relocation_target=target,
            sha256=None if kind==8 else digest(data[offset:offset+size])))
    return dict(layout=layout,sections=out)


def compare_sections(a,b):
    def selected(obj,predicate):
        # Preserve ordering/duplicates. No stripping addresses, relocations or operands.
        return [s for s in obj['sections'] if predicate(s)]
    comparisons={}
    for label,predicate in [('executable',lambda s:bool(s['flags']&4)),
        ('allocated',lambda s:bool(s['flags']&2)),
        ('relocation',lambda s:s['type'] in (4,9)),
        ('non_debug_relocation',lambda s:s['type'] in (4,9) and not (s['relocation_target'] or '').startswith('.debug')),
        ('symbols',lambda s:s['type'] in (2,11)),
        ('debug',lambda s:s['name'].startswith(('.debug','.rela.debug','.rel.debug')))]:
        left,right=selected(a,predicate),selected(b,predicate)
        comparisons[label]=dict(equal=left==right,left=left,right=right)
    return dict(section_comparisons=comparisons,
        machine_equal=a['layout']['machine']==b['layout']['machine'],
        interpretation='Identical executable sections do not establish identical relocated execution; data, relocations and external bindings also matter. Different bytes do not prove more instructions or cycles.')


def tool_candidates(cann):
    paths=[cann/p for p in ('aarch64-linux/ccec_compiler/bin/llvm-objdump','aarch64-linux/compiler/ccec_compiler/bin/llvm-objdump','compiler/ccec_compiler/bin/llvm-objdump','ccec_compiler/bin/llvm-objdump')]
    for name in ('llvm-objdump','ccec-objdump'):
        found=shutil.which(name)
        if found:paths.append(Path(found))
    return sorted({p.resolve() for p in paths if p.is_file()})


def command(argv,out):
    try:
        with out.open('wb') as log:
            result=subprocess.run(list(map(str,argv)),stdout=log,stderr=subprocess.STDOUT,timeout=90)
        return dict(argv=list(map(str,argv)),returncode=result.returncode,log=f.identity(out))
    except subprocess.TimeoutExpired:
        return dict(argv=list(map(str,argv)),returncode=None,status='TIMEOUT',log=f.identity(out))


def disassemble(tool,path,out):
    r=command([tool,'--disassemble','--reloc','--line-numbers',path],out)
    # These are unclassified static lines, never dynamic instruction counts/cycles.
    text=out.read_text(errors='replace') if out.stat().st_size<=64*1024*1024 else ''
    placeholders=r'<unknown>|<not\s+available>'
    lines=[line for line in re.findall(r'^\s*[0-9a-fA-F]+:\s+(?:[0-9a-fA-F]{2,8}\s+)+\S.*$',text,re.M)
           if not re.search(placeholders,line,re.I)]
    unavailable=len(re.findall(r'^\s*[0-9a-fA-F]+:[^\r\n]*<not\s+available>[^\r\n]*$',text,re.M|re.I))
    unknown=bool(re.search(placeholders+r'|unknown instruction|unsupported|can.t disassemble|invalid instruction',text,re.I))
    r.update(status='UNRESOLVED_TOOL_OUTPUT' if r['returncode']!=0 or unknown or not lines else 'DISASSEMBLY_COLLECTED_REQUIRES_REVIEW',
        instruction_like_lines=len(lines),unknown_diagnostic=unknown,unavailable_address_lines=unavailable,
        decode_limitation='TOOL_EMITTED_UNDECODED_ADDRESSES' if unavailable else 'NO_NOT_AVAILABLE_MARKER_OBSERVED',
        source_attribution='UNRESOLVED; inspect actual DWARF line records and generated-source binding',
        dynamic_execution_cost='UNRESOLVED')
    return r


def pair_artifacts(left,right):
    groups={}
    for side,items in [('left',left),('right',right)]:
        for item in items:
            key=(item['core_type'],item['elf']['layout']['machine'],item['elf']['layout']['elf_type'])
            groups.setdefault(key,{'left':[],'right':[]})[side].append(item)
    rows=[]
    for key,g in groups.items():
        row=dict(core_type=key[0],machine=key[1],elf_type=key[2],left_count=len(g['left']),right_count=len(g['right']))
        if len(g['left'])!=1 or len(g['right'])!=1:
            row.update(status='UNRESOLVED_AMBIGUOUS_ARTIFACT_PAIRING')
        else:
            a,b=g['left'][0],g['right'][0]
            row.update(status='ELF_BYTES_COMPARED_NOT_PERFORMANCE',left=a['binary'],right=b['binary'],
                whole_elf_equal=a['binary']['sha256']==b['binary']['sha256'],**compare_sections(a['elf'],b['elf']))
        rows.append(row)
    return rows


def main(args):
    report_path=args.build_report.resolve(strict=True)
    report=json.loads(report_path.read_text())
    protected=[report_path.parent,args.cann_root,HERE.parents[2]]
    if args.benchmark_report:protected.append(args.benchmark_report.resolve().parent)
    for v in report.get('variants',{}).values():
        protected.append(Path(v['opp']))
        protected.extend(Path(x['binary']['path']).parent for x in v['installed']['kernels'])
    root=f.fresh(args.output_dir,protected)
    r=dict(status='NOT_RUN',phase='identity',source_build_report=f.identity(report_path),
        sources=[f.identity(HERE/'inspect_binary.py'),f.identity(HERE.parents[1]/'draft_quant/simulator/bottleneck_elf.py')],
        new_cann_compile='NOT_RUN',npu='NOT_RUN',new_profile='NOT_RUN',new_kernel='NOT_CREATED',
        source_instruction_attribution='UNRESOLVED',performance_cause='UNRESOLVED')
    try:
        f.verify_tree(report)
        if report.get('status')!='STAGE4F_BUILT_NPU_NOT_RUN' or report.get('sources')!=f.sources():raise ValueError('Stage4F build/source identity mismatch')
        f.verify_audit(report['source_audit'],f.verify)
        if report.get('binary_comparison')!=f.binary_summary(report['variants']):raise ValueError('binary_comparison differs from artifacts')
        if set(report['variants'])!={'A0','A1','B0','B1'}:raise ValueError('incomplete variants')
        r['original_binary_comparison']=report['binary_comparison']
        if args.benchmark_report:
            bench=json.loads(args.benchmark_report.read_text())
            f_run.accepted_report(args.benchmark_report,'STAGE4F_CONTRASTS_MEASURED_DIAGNOSTIC_ONLY',bench,report)
            if bench['pair_build_report']!=f.identity(report_path):raise ValueError('benchmark belongs to a different build')
            r['benchmark_report']=f.identity(args.benchmark_report)
            r['observed_pair_statistics']=bench['comparison']['pairs']
            r['host_configurations']=sorted({x['kernel_configuration'] for x in bench['runs']})
            r['runtime_plans']=[x['plan'] for x in bench['runs']]

        candidates=tool_candidates(args.cann_root)
        r['tool_candidates']=[f.identity(p) for p in candidates]
        tool=args.objdump.resolve(strict=True) if args.objdump else (candidates[0] if len(candidates)==1 else None)
        r['tool_selection']='SELECTED_NOT_YET_CAPABILITY_VERIFIED' if tool else 'UNRESOLVED_MISSING_OR_AMBIGUOUS_USE_EXPLICIT_OBJDUMP'
        if tool:
            r['tool']=f.identity(tool);r['tool_version']=command([tool,'--version'],root/'objdump-version.txt')
        r['variants']={};r['phase']='elf_inspection'
        for v in ('A0','A1','B0','B1'):
            folder=root/v;folder.mkdir();items=[];r['variants'][v]=items
            for index,k in enumerate(report['variants'][v]['installed']['kernels']):
                path=f.verify(k['binary']);meta=json.loads(f.verify(k['metadata']).read_text())
                core=meta.get('coreType',meta.get('core_type','UNRESOLVED'))
                if not isinstance(core,str):core='UNRESOLVED'
                row=dict(binary=k['binary'],metadata=k['metadata'],core_type=core,
                    core_type_basis='Explicit compiler metadata only; no inference from Scalar/Vector activity',elf=sections(path))
                if tool:row['disassembly']=disassemble(tool,path,folder/(str(index)+'-disassembly.txt'))
                else:row['disassembly']=dict(status='NOT_RUN_TOOL_UNRESOLVED')
                items.append(row)
        r['pairs']={p:pair_artifacts(r['variants'][p+'0'],r['variants'][p+'1']) for p in ('A','B')}
        r.update(status='STAGE4G_OFFLINE_ELF_REVIEW_PARTIAL',phase='complete')
    except Exception as e:r.update(status='BLOCKED',error=f'{type(e).__name__}: {e}')
    finally:
        try:
            f.verify(r['source_build_report']);f.verify_tree(report)
            if r.get('benchmark_report'):f.verify(r['benchmark_report']);f.verify_tree(bench)
            for item in r['sources']:f.verify(item)
            if r.get('tool'):f.verify(r['tool'])
        except Exception as e:r.update(status='BLOCKED',integrity_error=str(e))
        r['artifacts']=[f.identity(p) for p in sorted(root.rglob('*.txt'))]
        f.write(root/'binary-analysis.json',r)
    print(r['status']+': '+str(root/'binary-analysis.json'))
    return int(r['status']=='BLOCKED')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark-report',type=Path);p.add_argument('--build-report',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--cann-root',type=Path,default=Path('/usr/local/Ascend/cann-9.0.0'));p.add_argument('--objdump',type=Path)
    raise SystemExit(main(p.parse_args()))
