#!/usr/bin/env python3
"""Read-only bounded Simulator analysis. No model/device/profiler execution."""
import argparse
import difflib
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from bounded_prepare import bounded_source
from simulator_prepare import focus_lines
from bottleneck_parse import csv_table,trace_schema,pc,symbolizer_frames,instruction_summary,trace_analysis,source_hotspots,compare_source_hotspots
from bottleneck_elf import elf_layout,address_space,llvm_frames


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def dump(path,value):
    def encode(item):
        if isinstance(item,Decimal):return str(item)
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,
        default=encode)+'\n')


class Evidence:
    def __init__(self,root):self.root=root.resolve();self.files={}
    def read(self,path,record=None,inside=True):
        path=Path(path).resolve()
        if inside and not path.is_relative_to(self.root):raise ValueError('evidence path escape: '+str(path))
        actual=dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path))
        if record and any(actual[k]!=record[k] for k in ('bytes','sha256')):raise ValueError('evidence size/SHA256 mismatch: '+str(path))
        if str(path) in self.files and self.files[str(path)]!=actual:raise ValueError('evidence changed during read')
        self.files[str(path)]=actual
        return path
    def absolute(self,item,inside=True):
        if not Path(item['path']).is_absolute():raise ValueError('expected absolute identity record')
        return self.read(item['path'],item,inside)
    def relative(self,root,item):
        p=Path(item['path'])
        if p.is_absolute() or not (root/p).resolve().is_relative_to(root.resolve()):raise ValueError('relative artifact escapes root')
        return self.read(root/p,item)
    def verify(self):
        for path,item in list(self.files.items()):self.read(path,item,inside=False)


def obj(path):return json.loads(path.read_text())


def load_mode(evidence,report,mode):
    root=evidence.root;item=report['runs'][mode];built=item['build'];result=item['bounded']
    folder=root/mode
    if obj(evidence.read(folder/'build-report.json'))!=built or obj(evidence.read(folder/'bounded/result.json'))!=result:raise ValueError('nested report identity mismatch')
    checks=result['checks']
    if checks['n_tile']!=report['n_tile'] or checks['elements']!=1024:raise ValueError('selected tile differs')
    if result['status']!='BOUNDED_SIMULATOR_DIAGNOSTIC_ONLY' or result['owner']!=0 or result['artifacts']['status']!='COLLECTED' or result['kernel_identity']['status']!='MATCHED' or checks['bit_mismatches']!=0 or checks['selected_tile_bitwise_equal'] is not True or checks['skipped_output_poison_intact'] is not True:raise ValueError('requires completed owner-core0 bounded bitwise evidence')
    budget=obj(evidence.read(folder/'bounded/simulator.budget.json'))
    if budget['status']!='COMMAND_COMPLETED' or budget.get('final_budget_error') is not None:raise ValueError('profile budget/completion gate failed')
    execution=obj(evidence.absolute(result['execution']))
    if execution['status']!='SIMULATOR_COMPLETED' or execution['guards_intact'] is not True or execution['input_readonly'] is not True:raise ValueError('guard/readonly gate failed')
    inventory=obj(evidence.read(folder/'bounded/inputs.json'))
    for rec in inventory['originals']:evidence.absolute(rec,inside=False)
    for rec in inventory['copies'].values():evidence.absolute(rec)
    actual=evidence.absolute(result['actual']).read_bytes()
    golden=evidence.absolute(inventory['copies']['expected.bin']).read_bytes();tile=report['n_tile'];lo=tile*64*2;hi=lo+128;stride=19456*2
    if len(actual)!=16*stride or len(golden)!=16*stride:raise ValueError('full ABI byte length changed')
    for row in range(16):
        a=actual[row*stride:(row+1)*stride];g=golden[row*stride:(row+1)*stride]
        if a[lo:hi]!=g[lo:hi] or a[:lo]!=b'\x7f'*lo or a[hi:]!=b'\x7f'*(stride-hi):raise ValueError('bounded output/poison no longer matches')
    cfg=obj(evidence.absolute(built['config']))
    if cfg['gate_prefetch_mode']!=mode or cfg['gate_copy_mode']!='loop' or cfg['k_tile']!=256 or cfg['scale_mode']!='broadcast':raise ValueError('not Off/Raw Loop K256 Broadcast')
    source_map=obj(evidence.relative(folder,built['source_map']));source=evidence.absolute(source_map['kernel'])
    adaptation=obj(evidence.read(folder/'bounded-source.json'))
    if adaptation!=built['simulator_source_adaptation'] or adaptation!=result['source_adaptation']:raise ValueError('source adaptation identity mismatch')
    text=source.read_text();original=text
    for insertion in adaptation['insertions']:
        if original.count(insertion)!=1:raise ValueError('bounded insertion not unique')
        original=original.replace(insertion,'',1)
    if hashlib.sha256(original.encode()).hexdigest()!=adaptation['original_sha256'] or digest(source)!=adaptation['bounded_sha256']:raise ValueError('bounded source SHA differs')
    if bounded_source(original,tile)[0]!=text:raise ValueError('unapproved bounded source changes')
    diff=''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='accepted/kernel.cpp',tofile='isolated/bounded/kernel.cpp'))
    if diff!=adaptation['diff'] or evidence.read(folder/'bounded-source.diff').read_text()!=diff:raise ValueError('bounded source diff changed')
    if adaptation['original_sha256']!=cfg['source_sha256']['op_kernel/d_flash_group_quant_linear.cpp']:raise ValueError('original source/build identity differs')
    if focus_lines(source)!=adaptation['focus_lines']:raise ValueError('source line map changed')
    binaries={k['sha256']:evidence.absolute(k) for k in built['kernels']}
    matched=[]
    for rec in result['kernel_identity']['binaries']:
        path=evidence.absolute(rec)
        if rec['sha256'] in binaries:matched.append(path)
    if len({digest(p) for p in matched})!=1:raise ValueError('ambiguous/missing profiled ELF identity')
    binary=matched[0]
    kernel=next(k for k in built['kernels'] if k['sha256']==digest(binary))
    if kernel.get('debug_info') is not True or kernel.get('debug_line') is not True:raise ValueError('no recorded DWARF info/line sections')
    readelf=evidence.relative(folder,kernel['readelf'])
    if '.debug_info' not in readelf.read_text() or '.debug_line' not in readelf.read_text():raise ValueError('missing DWARF ELF audit')
    paths={}
    for kind,records in result['artifacts']['files'].items():
        paths[kind]=[evidence.relative(folder/'bounded/profile',r) for r in records]
    code=[p for p in paths['code_hotspots'] if p.name=='core0_code_exe.csv']
    instr=[p for p in paths['instructions'] if p.name=='core0_instr_exe.csv']
    if len(code)!=1 or len(instr)!=1 or code[0].parent!=instr[0].parent:raise ValueError('ambiguous core0 CSV selection')
    trace=code[0].parent/'trace.json'
    if trace not in paths['trace']:raise ValueError('no hash-bound per-core0 trace; aggregate trace not substituted')
    return dict(source=source,focus=adaptation['focus_lines'],binary=binary,code=code[0],instr=instr[0],trace=trace,
                source_sha256=adaptation['bounded_sha256'],profile_root=folder/'bounded/profile',
                inputs={k:v['sha256'] for k,v in inventory['copies'].items()},config=cfg)


def symbolize(tool,binary,addresses,out):
    if not tool or not addresses:return {},dict(status='UNRESOLVED',reason='no addr2line tool or PC inventory')
    tool=Path(tool).resolve();args=[str(tool),'-a','-f','-C','-i','-e',str(binary)]
    request=''.join(f'0x{x:x}\n' for x in sorted(addresses));(out/'addr2line-input.txt').write_text(request)
    dump(out/'addr2line-command.json',dict(argv=args,tool_sha256=digest(tool),binary_sha256=digest(binary),address_policy='exact PC; no guessed relocation'))
    try:
        run=subprocess.run(args,input=request,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=out,timeout=120)
        (out/'addr2line-output.txt').write_text(run.stdout);(out/'addr2line-stderr.txt').write_text(run.stderr)
        if run.returncode:raise ValueError('addr2line exit '+str(run.returncode))
        mapping=symbolizer_frames(run.stdout,addresses)
        return mapping,dict(status='DWARF_DECODED' if any(mapping.values()) else 'UNRESOLVED',mapped_pcs=sum(bool(v) for v in mapping.values()),pc_count=len(addresses))
    except (ValueError,subprocess.TimeoutExpired,OSError) as error:return {},dict(status='UNRESOLVED',reason=str(error))


def discover_llvm(explicit,report,evidence):
    candidates=[];searched=[]
    if explicit:
        candidates=[Path(explicit)];searched=['explicit --llvm-symbolizer']
    else:
        found=shutil.which('llvm-symbolizer')
        if found:candidates=[Path(found)];searched=['PATH']
        else:
            tool=report.get('preflight',{}).get('tool')
            if tool:
                installed=evidence.absolute(tool,inside=False).parent.parent
                searched=[str(installed)]
                # Search only the previously identified installed msopprof
                # package, not the whole host or another guessed SDK checkout.
                candidates=list(installed.rglob('llvm-symbolizer'))
    candidates=sorted({p.resolve() for p in candidates if p.is_file() and os.access(p,os.X_OK)})
    records=[]
    for p in candidates:
        evidence.read(p,inside=False);records.append(evidence.files[str(p)])
    chosen=candidates[0] if len(candidates)==1 else None
    return chosen,dict(status='AVAILABLE' if chosen else 'UNAVAILABLE_OR_AMBIGUOUS',searched=searched,candidates=records)


def llvm_decode(tool,binary,addresses,out,prefix):
    if not tool or not addresses:return {},dict(status='UNRESOLVED',reason='no existing LLVM tool or eligible ELF addresses')
    argv=[str(tool),'--output-style=JSON','--inlines','--demangle','--obj='+str(binary)]
    request=''.join(f'0x{a:x}\n' for a in sorted(addresses))
    (out/(prefix+'-input.txt')).write_text(request)
    dump(out/(prefix+'-command.json'),dict(argv=argv,tool_sha256=digest(tool),binary_sha256=digest(binary),
         scope='ELF virtual address probe; not a runtime PC translation',addresses=[hex(a) for a in sorted(addresses)]))
    env=dict(os.environ);env.pop('LLVM_SYMBOLIZER_OPTS',None);env['DEBUGINFOD_URLS']=''
    try:
        run=subprocess.run(argv,input=request,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=out,env=env,timeout=120)
        (out/(prefix+'-output.json')).write_text(run.stdout);(out/(prefix+'-stderr.txt')).write_text(run.stderr)
        if run.returncode:raise ValueError('llvm-symbolizer exit '+str(run.returncode))
        mapping=llvm_frames(run.stdout,addresses)
        return mapping,dict(status='ELF_DWARF_DECODED' if any(mapping.values()) else 'UNRESOLVED',
            mapped_pcs=sum(bool(frames) for frames in mapping.values()),pc_count=len(addresses),frames={hex(a):f for a,f in mapping.items()})
    except (OSError,ValueError,subprocess.TimeoutExpired) as error:return {},dict(status='UNRESOLVED',reason=str(error))


def elf_diagnostic(spec,addresses,out,llvm,addr2line):
    try:layout=elf_layout(spec['binary'])
    except ValueError as error:layout=dict(status='UNRESOLVED_ELF_LAYOUT',reason=str(error))
    correspondence=address_space(layout,addresses)
    audit=dict(layout=layout,trace_pc_correspondence=correspondence,load_bias='UNRESOLVED_NOT_INFERRED')
    if llvm:
        try:
            version=subprocess.run([str(llvm),'--version'],text=True,capture_output=True,timeout=15)
            (out/'llvm-version.txt').write_text(version.stdout+version.stderr)
            audit['llvm_version_exit']=version.returncode
        except (OSError,subprocess.TimeoutExpired) as error:audit['llvm_version_error']=str(error)
    probes=set()
    if layout.get('status')=='ELF_LAYOUT_READ':
        text=layout['text']
        probes={text['address']+offset for offset in (0,4) if offset<text['bytes']}
    _,probe=llvm_decode(llvm,spec['binary'],probes,out,'elf-relative-probe')
    audit['llvm_elf_relative_probe']=probe
    # Probe success never supplies an offset. The observed device PCs are
    # outside .text/LOAD; no subtraction or address search is attempted.
    mapping={};symbols=dict(status='UNRESOLVED',mapped_pcs=0,pc_count=len(addresses),reason='runtime PC/ELF correspondence unverified')
    if correspondence['status']=='DIRECT_ELF_VIRTUAL_ADDRESSES':
        if llvm:
            mapping,symbols=llvm_decode(llvm,spec['binary'],addresses,out,'runtime-pc')
        else:mapping,symbols=symbolize(addr2line,spec['binary'],addresses,out)
    dump(out/'elf-diagnostic.json',audit)
    return mapping,symbols,audit


def analyze(args):
    report_path=args.report.resolve();root=report_path.parent;output=args.output_dir.absolute()
    if output.resolve()!=output or output==root or root in output.parents:raise ValueError('use a new output outside original evidence')
    output.mkdir(parents=True,exist_ok=False);output.chmod(0o700)
    evidence=Evidence(root);analysis=dict(status='PREPARING',scope='offline bounded one N64 full-K simulator diagnostic',modes={},
        full_gate_correctness='NOT_RUN',npu_correctness='NOT_RUN',hardware_performance='NOT_RUN',full_draft_token_kv='NOT_RUN')
    formats={}
    try:
        report=obj(evidence.read(report_path))
        if report['status']!='BOUNDED_SIMULATOR_DIAGNOSTIC_ONLY' or report['phase']!='complete' or report['paired_input_identity']!='MATCHED' or set(report['runs'])!={'off','raw'}:raise ValueError('need completed paired bounded report')
        specs={mode:load_mode(evidence,report,mode) for mode in ('off','raw')}
        if specs['off']['inputs']!=specs['raw']['inputs']:raise ValueError('Off/Raw frozen input mismatch')
        for key in ('source_sha256','core_limit','dequant_mode','pipeline_mode','kv_m80_mode','scale_mode','k_tile','gate_copy_mode'):
            if specs['off']['config'].get(key)!=specs['raw']['config'].get(key):raise ValueError('Off/Raw changes another variable: '+key)
        tool=args.addr2line or shutil.which('llvm-addr2line') or shutil.which('addr2line')
        llvm=None
        if not args.inspect_only:
            llvm,analysis['llvm_symbolizer_availability']=discover_llvm(getattr(args,'llvm_symbolizer',None),report,evidence)
        for mode,spec in specs.items():
            out=output/mode;out.mkdir()
            code_schema,code=csv_table(spec['code'],'code');instr_schema,rows=csv_table(spec['instr'],'instructions')
            trace_format=trace_schema(spec['trace'])
            formats[mode]=dict(code_path=str(spec['code']),instruction_path=str(spec['instr']),trace_path=str(spec['trace']),
                code=code_schema,instructions=instr_schema,trace=trace_format)
            if args.inspect_only:continue
            mapping,symbols,elf=elf_diagnostic(spec,{pc(r['addr']) for r in rows if pc(r['addr']) is not None},out,llvm,tool)
            code_result=source_hotspots(code,spec['source'],spec['focus'],spec['source_sha256'])
            summary=instruction_summary(rows,mapping,spec['source'],spec['focus'])
            trace=trace_analysis(spec['trace'],rows,mapping,spec['source'],spec['focus'])
            dump(out/'instructions.json',summary.pop('instructions'));dump(out/'timeline-intervals.json',trace.pop('intervals',[]))
            dump(out/'cachemiss-events.json',trace.get('cachemiss',{}).pop('events',[]))
            # Code CSV line totals can include inline caller attribution; keep
            # them separate and never add them to PC instruction totals.
            dump(out/'code-hotspots.json',code_result)
            visualization=[]
            for path in sorted(spec['profile_root'].rglob('visualize_data.bin')):
                evidence.read(path)
                visualization.append(dict(evidence.files[str(path.resolve())],status='CURRENT_HASH_ONLY_NOT_DECODED',historical_manifest_binding='NOT_RECORDED'))
            analysis['modes'][mode]=dict(csv_schema=instr_schema['status'],dwarf=symbols,elf=elf,
                source_hotspots=code_result,visualization=visualization,instructions=summary,trace=trace)
        if args.inspect_only:analysis['status']='FORMAT_EVIDENCE_ONLY'
        else:
            analysis['status']='OFFLINE_ANALYSIS_PARTIAL'
            analysis['comparison']=dict(kernel_elapsed_cycles='UNRESOLVED_NO_VERIFIED_CYCLE_CLOCK',
                synchronization_wait_fraction='UNRESOLVED_NO_DEPENDENCY_INTERVALS',
                raw_prefetch_earlier_issue='UNRESOLVED_PER_K_DATA_DEPENDENCIES',
                exposed_mte2_wait_reduction='UNRESOLVED',pipe_all_limits_overlap='UNRESOLVED',
                instruction_cycles_delta={})
            for pipe in sorted(set(analysis['modes']['off']['instructions']['by_reported_pipe'])|set(analysis['modes']['raw']['instructions']['by_reported_pipe'])):
                a=analysis['modes']['off']['instructions']['by_reported_pipe'].get(pipe)
                b=analysis['modes']['raw']['instructions']['by_reported_pipe'].get(pipe)
                analysis['comparison']['instruction_cycles_delta'][pipe]=b['instruction_cycles']-a['instruction_cycles'] if a and b else None
            left=analysis['modes']['off']['trace'];right=analysis['modes']['raw']['trace']
            if left.get('status')==right.get('status')=='PC_INSTRUCTION_COUNTS_MATCHED' and left['clock']==right['clock']:
                analysis['comparison']['observed_trace_span_delta_raw_units']=right['trace_instruction_span']-left['trace_instruction_span']
                analysis['comparison']['observed_trace_union_delta_raw_units']=right['interval_union']-left['interval_union']
            else:analysis['comparison']['trace_span_comparison']='UNRESOLVED'
            analysis['comparison']['instruction_cycle_sum_delta']=sum(analysis['comparison']['instruction_cycles_delta'].values(),Decimal(0)) if all(v is not None for v in analysis['comparison']['instruction_cycles_delta'].values()) else None
            analysis['comparison']['source_hotspot_rows']=compare_source_hotspots(analysis['modes']['off']['source_hotspots'],analysis['modes']['raw']['source_hotspots'])
    except Exception as error:analysis.update(status='BLOCKED',error=f'{type(error).__name__}: {error}')
    finally:
        try:evidence.verify()
        except Exception as error:analysis.update(status='BLOCKED',integrity_error=str(error))
        analysis['evidence']=evidence.files
        dump(output/'format-evidence.json',formats);dump(output/'bottleneck-analysis.json',analysis)
        lines=['# Bounded Gate offline bottleneck analysis','', 'Status: '+analysis['status'],
            'Read-only original evidence. No Simulator, model, NPU or compiler execution.','',
            'CSV cycles are instruction execution totals, not elapsed cycles; overlapping pipes cannot be added as kernel time.',
            'Trace interval unions/intersections use unconverted Trace ts/dur units, never NPU ms. Zero-width events may still have nonzero instruction cycles.',
            'Critical path, synchronization wait fraction, exposed MTE2 wait and next-K prefetch causality remain UNRESOLVED without dependency evidence.','']
        for mode,result in analysis['modes'].items():
            lines += [f"{mode}: instruction DWARF={result['dwarf']['status']}; source CSV={result['source_hotspots']['status']}; Trace={result['trace']['status']}",
                'ELF Machine: '+str(result['elf']['layout'].get('machine_hex'))+'; runtime PC address correspondence: '+result['elf']['trace_pc_correspondence']['status'],'',
                'Hash-verified source hotspots (individual rows; inclusive costs are NOT summed):','',
                '| Category | Source line | Reported instruction involvement | Reported cycles |','| --- | ---: | ---: | ---: |']
            for category,rows in result['source_hotspots']['by_category'].items():
                if category=='UNMAPPED':continue
                for row in rows[:8]:lines += [f"| {category} | {row['line']} | {row['call_count']} | {row['cycles']} |"]
            lines += ['', 'Unbound wrapper/SDK source rows: '+str(result['source_hotspots']['unmapped_rows'])+' (UNMAPPED).',
                '', '| Pipe | Instruction calls | Instruction cycles (not wall-time) |','| --- | ---: | ---: |']
            for pipe,values in result['instructions']['by_reported_pipe'].items():lines += [f"| {pipe} | {values['call_count']} | {values['instruction_cycles']} |"]
            lines += ['', '| Verified source operation | Instruction calls | Instruction cycles |','| --- | ---: | ---: |']
            for operation,values in result['instructions']['by_source_operation'].items():
                lines += [f"| {operation} | {values['call_count']} | {values['instruction_cycles']} |"]
            trace=result['trace']
            lines += ['', 'Trace interval evidence (raw ts/dur units, not cycles):',
                '- Observed instruction span: '+str(trace.get('trace_instruction_span')),
                '- Instruction interval union: '+str(trace.get('interval_union')),
                '- Zero-duration events: '+str(trace.get('zero_duration_events')),
                '- Independent CACHEMISS channel: '+json.dumps(trace.get('cachemiss',{}),default=str),
                '- MTE2 pairwise overlap: '+json.dumps(trace.get('mte2_pairwise_intersections',{}),default=str),
                '- Next-copy source/PC overlap: '+json.dumps(trace.get('prefetch_callsite_overlap',{}),default=str),
                '', 'Other instructions remain UNMAPPED. C++ API call counts and individual K tile IDs are not inferred.','']
        if analysis.get('comparison'):
            comparison={k:v for k,v in analysis['comparison'].items() if k!='source_hotspot_rows'}
            lines += ['Off/Raw differences (Raw minus Off):',json.dumps(comparison,indent=2,default=str),'',
                'Per-source-line paired differences are in JSON; source rows are not summed and are independent of instruction statistics.','']
        lines += ['Next evidence needed: supported DWARF reader if PCs remain unresolved; explicit cycle clock/end marker for elapsed cycles; verified wait/producer dependencies for exposed waiting. No Kernel optimization is proposed as proven by this report.','']
        if analysis.get('error'):lines += ['Error: '+analysis['error']]
        lines += ['','See format-evidence.json for original headers, metadata, full sampled args; per-mode files retain PC mappings and intervals.',
            'Full Gate / NPU correctness / Draft / hardware performance: NOT_RUN.']
        (output/'bottleneck-analysis.md').write_text('\n'.join(lines)+'\n')
    print(analysis['status']+': '+str(output/'bottleneck-analysis.json'))
    return int(analysis['status']=='BLOCKED')


def parse():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/gate-simulator-bounded-2g-racefix-20261020-103256/bounded-report.json'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--addr2line',help='Existing DWARF reader only; no installation, build, or address rebasing')
    parser.add_argument('--llvm-symbolizer',help='Existing tool path; otherwise inspect PATH and the evidence-bound msopprof installation')
    parser.add_argument('--inspect-only',action='store_true')
    return parser.parse_args()

if __name__=='__main__':raise SystemExit(analyze(parse()))
