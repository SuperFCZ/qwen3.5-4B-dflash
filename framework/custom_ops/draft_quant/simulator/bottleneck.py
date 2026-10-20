#!/usr/bin/env python3
"""Read-only bounded Simulator analysis. No model/device/profiler execution."""
import argparse
import difflib
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from bounded_prepare import bounded_source
from simulator_prepare import focus_lines
from bottleneck_parse import csv_table,trace_schema,pc,symbolizer_frames,instruction_summary,trace_analysis


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
        for mode,spec in specs.items():
            out=output/mode;out.mkdir()
            code_schema,code=csv_table(spec['code'],'code');instr_schema,rows=csv_table(spec['instr'],'instructions')
            trace_format=trace_schema(spec['trace'])
            formats[mode]=dict(code_path=str(spec['code']),instruction_path=str(spec['instr']),trace_path=str(spec['trace']),
                code=code_schema,instructions=instr_schema,trace=trace_format)
            if args.inspect_only:continue
            mapping,symbols=symbolize(tool,spec['binary'],{pc(r['addr']) for r in rows if pc(r['addr']) is not None},out)
            summary=instruction_summary(rows,mapping,spec['source'],spec['focus'])
            trace=trace_analysis(spec['trace'],rows,mapping,spec['source'],spec['focus'])
            dump(out/'instructions.json',summary.pop('instructions'));dump(out/'timeline-intervals.json',trace.pop('intervals',[]))
            # Code CSV line totals can include inline caller attribution; keep
            # them separate and never add them to PC instruction totals.
            dump(out/'code-hotspots.json',dict(rows=code,interpretation='Raw documented code rows; not a disjoint instruction partition. No additive hotspot-cycle claim.'))
            analysis['modes'][mode]=dict(csv_schema=instr_schema['status'],dwarf=symbols,instructions=summary,trace=trace)
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
            lines += [f"{mode}: DWARF={result['dwarf']['status']}; Trace={result['trace']['status']}",'', '| Pipe | Instruction calls | Instruction cycles (not wall-time) |','| --- | ---: | ---: |']
            for pipe,values in result['instructions']['by_reported_pipe'].items():lines += [f"| {pipe} | {values['call_count']} | {values['instruction_cycles']} |"]
            lines += ['', '| Verified source operation | Instruction calls | Instruction cycles |','| --- | ---: | ---: |']
            for operation,values in result['instructions']['by_source_operation'].items():
                lines += [f"| {operation} | {values['call_count']} | {values['instruction_cycles']} |"]
            trace=result['trace']
            lines += ['', 'Trace interval evidence (raw ts/dur units, not cycles):',
                '- Observed instruction span: '+str(trace.get('trace_instruction_span')),
                '- Instruction interval union: '+str(trace.get('interval_union')),
                '- Zero-duration events: '+str(trace.get('zero_duration_events')),
                '- MTE2 pairwise overlap: '+json.dumps(trace.get('mte2_pairwise_intersections',{}),default=str),
                '- Next-copy source/PC overlap: '+json.dumps(trace.get('prefetch_callsite_overlap',{}),default=str),
                '', 'Other instructions remain UNMAPPED. C++ API call counts and individual K tile IDs are not inferred.','']
        if analysis.get('comparison'):
            lines += ['Off/Raw differences (Raw minus Off):',json.dumps(analysis['comparison'],indent=2,default=str),'']
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
    parser.add_argument('--inspect-only',action='store_true')
    return parser.parse_args()

if __name__=='__main__':raise SystemExit(analyze(parse()))
