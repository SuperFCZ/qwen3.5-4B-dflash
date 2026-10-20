#!/usr/bin/env python3
"""One N64 tile, original full-K Gate ABI. Simulator diagnostic only."""
import argparse
import json
from pathlib import Path
import shutil

import run as base
from bounded_prepare import SHAPE,patch_kernel,tile_number,compare_tile,inspect_partial,schedule
from bounded_budget import BudgetCommands,usage,limits
from simulator_prepare import artifacts,profile_command


def inputs(source,folder):
    folder.mkdir()
    manifest=Path(source['bundle']);bundle=base.load_bundle(manifest,'a4')
    case=next(c for c in bundle['cases'] if c['name']=='layer-0-gate_up')
    base.validate_layout(manifest.parent,case,'a4')
    originals=[base.ident(manifest)]
    for name in ('x.bin','w_nz.bin','s_gn.bin'):
        src=base.checked_file(manifest.parent,case['files'][name]);originals.append(base.ident(src))
        shutil.copyfile(src,folder/name)
        if base.sha256(folder/name)!=case['files'][name]['sha256']:raise ValueError('input copy changed')
    golden=Path(source['golden']);originals.append(base.ident(golden))
    row=next(c for c in json.loads(golden.read_text())['cases'] if c['name']=='layer-0-gate_up')
    src=base.checked_file(golden.parent,row['outputs']['native_om'][0]);originals.append(base.ident(src))
    shutil.copyfile(src,folder/'expected.bin')
    if base.sha256(folder/'expected.bin')!=base.sha256(src):raise ValueError('Golden copy changed')
    inventory=dict(originals=originals,copies={p.name:base.ident(p) for p in folder.iterdir()})
    base.write_json(folder.parent/'inputs.json',inventory)
    return inventory


def verify_unchanged(records):
    for item in records:
        if base.ident(item['path'])!=item:raise ValueError('read-only evidence/input changed: '+item['path'])


def simulate(mode,runner,env,built,root,args,source,commands):
    folder=root/mode/'bounded';folder.mkdir();case=folder/'input'
    inventory=inputs(source,case)
    try:
        cmd=profile_command(args.msopprof,runner,folder/'profile',case,SHAPE,args.timeout_minutes)
        # Parse every launched core: owner is nTile % *actual* blockDim, which
        # is checked from original Host Tiling after execution. This filter
        # never determines how much work the patched kernel executes.
        cmd.remove('--core-id=0')
        commands(cmd,folder/'simulator.log',env,folder,args.timeout_minutes*60+30)
        ex=json.loads((case/'execution.json').read_text())
        if ex.get('status')!='SIMULATOR_COMPLETED' or ex.get('runtime')!='AscendCL camodel' or ex.get('repetitions')!=1 or ex.get('guards_intact') is not True or ex.get('input_readonly') is not True or ex.get('timing',{}).get('status')!='NOT_RUN':raise ValueError('simulator completion/guards/readonly/one-call gate failed')
        if tuple(ex.get(k) for k in ('m','k','n'))!=SHAPE:raise ValueError('Gate ABI shape changed')
        cfg=base.load_build_config(root/mode/'build-config.json')
        launch=base.launch_evidence(case/'application.log',SHAPE,cfg['core_limit'],ex['workspace_bytes'],cfg['dequant_mode'],cfg['pipeline_mode'],5,cfg['kv_m80_mode'],cfg['scale_mode'],256,cfg['k_tile_policy'],mode)
        check=compare_tile((case/'actual-0.bin').read_bytes(),(case/'expected.bin').read_bytes(),args.n_tile)
        collected=artifacts(folder/'profile',base.record)
        collected.update(tile_mapping='SOURCE_BOUNDED_SINGLE_N_TILE_FULL_K',
            limitation='One N tile only, unmodified full K body. Dynamic individual K/PC labels are not inferred from instruction counts.')
        binaries=[base.ident(p) for p in (folder/'profile').rglob('*.o') if p.is_file()]
        matched=any(p['sha256'] in {k['sha256'] for k in built['kernels']} for p in binaries)
        result=dict(status='BOUNDED_SIMULATOR_DIAGNOSTIC_ONLY',scope='partial N64 output only',checks=check,
            launch=launch,owner=args.n_tile%launch['block_dim'],planned_k_schedule=schedule(args.n_tile,launch['block_dim'],mode=='raw'),
            kernel_identity=dict(status='MATCHED' if matched else 'UNVERIFIED',binaries=binaries),artifacts=collected,
            source_adaptation=built['simulator_source_adaptation'],disk=usage(folder/'profile'),
            actual=base.ident(case/'actual-0.bin'),execution=base.ident(case/'execution.json'),
            full_gate_correctness='NOT_RUN',full_draft_token_kv='NOT_RUN',npu_correctness='NOT_RUN',hardware_performance='NOT_RUN')
        base.write_json(folder/'result.json',result)
        if not check['selected_tile_bitwise_equal'] or not check['skipped_output_poison_intact'] or not matched or collected['status']!='COLLECTED':raise ValueError('bounded output/poison/kernel identity/artifact gate failed')
        return result,inventory
    finally:verify_unchanged([*inventory['originals'],*inventory['copies'].values()])


def run(args):
    root=args.output_dir.absolute()
    if root.resolve()!=root:raise ValueError('output must not use symlinks')
    if root==args.partial_report.resolve().parent or args.partial_report.resolve().parent in root.parents:raise ValueError('never write into old evidence')
    root.mkdir(parents=True,exist_ok=False);root.chmod(0o700)
    report=dict(status='PREPARING',phase='read-only-partial-inspection',n_tile=args.n_tile,shape=list(SHAPE),runs={},
        full_gate_correctness='NOT_RUN',full_draft_token_kv='NOT_RUN',npu_correctness='NOT_RUN',hardware_performance='NOT_RUN')
    watched=[base.ident(base.HERE/n) for n in (*base.FROZEN,*base.BORROWED_RUNNER)]
    commands=BudgetCommands(root,args.max_output_mib*1024**2,args.max_file_mib*1024**2)
    def save():base.write_json(root/'bounded-report.json',report)
    save()
    try:
        partial=inspect_partial(args.partial_report,root/'partial-inspection.json',base.ident)
        report['partial_inspection']=base.record(root/'partial-inspection.json',root)
        watched += [partial['prior_report'],partial['prior_execution']]
        watched += [{k:v for k,v in item.items() if k in ('path','bytes','sha256')} for item in partial['artifacts']]
        # The interrupted binary must have originated in the same frozen source;
        # this is source provenance, not a claim that the interrupted run passed.
        oldmap=args.partial_report.resolve().parent/'off/source-map.json'
        old=json.loads(oldmap.read_text())
        if old['kernel']['sha256']!=base.sha256(base.HERE/'op_kernel/d_flash_group_quant_linear.cpp'):raise ValueError('partial trace source differs from frozen Kernel')
        watched += [base.ident(oldmap),base.ident(old['kernel']['path'])]
        if watched[-1]['sha256']!=old['kernel']['sha256']:raise ValueError('partial Kernel source changed')
        if args.inspect_only:
            report.update(status='PARTIAL_INSPECTED_NOT_EXECUTED',phase='complete')
        else:
            report['phase']='accepted-preflight';save()
            env,source,configs,prepared=base.preflight(args,root);report['preflight']=prepared
            def watch(value):
                if isinstance(value,dict):
                    if {'path','sha256','bytes'}<=value.keys() and Path(value['path']).is_absolute():watched.append({k:value[k] for k in ('path','sha256','bytes')})
                    for item in value.values():watch(item)
                elif isinstance(value,list):
                    for item in value:watch(item)
            watch(prepared)
            pair=[]
            for mode in ('off','raw'):
                report['phase']='build-'+mode;save()
                runner,app_env,built=base.build(mode,configs[mode],root,args,env,
                    kernel_adapter=lambda path:patch_kernel(path,args.n_tile),command_runner=commands)
                report['runs'][mode]={'build':built,'bounded':'NOT_RUN'}
                report['phase']='bounded-'+mode;save()
                result,inventory=simulate(mode,runner,app_env,built,root,args,source,commands)
                report['runs'][mode]['bounded']=result
                pair.append({name:record['sha256'] for name,record in inventory['copies'].items()})
            if pair[0]!=pair[1]:raise ValueError('Off/Raw inputs or Golden differ')
            report.update(status='BOUNDED_SIMULATOR_DIAGNOSTIC_ONLY',phase='complete',paired_input_identity='MATCHED')
    except Exception as error:report.update(status='BLOCKED',error=f'{type(error).__name__}: {error}')
    finally:
        try:verify_unchanged(watched);limits(root,commands.total,commands.file_limit)
        except Exception as error:report.update(status='BLOCKED',integrity_or_budget_error=str(error))
        report['disk']=usage(root);save()
        (root/'bounded-analysis.md').write_text('# Bounded Gate Simulator\n\n'+
            'Status: '+report['status']+'; phase: '+report['phase']+'\n\n'+
            f'N tile {args.n_tile}; full logical M16/K2560/N19456; only 1024 output elements compared.\n\n'+
            'Complete Gate / NPU / full Draft / hardware performance: NOT_RUN. Cycles are not hardware wall-time.\n\n'+
            'Original K loop and event lifecycle are byte-identical after removing the two audited scope insertions. '+
            'See bounded-source.diff/json, source-map.json and profiled binary hash; trace/CSV are never synthesized.\n\n'+
            'Per-file RLIMIT and polled total budget stop this process group on excess; incomplete artifacts never imply completion.\n\n'+
            str(report.get('error',''))+'\n')
    print(report['status']+': '+str(root/'bounded-report.json'),flush=True)
    return int(report['status']=='BLOCKED')


def parse():
    parser=argparse.ArgumentParser(description=__doc__)
    # Reuse defaults, not the unbounded run function or its execution stages.
    defaults=base.parse(['--output-dir','unused'])
    for name in ('msopprof','sim_lib','cann_root','a51_evidence','prefetch_evidence'):
        parser.add_argument('--'+name.replace('_','-'),type=Path,default=getattr(defaults,name))
    parser.add_argument('--partial-report',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/gate-simulator-full-20261020-084557/simulator-report.json'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--n-tile',type=int,default=0)
    parser.add_argument('--timeout-minutes',type=int,default=5)
    parser.add_argument('--max-output-mib',type=int,default=1024)
    parser.add_argument('--max-file-mib',type=int,default=256)
    parser.add_argument('--inspect-only',action='store_true')
    args=parser.parse_args();tile_number(args.n_tile)
    if not 1<=args.timeout_minutes<=20:parser.error('bounded timeout must be 1..20 minutes; default 5, no automatic escalation')
    if not 1<=args.max_file_mib<=args.max_output_mib:parser.error('require 1 <= per-file <= total disk budget')
    return args

if __name__=='__main__':raise SystemExit(run(parse()))
