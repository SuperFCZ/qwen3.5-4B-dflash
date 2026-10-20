"""Gate-only correctness gate, followed by existing strict continuous-v1 runner."""
import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import run_a2
from a2_common import checked_file,load_bundle,record,sha256,validate_execution,validate_layout,write_json
from a3_launch import load_build_config,launch_evidence
from reference import compare
from gate_prefetch_contract import check_plan

SHAPE=(16,2560,19456)
COPY_PREFIX='DFLASH_GROUP_QUANT_GATE_COPY '


def selected_layers(layers):
    if not isinstance(layers,(list,tuple)) or not layers or any(type(i) is not int or i not in range(5) for i in layers) or len(set(layers))!=len(layers):
        raise ValueError('layers must be nonempty, unique integers in 0..4')
    return tuple(sorted(layers))


def selected_cases(layers):
    return tuple(f'layer-{i}-gate_up' for i in selected_layers(layers))


def file_identity(path):
    path=Path(path).resolve()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha256(path))


def layer_identity(case,compiled,golden,bundle_root,native_root,golden_root):
    # Full capture/native ABI is checked by load_bundle/validate_om_manifest.
    # Save explicit per-layer binding as well, so a pair cannot swap weights,
    # Native OMs or Golden between identically shaped Gate projections.
    files={name:file_identity(checked_file(bundle_root,item)) for name,item in case['files'].items()}
    native={name:file_identity(checked_file(native_root,compiled[name])) for name in ('om','deployment','air_manifest')}
    old_om=golden.get('native_om',{})
    if old_om.get('sha256')!=native['om']['sha256'] or old_om.get('bytes')!=native['om']['bytes']:
        raise ValueError('frozen Golden Native OM differs for '+case['name'])
    outputs=golden['outputs']['native_om']+golden['bracket_outputs']['native_om']
    if len(outputs)!=4:raise ValueError('incomplete per-layer frozen Golden')
    return dict(name=case['name'],capture_files=files,native=native,
                golden=[file_identity(checked_file(golden_root,item)) for item in outputs])


def copy_evidence(log,mode):
    rows=[json.loads(s.split(COPY_PREFIX,1)[1]) for s in Path(log).read_text().splitlines() if COPY_PREFIX in s]
    expected=dict(version=1,mode=mode,selected=mode)
    if not rows or any(r!=expected for r in rows):raise ValueError('missing/wrong Gate copy launch evidence')
    return dict(**expected,q_params=[8,64,19392,0] if mode=='2d' else None,
                scale_params=[2,4,1212,0] if mode=='2d' else None,raw_calls_per_tile=2 if mode=='2d' else 10,
                tile_iterations=3040,native_bitwise_gate='unchanged')


def validate_untimed(r,device):
    wanted=dict(status='PASS',runtime='AscendCL ACLNN',op='DFlashGroupQuantLinear',cpu_fallback=False,
                input_readonly=True,guards_intact=True,repetitions=2,device_id=device,m=16,k=2560,n=19456,
                tile_n=64,tile_k=256,tile_m=16,weight_reuse_rows=16)
    if any(type(r.get(k)) is not type(v) or r[k]!=v for k,v in wanted.items()):raise ValueError('untimed correctness runner/guards/shape differs')
    t=r.get('timing',{})
    if t.get('status')!='NOT_RUN' or t.get('warmup')!=0 or t.get('repetitions')!=0:raise ValueError('correctness pre-gate must not benchmark')


def launch_plan(log,config,workspace,experiment):
    mode=config.get('gate_prefetch_mode','off')
    plan=launch_evidence(log,SHAPE,config['core_limit'],workspace,config['dequant_mode'],config['pipeline_mode'],5,config['kv_m80_mode'],config['scale_mode'],256,config['k_tile_policy'],mode)
    if experiment=='prefetch':check_plan(plan,mode)
    elif plan['selected_pipeline']!='serial-v1' or plan['raw_banks']!=1 or mode!='off':raise ValueError('copy experiment must not prefetch Gate')
    return plan


def run(args):
    names=selected_cases(args.layers)
    root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=False)
    config=load_build_config(args.build_config)
    experiment=getattr(args,'experiment','copy')
    if experiment not in ('copy','prefetch'):raise ValueError('unknown Gate experiment')
    if experiment=='prefetch' and config.get('gate_copy_mode')!='loop':raise ValueError('prefetch requires Loop copy')
    if experiment=='copy' and config.get('gate_prefetch_mode','off')!='off':raise ValueError('2D copy/prefetch experiments must be isolated')
    if (config.get('k_tile')!=256 or config.get('scale_mode')!='broadcast' or config.get('dequant_mode')!='batched' or
        config.get('gate_copy_mode') not in ('loop','2d') or args.warmup!=5 or args.repetitions!=30):raise ValueError('requires Gate-only K256/broadcast and 5/30')
    bundle=load_bundle(args.bundle,'a4');native=run_a2.validate_om_manifest(args.native_om_manifest,args.bundle,bundle,'a4')
    golden=json.loads(args.golden_suite.read_text())
    if (golden.get('status')!='PASS' or golden.get('context_rows')!=16 or bundle['context_rows']!=16 or
        golden.get('bundle_sha256')!=sha256(args.bundle) or golden.get('native_om_manifest_sha256')!=sha256(args.native_om_manifest)):
        raise ValueError('frozen Native output suite does not bind same C16 capture/OM')
    report=dict(abi=f'dflash-gate-{experiment}-run-v1',experiment=experiment,status='RUNNING',build=config,build_config_path=str(args.build_config),
        build_config_sha256=sha256(args.build_config),device_id=args.device_id,selected_cases=names,checks={},input_identities={},phase='untimed_correctness',
        inputs={key:file_identity(path) for key,path in (('capture_manifest',args.bundle),('native_manifest',args.native_om_manifest),('golden_suite',args.golden_suite))},
        full_a4_regression='NOT_RUN',full_draft_token_kv='NOT_RUN',decode_performance='NOT_RUN',performance='NOT_RUN')
    summary=root/'suite.json';write_json(summary,report)
    watched={p:sha256(p) for p in (args.bundle,args.native_om_manifest,args.golden_suite,args.build_config,args.runner,args.om_runner)}
    expected_by_name={}
    try:
        for name in names:
            case=next(c for c in bundle['cases'] if c['name']==name);old=next(c for c in golden['cases'] if c['name']==name)
            report['current_case']=name;write_json(summary,report)
            validate_layout(args.bundle.parent,case,'a4')
            if old['status']!='PASS':raise ValueError('frozen projection failed')
            if experiment=='prefetch':
                compiled=next(c for c in native['cases'] if c['name']==name)
                report['input_identities'][name]=layer_identity(case,compiled,old,args.bundle.parent,args.native_om_manifest.parent,args.golden_suite.parent)
            validate_execution(old['executions']['native_om'],SHAPE,args.device_id,'AscendCL native OM',5,30,'continuous-v1',256)
            outputs=old['outputs']['native_om']+old['bracket_outputs']['native_om']
            if len(outputs)!=4:raise ValueError('frozen native repeat/tail/postcheck incomplete')
            for item in outputs:
                file=checked_file(args.golden_suite.parent,item);watched[file]=item['sha256']
            golden_file=checked_file(args.golden_suite.parent,outputs[0])
            expected=golden_file.read_bytes();expected_by_name[name]=expected
            if any(checked_file(args.golden_suite.parent,o).read_bytes()!=expected for o in outputs):raise ValueError('frozen Native outputs drift')
            directory=root/'correctness'/name;directory.mkdir(parents=True)
            for file in ('x.bin','w_nz.bin','s_gn.bin'):(directory/file).symlink_to(checked_file(args.bundle.parent,case['files'][file]))
            command=[str(args.runner),str(args.device_id),str(directory),*map(str,SHAPE)]
            write_json(directory/'command.json',command)
            with (directory/'runner.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=args.timeout)
            execution=json.loads((directory/'execution.json').read_text());validate_untimed(execution,args.device_id)
            copy_evidence(directory/'runner.log',config['gate_copy_mode'])
            pre_plan=launch_plan(directory/'runner.log',config,execution['workspace_bytes'],experiment)
            comparisons=[compare(expected,(directory/f'actual-{i}.bin').read_bytes(),SimpleNamespace(m=16,n=19456)) for i in (0,1)]
            passed=all(c['bitwise_equal'] and c['finite'] and c['max_ulp']==0 for c in comparisons)
            report['checks'][name]=dict(status='PASS' if passed else 'FAIL',golden=dict(path=str(golden_file),sha256=sha256(golden_file),bytes=len(expected)),comparisons=comparisons,launch=pre_plan,execution=record(directory/'execution.json',root))
            write_json(summary,report)
            if not passed:raise ValueError('untimed Gate correctness failed; performance NOT_RUN')
        if tuple(report['checks'])!=names or any(c['status']!='PASS' for c in report['checks'].values()):raise ValueError('all selected layers must pass before timing')
        report.update(phase='timing',untimed_correctness='PASS');write_json(summary,report)
        child=SimpleNamespace(**vars(args),scope='a4',timing_protocol='continuous-v1',case_filter=names,k_tile=256)
        child.output_dir=root/'timing'
        if run_a2.run(child)!=0:raise ValueError('same-input Native/custom timing bracket correctness failed')
        measured=json.loads((child.output_dir/'suite.json').read_text());plans=[]
        if tuple(c['name'] for c in measured['cases'])!=names:raise ValueError('timing omitted/reordered selected Gate layers')
        for row in measured['cases']:
            for kind in ('custom','native_om'):
                for item in row['outputs'][kind]+row['bracket_outputs'][kind]:
                    if checked_file(child.output_dir,item).read_bytes()!=expected_by_name[row['name']]:raise ValueError('timed outputs differ from frozen Native pre-gate')
            log=child.output_dir/row['name']/'custom/runner.log'
            plan=launch_plan(log,config,row['executions']['custom']['workspace_bytes'],experiment)
            if plan!=report['checks'][row['name']]['launch']:raise ValueError('pre-gate/timing launch changed')
            plans.append(dict(name=row['name'],launch=plan,copy=copy_evidence(log,config['gate_copy_mode']),prefetch_resources=check_plan(plan,config['gate_prefetch_mode']) if experiment=='prefetch' else None))
        if any(sha256(p)!=digest for p,digest in watched.items()):raise ValueError('inputs/config/runners changed')
        for case in bundle['cases']:
            for file in case['files'].values():checked_file(args.bundle.parent,file)
        report.update(status='PASS',phase='complete',performance='MEASURED',timing=record(child.output_dir/'suite.json',root),launches=plans)
    except Exception as error:report.update(status='FAIL',error=f'{type(error).__name__}: {error}')
    write_json(summary,report);print(f"{report['status']}: {summary}; full Draft Token/KV NOT_RUN",flush=True)
    return int(report['status']!='PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('build-config','bundle','native-om-manifest','golden-suite','runner','om-runner','output-dir'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--experiment',choices=('copy','prefetch'),default='copy')
    p.add_argument('--layers',type=int,nargs='+',default=[0],choices=range(5));p.add_argument('--device-id',type=int,default=0)
    p.add_argument('--warmup',type=int,default=5);p.add_argument('--repetitions',type=int,default=30);p.add_argument('--timeout',type=int,default=1800)
    a=p.parse_args()
    if len(set(a.layers))!=len(a.layers):p.error('duplicate layers')
    raise SystemExit(run(a))
