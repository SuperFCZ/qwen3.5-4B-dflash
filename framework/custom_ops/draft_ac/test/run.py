#!/usr/bin/env python3
"""Single-op GE gates first; optional static real Draft replay. No production plan edits."""
import argparse
import gc
import json
import os
import sys
# Auxiliary only: explicit third-party imports are isolated in the preflight child.
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"]="0"
from pathlib import Path
import subprocess
from types import SimpleNamespace
from common import ROOT,HERE,checked_file,record,sha256,write_json
from a2_common import validate_layout,case_shape
from opp import build,preflight
from process_boundary import encode_args,decode_args,require_clean_npu,npu_modules,validate_preflight_result


def descriptors(names,values):
    return [dict(name=n,dtype=str(v.dtype).removeprefix('torch.'),shape=list(v.shape)) for n,v in zip(names,values)]


def execute(runner,om,abi,inputs,folder,device):
    folder.mkdir(parents=True,exist_ok=False); data=folder/'inputs'; data.mkdir()
    identity={}
    for name,path in inputs.items():
        target=data/(name+'.bin'); target.symlink_to(Path(path).resolve()); identity[name]=sha256(path)
    for side in ('inputs','outputs'):
        (folder/(side+'.tsv')).write_text(''.join(t['name']+' '+t['dtype']+' '+' '.join(map(str,t['shape']))+'\n' for t in abi[side]))
    command=[str(runner),str(device),str(om),str(folder),str(data)]
    write_json(folder/'command.json',command)
    with (folder/'runner.log').open('w') as log: subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    report=json.loads((folder/'execution.json').read_text())
    if report.get('status')!='PASS' or any(report.get(k) is not True for k in ('guards','readonly','repeat')): raise ValueError('runner guard/readonly/repeat failure')
    permutable=abi['inputs'][0]['name'] in ('x','hidden') and abi['inputs'][0]['shape'][0]>1
    if report.get('row_permutation_checked') is not permutable: raise ValueError('missing same-workspace changed-request check')
    timing=report['timing']
    if timing.get('protocol')!='continuous-v1' or timing['warmup']!=5 or timing['repetitions']!=30 or not timing['timed_tail_checked']:
        raise ValueError('timing/tail protocol differs')
    for name,path in inputs.items():
        if sha256(path)!=identity[name]: raise ValueError('input file changed')
    result={}
    for t in abi['outputs']:
        name=t['name']; first=(folder/(name+'-actual-0.bin')).read_bytes()
        for suffix in ('actual-1.bin','benchmark-last.bin','postcheck.bin'):
            if (folder/(name+'-'+suffix)).read_bytes()!=first: raise ValueError('output repeat/tail/postcheck mismatch')
        result[name]=first
    return report,result


def compare(expected,actual,folder):
    import numpy as np
    from qwen35_dflash.ascend310p.utils import atomic_write_json
    report=dict(status='PASS',scope='all logical output bytes, no FP16/ID tolerance',outputs={})
    if expected.keys()!=actual.keys(): raise ValueError('output names differ')
    for name,raw in expected.items():
        out=actual[name]; equal=raw==out; mismatch=None
        if not equal:
            if len(raw)!=len(out): mismatch=min(len(raw),len(out))
            else: mismatch=int(np.flatnonzero(np.frombuffer(raw,dtype='u1')!=np.frombuffer(out,dtype='u1'))[0])
        report['outputs'][name]=dict(equal=equal,bytes=len(out),first_byte_difference=mismatch,
            expected_sha256=__import__('hashlib').sha256(raw).hexdigest(),actual_sha256=__import__('hashlib').sha256(out).hexdigest())
        if not equal: report['status']='FAIL'
    atomic_write_json(folder/'comparison.json',report)
    if report['status']!='PASS': raise ValueError('bitwise comparison failed: '+str(folder/'comparison.json'))
    return report


def export_one(spec,folder,atc):
    import torch
    import torchair
    from qwen35_dflash.ascend310p.draft_ac_export import GE,converter_capabilities
    capabilities=[converter_capabilities(op,torchair) for op in spec.custom_ops if op.torch_op in GE]
    if capabilities:
        write_json(folder/'converter-capabilities.json',capabilities)
        print('A/C converter capabilities: '+str(folder/'converter-capabilities.json'),flush=True)
    from qwen35_dflash.ascend310p.exporter import export_air_bundle
    from qwen35_dflash.ascend310p.compiler import compile_air_bundle
    air=export_air_bundle(lambda _:(spec,),{},folder/'artifacts')
    compiled=compile_air_bundle(air['manifest_path'],atc_bin=atc,soc_version='Ascend310P3',
        extra_args=['--precision_mode=must_keep_origin_dtype','--deterministic=0'])
    graph=compiled['graphs'][0]
    om=checked_file(Path(compiled['manifest_path']).parent,graph['om'])
    torch._dynamo.reset(); gc.collect()
    return om,dict(air=record(Path(air['manifest_path']),folder),deployment=record(Path(compiled['manifest_path']),folder),om=record(om,folder))


def probes(args,root,runner,bundles):
    from resolve import head_capture
    import numpy as np
    import torch
    from qwen35_dflash.ascend310p.draft_ac_export import A,C,AUDIT,GE,POLICY,ensure_ops
    from qwen35_dflash.ascend310p.contracts import AirGraphSpec,CustomOpExportSpec
    ensure_ops(); device=f'npu:{args.device_id}'; results=[]
    def tensor(path,dtype,shape): return torch.from_numpy(np.fromfile(path,dtype=dtype).reshape(shape)).to(device)
    for gear,bundle_path,native_path in ((16,args.a16_bundle,args.a16_native),(64,args.a64_bundle,args.a64_native)):
        bundle=bundles[gear]; native=json.loads(native_path.read_text())
        if native.get('status')!='PASS' or native.get('bundle_sha256')!=sha256(bundle_path): raise ValueError('A Native manifest is not bound to this capture')
        for case in bundle['cases']:
            validate_layout(bundle_path.parent,case,'a4'); m,k,n=case_shape(case,'a4')
            folder=root/f'a-c{gear}-{case["name"]}'; folder.mkdir()
            native_case=next(c for c in native['cases'] if c['name']==case['name'] and c['status']=='PASS')
            native_om=checked_file(native_path.parent,native_case['om'])
            qpath=checked_file(bundle_path.parent,case['files']['q_nk.bin']); spath=checked_file(bundle_path.parent,case['files']['s_gn.bin'])
            xpath=checked_file(bundle_path.parent,case['files']['x.bin'])
            class Projection(torch.nn.Module):
                def __init__(self):
                    super().__init__(); self.register_buffer('weight',tensor(qpath,'i1',(n,k))); self.register_buffer('scale',tensor(spath,'<f2',(k//128,n)))
                def forward(self,x): return torch.ops.dflash_ac.group_quant_linear.default(x,self.weight,self.scale)
            model=Projection(); x=tensor(xpath,'<f2',(m,k))
            abi=dict(inputs=descriptors(['x'],[x]),outputs=[dict(name='y',dtype='float16',shape=[m,n])])
            control=dict(policy=POLICY,mode='a',a_count=1,c_count=0,audit_count=0,scope='single-op-A',projections=[dict(n=n,k=k,
                         weight_sha256=case['files']['q_nk.bin']['sha256'],scale_gn_sha256=case['files']['s_gn.bin']['sha256'])])
            spec=AirGraphSpec(name='ac_projection',role='diagnostic',model=model,example_args=(x,),input_names=('x',),output_names=('y',),
                              custom_ops=(CustomOpExportSpec(A,GE[A]),),metadata=dict(tensor_abi=abi,draft_ac=control))
            om,artifacts=export_one(spec,folder,args.atc)
            del spec,model,x; gc.collect(); torch.npu.empty_cache()
            nt,golden=execute(runner,native_om,abi,{'x':xpath},folder/'native',args.device_id)
            ct,actual=execute(runner,om,abi,{'x':xpath},folder/'custom',args.device_id)
            compare(golden,actual,folder)
            results.append(dict(name=f'c{gear}-{case["name"]}',status='PASS',artifacts=artifacts,native=nt,custom=ct))
            write_json(root/'a-probes.json',dict(status='RUNNING',cases=results))
    write_json(root/'a-probes.json',dict(status='PASS',cases=results))
    cap=head_capture(args.head_capture); native=json.loads(args.head_native.read_text())
    if cap.get('status')!='PASS' or cap.get('abi')!='dflash-draft-head-v1' or native.get('status')!='PASS' or native.get('capture_sha256')!=sha256(args.head_capture):
        raise ValueError('Head capture/Native identity differs')
    wpath=checked_file(args.head_capture.parent,cap['weight']['file'],248320*2560*2)
    head_results=[]
    for gear in (16,64):
        hpath=checked_file(args.head_capture.parent,cap['gears'][f'c{gear}']['hidden'],15*2560*2)
        h=tensor(hpath,'<f2',(15,2560)); w=tensor(wpath,'<f2',(248320,2560))
        inputs=descriptors(['hidden','weight'],[h,w]); golden={}; timings={}
        base=root/f'c-c{gear}'; base.mkdir()
        for kind,name,dtype,shape in (('top1','ids','int64',[15]),('logits','logits','float16',[15,248320])):
            item=next(c for c in native['models'] if c['m']==15 and c['kind']==kind and c['status']=='PASS')
            om=checked_file(args.head_native.parent,item['om'])
            t,out=execute(runner,om,dict(inputs=inputs,outputs=[dict(name=name,dtype=dtype,shape=shape)]),
                          {'hidden':hpath,'weight':wpath},base/('native-'+kind),args.device_id)
            golden.update(out); timings[kind]=t
        for audit in (False,True):
            class Head(torch.nn.Module):
                def forward(self,hidden,weight):
                    return torch.ops.dflash_ac.head_audit.default(hidden,weight) if audit else torch.ops.dflash_ac.head_top1.default(hidden,weight)
            outputs=[dict(name='ids',dtype='int64',shape=[15])]+([dict(name='logits',dtype='float16',shape=[15,248320])] if audit else [])
            control=dict(policy=POLICY,mode='c',a_count=0,c_count=0 if audit else 1,audit_count=int(audit),
                         runtime_head=True,head_weight_sha256=sha256(wpath),scope='single-op-C',projections=[])
            op=AUDIT if audit else C
            spec=AirGraphSpec(name='ac_head_audit' if audit else 'ac_head',role='diagnostic',model=Head(),example_args=(h,w),
                input_names=('hidden','weight'),output_names=tuple(t['name'] for t in outputs),custom_ops=(CustomOpExportSpec(op,GE[op]),),
                metadata=dict(tensor_abi=dict(inputs=inputs,outputs=outputs),draft_ac=control))
            folder=base/('audit' if audit else 'fused'); folder.mkdir(); om,artifacts=export_one(spec,folder,args.atc)
            ct,actual=execute(runner,om,spec.metadata['tensor_abi'],{'hidden':hpath,'weight':wpath},folder/'execute',args.device_id)
            compare({k:golden[k] for k in actual},actual,folder)
            head_results.append(dict(gear=gear,audit=audit,status='PASS',artifacts=artifacts,native=timings,custom=ct))
        del h,w,spec; gc.collect(); torch.npu.empty_cache()
    write_json(root/'c-probes.json',dict(status='PASS',cases=head_results))
    return results,head_results


def full_draft(args,root,runner,bundles):
    from resolve import validate_offline_nz,linked
    import numpy as np
    import torch
    import torchair
    from qwen35_dflash.ascend310p.quant_factory import create_quant_incremental_graphs
    from qwen35_dflash.ascend310p.draft_ac_export import integration_spec,projection_modules,tensor_hash
    from qwen35_dflash.ascend310p.draft_gears import verify_om_files,om_records
    from a2_capture import frozen_inputs
    config=json.loads(args.factory_config.read_text())
    deployment=json.loads(args.native_deployment.read_text())
    candidates=[g for g in deployment['graphs'] if g['name']=='draft' and g['metadata'].get('draft_quantization')=='w8a16']
    if deployment.get('status')!='PASS' or len(candidates)!=1: raise ValueError('need one original W8 Draft deployment')
    native=candidates[0]; verify_om_files(native,args.native_deployment.parent)
    air_path=linked(args.native_deployment.parent,deployment['air_manifest'])
    original_air=json.loads(air_path.read_text())
    exported=next(g for g in original_air['graphs'] if g['name']=='draft')
    if config!=original_air.get('factory_config'):
        raise ValueError('full Draft factory config differs from hash-bound AIR')
    for key in ('metadata','runtime_input_abi','custom_op_audit'):
        if exported.get(key)!=native.get(key): raise ValueError('Native AIR/Deployment differs: '+key)
    validate_offline_nz(config,exported,air_path.parent,args.a16_bundle,bundles[16])
    native_oms=[checked_file(args.native_deployment.parent,r) for r in om_records(native)]
    result=[]
    for gear,replay in ((16,args.replay16),(64,args.replay64)):
        golden=None
        for mode in ('off','a','c','ac'):
            folder=root/f'draft-{mode}-c{gear}'; folder.mkdir()
            create_quant_incremental_graphs.prepare_export(config,torchair)
            specs=create_quant_incremental_graphs(config); source=next(s for s in specs if s.name=='draft')
            if source.metadata['incremental_contract']!=native['metadata']['incremental_contract']:
                raise ValueError('reloaded Draft contract differs from frozen Native deployment')
            graph=source.model; layers=bundles[gear]['source']['feature_layers']
            names,arrays,_,origin=frozen_inputs(replay,graph.config,layers)
            if origin['context_rows']!=gear or tuple(names)!=source.input_names: raise ValueError('frozen Draft ABI/order/gear differs')
            if (origin['draft_om_sha256']!=native['om']['sha256'] or (gear==64 and origin['draft_static64_om_sha256']!=native['static_gear_oms'][1]['om']['sha256'])):
                raise ValueError('frozen replay is not bound to the provided Native Draft OM')
            for name,parent,attr in projection_modules(graph):
                m=getattr(parent,attr); captured=next(c for c in bundles[gear]['cases'] if c['name']==name)
                if tensor_hash(m.qweight)!=captured['files']['q_nk.bin']['sha256'] or tensor_hash(m.scales.t().contiguous())!=captured['files']['s_gn.bin']['sha256']:
                    raise ValueError('full graph checkpoint projection differs from A4 capture: '+name)
            if tensor_hash(graph.head.weight)!=json.loads(args.head_capture.read_text())['weight']['file']['sha256']:
                raise ValueError('full graph head differs from accepted Head capture')
            inputs=tuple(torch.from_numpy(a.copy()).to(f'npu:{args.device_id}') for a in arrays)
            selected=integration_spec(source,mode,inputs,gear)
            abi=selected.metadata['tensor_abi']; inp=folder/'input'; inp.mkdir()
            paths={}
            for name,array in zip(names,arrays): array.tofile(inp/(name+'.bin')); paths[name]=inp/(name+'.bin')
            om,artifacts=export_one(selected,folder,args.atc)
            del graph,source,specs,selected,inputs; gc.collect(); torch.npu.empty_cache()
            nt,current_native=execute(runner,native_oms[0 if gear==16 else 1],abi,paths,folder/'native',args.device_id)
            if golden is not None: compare(golden,current_native,folder/'native')
            golden=current_native
            ct,actual=execute(runner,om,abi,paths,folder/'candidate',args.device_id)
            compare(golden,actual,folder) # IDs AND all ten full KV buffers, including untouched rows.
            result.append(dict(gear=gear,mode=mode,status='PASS',scope='frozen real Draft only',native=nt,custom=ct,artifacts=artifacts,
                               frozen_report_sha256=sha256(replay),inputs={k:sha256(v) for k,v in paths.items()}))
            write_json(root/'draft-suite.json',dict(status='RUNNING',cases=result,end_to_end='NOT_RUN'))
    write_json(root/'draft-suite.json',dict(status='PASS',cases=result,end_to_end='NOT_RUN'))
    lines=['# Frozen Draft OM timing', '', 'Same frozen inputs; continuous-v1, warmup=5, repetitions=30. End-to-end generation: NOT_RUN.', '',
           '| Gear | Mode | Native median/p95 ms | Candidate median/p95 ms | Native/candidate median |', '|---|---|---:|---:|---:|']
    for row in result:
        n=row['native']['timing']['execute_sync']; c=row['custom']['timing']['execute_sync']
        lines.append(f"| C{row['gear']} | {row['mode']} | {n['median_ms']:.4f}/{n['p95_ms']:.4f} | {c['median_ms']:.4f}/{c['p95_ms']:.4f} | {n['median_ms']/c['median_ms']:.4f} |")
    (root/'draft-timing.md').write_text('\n'.join(lines)+'\n')
    return result


def validate_device_startup(root,opp):
    require_clean_npu('device worker before joint OPP load')
    install=root/'joint/opp'; vendor=install/'vendors/customize'
    if opp.get('status')!='BUILT_NOT_NPU_VALIDATED' or Path(opp['install']).resolve()!=install:
        raise ValueError("device worker requires this run's completed joint OPP")
    if json.loads((root/'joint/opp.json').read_text())!=opp or preflight(install)!=opp['preflight']:
        raise ValueError('joint OPP changed before device startup')
    for key,expected in (('ASCEND_CUSTOM_OPP_PATH',vendor),('LD_LIBRARY_PATH',vendor/'op_api/lib')):
        first=os.environ.get(key,'').split(os.pathsep)[0]
        if not first or Path(first).resolve()!=expected:
            raise ValueError('device worker loader environment differs: '+key)


def device_worker(path):
    state=json.loads(Path(path).read_text())
    args=decode_args(state)
    root=args.output_dir.resolve(); os.environ['AI_RUN_DIR']=str(root)
    report=state['report']; provenance=state['provenance']; opp=state['opp']
    try:
        report['phase']='device_startup'
        if Path(path).resolve()!=root/'worker-input.json': raise ValueError('worker input must be in its new evidence directory')
        validate_device_startup(root,opp)
        write_json(root/'device-process.json',dict(pid=os.getpid(),npu_modules=npu_modules(),
            ASCEND_CUSTOM_OPP_PATH=os.environ.get('ASCEND_CUSTOM_OPP_PATH'),LD_LIBRARY_PATH=os.environ.get('LD_LIBRARY_PATH')))
        from a2_common import load_bundle
        bundles={16:load_bundle(args.a16_bundle,'a4'),64:load_bundle(args.a64_bundle,'a4')}
        # No torch_npu/GE import before selecting the joint OPP.
        import torch
        import torch_npu
        torch.npu.set_device(args.device_id)
        if '310P' not in torch.npu.get_device_name(args.device_id): raise ValueError('requires real 310P')
        report['opp']=record(root/'joint/opp.json',root)
        cann=os.environ.get('CANN_ROOT',os.environ.get('ASCEND_HOME_PATH','/usr/local/Ascend/ascend-toolkit/latest'))
        report['phase']='runner_build'; write_json(root/'suite.json',report)
        subprocess.run(['cmake','-S',str(HERE/'test'),'-B',str(root/'runner'),'-DCANN_ROOT='+cann],check=True)
        subprocess.run(['cmake','--build',str(root/'runner'),'--parallel'],check=True)
        runner=root/'runner/ac_om_runner'; report['runner_sha256']=sha256(runner)
        import platform
        report['environment']=dict(torch=str(torch.__version__),torch_npu=str(torch_npu.__version__),device_id=args.device_id,
            device_name=torch.npu.get_device_name(args.device_id),platform=platform.platform(),cann_root=cann,
            ASCEND_CUSTOM_OPP_PATH=os.environ.get('ASCEND_CUSTOM_OPP_PATH'),LD_LIBRARY_PATH=os.environ.get('LD_LIBRARY_PATH'),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        report['source_sha256']={str(p.relative_to(ROOT)):sha256(p) for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.py','.cpp','.h','.sh','.txt')}
        for name in ('draft_ac_export.py','draft_ac_constants.py','incremental.py','quant_factory.py','custom_op_export.py','runtime_input_export.py','exporter.py','compiler.py'):
            path=ROOT/'framework/python/qwen35_dflash/ascend310p'/name; report['source_sha256'][str(path.relative_to(ROOT))]=sha256(path)
        report['phase']='single_op_graph'; write_json(root/'suite.json',report)
        probes(args,root,runner,bundles); report['single_op_graph']='PASS'
        if args.full_draft:
            report['phase']='full_draft'; write_json(root/'suite.json',report)
            full_draft(args,root,runner,bundles); report['full_draft']='PASS'
        for entry in provenance['records'].values():
            if sha256(entry['path'])!=entry['sha256']: raise ValueError('source evidence changed during run')
        if preflight(Path(opp['install']))!=opp['preflight']: raise ValueError('joint OPP changed during run')
        report.update(status='PASS',phase='complete')
    except Exception as error:
        import traceback
        traceback.print_exc()
        report.update(status='FAIL',error=f'{type(error).__name__}: {error}')
    write_json(root/'suite.json',report)
    print(f"{report['status']}: {root/'suite.json'}; end-to-end NOT_RUN",flush=True)
    return 0 if report['status']=='PASS' else 1


def run(args):
    root=args.output_dir.resolve(); root.mkdir(parents=True,exist_ok=False)
    os.environ['AI_RUN_DIR']=str(root)
    write_json(root/'request.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    report=dict(status='RUNNING',single_op_graph='NOT_RUN',full_draft='NOT_RUN',end_to_end='NOT_RUN')
    try:
        report['phase']='provenance'
        require_clean_npu('coordinator before preflight')
        request=encode_args(args); write_json(root/'preflight-input.json',request)
        command=[sys.executable,'-B','-u',str(HERE/'test/preflight_worker.py'),str(root/'preflight-input.json')]
        write_json(root/'preflight-command.json',command)
        write_json(root/'suite.json',report)
        with (root/'preflight.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((root/'preflight-result.json').read_text())
        args,provenance=validate_preflight_result(request,result,root)
        require_clean_npu('coordinator after preflight')
        write_json(root/'coordinator-process.json',dict(pid=os.getpid(),preflight_pid=result['pid'],npu_modules=npu_modules()))
        report['provenance']=record(root/'preflight.json',root)
        report['preflight_imports']=record(root/'preflight-imports.json',root)
        report['coordinator_process']=record(root/'coordinator-process.json',root)
        if args.preflight_only:
            report.update(status='PREFLIGHT_ONLY',phase='complete')
            write_json(root/'suite.json',report)
            print(f"PREFLIGHT_ONLY: {root/'suite.json'}; device stages NOT_RUN",flush=True)
            return 0
        require_clean_npu('coordinator before joint OPP build')
        report['phase']='joint_opp'; write_json(root/'suite.json',report)
        opp=build(SimpleNamespace(a_build_config=args.a_build_config,c_release=args.c_release,output_dir=root/'joint'))
        require_clean_npu('coordinator after joint OPP build')
        if opp.get('status')!='BUILT_NOT_NPU_VALIDATED': raise ValueError('joint OPP build did not complete')
        if preflight(Path(opp['install']))!=opp['preflight']: raise ValueError('joint OPP changed after build')
        vendor=Path(opp['preflight']['vendor'])
        os.environ['ASCEND_CUSTOM_OPP_PATH']=str(vendor)+os.pathsep+os.environ.get('ASCEND_CUSTOM_OPP_PATH','')
        os.environ['LD_LIBRARY_PATH']=str(vendor/'op_api/lib')+os.pathsep+os.environ.get('LD_LIBRARY_PATH','')
        state=root/'worker-input.json'
        write_json(state,dict(**encode_args(args),
                             report=report,provenance=provenance,opp=opp))
        write_json(root/'suite.json',report)
        # Fresh ELF loader: changing LD_LIBRARY_PATH inside Python is insufficient.
        os.execve(sys.executable,[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--device-worker',str(state)],os.environ.copy())
    except Exception as error:
        import traceback
        traceback.print_exc()
        report.update(status='FAIL',error=f'{type(error).__name__}: {error}')
    write_json(root/'suite.json',report)
    print(f"{report['status']}: {root/'suite.json'}; end-to-end NOT_RUN",flush=True)
    return 0 if report['status']=='PASS' else 1

if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--device-worker': raise SystemExit(device_worker(sys.argv[2]))
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('a51-suite','c-release','head-capture','head-native','native-deployment','replay16','replay64','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('factory-config','air-manifest','a16-bundle','a64-bundle'): p.add_argument('--'+name,type=Path)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--full-draft',action='store_true'); p.add_argument('--device-id',type=int,default=0); p.add_argument('--atc',default='atc')
    args=p.parse_args()
    for k,v in vars(args).items():
        if isinstance(v,Path): setattr(args,k,v.resolve())
    raise SystemExit(run(args))
