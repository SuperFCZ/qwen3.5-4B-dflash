"""Fresh Native Base import boundary; export only a complete A-only static Draft."""
import gc,os,sys
from c8_common import c
from pathlib import Path
from types import SimpleNamespace
from sidecar_atc import SidecarAtcRunner
from split_atc import one_capture
from raw_atc_interface_identity import canonical_worker_report
# c is supplied by the isolated Core8 bootstrap


def main(path):
    req=c.read(path);root=Path(req['output']);root.mkdir(parents=True,exist_ok=False)
    result=dict(status='BLOCKED',output_root=str(root),draft_execution='NOT_RUN',device_preparation='MAY_USE_NATIVE_TENSOR_OPERATIONS',profile='NOT_RUN',production='NOT_APPROVED')
    boundary=None;capture=None
    result.update(variant=req["variant"],gear=req["gear"],continuation_sources=req["continuation_sources"],private_opp=req["private_opp"])
    try:
        c.clean();c.verify_tree(req);c.verify_opp(req["private_opp"])
        from native_base import activate
        h=req['evidence']['H'];args=dict(h['state']['args'],output_dir=str(root),draft_only=True,full_draft=False,preflight_only=False)
        state=dict(args=args)
        boundary,imports=activate(state,c.REPO)
        imports['export_settings'].update(variants=['a'],gears=[req['gear']],scope='new A-only B/R only')
        result['import_boundary']=imports
        from draft_only import require_draft_interfaces,load_draft,native_history
        a=SimpleNamespace(**{k:Path(v) if k in h['state']['path_args'] else v for k,v in args.items()})
        # Selected OPP environment has been installed before this process started.
        result['interfaces']=require_draft_interfaces(a,root/'draft-interface.json')
        import torch
        import torch_npu
        torch.npu.set_device(req['device'])
        if torch.npu.get_device_name(req['device'])!='Ascend310P3':raise ValueError('requires same310P3 device')
        result['environment']=dict(torch=str(torch.__version__),torch_npu=str(torch_npu.__version__),device_id=req['device'],device_name=torch.npu.get_device_name(req['device']),
            ASCEND_CUSTOM_OPP_PATH=os.environ.get('ASCEND_CUSTOM_OPP_PATH'),LD_LIBRARY_PATH=os.environ.get('LD_LIBRARY_PATH'))
        old=req['evidence']['H']['environment']
        for key in ('torch','torch_npu','device_id','device_name'):
            if result['environment'][key]!=old[key]:raise ValueError('model runtime changed: '+key)
        config,native,_=native_history(a)
        source=load_draft(config,native);graph=source.model
        from a2_common import load_bundle,checked_file
        from a2_capture import frozen_inputs
        from qwen35_dflash.ascend310p.draft_ac_export import integration_spec,projection_modules,tensor_hash
        gear=req['gear'];bundle=load_bundle(getattr(a,f'a{gear}_bundle'),'a4')
        names,arrays,_,origin=frozen_inputs(getattr(a,f'replay{gear}'),graph.config,bundle['source']['feature_layers'])
        expected=req['evidence']['H']['gears'][str(gear)]
        if list(names)!=[x['name'] for x in expected['abi']['inputs']] or origin['context_rows']!=gear:raise ValueError('frozen input gear/ABI differs')
        for name,value in zip(names,arrays):
            import hashlib
            if hashlib.sha256(value.tobytes()).hexdigest()!=expected['inputs'][name]['sha256']:raise ValueError('replayed input differs: '+name)
        bindings=[]
        for name,parent,attr in projection_modules(graph):
            module=getattr(parent,attr);captured=next(x for x in bundle['cases'] if x['name']==name)
            q=tensor_hash(module.qweight);s=tensor_hash(module.scales.t().contiguous())
            if q!=captured['files']['q_nk.bin']['sha256'] or s!=captured['files']['s_gn.bin']['sha256']:raise ValueError('projection checkpoint differs: '+name)
            bindings.append(dict(name=name,weight_sha256=q,scale_gn_sha256=s))
        cap=c.read(a.head_capture)
        if tensor_hash(graph.head.weight)!=cap['weight']['file']['sha256']:raise ValueError('Head weight changed')
        result['weight_bindings']=bindings;result['head_sha256']=tensor_hash(graph.head.weight)
        inputs=tuple(torch.from_numpy(x.copy()).to(f'npu:{req["device"]}') for x in arrays)
        selected=integration_spec(source,'a',inputs,gear)
        if selected.metadata['tensor_abi']!=expected['abi']:raise ValueError('full graph ABI changed')
        from split_export import export_one
        capture=SidecarAtcRunner(root/'atc-streams',req['atc_binary'],root,identity_text=req['compiler_identity_text'])
        om,artifacts=export_one(selected,root,a.atc,draft_only=True,runner=capture)
        deployment=c.read(checked_file(root,artifacts['deployment']))
        atc_log=(Path(os.environ['AI_RUN_DIR'])/deployment['graphs'][0]['atc_log']).resolve(strict=True)
        if Path(os.environ['AI_RUN_DIR']).resolve() not in atc_log.parents:raise ValueError('ATC log outside experiment')
        result['atc_log']=c.identity(atc_log)
        result['atc_command']=deployment['graphs'][0]['atc_command']
        split=one_capture(capture.records,result['atc_command'])
        result['compiler_stdout_log']=result['atc_log']
        if atc_log.read_bytes()!=c.verify(split['stdout']).read_bytes():raise ValueError('compiler stdout copy differs')
        result['atc_log']=split['stdout'];result['atc_stderr']=split['stderr'];result['host_evidence']=split['host_evidence']
        airpath=checked_file(root,artifacts['air']);air=c.read(airpath);g=air['graphs'][0]
        counts=g['runtime_input_abi']['draft_ac']['counts']
        if counts!={'DFlashGroupQuantLinear':26,'DFlashDraftLmHeadTop1':0,'DFlashDraftLmHeadTop1Audit':0}:raise ValueError('not26 A-only')
        # Same deterministic logical graph, with only owned output-folder prefixes
        # normalized; do not erase arbitrary metadata/attributes to force equality.
        ir=[checked_file(airpath.parent,p) for p in g['payload_files'] if Path(p['path']).name=='dynamo.pbtxt']
        if len(ir)!=1:raise ValueError('unique full graph IR required')
        normalized=ir[0].read_bytes().replace(str(root).encode(),b'<VARIANT_ROOT>')
        result.update(status='A_ONLY_OM_COMPILED_NOT_EXECUTED',artifacts={k:c.identity(checked_file(root,v)) for k,v in artifacts.items()},
            om=c.identity(om),abi=expected['abi'],counts=counts,graph=c.identity(ir[0]),logical_ir_sha256=hashlib.sha256(normalized).hexdigest())
        del selected,source,graph,inputs;gc.collect();torch.npu.empty_cache()
    except Exception as e:
        import traceback
        traceback.print_exc();result['error']=f'{type(e).__name__}: {e}'
    finally:
        result['atc_captures']=capture.records if capture else []
        try:
            if boundary:
                result['import_boundary']['modules']=boundary.verify_loaded()
                result['import_boundary']['receiver']=boundary.receiver
                result['import_boundary']['status']='PASS' if result['status']!='BLOCKED' else 'FAIL'
            if 'interfaces' in result:
                result=canonical_worker_report(result,root/'report.json',req['output'],req['evidence'])
            c.verify_tree(req);c.verify_tree(result);c.verify_opp(req['private_opp'])
        except Exception as e:result.update(status='BLOCKED',integrity_error=str(e))
        c.write(root/'report.json',result)
    return int(result['status']=='BLOCKED')


if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
