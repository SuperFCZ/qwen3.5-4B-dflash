#!/usr/bin/env python3
"""Only a-c16-fc: reuse accepted OPP/runner, fresh verified Native/Extension workers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from common import ROOT,HERE,checked_file,record,sha256,write_json
from process_boundary import decode_args,encode_args,require_clean_npu,validate_preflight_result
from opp import preflight,validate_sources


def load_reuse(source):
    source=Path(source).resolve()
    state_path=source/'worker-input.json'; state=json.loads(state_path.read_text())
    old_args=decode_args(state)
    if not getattr(old_args,'draft_only',False) or not getattr(old_args,'native_base',None):
        raise ValueError('reuse requires a historical Draft-only run')
    accepted_path=checked_file(source,state['native_boundary'])
    accepted=json.loads(accepted_path.read_text())
    if accepted.get('status')!='PASS' or accepted.get('accepted_args')!=state['args']:
        raise ValueError('old worker arguments differ from accepted Native Preflight')
    if accepted.get('request_sha256')!=sha256(source/'preflight-input.json'):
        raise ValueError('old Native boundary request changed')
    suite_path=source/'suite.json'; suite=json.loads(suite_path.read_text())
    opp_path=source/'joint/opp.json'; opp=json.loads(opp_path.read_text())
    if state['opp']!=opp or checked_file(source,suite['opp'])!=opp_path:
        raise ValueError('reuse OPP is not bound to the completed source build')
    if opp.get('status')!='BUILT_NOT_NPU_VALIDATED' or Path(opp['install']).resolve()!=source/'joint/opp':
        raise ValueError('source joint OPP was not installed in its isolated directory')
    if preflight(Path(opp['install']))!=opp['preflight']:
        raise ValueError('installed joint OPP changed')
    a,c=validate_sources(old_args.a_build_config,old_args.c_release)
    if opp['a_config']!=a or opp['c_config']!=c:
        raise ValueError('reused OPP frozen build configuration differs')
    runner=source/'runner/ac_om_runner'
    if runner.is_symlink() or not runner.is_file() or sha256(runner)!=suite['runner_sha256']:
        raise ValueError('reused C++ runner missing or changed')
    device_path=source/'device-process.json';device=json.loads(device_path.read_text())
    vendor=Path(opp['install'])/'vendors/customize'
    for key,want in (('ASCEND_CUSTOM_OPP_PATH',vendor),('LD_LIBRARY_PATH',vendor/'op_api/lib')):
        if Path(device[key].split(os.pathsep)[0]).resolve()!=want:
            raise ValueError('source device environment did not select the accepted joint OPP')
    inputs=[state_path,accepted_path,suite_path,opp_path,runner,device_path,source/'preflight-input.json']
    reuse=dict(source=str(source),opp=opp,runner=str(runner),environment={k:device[k] for k in ('ASCEND_CUSTOM_OPP_PATH','LD_LIBRARY_PATH')},
               records=[dict(path=str(p),sha256=sha256(p)) for p in inputs])
    return old_args,reuse


def verify_reuse(reuse):
    for item in reuse['records']:
        if sha256(item['path'])!=item['sha256']: raise ValueError('source reuse evidence changed: '+item['path'])
    if preflight(Path(reuse['opp']['install']))!=reuse['opp']['preflight']:
        raise ValueError('reused OPP binaries/registration changed')


def coordinator(options):
    require_clean_npu('single-graph coordinator')
    args,reuse=load_reuse(options.source_evidence)
    root=options.output_dir.resolve()
    if root.is_relative_to(Path(args.native_base).resolve()) or root.is_relative_to(options.source_evidence.resolve()):
        raise ValueError('new evidence must be outside Native Base and source evidence')
    root.mkdir(parents=True,exist_ok=False)
    args.output_dir=root; args.preflight_only=False; args.single_graph='a-c16-fc'; args.air_only=options.air_only
    args.trace_bytecode=options.trace_bytecode
    report=dict(status='RUNNING',scope='only a-c16-fc; no other 55 graphs or full Draft',air='NOT_RUN',atc='NOT_RUN',
                npu='NOT_RUN',full_draft='NOT_RUN',end_to_end='NOT_RUN',phase='preflight')
    write_json(root/'suite.json',report);write_json(root/'reuse.json',reuse)
    env=os.environ.copy();env['AI_RUN_DIR']=str(root);env['TORCH_DEVICE_BACKEND_AUTOLOAD']='0';env['PYTHONDONTWRITEBYTECODE']='1'
    env['TORCH_LOGS']='+dynamo,trace_source,trace_bytecode' if options.trace_bytecode else '+dynamo'
    env['TORCHDYNAMO_VERBOSE']='1'
    try:
        request=encode_args(args);write_json(root/'preflight-input.json',request)
        command=[sys.executable,'-B','-u',str(HERE/'test/native_worker.py'),'preflight',str(root/'preflight-input.json')]
        write_json(root/'preflight-command.json',command)
        with (root/'preflight.log').open('w') as out: subprocess.run(command,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
        result=json.loads((root/'preflight-result.json').read_text())
        args,provenance=validate_preflight_result(request,result,root)
        from native_base import validate_preflight_boundary
        boundary=validate_preflight_boundary(root,request,accepted_args=encode_args(args)['args'])
        verify_reuse(reuse)
        report['phase']='single_graph';write_json(root/'suite.json',report)
        state=root/'worker-input.json'
        write_json(state,dict(**encode_args(args),native_boundary=boundary,provenance=provenance,reuse=reuse,report=report))
        env.update(reuse['environment']) # Before the NEW interpreter/ELF loader starts.
        command=[sys.executable,'-B','-u',str(HERE/'test/native_worker.py'),'single',str(state)]
        write_json(root/'device-command.json',command)
        with (root/'single-graph.log').open('w') as out:
            done=subprocess.run(command,env=env,stdout=out,stderr=subprocess.STDOUT)
        if done.returncode: raise RuntimeError('single graph failed; see single-graph.log and exception.json')
        return 0
    except Exception as error:
        from generator_diagnostics import save_exception
        # Do not overwrite device traceback with the coordinator's generic error.
        if not (root/'exception.json').exists(): save_exception(error,root)
        report=json.loads((root/'suite.json').read_text());report.update(status='FAIL',coordinator_error=str(error));write_json(root/'suite.json',report)
        print(str(error),file=sys.stderr);return 1
    finally:
        print('Evidence: '+str(root),flush=True)


def device(path):
    state=json.loads(Path(path).read_text());args=decode_args(state);root=args.output_dir.resolve()
    os.environ['AI_RUN_DIR']=str(root);report=state['report'];reuse=state['reuse']
    try:
        if Path(path).resolve()!=root/'worker-input.json' or args.single_graph!='a-c16-fc':
            raise ValueError('single-graph scope/request mismatch')
        require_clean_npu('reused OPP selection before single-graph torch_npu import')
        from native_base import active
        if active() is None: raise RuntimeError('single graph requires the verified Native import worker')
        verify_reuse(reuse)
        for key,value in reuse['environment'].items():
            if os.environ.get(key)!=value: raise ValueError('reused OPP loader environment differs: '+key)
        from draft_only import require_draft_interfaces
        interface=require_draft_interfaces(args,root/'draft-interface-device.json')
        report['target_interface']=interface['target_interface']
        import torch
        import torch_npu
        torch.npu.set_device(args.device_id)
        if '310P' not in torch.npu.get_device_name(args.device_id): raise ValueError('requires real Ascend310P')
        from generator_diagnostics import environment_report
        environment_report(root)
        from a2_common import load_bundle
        from run import probes
        bundles={16:load_bundle(args.a16_bundle,'a4')}
        cases,heads=probes(args,root,Path(reuse['runner']),bundles,only_first=True,air_only=args.air_only)
        if len(cases)!=1 or heads: raise ValueError('diagnostic executed outside a-c16-fc')
        for item in state['provenance']['records'].values():
            if sha256(item['path'])!=item['sha256']: raise ValueError('input provenance changed')
        verify_reuse(reuse); report['modules']=active().verify_loaded()
        report.update(status='AIR_ONLY' if args.air_only else 'PASS',air='PASS',
                      atc='NOT_RUN' if args.air_only else 'PASS',npu='NOT_RUN' if args.air_only else 'PASS',cases=cases,phase='complete')
    except Exception as error:
        from generator_diagnostics import save_exception
        import traceback
        traceback.print_exc();save_exception(error,root);report.update(status='FAIL',error=f'{type(error).__name__}: {error}')
        stages=root/'a-c16-fc/export-stages.json'
        if stages.is_file():
            observed=json.loads(stages.read_text())
            for key in ('air','atc'): report[key]='FAIL' if observed[key]=='RUNNING' else observed[key]
            if observed['atc']=='PASS': report['npu']='FAIL'
    write_json(root/'suite.json',report)
    print(f"{report['status']}: only a-c16-fc; full Draft/end-to-end NOT_RUN",flush=True)
    return 0 if report['status'] in ('PASS','AIR_ONLY') else 1


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--device-worker': raise SystemExit(device(sys.argv[2]))
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-evidence',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--air-only',action='store_true',help='Only export AIR; never label ATC/NPU PASS')
    p.add_argument('--trace-bytecode',action='store_true',help='Also enable trace_source/trace_bytecode logs')
    raise SystemExit(coordinator(p.parse_args()))
