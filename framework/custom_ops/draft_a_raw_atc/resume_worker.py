"""ATC-only compile of a byte-identical frozen AIR mirror. No model execution."""
from pathlib import Path
import sys
import traceback
from split_atc import c,SplitAtcRunner,one_capture
from atc_evidence import clone_air,resumed_command,compare_om


def main(request):
    req=c.read(request);root=Path(req['output']);root.mkdir(parents=True,exist_ok=False)
    r=dict(status='BLOCKED',output_root=str(root),variant='B',gear=16,npu='NOT_RUN',draft_execution='NOT_RUN',
        continuation_sources=req['continuation_sources'],private_opp=req['private_opp'],atc_captures=[])
    capture=None;boundary=None
    try:
        c.clean();c.verify_tree(req);c.verify_opp(req['private_opp'])
        from native_base import activate
        args=dict(req['evidence']['H']['state']['args'],output_dir=str(root),draft_only=True,full_draft=False,preflight_only=False)
        boundary,imports=activate(dict(args=args),c.REPO);r['import_boundary']=imports
        frozen=req['frozen'];model=frozen['model']
        manifest=clone_air(frozen,root/'artifacts');air=c.read(manifest)
        from qwen35_dflash.ascend310p.runtime_input_export import validated_runtime_input_abi
        from qwen35_dflash.ascend310p.draft_ac_export_pipeline import compile_air_bundle
        for graph in air['graphs']:validated_runtime_input_abi(graph,required=True,allow_test_double=False)
        planned=resumed_command(frozen,manifest,manifest.parent)
        capture=SplitAtcRunner(root/'atc-streams',frozen['atc_binary'],root,
            identity_text=frozen['compiler_identity_text'],expected_command=planned)
        old=c.read(c.verify(frozen['deployment']))
        compiled=compile_air_bundle(manifest,atc_bin=frozen['atc_binary']['path'],soc_version='Ascend310P3',
            extra_args=old['compiler']['extra_args'],runner=capture,atc_identity=frozen['compiler_identity_text'])
        dep=Path(compiled['manifest_path']);g=compiled['graphs'][0]
        split=one_capture(capture.records,g['atc_command'])
        om=c.identity(c.linked(dep.parent,g['om']))
        original_air=Path(frozen['air']['path']).parent
        graph_rel=Path(model['graph']['path']).relative_to(original_air)
        r.update(status='A_ONLY_ATC_RESUMED_NOT_EXECUTED',atc_command=g['atc_command'],atc_log=split['stdout'],atc_stderr=split['stderr'],
            compiler_stdout_log=c.identity(Path(__import__('os').environ['AI_RUN_DIR'])/g['atc_log']),
            om=om,artifacts=dict(air=c.identity(manifest),deployment=c.identity(dep),om=om),
            graph=c.identity(manifest.parent/graph_rel),logical_ir_source_root=model['output_root'],
            logical_ir_sha256=model['logical_ir_sha256'],abi=model['abi'],counts=model['counts'],
            weight_bindings=model['weight_bindings'],head_sha256=model['head_sha256'],
            frozen_worker=frozen['worker_report'],frozen_air=frozen['air'],payloads=frozen['payloads'],
            historical_interfaces=model['interfaces'],om_comparison=compare_om(model['om'],om),
            input_storage='Byte-identical private AIR/payload copies; original absolute FileConstant targets remain frozen read-only')
        if c.verify(r['compiler_stdout_log']).read_bytes()!=c.verify(r['atc_log']).read_bytes():raise ValueError('compiler stdout copy differs')
    except Exception as error:
        r.update(status='BLOCKED',error=f'{type(error).__name__}: {error}');(root/'traceback.txt').write_text(traceback.format_exc())
    finally:
        r['atc_captures']=capture.records if capture else []
        try:
            if boundary:
                r['import_boundary']['modules']=boundary.verify_loaded()
                r['import_boundary']['status']='PASS' if r['status']!='BLOCKED' else 'FAIL'
            c.verify_tree(req);c.verify_tree(r);c.verify_opp(req['private_opp'])
        except Exception as error:r.update(status='BLOCKED',integrity_error=str(error))
        c.write(root/'report.json',r)
    return int(r['status']=='BLOCKED')


if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
