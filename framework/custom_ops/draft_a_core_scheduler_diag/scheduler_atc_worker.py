import sys,traceback
from pathlib import Path
from scheduler_common import c
from isolated_atc import IsolatedAtcRunner
from split_atc import one_capture
from atc_evidence import clone_air,resumed_command
from atc_dependency_identity import plan as atc_plan
import platform_runtime_env as runtime_env


def main(path):
    q=c.read(path);root=Path(q['output']);root.mkdir(parents=True,exist_ok=False);r={'status':'BLOCKED','npu':'NOT_RUN','execute_calls':0};boundary=None;runner=None
    try:
        c.clean();c.verify_tree(q);c.verify_opp(q['opp']);runtime_env.assert_invocation(q['environment'],q['runtime_environment'])
        from native_base import activate
        args=dict(q['evidence']['H']['state']['args'],output_dir=str(root),draft_only=True,full_draft=False)
        boundary,r['boundary']=activate(dict(args=args),c.REPO)
        air=clone_air(q['frozen'],root/'artifacts');platform=root/'platform.jsonl';platform.touch(exist_ok=False)
        scheduling=root/'scheduling.jsonl';scheduling.touch(exist_ok=False)
        # This independent adapter explicitly permits exactly one additional diagnostic path.
        env=dict(q['environment'],DFLASH_PLATFORM_DIAG_PATH=str(platform),DFLASH_PLATFORM_DIAG_ROUTE='ATC')
        runtime_env.assert_invocation(env,q['runtime_environment']);env['DFLASH_SCHEDULER_EVIDENCE_PATH']=str(scheduling)
        r['child_environment']=env
        chain=atc_plan(q['frozen']['atc_binary'],q['frozen']['command'][0],q['runtime_environment']['cann_root'],env)
        r['elf_environment']=dict(env,**chain['wrapper_exports'])
        from qwen35_dflash.ascend310p.draft_ac_export_pipeline import compile_air_bundle
        from qwen35_dflash.ascend310p.runtime_input_export import validated_runtime_input_abi
        for g in c.read(air)['graphs']:validated_runtime_input_abi(g,required=True,allow_test_double=False)
        # Add the approved path to a local binding copy; never alter the shared runtime policy.
        local_binding=dict(q['runtime_environment'],values=dict(q['runtime_environment']['values'],DFLASH_SCHEDULER_EVIDENCE_PATH=str(scheduling)))
        runner=IsolatedAtcRunner(root/'atc',q['frozen']['atc_binary'],root,env=env,binding=local_binding,
            identity_text=q['frozen']['compiler_identity_text'],expected_command=resumed_command(q['frozen'],air,air.parent))
        old=c.read(c.verify(q['frozen']['deployment']))
        compiled=compile_air_bundle(air,atc_bin=q['frozen']['atc_binary']['path'],soc_version='Ascend310P3',extra_args=old['compiler']['extra_args'],runner=runner,atc_identity=q['frozen']['compiler_identity_text'])
        g=compiled['graphs'][0];one_capture(runner.records,g['atc_command']);dp=Path(compiled['manifest_path'])
        r.update(status='SCHEDULER_SINGLE_FC_ATC_OBSERVED',platform=c.identity(platform),scheduling=c.identity(scheduling),deployment=c.identity(dp),om=c.identity(c.linked(dp.parent,g['om'])),atc_command=g['atc_command'])
    except Exception as e:r['error']=str(e);(root/'traceback.txt').write_text(traceback.format_exc())
    finally:
        r['captures']=runner.records if runner else []
        try:
            if boundary:r['boundary']['modules']=boundary.verify_loaded()
            c.verify_tree(q);c.verify_tree(r);c.verify_opp(q['opp'])
        except Exception as e:r.update(status='BLOCKED',integrity_error=str(e))
        c.write(root/'report.json',r)
    return int(r['status']=='BLOCKED')


if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
