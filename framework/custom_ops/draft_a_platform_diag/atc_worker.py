"""Recompile one frozen FC projection AIR, not a complete Draft OM."""
import os,sys,traceback
from pathlib import Path
from platform_common import c
from split_atc import SplitAtcRunner,one_capture
from atc_evidence import clone_air,resumed_command


def main(path):
    q=c.read(path);root=Path(q['output']);root.mkdir(parents=True,exist_ok=False)
    r=dict(status='BLOCKED',npu='NOT_RUN',kernel_execute_calls=0);boundary=None;runner=None
    try:
        c.clean();c.verify_tree(q);c.verify_opp(q['opp'])
        from native_base import activate
        args=dict(q['evidence']['H']['state']['args'],output_dir=str(root),draft_only=True,full_draft=False)
        boundary,r['boundary']=activate(dict(args=args),c.REPO)
        air=clone_air(q['frozen'],root/'artifacts')
        diag=root/'platform.jsonl';diag.touch(exist_ok=False)
        os.environ['DFLASH_PLATFORM_DIAG_PATH']=str(diag);os.environ['DFLASH_PLATFORM_DIAG_ROUTE']='ATC'
        from qwen35_dflash.ascend310p.draft_ac_export_pipeline import compile_air_bundle
        from qwen35_dflash.ascend310p.runtime_input_export import validated_runtime_input_abi
        for g in c.read(air)['graphs']:validated_runtime_input_abi(g,required=True,allow_test_double=False)
        runner=SplitAtcRunner(root/'atc',q['frozen']['atc_binary'],root,identity_text=q['frozen']['compiler_identity_text'],expected_command=resumed_command(q['frozen'],air,air.parent))
        old=c.read(c.verify(q['frozen']['deployment']))
        compiled=compile_air_bundle(air,atc_bin=q['frozen']['atc_binary']['path'],soc_version='Ascend310P3',extra_args=old['compiler']['extra_args'],runner=runner,atc_identity=q['frozen']['compiler_identity_text'])
        g=compiled['graphs'][0];one_capture(runner.records,g['atc_command'])
        dp=Path(compiled['manifest_path']);om=c.linked(dp.parent,g['om'])
        r.update(status='SINGLE_FC_ATC_COMPILED_NOT_EXECUTED',platform=c.identity(diag),deployment=c.identity(dp),om=c.identity(om),air=c.identity(air),atc_command=g['atc_command'])
    except Exception as e:r['error']=str(e);(root/'traceback.txt').write_text(traceback.format_exc())
    finally:
        r['captures']=runner.records if runner else []
        try:
            if boundary:
                r['boundary']['modules']=boundary.verify_loaded()
                r['boundary']['status']='PASS' if r['status']!='BLOCKED' else 'FAIL'
            c.verify_tree(q);c.verify_tree(r);c.verify_opp(q['opp'])
        except Exception as e:r.update(status='BLOCKED',integrity_error=str(e))
        c.write(root/'report.json',r)
    return int(r['status']=='BLOCKED')


if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
