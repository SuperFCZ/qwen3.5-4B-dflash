import argparse
from pathlib import Path
import traceback
from profile_common import admit,c,require,sources
from collect_profile import run
from attribution import aggregate,recover

PHASES=('admit','profile-c16','profile-c64','analyze')
STATUS=dict(zip(PHASES,('A_OM_PROFILE_ADMITTED','A_C16_RUNNER_CAPTURE_VERIFIED','A_C64_RUNNER_CAPTURE_VERIFIED','A_OM_PROFILE_ANALYZED')))


def artifacts(root):
    result=[]
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            # Inputs and Golden are the explicitly bound external references, never raw profiler evidence.
            if path.parent in (root/'application/inputs',root/'application/golden'):continue
            require(root in path.resolve(strict=True).parents,'profiler symlink escapes stage')
        if path.is_file():result.append(c.identity(path))
    return result


def main(args):
    root=args.root.resolve();old=args.old.resolve();resume_root=args.resume.resolve();phase=args.phase
    out=c.fresh(root/phase,[old,resume_root]);r=dict(phase=phase,status='BLOCKED',sources=sources(),predecessors={},old=str(old),
        resume=str(resume_root),execute='NOT_RUN',formal_benchmark='NOT_RUN',launch_authorized=False)
    try:
        reports={}
        needed={'admit':(),'profile-c16':('admit',),'profile-c64':('admit',),'analyze':PHASES[:3]}[phase]
        for name in needed:
            path=root/name/'profile-report.json';report=c.read(path);c.verify_tree(report)
            require(report['phase']==name and report['status']==STATUS[name] and report['sources']==r['sources'] and
                    (report['old'],report['resume'])==(str(old),str(resume_root)) and report['predecessors']==({} if name=='admit' else {'admit':r['predecessors']['admit']}),
                    'invalid/mixed predecessor: '+name)
            r['predecessors'][name]=c.identity(path);reports[name]=report
        if phase=='admit':r['binding']=admit(old,resume_root)
        elif phase.startswith('profile-'):
            gear=phase.split('c')[-1];b=reports['admit']['binding'];r.update(gear=gear,baseline=b['cases'][gear]['baseline'])
            run(out,b,gear,r)
            t,tables,notes=recover(out/'profiler',r['runner_pid'])
            require(t['assignment_evidence']['real_event_count']>0,'no real device tasks in capture')
            c.write(out/'windows.json',t)
            r.update(execute='38_PROFILED_CALLS_GOLDEN_VERIFIED',windows=c.identity(out/'windows.json'),device_assignment=t['device_assignment'],
                     csv_sources=[dict(file=x['file'],rows=len(x['rows']),rejected=x['rejected_rows']) for x in tables],other_timelines=notes)
            c.verify_tree(b);c.verify_opp(b['selected']['opp'])
        else:
            r['gears']={}
            for gear in ('16','64'):
                src=reports['profile-c'+gear];folder=out/('c'+gear);folder.mkdir()
                report=aggregate(folder,c.read(c.verify(src['windows'])),c.verify(src['graph']),src['baseline'])
                r['gears'][gear]=dict(report=c.identity(folder/'profile-report.json'),attribution=report['attribution'])
            (out/'profile-hotspots.md').write_text('# Full Draft A-only\n\n'+
                '\n\n'.join('## C'+gear+'\n'+(out/('c'+gear)/'profile-hotspots.md').read_text() for gear in ('16','64')))
        require(sources()==r['sources'],'profile source drift');r['status']=STATUS[phase]
    except Exception as err:
        r['error']=repr(err);(out/'traceback.txt').write_text(traceback.format_exc())
        if r['launch_authorized']:r['execute']='LAUNCH_AUTHORIZED_CAPTURE_NOT_ACCEPTED'
    finally:
        try:
            r['files']=artifacts(out);c.verify_tree(r);c.clean()
        except Exception as err:r.update(status='BLOCKED',integrity_error=repr(err))
        c.write(out/'profile-report.json',r)
    print(r['status'],r.get('error'),r.get('integrity_error'),out/'profile-report.json');return int(r['status']=='BLOCKED')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='One manual A-only OM profiling stage');parser.add_argument('phase',choices=PHASES)
    parser.add_argument('--root',required=True,type=Path);prefix=Path('/home/w00949577/qwen35-dflash-run')
    parser.add_argument('--old',type=Path,default=prefix/'a-core-scheduler-flow-F6ia0q');parser.add_argument('--resume',type=Path,default=prefix/'a-scheduler-om-resume-0foBgF')
    raise SystemExit(main(parser.parse_args()))
