"""Resume the frozen OM flow with a separately sealed OM-only loader manifest."""
import argparse
import inspect
import sys
import traceback
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'draft_a_core_scheduler_flow'))
import run_flow as flow
from flow_common import c,host,require
from om_run import full

POLICY=HERE/'observed.json'
PHASES=('admit','full-smoke','benchmark')
STATUS=dict(zip(PHASES,('OM_RESUME_ADMITTED','OM_RESUME_TOKEN_KV_MATCH','OM_RESUME_PAIRED_MEASURED')))


def sources():
    return [c.identity(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]


def pin_check(pin,row):
    require(row['alias']==row['file']['path']==pin['alias'] and row['links']==[] and
            all(row['file'][key]==pin[key] for key in ('bytes','sha256')) and row['stat']['mode']==pin['mode'] and
            row['stat']['inode']==row['backing']['inode']==pin['inode'] and row['backing']['device']==pin['maps_device'],
            'observed OM DSO identity changed: '+pin['alias'])


def check_maps(groups,records):
    require(bool(groups),'empty maps')
    for alias,segs in groups.items():
        row=records.get(alias)
        require(row is not None,'unbound OM DSO: '+alias)
        host.dsos.verify_record(row)
        require(all(seg['device']==row['backing']['device'] and seg['inode']==row['backing']['inode'] for seg in segs),
                'unbound mapped backing: '+alias)


def combined(selected,cann,extras,policy):
    manifest=c.read(c.verify(selected['manifest']));host.dsos.verify_manifest(manifest)
    require(manifest['private_opp']==selected['opp'] and manifest['cann_root']==policy['cann_root']==cann,'manifest/OPP root mismatch')
    records={row['alias']:row for row in manifest['records']}
    pins={row['alias']:row for row in policy['records']}
    require(len(pins)==22 and set(extras)==set(pins) and not (set(records)&set(extras)),'OM extension inventory differs')
    for alias,row in extras.items():
        require(row['owner']==cann,'extension owner mismatch')
        pin_check(pins[alias],row)
    return dict(records,**extras)


def historical(old):
    stages={};refs={};source=flow.sources()
    for phase in flow.STAGES[:5]:
        path=old/phase/'report.json';r=c.read(path);c.verify_tree(r)
        require(r['phase']==phase and r['status']==flow.STATUS[phase] and r['sources']==source and
                r['predecessors']==refs and not r.get('error') and not r.get('integrity_error'),'invalid old stage: '+phase)
        flow.inventory(phase,r,stages);stages[phase]=r;refs[phase]=c.identity(path)
    path=old/'full-smoke/report.json';failed=c.read(path);c.verify_tree(failed)
    require(failed['predecessors']==refs and failed['sources']==source and failed['status']=='BLOCKED' and
            failed['phase']=='full-smoke' and not failed.get('integrity_error') and len(failed['runs'])==1 and
            'unbound OM DSO:' in failed['error'],'not the expected failed full smoke')
    row=failed['runs'][0];require((row['variant'],row['gear'],row['round'])==('A','16',0) and
                                row['om']==stages['export']['models']['A/16']['om'],'wrong blocked run')
    wanted=old/'full-smoke/0/A/16/pre-maps.txt'
    require(c.identity(wanted) in row['files'] and not (wanted.parent/'post-maps.txt').exists(),'wrong blocked maps')
    refs['failed_full_smoke']=c.identity(path)
    return stages,refs,wanted


def seal(policy,path,built,cann):
    require(c.sha(path)==policy['pre_maps_sha256'] and cann==policy['cann_root'],'old maps or CANN root differs')
    groups=host.dsos.maps_rows(path.read_text())
    pins={pin['alias']:pin for pin in policy['records']}
    manifest=c.read(c.verify(built['variants']['A']['manifest']));host.dsos.verify_manifest(manifest)
    require(set(groups)-{row['alias'] for row in manifest['records']}==set(pins),'observed extension set differs')
    extras={}
    for alias,pin in pins.items():
        row=host.dsos.file_snapshot(alias,cann)
        row.update(owner=cann,backing=host.dsos.readonly_backing(Path(alias),row['file']))
        pin_check(pin,row);extras[alias]=row
    for variant,selected in built['variants'].items():
        records=combined(selected,cann,extras,policy)
        if variant=='A':check_maps(groups,records)
    return extras


def main(args):
    old=args.old.resolve(strict=True);base=args.root.resolve()
    require(old!=base and old not in base.parents and base not in old.parents,'need independent resume root')
    phase=args.phase;out=c.fresh(base/phase,[old])
    r=dict(phase=phase,status='BLOCKED',sources=sources(),npu='NOT_RUN',predecessors={})
    try:
        require(inspect.getsource(full)==inspect.getsource(flow.full).replace('r,smoke=None):','r,smoke=None,*,mapped):'),'OM protocol drift')
        stages,refs,path=historical(old);r['history']=refs;r['old_root']=str(old)
        prior=c.read(c.verify(stages['source']['binding']['host_source']))['binding']['prior']
        cann=prior['current']['cann_root'];policy=c.read(POLICY);prev={}
        for name in PHASES[:PHASES.index(phase)]:
            file=base/name/'report.json';entry=c.read(file);c.verify_tree(entry)
            require(entry['phase']==name and entry['status']==STATUS[name] and not entry.get('integrity_error') and
                    entry['predecessors']==r['predecessors'] and entry['history']==refs and entry['sources']==r['sources'],'wrong continuation: '+name)
            prev[name]=entry;r['predecessors'][name]=c.identity(file)
        if phase=='admit':
            r['extra_dsos']=seal(policy,path,stages['build'],cann);r['old_pre_maps']=c.identity(path)
        else:
            extras=prev['admit']['extra_dsos']
            def mapped(snapshot,selected,root,om=False):
                require(om and root==cann,'OM-only verifier');check_maps(host.dsos.maps_rows(snapshot.read_text()),combined(selected,cann,extras,policy))
            smoke=prev.get('full-smoke')
            if smoke:
                flow.inventory('full-smoke',smoke,stages)
                for row in smoke['runs']:
                    spec=prior['current']['evidence']['H']['gears'][row['gear']]
                    require(flow.verify_run(row,spec,0,False)==row['validation'],'Smoke evidence drift')
            r['npu']='ATTEMPTED'
            full(out,prior,stages['build'],stages['export'],r,smoke,mapped=mapped)
            flow.inventory(phase,r,stages);r['npu']='COMPLETED'
        r['status']=STATUS[phase]
    except Exception as err:
        r['error']=repr(err);(out/'traceback.txt').write_text(traceback.format_exc())
    finally:
        try:c.verify_tree(r);require(r['sources']==sources(),'continuation source drift');c.clean()
        except Exception as err:r.update(status='BLOCKED',integrity_error=str(err))
        c.write(out/'report.json',r)
    print(r['status'],r.get('error'),r.get('integrity_error'),out/'report.json')
    return int(r['status']=='BLOCKED')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=PHASES);parser.add_argument('--old',required=True,type=Path);parser.add_argument('--root',required=True,type=Path)
    raise SystemExit(main(parser.parse_args()))
