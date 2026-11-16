#!/usr/bin/env python3
"""Read-only source continuation for an accepted f2688ad ATC Admission."""
import argparse
import copy
import hashlib
from pathlib import Path
import subprocess
import sys
import traceback
import raw_env_contract as ec
c=ec.c
ATC=ec.HERE.parent/'draft_a_raw_atc';sys.path.insert(0,str(ATC))
from split_atc import sources as atc_sources
from run_atc import bound_admission,STATES

# The approved old environment gate was unchanged between 551221b and f2688ad.
BASE='f2688ad377787ca7c388751a89b179b935a4a21d'
PREFIX='framework/custom_ops/draft_a_raw_env'


def git_bytes(*args):
    return subprocess.check_output(['git','--no-pager','-C',str(c.REPO),*args],stderr=subprocess.PIPE)


def archive_previous_sources(items,root):
    paths=git_bytes('ls-tree','-r','--name-only',BASE,'--',PREFIX).decode().splitlines()
    expected={str(c.REPO/p):p for p in paths};actual={x['path']:x for x in items}
    if len(actual)!=len(items) or set(actual)!=set(expected):raise ValueError('old environment inventory is not the reviewed baseline')
    sealed=[]
    for absolute,rel in expected.items():
        item=actual[absolute]
        if set(item)!={'path','bytes','sha256'}:raise ValueError('old environment source schema differs')
        blob=git_bytes('cat-file','blob',BASE+':'+rel)
        if len(blob)!=item['bytes'] or hashlib.sha256(blob).hexdigest()!=item['sha256']:
            raise ValueError('old environment source does not match immutable Git blob: '+rel)
        dest=root/'previous-environment'/Path(rel).relative_to(PREFIX)
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(blob)
        sealed.append(dict(repository_path=rel,previous_live_path=absolute,git_commit=BASE,archive=c.identity(dest)))
    return sealed


def renew(old,old_path,root):
    if old.get('status')!=STATES['admit'] or old.get('error') or old.get('integrity_error'):
        raise ValueError('previous Admission was not PASS')
    if old['continuation_sources']!=atc_sources():raise ValueError('ATC continuation source changed; renewal is not authorized for it')
    archives=archive_previous_sources(old['environment_sources'],root)
    # Old environment sources are verified against their immutable Git blobs.
    # Every other live artifact/identity retains its original strict verifier.
    view=copy.deepcopy(old);view['environment_sources']=ec.sources()
    c.verify_tree(view);bound_admission(view)
    view['environment_transition']=dict(previous_admission=c.identity(old_path),baseline_commit=BASE,
        previous_source_archives=archives,reason='Owned process-lifetime module snapshots; no model/OPP/AIR/OM change',
        scope='Read-only Admission revalidation; Import-only/ATC/NPU NOT_RUN')
    return view


def main(a):
    root=c.fresh(a.output_dir,[a.admission_report.parent])
    report=dict(status='BLOCKED',phase='renew-admit',atc='NOT_RUN',npu='NOT_RUN',draft_execution='NOT_RUN',
        environment_sources=ec.sources(),previous_admission=c.identity(a.admission_report))
    try:
        c.clean();report=renew(c.read(a.admission_report),a.admission_report,root)
    except Exception as e:
        report.update(status='BLOCKED',error=f'{type(e).__name__}: {e}');(root/'traceback.txt').write_text(traceback.format_exc())
    finally:
        try:
            c.verify_tree(report)
            if report['environment_sources']!=ec.sources():raise ValueError('current environment source changed during renewal')
            c.clean()
        except Exception as e:report.update(status='BLOCKED',integrity_error=str(e))
        c.write(root/'report.json',report)
    print(__import__('json').dumps({k:report.get(k) for k in ('status','phase','error','integrity_error','atc','npu','source_root')},indent=2))
    print('REPORT:',root/'report.json');return int(report['status']=='BLOCKED')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--admission-report',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.admission_report=a.admission_report.resolve();a.output_dir=a.output_dir.resolve()
    raise SystemExit(main(a))
