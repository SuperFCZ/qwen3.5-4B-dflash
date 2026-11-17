import os
from pathlib import Path
import tempfile
from c8_common import c
from resource8 import probe_view,canonical,compare
from raw_probes import validate as original_validate
from sidecar_atc import ENV


def validate(row,case,cfg,device):
    with tempfile.TemporaryDirectory(prefix='core8-probe-check-') as tmp:
        log=Path(tmp)/'host.log';probe_view(c.verify(row['host_evidence']),log)
        adapted=dict(row,log=c.identity(log))
        result=original_validate(adapted,case,cfg,device)
    canonical(result['plan'],cfg,row['variant'])
    result['workspace_bytes']=c.read(c.verify(row['execution']))['workspace_bytes']
    return result


def run_one(root,prior,package,runner,variant,gear,case,device,rows):
    root.mkdir(parents=True,exist_ok=False)
    for name,item in case['inputs'].items():(root/name).symlink_to(c.verify(item))
    sidecar=root/'host-evidence.jsonl';fd=os.open(str(sidecar),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
    env=c.selected_env(prior['current']['evidence'],package['opp']['install']);env[ENV]=str(sidecar)
    row=dict(variant=variant,gear=gear,name=case['name'],shape=case['shape'],runner=runner,opp=package['opp'])
    rows.append(row);c.verify_opp(package['opp'])
    try:c.command([c.verify(runner),device,root,*case['shape']],root/'runner.log',env)
    finally:row['retained_files']=[c.identity(p) for p in sorted(root.iterdir()) if p.is_file()]
    if 'DFLASH_GROUP_QUANT_EVIDENCE_ERROR' in (root/'runner.log').read_text():raise ValueError('probe Host evidence failure')
    row.update(execution=c.identity(root/'execution.json'),host_evidence=c.identity(sidecar),outputs=[c.identity(root/f'actual-{i}.bin') for i in (0,1)])
    row['validation']=validate(row,case,prior['current']['evidence'][variant]['configuration'],device)
    c.verify_opp(package['opp'])
    return row


def validate_all(report,prior,packages,runners):
    rows=report['runs'];wanted=[(v,g,x['name']) for v in ('B','R') for g in (16,64) for x in prior['current']['evidence']['projections'][str(g)]]
    if len(wanted)!=104 or [(x['variant'],x['gear'],x['name']) for x in rows]!=wanted:raise ValueError('104 fresh 8-core probes required')
    plans={v:{str(g):{} for g in (16,64)} for v in ('B','R')}
    for row in rows:
        v=row['variant'];g=row['gear'];cfg=prior['current']['evidence'][v]['configuration']
        if row['opp']!=packages[v]['opp'] or row['runner']!=runners[v]:raise ValueError('probe package/Runner identity differs')
        case=next(x for x in prior['current']['evidence']['projections'][str(g)] if x['name']==row['name'])
        result=validate(row,case,cfg,prior['current']['device'])
        if result!=row['validation']:raise ValueError('probe result drift')
        key=','.join(map(str,case['shape']));dest=plans[v][str(g)]
        if key in dest and dest[key]!=result['plan']:raise ValueError('same-shape new probe plans conflict')
        dest[key]=result['plan']
    for b,r in zip(rows[:52],rows[52:]):
        if b['validation']['workspace_bytes']!=r['validation']['workspace_bytes']:raise ValueError('B8/R8 actual Probe Workspace differs')
    for gear in ('16','64'):compare(plans['B'][gear],plans['R'][gear])
    return plans
