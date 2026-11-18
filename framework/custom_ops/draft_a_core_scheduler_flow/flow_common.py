"""Shared evidence for one strictly ordered multi-projection experiment."""
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
SMOKE=HERE.parent/'draft_a_core_scheduler_smoke'
sys.path.insert(0,str(SMOKE))
import run_smoke as fc
from fc_contract import c,host,require
from scheduler_derive import audit_host as original_audit
from atc_evidence import payloads, COUNTS, option
import a4_contract

STAGES=('source','build','host-query','projection-smoke','export','full-smoke','benchmark')
STATUS={name:'FLOW_'+name.upper().replace('-','_')+'_PASS' for name in STAGES}


def sources():
    files=set(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.update(c.QUANT/'test'/name for name in ('a4_contract.py','reference.py','runner_common.h'))
    files.update(c.HERE/name for name in ('raw_protocol.py','raw_common.py'))
    files.update(path for folder in ('draft_a_raw_env','draft_a_raw_atc','draft_a_raw_sidecar') for path in (HERE.parent/folder).glob('*.py'))
    return [c.identity(p) for p in sorted(files)]+fc.sources()


def audit_host():
    old=original_audit()
    return dict(original=old['original'],candidate=c.identity(HERE/'op_host/d_flash_group_quant_linear.cpp'),
                writer=c.identity(HERE/'op_host/scheduler_evidence.h'),platform_writer=c.identity(HERE/'op_host/platform_diag.h'))


def cases(evidence):
    out={}
    for gear in ('16','64'):
        rows=evidence['projections'][gear]
        require(len(rows)==26 and {case['name'] for case in rows}==set(a4_contract.CASE_NAMES),'projection inventory')
        shapes={x['name']:[x[k] for k in ('m','k','n')] for x in a4_contract.cases(int(gear))}
        for case in rows:
            m,k,n=case['shape'];name=case['name']
            require(case['shape']==shapes[name],'frozen shape differs from projection ABI')
            require(Path(name).name==name and name not in ('.','..') and min(m,k,n)>0 and k%128==0 and n%64==0,'bad projection')
            require({key:item['bytes'] for key,item in case['inputs'].items()}=={'x.bin':m*k*2,'w_nz.bin':k*n,'s_gn.bin':k//128*n*2}
                    and case['golden']['bytes']==m*n*2,'projection byte contract')
            out[gear+'/'+name]=case
    require({key.split('/')[1] for key in out if key.startswith('16/')}=={key.split('/')[1] for key in out if key.startswith('64/')},'gear inventory differs')
    for name in a4_contract.CASE_NAMES:
        for key in ('w_nz.bin','s_gn.bin'):
            require(out['16/'+name]['inputs'][key]['sha256']==out['64/'+name]['inputs'][key]['sha256'],'checkpoint/scale drift across gears')
    return out


def admit(host_root,smoke_root):
    reports={phase:fc.predecessor(smoke_root/phase/'report.json',phase) for phase in fc.STATES}
    refs={phase:c.identity(smoke_root/phase/'report.json') for phase in reports}
    for i,phase in enumerate(fc.STATES):
        for prev in tuple(fc.STATES)[:i]:require(reports[phase][prev+'_report']==refs[prev],'mixed FC stages')
    bound=reports['source']['binding'];host_source=c.verify(bound['host_reports']['source'])
    require(host_source==host_root/'source/report.json' and all(c.identity(host_root/name/'report.json')==item for name,item in bound['host_reports'].items()),'mixed Host roots')
    require(fc.replay(bound,reports['smoke']['runs'],reports['build'])==reports['compare']['comparison'],'single FC replay differs')
    binding=c.read(host_source)['binding'];prior=binding['prior'];e=prior['current']['evidence'];projection=cases(e)
    h=e['H'];suite=c.read(c.verify(h['reports']['draft-suite.json']));history=Path(h['state_reference_root']);models={}
    for row in suite['cases']:
        if row['mode']!='a':continue
        gear=str(row['gear']);folder=history/('draft-a-c'+gear);spec=h['gears'][gear]
        paths={key:c.linked(folder,row['artifacts'][key]) for key in ('air','deployment','om')}
        dep=c.read(paths['deployment']);air=c.read(paths['air']);graph=air['graphs'][0];built=dep['graphs'][0]
        require(gear not in models and row['status']=='PASS' and c.identity(paths['om'])==spec['om'] and c.linked(paths['deployment'].parent,built['om'])==paths['om'] and c.linked(paths['deployment'].parent,dep['air_manifest'])==paths['air'],'full graph chain')
        require(graph['runtime_input_abi']['draft_ac']['counts']==COUNTS and graph['metadata']['tensor_abi']==spec['abi'] and all(graph[key]==built[key] for key in ('metadata','runtime_input_abi','custom_op_audit')),'full graph ABI/counts')
        cmd=built['atc_command'];atc=e['tools']['atc']
        require(c.identity(cmd[0])==atc and Path(option(cmd,'--model')).resolve()==c.linked(paths['air'].parent,graph['air']),'ATC model/identity')
        models[gear]=dict(air=c.identity(paths['air']),deployment=c.identity(paths['deployment']),om=spec['om'],payloads=payloads(paths['air'],air),command=cmd,atc_binary=atc,compiler_identity_text=dep['compiler']['identity'])
    require(set(models)=={'16','64'},'missing full AIR')
    return dict(host_source=bound['host_reports']['source'],fc_reports=refs,cases=projection,models=models,
                single_fc={v:{'query':row['query'],'api':row['api'],'kernels':sorted(x['sha256'] for x in row['opp']['preflight']['kernels'])}
                           for v,row in bound['variants'].items()},standard_manifest=binding['standard_manifest'])


def build_env(prior,selected):
    env,identity=host.envs.for_prior(prior,selected['opp']['install'])
    require(identity['values']==selected['env']['values'],'loader environment drift')
    host.envs.verify_binding(selected['env'])
    return env


def mapped(path,selected,cann,om=False):
    manifest=c.read(c.verify(selected['manifest']))
    if not om:return host.loaded_libraries(path,selected['opp'],Path(cann),manifest)
    # Static OM need not load a private Host/API DSO. All mapped DSOs still need an exact identity.
    host.dsos.verify_manifest(manifest);known={row['alias']:row for row in manifest['records']}
    groups=host.dsos.maps_rows(path.read_text());require(bool(groups),'empty OM mappings')
    for alias,segments in groups.items():
        row=known.get(alias);require(row is not None,'unbound OM DSO: '+alias);host.dsos.verify_record(row)
        require(all(seg['device']==row['backing']['device'] and seg['inode']==row['backing']['inode'] for seg in segments),'mapped backing differs: '+alias)
    return sorted(groups)
