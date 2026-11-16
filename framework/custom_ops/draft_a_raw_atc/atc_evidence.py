"""Read-only admission of the compiled B-C16 worker inside a failed Export."""
import hashlib
from pathlib import Path
import shutil
from split_atc import c,one_capture,verify_resources

COUNTS={'DFlashGroupQuantLinear':26,'DFlashDraftLmHeadTop1':0,'DFlashDraftLmHeadTop1Audit':0}


def find_export(root):
    found=[]
    for rel in ('report.json','export/report.json','resume/export/report.json','export/export/report.json'):
        p=Path(root)/rel
        if p.is_file():
            r=c.read(p)
            if r.get('phase')=='export' and 'evidence' in r and 'precheck_report' in r:found.append(p.resolve())
    if len(found)!=1:raise ValueError('need exactly one original Export report at bounded known paths; use --failed-export-report')
    return found[0]


def option(command,key):
    values=[x.split('=',1)[1] for x in command if x.startswith(key+'=')]
    if len(values)!=1:raise ValueError('exactly one ATC option required: '+key)
    return values[0]


def payloads(air_path,air):
    if air.get('status')!='PASS' or air.get('artifact_kind')!='qwen35-dflash-torchair-bundle' or len(air['graphs'])!=1:
        raise ValueError('frozen AIR bundle is not accepted')
    graph=air['graphs'][0];records={}
    for item in graph['payload_files']+[graph['air']]:
        rel=Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('AIR payload path must remain within bundle')
        p=c.linked(air_path.parent,item)
        if air_path.parent not in p.parents:raise ValueError('AIR payload escapes frozen bundle')
        if str(rel) in records and records[str(rel)]!=c.identity(p):raise ValueError('conflicting payload identity')
        records[str(rel)]=c.identity(p)
    return records


def old_worker(failed_path,prior):
    outer=c.read(failed_path);c.verify_tree(outer)
    if outer.get('status')!='BLOCKED' or outer.get('phase')!='export' or outer.get('integrity_error'):
        raise ValueError('only failed resource collection after a successful worker may resume')
    if outer['sources']!=c.sources() or outer['evidence']!=prior['current']['evidence']:raise ValueError('failed Export source/evidence differs')
    for stage,item in prior['references'].items():
        if outer.get(stage+'_report')!=item:raise ValueError('failed Export predecessor differs')
    root=failed_path.parent;request_path=root/'B-c16-request.json';request=c.read(request_path)
    if any(request[k]!=v for k,v in dict(variant='B',gear=16,device=prior['current']['device'],sources=c.sources(),evidence=prior['current']['evidence']).items()):
        raise ValueError('B-C16 request source/selection differs')
    child=root/'B/c16/report.json';model=c.read(child);c.verify_tree(model)
    if (model.get('status')!='A_ONLY_OM_COMPILED_NOT_EXECUTED' or model.get('error') or model.get('integrity_error') or
            model.get('import_boundary',{}).get('status')!='PASS' or model.get('interfaces',{}).get('status')!='PASS'):
        raise ValueError('B-C16 worker/interface did not pass')
    if Path(model['output_root']).resolve()!=child.parent or Path(request['output']).resolve()!=child.parent:raise ValueError('worker directory differs')
    env=c.selected_env(prior['current']['evidence'],prior['reports']['build']['variants']['B']['opp']['install'])
    for k in ('ASCEND_CUSTOM_OPP_PATH','LD_LIBRARY_PATH'):
        if model['environment'][k]!=env[k]:raise ValueError('worker private B OPP environment differs: '+k)
    if model['counts']!=COUNTS or model['abi']!=prior['current']['evidence']['H']['gears']['16']['abi']:raise ValueError('frozen full A-only ABI/counts differ')
    graph=c.verify(model['graph']);logical=graph.read_bytes().replace(model['output_root'].encode(),b'<VARIANT_ROOT>')
    if hashlib.sha256(logical).hexdigest()!=model['logical_ir_sha256']:raise ValueError('frozen logical graph differs')
    ap=c.verify(model['artifacts']['air']);air=c.read(ap);files=payloads(ap,air)
    dp=c.verify(model['artifacts']['deployment']);dep=c.read(dp)
    if dep.get('status')!='PASS' or len(dep['graphs'])!=1 or c.linked(dp.parent,dep['air_manifest'])!=ap:raise ValueError('frozen Deployment/AIR binding differs')
    g=air['graphs'][0];compiled=dep['graphs'][0]
    control=g['metadata']['draft_ac'];audit=g['runtime_input_abi']['draft_ac']
    observed={row['ge_op_type']:row['ge_node_occurrences'] for row in g['custom_op_audit']}
    if (control.get('mode')!='a' or control.get('gear')!=16 or g['metadata']['tensor_abi']!=model['abi'] or
            g['runtime_input_abi'].get('status')!='PASS' or audit.get('status')!='PASS' or audit.get('counts')!=COUNTS or
            any(observed.get(k,0)!=v for k,v in COUNTS.items()) or observed.get('WeightQuantBatchMatmulV2',0)!=0):
        raise ValueError('frozen AIR full A-only audit differs')
    for k in ('metadata','runtime_input_abi','custom_op_audit'):
        if compiled[k]!=g[k]:raise ValueError('frozen graph audit differs: '+k)
    if c.linked(dp.parent,compiled['om'])!=c.verify(model['om']):raise ValueError('frozen OM differs')
    command=model['atc_command']
    if command!=compiled['atc_command']:raise ValueError('original ATC command differs')
    if option(command,'--model')!=str(c.linked(ap.parent,g['air'])):raise ValueError('ATC model differs from frozen AIR')
    if option(command,'--output')+'.om'!=model['om']['path']:raise ValueError('ATC output differs from old OM')
    atc=prior['current']['evidence']['tools']['atc']
    if c.identity(command[0])!=atc or c.identity(dep['compiler']['path'])!=atc:raise ValueError('original ATC binary differs')
    if not isinstance(dep['compiler'].get('identity'),str) or not dep['compiler']['identity'].strip():raise ValueError('original ATC identity text missing')
    for key,value in (('--mode','0'),('--framework','1'),('--soc_version','Ascend310P3'),('--precision_mode','must_keep_origin_dtype'),('--deterministic','0')):
        if option(command,key)!=value:raise ValueError('original ATC policy differs: '+key)
    return dict(worker_report=c.identity(child),request=c.identity(request_path),failed_export=c.identity(failed_path),
        model=model,air=c.identity(ap),deployment=c.identity(dp),payloads=files,atc_binary=atc,command=command,
        compiler_identity_text=dep['compiler']['identity'],
        worker_hash_scope='First admission seal; original artifact references reverified. Failed outer collector had not saved a Worker report hash.')


def clone_air(frozen,dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    source=c.verify(frozen['air']);target=dest/source.name
    shutil.copyfile(source,target)
    if c.sha(target)!=frozen['air']['sha256']:raise ValueError('AIR manifest copy differs')
    for rel,identity in frozen['payloads'].items():
        source=c.verify(identity);p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():raise ValueError('AIR copy destination overlaps another payload')
        shutil.copyfile(source,p)
        if c.sha(p)!=identity['sha256'] or p.stat().st_size!=identity['bytes']:raise ValueError('AIR payload copy differs')
    c.verify_tree(frozen)
    return target


def resumed_command(frozen,manifest,dest):
    air=c.read(manifest);model=Path(manifest).parent/air['graphs'][0]['air']['path']
    prefix=Path(dest)/'om'/Path(option(frozen['command'],'--output')).name
    command=list(frozen['command'])
    changes={'--model':str(model),'--output':str(prefix)}
    for key,value in changes.items():
        old=option(command,key);command[command.index(key+'='+old)]=key+'='+value
    return command


def compare_om(old,new):
    c.verify(old);c.verify(new)
    equal=old['sha256']==new['sha256'] and old['bytes']==new['bytes']
    return dict(old=old,new=new,exact_sha_equal=equal,status='EXACT_FILE_MATCH' if equal else 'DIFFERENT_BINARY_UNATTRIBUTED',
        interpretation='Only input/output path relocation and stream capture changed; hashes alone do not prove instruction or numerical equivalence. NPU NOT_RUN.')


def verify_model(model,prior):
    c.verify_tree(model)
    c.verify_opp(model['private_opp'])
    if model['private_opp']!=prior['reports']['build']['variants'][model['variant']]['opp']:raise ValueError('model private OPP binding differs')
    if 'worker_report' in model:
        raw=c.read(c.verify(model['worker_report']))
        if raw!={k:v for k,v in model.items() if k not in ('worker_report','host_resources')}:
            raise ValueError('worker report/model differs')
    if model.get('status') not in ('A_ONLY_OM_COMPILED_NOT_EXECUTED','A_ONLY_ATC_RESUMED_NOT_EXECUTED') or model.get('error') or model.get('integrity_error'):
        raise ValueError('model worker failed')
    v=model['variant'];g=model['gear']
    if model['counts']!=COUNTS or model['abi']!=prior['current']['evidence']['H']['gears'][str(g)]['abi']:raise ValueError('model ABI/count mismatch')
    graph=c.verify(model['graph']);logical_root=model.get('logical_ir_source_root',model['output_root'])
    if hashlib.sha256(graph.read_bytes().replace(logical_root.encode(),b'<VARIANT_ROOT>')).hexdigest()!=model['logical_ir_sha256']:
        raise ValueError('model logical IR changed')
    capture=one_capture(model['atc_captures'],model['atc_command'])
    if model['atc_log']!=capture['stdout'] or model['atc_stderr']!=capture['stderr']:raise ValueError('ATC log identities differ')
    verify_resources(model['host_resources'],g,prior['current']['evidence'][v]['configuration'])
    from raw_resources import accepted_gate
    accepted_gate(model['host_resources']['plans']['16,2560,19456'],prior['current']['evidence'],v)
