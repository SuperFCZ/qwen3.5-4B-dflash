"""Bind the two known interface-report references without a cwd fallback."""
import copy
from pathlib import Path
from split_atc import c


def bind_record(item,expected_files,field):
    if not isinstance(item,dict) or set(item)!={'path','bytes','sha256'}:
        raise ValueError('interface identity schema differs: '+field)
    declarations=[Path(p) for p in expected_files]
    if not declarations or any(not p.is_absolute() for p in declarations):
        raise ValueError('interface identity requires explicit absolute owner: '+field)
    # Owner declarations, not a search for an existing or hash-matching file.
    targets={p.resolve() for p in declarations}
    if len(targets)!=1:
        raise ValueError('ambiguous interface identity roots: '+field+' '+str(sorted(map(str,targets))))
    target=next(iter(targets));raw=Path(item['path'])
    if not raw.is_absolute() and raw!=Path(target.name):
        raise ValueError('unexpected relative interface reference: '+field+' '+str(raw))
    actual=c.linked(target.parent,item)
    if actual!=target:raise ValueError('interface reference differs from declared owner: '+field)
    bound=dict(item,path=str(target))  # Keep the original expected bytes/SHA.
    c.verify(bound)
    return bound


def canonical_worker_report(model,report_path,request_output,evidence):
    """New in-memory copy only. The Worker and interface JSON files stay intact."""
    report_path=Path(report_path);roots=[report_path.parent,Path(request_output),Path(model['output_root'])]
    if any(not p.is_absolute() for p in roots):raise ValueError('Worker owner context must be absolute')
    resolved={p.resolve() for p in roots}
    if len(resolved)!=1:raise ValueError('ambiguous Worker/interface report roots: '+str(sorted(map(str,resolved))))
    root=next(iter(resolved));h=evidence['H']
    native=c.verify(h['native_air']);deployment=c.verify(h['native_deployment'])
    dep_air=c.linked(deployment.parent,c.read(deployment)['air_manifest'])
    expected=dict(source_lock=[p/'source-lock.json' for p in roots],native_air=[native,dep_air])

    def normalize(value):
        result=copy.deepcopy(value)
        for field,targets in expected.items():result[field]=bind_record(result[field],targets,'$.interfaces.'+field)
        return result

    # The producer writes these references relative to two DIFFERENT roots:
    # source_report -> draft-interface.json.parent; native_air -> AIR.parent.
    interface_path=root/'draft-interface.json'
    raw=c.read(interface_path)
    interfaces=normalize(model['interfaces']);producer=normalize(raw)
    if Path(interfaces['source_lock']['path']).parent!=root:
        raise ValueError('interface source_lock escapes its Worker root')
    if interfaces!=producer:raise ValueError('Worker interfaces differ from its producer report')
    result=copy.deepcopy(model);result['interfaces']=interfaces
    result['interface_identity_binding']=dict(producer_report=c.identity(interface_path),
        worker_root=str(root),native_air_root=str(native.parent),native_deployment=dict(h['native_deployment']),
        declared_paths={field:raw[field]['path'] for field in expected},
        scope='Known interface references only; original expected hashes retained; no file writes')
    return result
