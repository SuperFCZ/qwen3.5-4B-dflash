"""Resolve existing six-layer FP16 artifacts; never create a capture or golden."""
import copy
import json
import math
import os
from pathlib import Path
import struct

from common import checked_file, sha256
from a2_common import snapshot_contract, replay_context_rows


def load(path):
    return json.loads(Path(path).read_text())


def inventory(root):
    """Bounded metadata-only discovery. Ambiguity is an error, not newest-wins."""
    deployments, replays = [], []
    for directory, dirs, files in os.walk(root, followlinks=False):
        depth = len(Path(directory).relative_to(root).parts)
        dirs[:] = sorted(d for d in dirs if depth < 5 and d not in
                         ('.git', 'profiler', 'weights', 'checkpoint', 'checkpoints'))
        for name in sorted(files):
            path = Path(directory)/name
            if path.suffix != '.json' or path.stat().st_size > 4*1024*1024:
                continue
            try:
                value = load(path)
                if not isinstance(value, dict):
                    continue
                if value.get('status') == 'PASS' and any(
                    g.get('name') == 'draft' and g.get('metadata', {}).get('draft_quantization') == 'fp16'
                    for g in value.get('graphs', []) if isinstance(g, dict)):
                    if 'air_manifest' in value:
                        deployments.append(path.resolve())
                if value.get('status') == 'PASS_REPLAY_CHECKS' and value.get('fake_acl') is False:
                    replays.append(path.resolve())
            except (ValueError, OSError, TypeError):
                continue
    return sorted(set(deployments)), sorted(set(replays))


def draft_graph(manifest):
    graphs = [g for g in manifest['graphs'] if g['name'] == 'draft']
    if len(graphs) != 1:
        raise ValueError('need exactly one historical FP16 Draft graph')
    return graphs[0]


def deployment(path):
    path = Path(path).resolve(); dep = load(path)
    if dep.get('status') != 'PASS' or dep.get('target', {}).get('soc_version') != 'Ascend310P3':
        raise ValueError('FP16 deployment must be accepted Ascend310P3')
    air_path = checked_file(path.parent, dep['air_manifest']); air = load(air_path)
    g = draft_graph(dep); original = draft_graph(air)
    if air.get('status') != 'PASS' or any(g.get(k) != original.get(k) for k in
                                        ('metadata', 'runtime_input_abi', 'custom_op_audit')):
        raise ValueError('FP16 AIR/Deployment identity differs')
    meta = g['metadata']; contract = meta['incremental_contract']; abi = meta['tensor_abi']
    names = ['draft_top1']+[f'd{i}_{k}' for i in range(6) for k in ('key', 'value')]
    if (meta.get('draft_quantization') != 'fp16' or contract.get('draft_quantization') != 'fp16'
        or len(abi['inputs']) != 17 or [t['name'] for t in abi['outputs']] != names
        or len(contract['draft_states']) != 12):
        raise ValueError('expected separate six-layer FP16 17-input/13-output ABI')
    if any('DFlash' in a.get('ge_op_type', '') or a.get('ge_op_type') == 'WeightQuantBatchMatmulV2'
           for a in g.get('custom_op_audit', [])):
        raise ValueError('FP16 graph contains quant/custom A/C substitution')
    watch = {str(p): sha256(p) for p in (path, air_path)}
    payloads = [checked_file(air_path.parent, p) for p in original['payload_files']]
    if not payloads:
        raise ValueError('FP16 immutable AIR payload inventory is empty')
    for p in payloads:
        watch[str(p)] = sha256(p)
    # The existing accepted multi-IO runner is static. Never silently execute
    # a dynamic OM at a guessed gear, or run Target Prefill to obtain inputs.
    gears = g.get('static_gear_oms')
    if not gears or [r.get('rows') for r in gears] != [16, 64] or gears[0]['om'] != g['om']:
        raise ValueError('FP16_DYNAMIC_OM_UNSUPPORTED_BY_REUSED_RUNNER: need existing accepted '
                         'static C16/C64 OMs; no recompile or Target replay performed')
    selected = {}; options = []
    for row in gears:
        command = row['atc_command']
        if any(x.startswith('--dynamic_') for x in command) or '--input_format=ND' not in command:
            raise ValueError('FP16 static OM ATC contract differs')
        shape_args = [x for x in command if x.startswith('--input_shape=')]
        bindings = g['runtime_input_abi'].get('bindings', [])
        data_names = [b['data_node_name'] for b in bindings]
        if len(data_names) != 17 or meta.get('dynamic_input_axes') != {'features': [1]}:
            raise ValueError('FP16 static gear Data bindings/axes missing')
        shapes = []
        for t, name in zip(abi['inputs'], data_names):
            shape = list(t['shape'])
            if t['name'] == 'features': shape[1] = row['rows']
            shapes.append(name+':'+','.join(map(str, shape)))
        if shape_args != ['--input_shape='+';'.join(shapes)]:
            raise ValueError('FP16 static gear shape binding differs')
        options.append(sorted(x for x in command[1:] if not x.startswith(('--input_shape=', '--output='))))
        selected[row['rows']] = checked_file(path.parent, row['om'])
        watch[str(selected[row['rows']])] = row['om']['sha256']
    if options[0] != options[1] or selected[16] == selected[64]:
        raise ValueError('FP16 gear compiler settings differ or OMs alias')
    return dict(path=path, graph=g, air=air, oms=selected, watch=watch, payloads=payloads)


def replay(base, path, gear):
    path = Path(path).resolve(); r = load(path); g = base['graph']
    if (r.get('status') != 'PASS_REPLAY_CHECKS' or r.get('fake_acl') is not False
        or r.get('kv_output_audit', {}).get('context_rows') != gear
        or r.get('draft_om_sha256') != g['om']['sha256']
        or r.get('draft_static64_om_sha256') != g['static_gear_oms'][1]['om']['sha256']):
        raise ValueError('FP16 replay status/gear/OM identity differs')
    phases = r.get('phases', [])
    keys = ('token_mismatch_iterations', 'input_restore_mismatch_iterations',
            'input_mutation_iterations', 'full_output_hash_mismatch_iterations', 'valid_kv_mismatch_iterations')
    if (len(phases) != 2 or {p['phase'] for p in phases} != {'isolated', 'interleaved_prefill'}
        or any(p.get('iterations', 0) < 2 or any(p.get(k) != 0 for k in keys) for p in phases)):
        raise ValueError('FP16 frozen replay must have strict repeat/readonly/full-output checks')
    folder = Path(r['input_directory']).resolve()
    om_hash, specs = snapshot_contract((folder/'contract.txt').read_text())
    if om_hash != r['draft_om_sha256'] or (folder/'contract.txt').read_text().splitlines()[3] != 'static64 '+r['draft_static64_om_sha256']:
        raise ValueError('FP16 snapshot contract OM binding differs')
    abi = copy.deepcopy(g['metadata']['tensor_abi']); inputs = {}
    widths = {'float16':2, 'int16':2, 'int64':8, 'float32':4}
    watch = dict(base['watch']); watch[str(path)] = sha256(path)
    for p in (folder/'contract.txt', folder/'sha256.txt'): watch[str(p)] = sha256(p)
    disk_hashes = dict(line.split() for line in (folder/'sha256.txt').read_text().splitlines())
    if disk_hashes != r['snapshot_sha256'] or set(specs) != {t['name'] for t in abi['inputs']}:
        raise ValueError('FP16 input hash inventory differs')
    for t in abi['inputs']:
        name=t['name']; spec=specs[name]; p=folder/(name+'.bin')
        if (spec['dtype'] != t['dtype'] or spec['shape'] != t['shape']
            or p.stat().st_size != spec['bytes'] or sha256(p) != disk_hashes[name]):
            raise ValueError('FP16 snapshot shape/hash differs: '+name)
        watch[str(p)] = disk_hashes[name]; inputs[name] = p
    valid_path=inputs['valid_rows']; valid_type=specs['valid_rows']['dtype']
    if valid_type not in ('int64','int16'):
        raise ValueError('unsupported frozen valid_rows dtype')
    valid=struct.unpack('<q' if valid_type=='int64' else '<h', valid_path.read_bytes())[0]
    features=next(t for t in abi['inputs'] if t['name']=='features')
    replay_context_rows(valid, features['shape'][1], gear)
    if features['shape'][0] != 1 or features['shape'][2] != g['metadata']['incremental_contract']['feature_width']:
        raise ValueError('FP16 feature width differs from its own contract')
    features['shape'][1] = gear  # packed prefix from original maximum-allocation snapshot
    golden = {}
    if set(r['reference_output_sha256']) != {t['name'] for t in abi['outputs']}:
        raise ValueError('FP16 golden output inventory differs')
    for t in abi['outputs']:
        name=t['name']; p=folder.parent/'outputs/reference'/(name+'.bin')
        if p.stat().st_size != math.prod(t['shape'])*widths[t['dtype']] or sha256(p) != r['reference_output_sha256'][name]:
            raise ValueError('FP16 golden missing/changed: '+name)
        golden[name] = p; watch[str(p)] = sha256(p)
    return dict(model='fp16', gear=gear, layers=6, abi=abi, inputs=inputs, golden=golden,
                om=base['oms'][gear], payloads=base['payloads'], watch=watch,
                identity=dict(deployment=str(base['path']), replay=str(path), metadata=g['metadata'],
                              factory_config=base['air'].get('factory_config')))


def resolve(search_root, deployment_path=None, replay_paths=None):
    deployments, replays = inventory(search_root)
    if deployment_path: deployments = [Path(deployment_path).resolve()]
    candidates=[]; rejected=[]
    for path in deployments:
        try:
            base=deployment(path); cases=[]
            for gear in (16,64):
                matching=[]
                for rp in ([replay_paths[gear]] if replay_paths and gear in replay_paths else replays):
                    r=load(rp)
                    if r.get('draft_om_sha256') == base['graph']['om']['sha256'] and r.get('kv_output_audit',{}).get('context_rows') == gear:
                        matching.append(replay(base,rp,gear))
                if len(matching)!=1:
                    raise ValueError(f'C{gear}: need one exact frozen FP16 replay; found {len(matching)}; use explicit --fp16-c{gear}-replay')
                cases+=matching
            candidates.append(cases)
        except (ValueError,KeyError,OSError,TypeError) as error:
            rejected.append(dict(path=str(path),error=str(error)))
    diagnostic=dict(deployments=list(map(str,deployments)),replays=list(map(str,replays)),rejected=rejected)
    if len(candidates)!=1:
        raise ValueError('FP16_ARTIFACTS_BLOCKED: need one unambiguous accepted deployment + C16/C64 snapshots/goldens; '
                         +json.dumps(diagnostic,ensure_ascii=False))
    return candidates[0],diagnostic
