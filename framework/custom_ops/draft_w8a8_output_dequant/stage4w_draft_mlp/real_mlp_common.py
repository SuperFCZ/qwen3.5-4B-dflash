"""Independent real-Draft evidence; historical implementations remain read-only."""
from pathlib import Path
import ast
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BASE = HERE.parent
sys.path[:0] = [str(REPO / 'framework/custom_ops/draft_quant/test'),
               str(BASE / 'stage4v_bnz_fp16_paired')]
import pair_evidence as pe
from reference_v import encode, quantize_weight, compute, stats as errors
from nz_layout import pack_tile, unpack_tile
from explore_numeric import observe, timing, stats
from w8_reference import activation
from a2_common import load_bundle, checked_file, validate_layout, snapshot_contract, replay_context_rows
__all__ = ['encode','quantize_weight','compute','pack_tile','unpack_tile','observe','timing','stats','checked_file']

identity, verify, read, write, verify_tree = pe.identity, pe.verify, pe.read, pe.write, pe.verify_tree
SHAPES = {'gate_up': (16, 2560, 19456), 'down': (16, 9728, 2560)}


def sources():
    paths = [p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [REPO / p for p in ('models/dflash_v1/draft_quantization.py',
        'models/dflash_v1/modeling_dflash.py', 'models/dflash_v1/weight_quant_matmul.py',
        'framework/python/qwen35_dflash/ascend310p/incremental.py',
        'framework/python/qwen35_dflash/ascend310p/quant_factory.py',
        'framework/custom_ops/draft_quant/test/a2_capture.py')]
    # Imported arithmetic/layout is also part of this experiment's source lock.
    paths += [Path(sys.modules[x].__file__) for x in
              ('reference_v', 'nz_layout', 'w8_reference', 'explore_numeric', 'a2_common')]
    return [identity(p) for p in sorted(set(paths))]


def fresh(path):
    path = Path(path).resolve()
    if path == REPO or REPO in path.parents:
        raise ValueError('experiment outputs must be outside repository')
    path.mkdir(parents=True, exist_ok=False)
    return path


def lock():
    tree = ast.parse((REPO / 'models/dflash_v1/draft_quantization.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'QUANTIZED_DRAFTS' for t in node.targets):
            return ast.literal_eval(node.value)['w8a16']
    raise ValueError('checkpoint source lock missing')


def checkpoint(root):
    root = Path(root).resolve(strict=True)
    expected = lock()
    items = {n: identity(root / n) for n in ('config.json', 'model.safetensors')}
    for n, key in [('config.json', 'config_sha256'), ('model.safetensors', 'model_sha256')]:
        if items[n]['sha256'] != expected[key] or not verify(items[n]).is_relative_to(root):
            raise ValueError('pinned W8 checkpoint differs: ' + n)
    if items['model.safetensors']['bytes'] != expected['model_bytes']:
        raise ValueError('checkpoint size differs')
    return dict(source=expected, files=items)


def tensor_view(file, name):
    """Read only the pinned single-file safetensors I32/BF16 entries, no torch required."""
    path = verify(file)
    with path.open('rb') as f:
        length = int.from_bytes(f.read(8), 'little')
        if not 0 < length <= min(16 * 1024 * 1024, file['bytes'] - 8):
            raise ValueError('invalid safetensors header length')
        metadata = json.loads(f.read(length))
    item = metadata[name]
    dtype = {'I32': '<i4', 'BF16': '<u2'}[item['dtype']]
    lo, hi = item['data_offsets']
    shape = item['shape']
    if (any(type(v) is not int or v <= 0 for v in shape) or
        lo < 0 or hi - lo != np.prod(shape) * np.dtype(dtype).itemsize or
        8 + length + hi > file['bytes']):
        raise ValueError('safetensors bounds/shape')
    raw = np.memmap(path, mode='r', dtype=dtype, offset=8 + length + lo, shape=tuple(shape))
    if item['dtype'] == 'BF16':
        return (raw.astype('<u4') << 16).view('<f4').astype('<f2')
    return raw


def checkpoint_projection(ckpt, kind):
    names = ('gate_proj', 'up_proj') if kind == 'gate_up' else ('down_proj',)
    qs, scales = [], []
    for name in names:
        prefix = 'layers.0.mlp.' + name
        packed = tensor_view(ckpt['files']['model.safetensors'], prefix + '.weight_packed')
        q = (((packed[..., None] >> (np.arange(4, dtype=np.int32) * 8)) & 255) - 128).astype('i1')
        qs.append(q.reshape(packed.shape[0], -1))
        scales.append(tensor_view(ckpt['files']['model.safetensors'], prefix + '.weight_scale'))
    q, s = np.concatenate(qs), np.concatenate(scales).T.copy()
    _, k, n = SHAPES[kind]
    if q.shape != (n, k) or s.shape != (k // 128, n) or not np.isfinite(s).all() or np.any(s <= 0):
        raise ValueError('checkpoint projection shape/Scale invalid')
    return q, s


def capture_binding(path, ckpt):
    """Accept actual A2 eager replay, never generated random X or an OM parity claim."""
    path = Path(path).resolve()
    b = load_bundle(path)
    audit = b['checkpoint']
    if (audit['source'] != ckpt['source'] or audit.get('execution') != 'cann-weight-quant-batchmatmul-v2'
        or audit.get('matmul_backend') != 'weight_quant' or audit.get('activation_dtype') != 'float16'):
        raise ValueError('capture not from pinned native W8A16 checkpoint/path')
    for name, sha in b['sources'].items():
        p = (REPO / name).resolve()
        if not p.is_relative_to(REPO) or identity(p)['sha256'] != sha:
            raise ValueError('capture production source drift')
    src = b['source']
    replay = read(Path(src['report']))
    if identity(src['report'])['sha256'] != src['report_sha256'] or replay.get('fake_acl') is not False:
        raise ValueError('real replay report missing/changed')
    directory = Path(replay['input_directory']).resolve()
    contract = identity(directory / 'contract.txt')
    om_hash, specs = snapshot_contract(verify(contract).read_text())
    if contract['sha256'] != src['contract_sha256'] or om_hash != src['draft_om_sha256']:
        raise ValueError('real replay contract differs')
    originals = []
    names={'features','start_position','valid_rows','anchor','proposal_count'}
    names.update(f'd{i}_{kind}' for i in range(5) for kind in ('key','value'))
    if set(src['snapshot_sha256'])!=names:
        raise ValueError('real replay input inventory incomplete')
    for name, digest in src['snapshot_sha256'].items():
        item = identity(directory / (name + '.bin'))
        if item['sha256'] != digest or replay['snapshot_sha256'][name] != digest or item['bytes'] != specs[name]['bytes']:
            raise ValueError('snapshot identity differs: ' + name)
        originals.append(item)
    valid=np.fromfile(directory/'valid_rows.bin','<i2')
    if valid.shape!=(1,) or specs['valid_rows']['dtype']!='int16':
        raise ValueError('replay valid-row ABI differs')
    rows=replay_context_rows(int(valid[0]),specs['features']['shape'][1],replay.get('kv_output_audit',{}).get('context_rows'))
    if src['context_rows']!=rows or b['context_rows'] != rows:
        raise ValueError('context gear differs')
    embedding = identity(b['embedding']['shard'])
    if embedding['sha256'] != b['embedding']['shard_sha256']:
        raise ValueError('embedding checkpoint changed')
    cases = {c['projection']: c for c in b['cases'] if c['layer'] == 0}
    for c in cases.values():
        validate_layout(path.parent, c)
    captured={kind:{name:identity(checked_file(path.parent,item)) for name,item in case['files'].items()} for kind,case in cases.items()}
    return dict(manifest=identity(path), replay=identity(src['report']), contract=contract,
        snapshot=originals, embedding=embedding, context_rows=src['context_rows'],
        captured_layer0_files=captured,
        activation_sha256=cases['gate_up']['files']['x.bin']['sha256']), b, cases


def coverage(rows):
    for context in (16, 64):
        selected = [r for r in rows if r['context_rows'] == context]
        if len(selected) < 2 or len({r['activation_sha256'] for r in selected}) < 2 or len({r['replay']['sha256'] for r in selected}) < 2:
            raise ValueError('need at least two distinct real activations/replays for C' + str(context))


def effective_weight(q, scale):
    n, k = q.shape
    if q.dtype != np.int8 or k % 128 or scale.shape != (k // 128, n):
        raise ValueError('Group128 weight layout')
    w = (q.reshape(n, k // 128, 128).astype(np.float64) * scale.T[..., None].astype(np.float64)).astype('<f2').reshape(n, k)
    if not np.isfinite(w).all():
        raise ValueError('nonfinite effective FP16 weight')
    return w


def shared_activation(x):
    if x.dtype != np.float16 or x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('finite FP16 matrix required')
    if x.shape == (16, 2560):
        q, scale, detail = activation(x)  # exact already-reviewed Full-X recipe
    else:  # CPU Down design only, not an accepted device shape
        peak = float(np.max(np.abs(x.astype(np.float64))))
        scale = peak / 127 if peak else 1.
        rounded = np.rint(x.astype(np.float64) / scale)
        q = np.clip(rounded, -127, 127).astype('i1')
        detail = dict(scale_scope=list(x.shape), clipped_ratio=float(np.mean(np.abs(rounded) > 127)),
                      endpoint_ratio=float(np.mean(np.abs(q.astype(np.int16)) == 127)))
    row_peak = np.max(np.abs(x.astype(np.float64)), axis=1)
    return q, scale, dict(detail, shared_scale=scale, row_max_abs=row_peak.tolist(),
        hypothetical_row_scales=np.where(row_peak == 0, 1., row_peak / 127).tolist(),
        row_endpoint_ratio=np.mean(np.abs(q.astype(np.int16))==127,axis=1).tolist(),
        reconstruction_error=errors(x, q.astype(np.float64) * scale),
        rowwise_scale_supported=False, online_npu_quantization='NOT_RUN')


def fp64_mm(x, w, block=256):
    result = np.empty((x.shape[0], w.shape[0]), '<f8')
    for n in range(0, w.shape[0], block):
        result[:, n:n+block] = x.astype(np.float64) @ w[n:n+block].astype(np.float64).T
    if not np.isfinite(result).all():
        raise ValueError('nonfinite FP64 reference')
    return result


def swiglu_cpu(y):
    if y.dtype != np.float16 or y.shape[1] % 2 or not np.isfinite(y).all():
        raise ValueError('FP16 gate/up required')
    gate, up = np.split(y, 2, axis=1)
    z = gate.astype(np.float64)
    sigmoid = np.empty_like(z)
    pos = z >= 0
    sigmoid[pos] = 1 / (1 + np.exp(-z[pos]))
    e = np.exp(z[~pos]); sigmoid[~pos] = e / (1 + e)
    silu = (z * sigmoid).astype('<f2')
    product = (silu.astype(np.float64) * up.astype(np.float64)).astype('<f2')
    if not np.isfinite(product).all():
        raise ValueError('nonfinite SwiGLU reference')
    return silu, product  # CPU diagnostic, not a torch/NPU SiLU bitwise oracle


def array(item, dtype, shape):
    if item['bytes'] != int(np.prod(shape)) * np.dtype(dtype).itemsize:
        raise ValueError('array byte count')
    return np.memmap(verify(item), mode='r', dtype=dtype, shape=shape)


def save(root, name, value):
    path = root / name
    with path.open('xb') as f:
        value.tofile(f)
    return identity(path)


def prepared(path):
    r = read(path); verify_tree(r)
    if r.get('status') != 'REAL_MLP_CPU_PREPARED_NPU_NOT_RUN' or r.get('error') or r.get('integrity_error') or r['sources'] != sources():
        raise ValueError('Prepare incomplete/source drift')
    coverage([c['capture'] for c in r['cases']])
    return r
