#!/usr/bin/env python3
"""Checkpoint-checked C16/C64 captures -> independent Layer0 references and B-NZ."""
import argparse
from pathlib import Path
import time
import numpy as np
import real_mlp_common as c


def convert_weights(root, kind, q, scale):
    root.mkdir()
    start = time.perf_counter_ns()
    w = c.effective_weight(q, scale)
    quant = np.empty(q.shape, 'i1'); scales = np.empty(q.shape[0], '<f8')
    clipped = endpoints = 0
    for n in range(0, q.shape[0], 256):
        b, s, detail = c.quantize_weight(w[n:n+256])
        quant[n:n+256] = b; scales[n:n+256] = s
        clipped += detail['clipped']; endpoints += detail['endpoint_count']
    files = {name: c.save(root, name, value) for name, value in
             [('q_group_nk.bin', q), ('s_group_gn.bin', scale), ('w_effective_nk.bin', w),
              ('q_fullk_nk.bin', quant), ('weight_scale_f64.bin', scales)]}
    result = dict(files=files, shape=list(q.shape), clipped_ratio=clipped/q.size,
        endpoint_ratio=endpoints/q.size, weight_error=c.errors(w, quant.astype(np.float64)*scales[:, None]),
        scale_min=float(scales.min()), scale_max=float(scales.max()),
        scale_percentiles={str(p):float(np.percentile(scales,p)) for p in (0,1,50,95,99,100)},
        effective_weight_rule='RN16(INT8_Group128 * FP16(checkpoint_BF16_scale))',
        fullk_rule='max(abs(effective_FP16_weight), K)/127 per output channel; FP64 RNE clip[-127,127]',
        conversion_prepare_wall_ms=(time.perf_counter_ns()-start)/1e6,
        conversion_timing_scope='CPU conversion, diagnostics, writes and hashing; single offline observation')
    if kind == 'gate_up':
        start = time.perf_counter_ns()
        pack_ns=inverse_ns=write_ns=0
        path = root/'b_nz.bin'
        with path.open('xb') as f:
            for t in range(76):
                tick=time.perf_counter_ns()
                b = quant[t*256:(t+1)*256].T.copy()
                z = c.pack_tile(b)
                pack_ns+=time.perf_counter_ns()-tick;tick=time.perf_counter_ns()
                if not np.array_equal(c.unpack_tile(z), b):
                    raise ValueError('B-NZ roundtrip failed')
                inverse_ns+=time.perf_counter_ns()-tick;tick=time.perf_counter_ns()
                z.tofile(f)
                write_ns+=time.perf_counter_ns()-tick
        result.update(packed_b=c.identity(path), physical_shape=[76,80,16,16,32], padding_bytes=0,
            prepack=dict(permutation_cpu_ms=pack_ns/1e6,inverse_cpu_ms=inverse_ns/1e6,write_wall_ms=write_ns/1e6,
                total_wall_ms=(time.perf_counter_ns()-start)/1e6,fsync_timing=False,scope='CPU static preparation, not resident Kernel'),
            packing='NZ[t,k//32,n//16,n%16,k%32] = q_fullk[t*256+n,k]')
    else:
        result['device_kernel'] = 'BLOCKED_FIXED_ABI_K9728_N2560_NOT_SUPPORTED'
    return result, w, quant, scales


def fullk_cpu(x, q, sw):
    start = time.perf_counter_ns(); a, sa, detail = c.shared_activation(x)
    quant_ms = (time.perf_counter_ns()-start)/1e6
    start = time.perf_counter_ns()
    product = np.float64(sa)*sw
    combined = product.astype('<f4')
    if np.any((product > 0) & (combined == 0)):
        raise ValueError('positive combined Scale underflows FP32')
    words, effective = c.encode(combined)
    scale_ms = (time.perf_counter_ns()-start)/1e6
    dot = np.empty((x.shape[0], q.shape[0]), '<i4'); y = np.empty(dot.shape, '<f2')
    for n in range(0, q.shape[0], 256):
        dot[:, n:n+256], y[:, n:n+256] = c.compute(a, q[n:n+256].T.copy(), words[n:n+256])
    return dict(a=a, words=words, combined=combined, effective=effective, dot=dot, y=y), dict(
        activation=detail, combined_min=float(combined.min()), combined_max=float(combined.max()),
        scale_encoding_error=c.errors(combined, effective),
        activation_quant_cpu_ms=quant_ms, scale_generate_encode_cpu_ms=scale_ms,
        timing_scope='NUMPY_CPU_SINGLE_PREPARE_OBSERVATION_NOT_NPU_OR_MLP_TIMING')


def main(a):
    root = c.fresh(a.output_dir)
    r = dict(status='NOT_RUN', sources=c.sources(), npu='NOT_RUN', quality='UNRESOLVED',
        benchmark='NOT_RUN', full_w8a8_mlp='NOT_RUN_DOWN_KERNEL_UNSUPPORTED', cases=[])
    try:
        r['checkpoint'] = c.checkpoint(a.draft_dir)
        bound = [c.capture_binding(p, r['checkpoint']) for p in a.capture]
        c.coverage([v[0] for v in bound])
        if len({v[0]['manifest']['sha256'] for v in bound}) != len(bound):
            raise ValueError('duplicate captures')
        weights = {}; r['weights'] = {}
        for kind in ('gate_up', 'down'):
            q, s = c.checkpoint_projection(r['checkpoint'], kind)
            for proof, bundle, cases in bound:
                files = cases[kind]['files']; base = c.verify(proof['manifest']).parent
                cq = np.memmap(c.checked_file(base, files['q_nk.bin']), 'i1', mode='r', shape=q.shape)
                cs = np.memmap(c.checked_file(base, files['s_gn.bin']), '<f2', mode='r', shape=s.shape)
                if not np.array_equal(cq, q) or cs.tobytes() != s.tobytes():
                    raise ValueError('captured projection differs from pinned checkpoint: ' + kind)
            record, w, quant, sw = convert_weights(root/kind, kind, q, s)
            r['weights'][kind] = record; weights[kind] = (w, quant, sw)
        for i, (proof, bundle, cases) in enumerate(bound):
            dest = root/f'sample-{i:03d}'; dest.mkdir()
            item = dict(id=dest.name, capture=proof, files={}, diagnostics={})
            r['cases'].append(item)
            for kind in ('gate_up', 'down'):
                m, k, n = c.SHAPES[kind]; files = cases[kind]['files']; base = c.verify(proof['manifest']).parent
                for old, new, shape in [('x.bin', kind+'_x.bin', (m,k)), ('eager-0.bin', kind+'_w8a16.bin', (m,n))]:
                    value = np.fromfile(c.checked_file(base, files[old]), '<f2').reshape(shape)
                    if not np.isfinite(value).all():raise ValueError('nonfinite real capture')
                    item['files'][new] = c.save(dest, new, value)
            x = c.array(item['files']['gate_up_x.bin'], '<f2', (16,2560))
            w, q, sw = weights['gate_up']; result, diag = fullk_cpu(x, q, sw)
            ref = c.fp64_mm(x, w)
            for name, value in [('a_i8.bin', result['a']), ('quant_u64_doc_candidate.bin', result['words']),
                ('combined_f32.bin', result['combined']), ('effective_f32.bin', result['effective']),
                ('dot_i32.bin', result['dot']), ('cpu_fullk_fp16.bin', result['y']), ('gate_up_reference_f64.bin', ref)]:
                item['files'][name] = c.save(dest, name, value)
            original = c.array(item['files']['gate_up_w8a16.bin'], '<f2', (16,19456))
            diag.update(cpu_fullk_vs_effective_fp64=c.errors(ref,result['y']), original_w8a16_vs_effective_fp64=c.errors(ref,original))
            item['diagnostics']['gate_up'] = diag
            # Independent complete CPU design: actual NPU SiLU is assessed separately.
            _, z_fp = c.swiglu_cpu(ref.astype('<f2'))
            _, z_w8 = c.swiglu_cpu(result['y'])
            dw, dq, ds = weights['down']
            down_ref = c.fp64_mm(z_fp, dw)
            down_w8, down_diag = fullk_cpu(z_w8, dq, ds)
            native_down_x = c.array(item['files']['down_x.bin'], '<f2', (16,9728))
            native_down = c.array(item['files']['down_w8a16.bin'], '<f2', (16,2560))
            for name,value in [('cpu_swiglu_fp16.bin',z_fp),('cpu_swiglu_w8a8.bin',z_w8),
                ('mlp_reference_f64.bin',down_ref),('cpu_full_w8a8_mlp.bin',down_w8['y']),
                ('down_at_native_x_f64.bin',c.fp64_mm(native_down_x,dw))]:
                item['files'][name]=c.save(dest,name,value)
            item['diagnostics']['mlp_cpu_design_only'] = dict(down=down_diag,
                w8a8_vs_effective_fp64=c.errors(down_ref,down_w8['y']),
                original_w8a16_vs_effective_fp64=c.errors(down_ref,native_down),
                swiglu_vs_native_input=c.errors(native_down_x,z_w8),
                semantics='CPU RN16 SiLU then RN16 multiply; Down Full-K INT64/FP32/RN16 candidate, NOT_NPU')
        r['status'] = 'REAL_MLP_CPU_PREPARED_NPU_NOT_RUN'
    except Exception as exc:
        r.update(status='BLOCKED', error=f'{type(exc).__name__}: {exc}')
    finally:
        try:
            c.verify_tree(r)
            if r['sources'] != c.sources():raise ValueError('source changed')
        except Exception as exc:r.update(status='BLOCKED', integrity_error=str(exc))
        c.write(root/'prepare-report.json',r)
        c.write(root/'preprocessing-inventory.json',dict(status=r['status'], weights=r.get('weights'),
            activation='CPU FP64 maxabs/RNE shared over16 rows; online NPU NOT_RUN',
            combined_scale='CPU FP64 sA*sW -> FP32 -> original UINT64 encoding; per input',
            weight_preparation='static checkpoint decode, RN16 W_eff, Full-K requant, B-NZ',
            upload='input payload per sample 50,003,968 B; excluded from resident timing; B reusable',
            complete_w8a8_mlp='NOT_RUN_DOWN_KERNEL_UNSUPPORTED', reserve='UNCHANGED'))
    print(r['status']); return int(r['status']=='BLOCKED')


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--draft-dir',type=Path,required=True)
    p.add_argument('--capture',type=Path,action='append',required=True,help='A2 manifest; at least two real samples per C16/C64')
    p.add_argument('--output-dir',type=Path,required=True)
    return p.parse_args(argv)


if __name__=='__main__':raise SystemExit(main(parse()))
