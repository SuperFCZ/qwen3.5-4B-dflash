#!/usr/bin/env python3
"""Measure the current CPU online quantization/encoding; never call it NPU timing."""
import argparse
from pathlib import Path
import time
import numpy as np
import real_mlp_common as c


def main(a):
    root=c.fresh(a.output_dir);r=dict(status='NOT_RUN',sources=c.sources(),prepare_report=c.identity(a.prepare_report),npu='NOT_RUN')
    try:
        prep=c.prepared(a.prepare_report);r['cases']=[]
        sw=c.array(prep['weights']['gate_up']['files']['weight_scale_f64.bin'],'<f8',(19456,))
        for case in prep['cases']:
            x=c.array(case['files']['gate_up_x.bin'],'<f2',(16,2560)).copy()
            qa=c.array(case['files']['a_i8.bin'],'i1',(16,2560));words=c.array(case['files']['quant_u64_doc_candidate.bin'],'<u8',(19456,))
            # Exclude diagnostic metrics/percentiles from the quantizer benchmark.
            sa=float(np.max(np.abs(x.astype(np.float64))))/127 or 1.
            def quant():
                v=x.astype(np.float64);peak=float(np.max(np.abs(v)));s=peak/127 if peak else 1.
                return np.clip(np.rint(v/s),-127,127).astype('i1'),s
            jobs={'activation_quant_cpu':quant,'combined_scale_and_encode_cpu':lambda:c.encode((sw*sa).astype('<f4'))}
            record=dict(case=case['id'],rounds={});r['cases'].append(record)
            for name,fn in jobs.items():
                rows=[]
                for rnd in range(3):
                    for _ in range(5):fn()
                    samples=[]
                    for _ in range(30):
                        start=time.perf_counter_ns();out=fn();samples.append((time.perf_counter_ns()-start)/1e6)
                    if name=='activation_quant_cpu':
                        if out[0].tobytes()!=qa.tobytes() or out[1]!=sa:raise ValueError('quantizer repeat differs')
                    elif out[0].tobytes()!=words.tobytes():raise ValueError('encoding repeat differs')
                    rows.append(dict(round=rnd,statistics=c.stats(samples,30)))
                record['rounds'][name]=rows
                record[name]=c.stats([s for row in rows for s in row['statistics']['samples_ms']],90)
        r.update(status='CPU_ONLINE_PREPROCESSING_MEASURED',scope='NUMPY_CPU_HOT_INPUT_ONLY',
            online_npu_preprocessing='NOT_RUN',upload='NOT_TIMED',complete_mlp='NOT_TIMED',
            integration_note='Not overlap-safe/integrated with resident NPU Matmul; do not add independent medians as MLP wall time')
    except Exception as exc:r.update(status='BLOCKED',error=f'{type(exc).__name__}: {exc}')
    finally:
        try:
            c.verify_tree(r);c.prepared(a.prepare_report)
            if r['sources']!=c.sources():raise ValueError('source changed')
        except Exception as exc:r.update(status='BLOCKED',integrity_error=str(exc))
        c.write(root/'report.json',r)
    print(r['status']);return int(r['status']=='BLOCKED')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-report',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    raise SystemExit(main(p.parse_args()))
