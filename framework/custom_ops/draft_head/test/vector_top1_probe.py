#!/usr/bin/env python3
"""Raw FP16 tile semantics on NPU, separate from the 41-case Head acceptance."""
import argparse
from pathlib import Path
import subprocess
import numpy as np
from common import write_json,record,sha256


def winner(row):
    # Integer oracle mirrors Better's mathematics without FP operations/NaN conversion.
    def key(b):
        b=int(b); nan=(b&0x7fff)>0x7c00
        return (nan,0 if nan else (-(b&0x7fff) if b&0x8000 else b))
    return max(range(64),key=lambda i:key(row[i]))


def rows():
    cases=[]
    for v in (0,0x8000,1,0x8001,0x7c00,0xfc00,0x7e01,0xfe21,0x7c01,0xfc01):
        cases.append(np.full(64,v,dtype='<u2'))
    for a,b in ((0,0x8000),(1,2),(0x8002,0x8001),(0x7c00,0x7e12),(0xfe13,0x7e01),(0x7c01,0xfc01)):
        for i in (0,1,31,32,62,63):
            row=np.full(64,a,dtype='<u2'); row[i]=b; cases.append(row)
    # Every FP16 bit pattern appears, with shuffled lane order and mixed boundaries.
    rng=np.random.default_rng(20261013)
    all_bits=np.arange(65536,dtype='<u2'); rng.shuffle(all_bits)
    cases.extend(all_bits.reshape(-1,64))
    # Also test ordered NON-NaN blocks: a NaN must not hide finite/subnormal
    # ordering failures in the exhaustive shuffled corpus.
    finite=np.arange(65536,dtype='<u2'); finite=finite[(finite&0x7fff)<=0x7c00]
    finite=np.pad(finite,(0,(-len(finite))%64),mode='edge')
    cases.extend(finite.reshape(-1,64))
    cases.extend(rng.integers(0,65536,size=(128,64),dtype=np.uint16))
    return np.array(cases,dtype='<u2')


def trace_report(raw_half,trace_bytes):
    trace=np.frombuffer(trace_bytes,dtype='<u4').reshape(15,240)
    result=[]; errors=[]
    def first(mask): return (mask&-mask).bit_length()-1 if mask else 64
    def rank(bits): return -(int(bits)&0x7fff) if int(bits)&0x8000 else int(bits)
    for r,words in enumerate(trace):
        raw=raw_half[r]; nan=(raw&0x7fff)>0x7c00
        expected_nan=sum(1<<i for i,v in enumerate(nan) if v)
        expected=winner(raw); expected_eq=0 if expected_nan else sum(1<<i for i,v in enumerate(raw) if rank(v)==rank(raw[expected]))
        masks={}
        for name,offset,scalar in (('legacy_ne',64,96),('legacy_eq',72,98),('raw_nan',80,100),('selected_eq',88,102)):
            memory=int(words[offset])|(int(words[offset+1])<<32)
            read=int(words[scalar])|(int(words[scalar+1])<<32)
            masks[name]=dict(words_u32=[f'0x{int(v):08x}' for v in words[offset:offset+8]],
                             memory_low64=f'0x{memory:016x}',scalar_low64=f'0x{read:016x}',first_bit=first(read))
            if memory!=read: errors.append(dict(row=r,rule='mask memory/scalar layout mismatch',mask=name))
        new_nan=int(masks['raw_nan']['scalar_low64'],16)
        new_eq=int(masks['selected_eq']['scalar_low64'],16)
        if new_nan!=expected_nan: errors.append(dict(row=r,rule='raw FP16 NaN mask mismatch',expected=f'0x{expected_nan:016x}'))
        if int(words[107])!=int(not expected_nan): errors.append(dict(row=r,rule='wrong selected EQ execution flag'))
        if new_eq!=expected_eq: errors.append(dict(row=r,rule='numeric equality/min-ID mask mismatch',expected=f'0x{expected_eq:016x}'))
        if int(words[108])!=expected: errors.append(dict(row=r,rule='trace winner mismatch',expected=expected,actual=int(words[108])))
        if int(words[109])!=0x48543102: errors.append(dict(row=r,rule='missing diagnostic ABI/version'))
        legacy_nan=int(masks['legacy_ne']['scalar_low64'],16); legacy_eq=int(masks['legacy_eq']['scalar_low64'],16)
        if int(words[106])!=first(legacy_nan or legacy_eq) or int(words[110])!=first(legacy_nan) or int(words[111])!=first(legacy_eq):
            errors.append(dict(row=r,rule='legacy path trace inconsistent'))
        # For non-NaNs the widening must preserve every FP16 value exactly,
        # including zero sign and subnormal. NaN widening payload is recorded,
        # but the selected classifier does not depend on it.
        with np.errstate(invalid='ignore'):
            widened=raw.view('<f2').astype('<f4').view('<u4')
        for stage,actual in (('legacy_full_tile',words[:64]),('selected_per_row',words[176:240])):
            wrong=np.flatnonzero((~nan)&(actual!=widened))
            if len(wrong): errors.append(dict(row=r,rule='non-NaN FP16 widening changed value/bits',stage=stage,lanes=wrong.tolist()))
        markers=words[112:176].copy().view('<f4')
        magnitude=(raw&0x7fff).astype('int32')
        valid=np.isfinite(markers)&np.where(magnitude>0x7c00,markers>0,np.where(magnitude==0x7c00,markers==0,markers<0))
        if not valid.all(): errors.append(dict(row=r,rule='finite classification marker has wrong sign',lanes=np.flatnonzero(~valid).tolist()))
        f32_nan=((words[:64]&0x7f800000)==0x7f800000)&((words[:64]&0x007fffff)!=0)
        diagnosis='legacy trace recorded; no general NaN semantics inferred'
        if nan.all() and f32_nan.all() and not legacy_nan and not legacy_eq:
            diagnosis='NE=0 for widened NaNs; EQ=0; HeadFirstBit(EQ=0)=64, mapped to -1 by old caller'
        result.append(dict(row=r,fp16_bits=[f'0x{int(v):04x}' for v in raw],
            legacy_widened_fp32_bits=[f'0x{int(v):08x}' for v in words[:64]],
            selected_widened_fp32_bits=[f'0x{int(v):08x}' for v in words[176:240]],masks=masks,
            classifier_fp32_bits=[f'0x{int(v):08x}' for v in words[112:176]],
            legacy_reduce_fp32_bits=f'0x{int(words[104]):08x}',legacy_reduce_index_bits=f'0x{int(words[105]):08x}',
            legacy_winner=int(words[106]),selected_eq_executed=bool(words[107]),selected_winner=int(words[108]),
            diagnosis=diagnosis))
    return dict(abi='head-top1-mask-trace-v2',status='FAIL' if errors else 'PASS',rows=result,errors=errors,
                legacy_comparison='diagnostic only; never a required IEEE NaN oracle')


def run(args):
    root=args.output_dir.resolve(); root.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',scope='Vector Top1 raw-FP16 primitive only',full_41_case_acceptance='NOT_RUN',
                runner_sha256=sha256(args.runner),batches=[])
    samples=rows(); report['rows']=len(samples)
    try:
        for start in range(0,len(samples),15):
            batch=np.zeros((15,64),dtype='<u2'); batch[:len(samples[start:start+15])]=samples[start:start+15]
            expected=np.array([winner(row) for row in batch],dtype='<i8')
            folder=root/f'batch-{start:04d}'; folder.mkdir(); batch.tofile(folder/'logits.bin')
            with (folder/'runner.log').open('w') as log:
                subprocess.run([str(args.runner.resolve()),str(args.device_id),str(folder)],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=120)
            for repeat in range(3):
                trace=(folder/f'trace-{repeat}.bin').read_bytes()
                diagnostic=trace_report(batch,trace)
                write_json(folder/f'diagnostics-{repeat}.json',diagnostic)
                if repeat and trace!=(folder/'trace-0.bin').read_bytes(): raise ValueError('trace repeat differs')
                if diagnostic['status']!='PASS': raise ValueError(dict(batch=start,diagnostics=str(folder/f'diagnostics-{repeat}.json'),errors=diagnostic['errors']))
                raw=(folder/f'ids-{repeat}.bin').read_bytes()
                if raw!=expected.tobytes():
                    actual=np.frombuffer(raw,dtype='<i8'); row=int(np.flatnonzero(actual!=expected)[0])
                    raise ValueError(dict(batch=start,row=row,expected=int(expected[row]),actual=int(actual[row]),
                                          fp16_bits=[f'0x{int(x):04x}' for x in batch[row]]))
            report['batches'].append(dict(start=start,input=record(folder/'logits.bin',root),output=record(folder/'ids-0.bin',root),trace=record(folder/'trace-0.bin',root),
                                          diagnostics=record(folder/'diagnostics-0.json',root)))
            write_json(root/'suite.json',report)
        if sha256(args.runner)!=report['runner_sha256']: raise ValueError('runner changed')
        report['status']='PASS'
    except Exception as e: report.update(status='FAIL',error=str(e))
    write_json(root/'suite.json',report)
    return 0 if report['status']=='PASS' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--runner',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device-id',type=int,default=0); raise SystemExit(run(p.parse_args()))
