"""CPU arithmetic counterexamples and conditional traffic, NOT hardware simulation."""
import math
import struct


def fp16(x):
    return struct.unpack('<e', struct.pack('<e', x))[0]


def fp32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bits(x):
    return f'0x{struct.unpack("<H", struct.pack("<e", x))[0]:04x}'


def evaluate(groups, output_group_half=False):
    """Sparse (x,q) entries padded with zero to K128; real finite FP16 inputs.

    Baseline: FP16 effective weights, sequential FP32 full K, final FP16.
    Alternative: exact INT8->FP16 codes, group FP32 sum, ideal FP32 output
    scale multiply, sequential FP32 group sum, final FP16. This grants the
    proposed path ideal capabilities, without claiming SDK support/encoding.
    """
    baseline=alternative=0.0
    for scale,terms in groups:
        if len(terms)>128 or not math.isfinite(scale) or fp16(scale)!=scale or scale<=0:
            raise ValueError('requires positive FP16 scale and at most 128 terms')
        partial=0.0
        for x,q in terms:
            if not math.isfinite(x) or fp16(x)!=x or type(q) is not int or not -128<=q<=127:
                raise ValueError('requires FP16 activation and signed INT8')
            weight=fp16(q*scale)
            baseline=fp32(baseline+fp32(x*weight))
            partial=fp32(partial+fp32(x*q))
        scaled=fp32(partial*scale)
        if output_group_half:scaled=fp16(scaled)
        alternative=fp32(alternative+scaled)
    a,b=fp16(baseline),fp16(alternative)
    return dict(baseline_fp32=baseline,alternative_fp32=alternative,
                baseline_fp16=a,alternative_fp16=b,baseline_bits=bits(a),alternative_bits=bits(b),
                bitwise_equal=bits(a)==bits(b))


def counterexamples():
    cases=[
        ('weight_rounding_moved_after_dot',[(fp16(1+1/1024),[(1.5,3)])],False),
        ('per_group_fp16_loses_residual',[(1.0,[(65504.,1),(1.,1)]),(1.0,[(-65504.,1)])],True),
        ('fp32_grouping_changes_reduction',[(1.0,[(65504.,127)]),(1.0,[(-65504.,127),(.25,1)])],False),
    ]
    return dict(status='CPU_COUNTEREXAMPLES',hardware='NOT_RUN',
        scope='Sparse groups embed in M16/N64 and 20 K128 groups with zero terms in remaining groups. No real Capture/Golden changed.',
        model='Ideal IEEE FP32 scaling; no assumption about FixPipe encoded scale, saturation, subnormal or NaN behavior',
        cases=[dict(name=name,groups=groups,group_output_fp16=half,result=evaluate(groups,half))
               for name,groups,half in cases],
        conclusion='Output-scale factoring is not universally bitwise equivalent. Frozen workload equivalence remains UNRESOLVED; no tolerance authorized.')


def traffic(m=16,k=2560,n=19456,tile_n=64,group=128):
    if any(type(v) is not int or v<=0 for v in (m,k,n,tile_n,group)) or k%group or n%tile_n:
        raise ValueError('positive aligned dimensions required')
    g=k//group;p=m*tile_n*4;tiles=n//tile_n;full=p*tiles
    return dict(status='CONDITIONAL_BYTE_ACCOUNTING_NOT_MEASUREMENT',groups=g,n_tiles=tiles,
        shape=[m,k,n],partial_fp32_bytes_per_n_tile=p,final_fp16_bytes=m*n*2,
        baseline_logical_weights=dict(int8=n*k,fp16_gn=g*n*2),
        scaled_fp32_partial_spill=dict(
            assumption='Each scaled group output written once to GM and read once by reducer; FP32 accumulator retained on chip',
            gm_write_bytes=g*full,gm_read_bytes=g*full,extra_gm_bytes=2*g*full,
            streamed_partial_buffer_bytes=p,onchip_accumulator_bytes=p,
            all_group_slab_bytes_per_n_tile=g*p,all_group_slab_bytes_full_gate=g*full),
        gm_accumulator_variant=dict(
            assumption='Same partial spills plus first partial copied to GM accumulator; later groups read/update it; final read converts to FP16; no zero-fill counted',
            partial_gm_bytes=2*g*full,accumulator_read_bytes=g*full,accumulator_write_bytes=g*full,
            extra_gm_bytes=4*g*full),
        onchip_scaled_fp32_return=dict(extra_partial_gm_bytes=0,feasibility='UNRESOLVED',
            requirement='Prove scaled FP32 CO1 return, separate accumulator residency, scheduler lifetime and synchronization; not inferred from FP16 output support'),
        logical_reduction_additions=(g-1)*m*n,logical_group_scalings=g*m*n,
        scale_storage_hypotheses={str(width)+'_byte_parameter':dict(bytes=g*n*width,extra_vs_fp16=g*n*(width-2)) for width in (4,8)},
        scale_encoding='UNRESOLVED; 4/8-byte entries above are scenarios, not asserted supported encodings',
        launches='UNRESOLVED; group producer/reducer may be integrated or separate; no asserted launch count',
        latency='UNRESOLVED; byte counts do not establish bandwidth, overlap, accumulator latency or benefit',
        boundary='Extra GM bytes exclude unchanged final FP16 write, weight/X reads, padding, scale fetch/reload, cache and coherence traffic')
