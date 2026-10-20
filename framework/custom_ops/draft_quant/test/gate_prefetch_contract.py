"""Strict single-variable Gate raw-prefetch resource and launch contract."""
SHAPE=(16,2560,19456)


def check_plan(plan,mode):
    if mode not in ('off','raw'):raise ValueError('unknown Gate prefetch mode')
    enabled=mode=='raw'
    wanted=dict(m=16,k=2560,n=19456,tile_m=16,tile_n=64,tile_k=256,weight_reuse_rows=16,
                raw_banks=2 if enabled else 1,selected_pipeline='raw-prefetch-v1' if enabled else 'serial-v1',
                user_ub_bytes=76544 if enabled else 59904,matmul_ub_bytes=185600 if enabled else 202240,
                gate_prefetch=dict(version=1,mode=mode,enabled=enabled))
    if any(plan.get(k)!=v for k,v in wanted.items()):raise ValueError('Gate prefetch shape/UB/schedule/evidence differs')
    cube=plan.get('cube',{})
    if any(cube.get(k)!=v for k,v in dict(a_source='GM',base_m=16,base_n=64,base_k=256,weight_dequantizations_per_n_tile=10).items()):raise ValueError('Gate Cube/full-K contract differs')
    return dict(raw_codes_bytes=32768 if enabled else 16384,raw_scale_bytes=512 if enabled else 256,
                single_w16_bytes=32768,broadcast_bytes=2304,sdk_reserved_bytes=8192,
                matmul_ub_bytes=wanted['matmul_ub_bytes'],user_ub_bytes=wanted['user_ub_bytes'],
                raw_banks=wanted['raw_banks'],reserved_event_types=['MTE2_S','MTE2_V'] if enabled else [],
                weight_handoff='V_S wait' if enabled else 'PIPE_ALL',tile_consumer_fence='PIPE_ALL after every Iterate',
                scope='Host plan plus source dependency contract, not measured device overlap')


def match_plans(before,after):
    check_plan(before,'off');check_plan(after,'raw')
    allowed={'raw_banks','selected_pipeline','user_ub_bytes','matmul_ub_bytes','gate_prefetch'}
    if {k:v for k,v in before.items() if k not in allowed}!={k:v for k,v in after.items() if k not in allowed}:
        raise ValueError('non-prefetch launch/Cube/tiling/core/workspace changed')
