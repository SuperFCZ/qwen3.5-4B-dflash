"""Untimed ACLNN checks in both private packages; full OM still closes the GE path."""
import raw_common as c
from a3_launch import launch_evidence
from gate_prefetch_contract import match_plans


def validate(row,case,cfg,device):
    e=c.read(c.verify(row['execution']));m,k,n=case['shape']
    # The accepted untimed predicate is Gate-specific; other real shapes keep
    # exactly the same safety checks and their actual shape/selected K tile.
    from a3_launch import selected_k_tile
    want=dict(status='PASS',runtime='AscendCL ACLNN',op='DFlashGroupQuantLinear',cpu_fallback=False,input_readonly=True,guards_intact=True,repetitions=2,device_id=device,m=m,k=k,n=n,tile_n=64,tile_k=selected_k_tile(case['shape'],256))
    if any(e.get(key)!=v for key,v in want.items()):raise ValueError('untimed projection safety/shape mismatch')
    t=e['timing']
    if t['warmup']!=0 or t['repetitions']!=0 or t['status']!='NOT_RUN' or t['execute_sync']['samples_ms']:raise ValueError('projection gate must be untimed')
    expected=c.verify(case['golden']).read_bytes()
    if len(expected)!=m*n*2 or len(row['outputs'])!=2 or any(c.verify(x).read_bytes()!=expected for x in row['outputs']):raise ValueError('projection frozen Native bits differ')
    plan=launch_evidence(c.verify(row['log']),case['shape'],cfg['core_limit'],e['workspace_bytes'],cfg['dequant_mode'],cfg['pipeline_mode'],cfg['launch_version'],cfg['kv_m80_mode'],cfg['scale_mode'],cfg['k_tile'],cfg['k_tile_policy'],cfg['gate_prefetch_mode'])
    return dict(status='PROJECTION_BITS_AND_SAFETY_MATCH',plan=plan)


def compare(a,b):
    if a['shape']!=b['shape']:raise ValueError('projection shape changed')
    x=a['validation']['plan'];y=b['validation']['plan']
    if a['shape']==[16,2560,19456]:match_plans(x,y)
    else:
        if x['gate_prefetch']['enabled'] or y['gate_prefetch']['enabled']:raise ValueError('non-Gate Raw enabled')
        if {k:v for k,v in x.items() if k!='gate_prefetch'}!={k:v for k,v in y.items() if k!='gate_prefetch'}:raise ValueError('non-Gate resource drift')
