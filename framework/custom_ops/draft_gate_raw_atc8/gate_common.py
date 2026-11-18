"""Isolated Gate-only experiment; imports perform no device operations."""
import hashlib
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
CUSTOM=HERE.parent
sys.path.insert(0,str(CUSTOM/'draft_a_core_scheduler_om_resume'))
import resume
from resume import c,flow,host,require
from opp_preflight import check_root
from atc_evidence import payloads,option
from flow_build import tree,preset_options
from raw_protocol import stats

__all__=['c','flow','host','require','resume','check_root','tree','preset_options','payloads','option','stats']

GATE='DFlashGroupQuantLinear'
STEM='d_flash_group_quant_linear'
HEADER=STEM+'_build_config.h'
SHAPE=[16,2560,19456]
SHAPE_KEY='16x2560x19456'
LAYERS=tuple(f'layer-{i}-gate_up' for i in range(5))
GEARS=('16','64')
MODES=('off','raw')
ABI=dict(inputs=[dict(name='x',dtype='float16',shape=[16,2560])],outputs=[dict(name='y',dtype='float16',shape=[16,19456])])
COUNT_KEYS=(GATE,'DFlashDraftLmHeadTop1','DFlashDraftLmHeadTop1Audit')
OFF_HEADER='22d5541d1739f799588fbc076c866363d131d68a95e6e148a7cc0d1c02d692a1'
RAW_HEADER='26017103ea631e701088f9d7e44e8645737228b4ba869b062c5726b47bb543e6'
KERNEL_SOURCE='a4861bb5f7e38aceced9e4d89fc6c2519cb4a80e0914d7dd418c50389e8f1a4f'
OFF_ELF='831a9ce01371ef0a70962e6576489f8d16ec07daa253b99f1b2e5e627f71bca3'
API='b55f51b0eab58f780fbf287bc35162cd9ba718c3c3133dd694068574f81d5505'
OFF_OM={'16':'8e042a09444a9fba6b9a9aa3ddf11ce325698bd51861bc98f3ce7bdf38e20aaa',
        '64':'3f2917b8bdaf83c06906a5359e1203362c1f65d3cf1d7f1a3e35a432152502c7'}
OFF_PARENT='ee2d5b1e3c35105c56aeaeec79c79d0bef64074ee774d3715618e7c2390b751a'
SDK_SCHEMA=CUSTOM/'draft_w8a8_output_dequant/stage4v_external_co1_host/test/fixtures/tcube-schema.json'
SCHEMA_SHA='f1688b61034f91c7518f65dcf73382ee9b007ab2c461fc8244cb6b883bd1d944'
OFF_CONFIG=dict(k_tile=256,scale_mode='broadcast',pipeline_mode='prefetch',dequant_mode='batched',gate_copy_mode='loop',
                kv_m80_mode='a-ub',gate_prefetch_mode='off',core_limit=0,header_sha256=OFF_HEADER)

STAGES=('source','build','export','c16-smoke','c16-forward','c16-reverse','c64-smoke','c64-forward','c64-reverse','compare',
        'full-export','full-smoke','full-benchmark')
STATE={phase:'GATE8_'+phase.upper().replace('-','_')+'_PASS' for phase in STAGES}
ELIGIBLE='ELIGIBLE_FOR_FULL_DRAFT_EXPERIMENT'
NO_GO='NO_GO_FOR_FULL_DRAFT'


def sha(data):return hashlib.sha256(data).hexdigest()


def sources():
    own=[c.identity(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    extra=[SDK_SCHEMA,c.QUANT/'test/gate_prefetch_contract.py',c.QUANT/'test/opp_preflight.py',c.QUANT/'test/a3_launch.py',
           c.AC/'test/opp.py',c.AC/'test/c_profile_graph.py',c.HERE/'raw_evidence.py']
    return own+[c.identity(p) for p in extra]+resume.sources()+flow.sources()


def one(items,message):
    items=list(items)
    require(len(items)==1,message+' (count='+str(len(items))+')')
    return items[0]


def check_config(cfg,mode):
    want=dict(OFF_CONFIG,gate_prefetch_mode=mode,header_sha256={'off':OFF_HEADER,'raw':RAW_HEADER}[mode])
    require(all(type(cfg.get(k)) is type(v) and cfg.get(k)==v for k,v in want.items()),'configuration differs from '+mode+' contract')
    require(cfg['source_sha256']['op_kernel/'+STEM+'.cpp']==KERNEL_SOURCE,'unreviewed math source')


def derive_header(data):
    old=b'#define DFLASH_GROUP_QUANT_GATE_PREFETCH_MODE 0U';new=old[:-2]+b'1U'
    require(sha(data)==OFF_HEADER and data.count(old)==1,'unbound generated Off header')
    raw=data.replace(old,new);require(sha(raw)==RAW_HEADER,'Raw header differs beyond approved bit')
    return raw


def gate_kernel(opp):
    row=one((r for r in opp['preflight']['kernels'] if r['op']==GATE),'expected a unique Gate ELF')
    binary=c.identity(row['binary']);meta=c.identity(row['json'])
    require(binary['sha256']==row['sha256'] and binary['bytes']==row['bytes'] and
            meta['sha256']==row['json_sha256'] and meta['bytes']==row['json_bytes'],'installed Gate object/metadata drift')
    info=c.read(meta['path'])
    require(info.get('binFileName')==Path(binary['path']).stem and info.get('binFileSuffix')=='.o' and info.get('sha256')==binary['sha256'],'SDK kernel metadata/ELF mismatch')
    return dict(elf=binary,metadata=meta,bin_name=info['binFileName'])


def config_graph(graph,cases,full=False):
    count=26 if full else 1;control=graph['metadata']['draft_ac'];audit=graph['runtime_input_abi']['draft_ac']
    wanted=dict(zip(COUNT_KEYS,(count,0,0)))
    require(control['mode']=='a' and control['a_count']==count and control['c_count']==control.get('audit_count',0)==0 and
            graph['runtime_input_abi']['status']=='PASS' and audit['status']=='PASS' and audit['counts']==wanted,'wrong A-only graph')
    observed={r['ge_op_type']:r['ge_node_occurrences'] for r in graph['custom_op_audit']}
    require(all(observed.get(k,0)==v for k,v in wanted.items()) and observed.get('WeightQuantBatchMatmulV2',0)==0,'wrong graph node inventory')
    if not full:require(graph['role']=='diagnostic' and control['scope']=='single-op-A' and graph['metadata']['tensor_abi']==ABI and graph['input_names']==['x'] and graph['output_names']==['y'],'single Gate ABI differs')
    nodes=[n for n in audit['nodes'] if n['ge_type']==GATE];matches={}
    require(len(nodes)==count and len({n['name'] for n in nodes})==count,'immutable node count differs')
    for layer,case in cases.items():
        inputs=case['inputs']
        node=one((n for n in nodes if (n['nz_sha256'],n['scale_gn_sha256'])==(inputs['w_nz.bin']['sha256'],inputs['s_gn.bin']['sha256'])),
                 'unproven NZ/Scale -> '+layer)
        require(case['shape']==SHAPE and (node['k'],node['n'],node['nz_bytes'])==(2560,19456,49807360) and node['roundtrip']=='BIT_EXACT' and
                node['scale_shape']==[20,19456] and node['storage_shape']==[80,1216,16,32],'Gate immutable layout differs')
        matches[layer]=node
    require(len({node['name'] for node in matches.values()})==len(matches),'aliased Gate nodes')
    return matches


def case_contract(cases):
    require(set(cases)==set(GEARS),'need two independent gears')
    for gear,rows in cases.items():
        require(set(rows)==set(LAYERS),'missing or extra layer: '+gear)
        for layer,row in rows.items():
            require(row['name']==layer and row['shape']==SHAPE and row['golden']['bytes']==622592 and
                    {k:v['bytes'] for k,v in row['inputs'].items()}=={'x.bin':81920,'w_nz.bin':49807360,'s_gn.bin':778240},'Gate tensor contract differs')
        require(len({row['golden']['sha256'] for row in rows.values()})==5,'duplicate Native Golden across layers')
        for key in ('w_nz.bin','s_gn.bin','x.bin'):
            require(len({row['inputs'][key]['sha256'] for row in rows.values()})==5,'duplicate '+key+' across layers')
    for layer in LAYERS:
        left,right=[cases[g][layer] for g in GEARS]
        for key in ('w_nz.bin','s_gn.bin'):require(left['inputs'][key]['sha256']==right['inputs'][key]['sha256'],'gear constant drift')
        require(left['inputs']['x.bin']['sha256']!=right['inputs']['x.bin']['sha256'] and left['golden']['sha256']!=right['golden']['sha256'],'gear capture/Golden substitution')


def command(argv,log,env,cwd=None):
    import subprocess
    args=list(map(str,argv));c.write(log.with_suffix('.command.json'),dict(argv=args,cwd=str(cwd or Path.cwd()),environment=env))
    with log.open('xb') as f:rc=subprocess.run(args,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    if rc:raise RuntimeError(f'returncode {rc}: {log}')


def model_inventory(models,full=False):
    keys=GEARS if full else LAYERS
    require(set(models)==set(MODES) and all(set(models[m])==set(keys) for m in MODES),'incomplete model inventory')
