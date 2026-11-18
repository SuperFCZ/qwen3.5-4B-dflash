"""One-FC admission and replay contracts; no ACL imports."""
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
HOST = HERE.parent / 'draft_a_core_scheduler_diag'
sys.path.insert(0, str(HOST))
import run_scheduler as host
from scheduler_common import c
from platform_contracts import unique_object
from reference import compare, ordered_half

SIZES = {'x.bin': 16*12800*2, 'w_nz.bin': 12800*2560, 's_gn.bin': 100*2560*2}
DIMS = SimpleNamespace(m=16, n=2560)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def lines(path):
    return [json.loads(line, object_pairs_hook=unique_object) for line in path.read_text().splitlines()]


def sources():
    files = [path for path in HERE.rglob('*') if path.is_file() and '__pycache__' not in path.parts]
    files += [c.QUANT/'test/runner_common.h', c.QUANT/'test/reference.py']
    return [c.identity(path) for path in sorted(files)] + host.sources() + host.borrowed_sources()


def fixed_query(platform, schedule, variant):
    require([platform[key] for key in ('GetCoreNum','GetCoreNumAic','GetCoreNumAiv','blockDim','coreLimit')]
            == [7,8,8,{'A':7,'B':8}[variant],0], 'different Runtime core policy')
    require(schedule['variant']==variant and schedule['slice_num']==schedule['calc_tsch_block_dim']==8 and
            schedule['selected_block_dim']==platform['blockDim'] and platform['workspace']==2097152,
            'different schedule or Workspace')


def live_callback(folder, query, variant):
    for filename,key in [('platform.jsonl','platform'),('scheduling.jsonl','schedule')]:
        require(lines(folder/filename)==[query[key]], 'current GetWorkspaceSize callback differs: '+filename)
    fixed_query(query['platform'], query['schedule'], variant)


def bind_history(root):
    phases=('source','build','atc','aclnn','compare')
    refs={phase:c.identity(root/phase/'report.json') for phase in phases}
    reports={phase:host.stage(refs[phase]['path'],phase) for phase in phases}
    for phase in phases[1:]:
        keys={'build':['source'],'atc':['source','build'],'aclnn':['source','build'],
              'compare':['source','build','atc','aclnn']}[phase]
        for key in keys:
            require(reports[phase][key+'_report']==refs[key], 'mixed historical reports: '+phase)
    source=reports['source']['binding'];case=source['case'];prior=source['prior']
    require(case['shape']==[16,12800,2560] and case['name']=='fc' and
            [case]==[row for row in prior['current']['evidence']['projections']['16'] if row['name']=='fc'],
            'not the frozen C16 FC case')
    require({key:c.verify(value).stat().st_size for key,value in case['inputs'].items()}==SIZES and
            c.verify(case['golden']).stat().st_size==81920, 'wrong frozen tensor size')
    variants={}; signatures=[]
    for variant in ('A','B'):
        build=reports['build']['variants'][variant]; package=build['package'];opp=package['opp']
        require(c.read(c.verify(build['package_report']))==package and package['variant']==variant and
                package['opp_comparison']['all_non_host_files_equal'] is True and
                package['opp_comparison']['all_kernel_bytes_equal'] is True, 'unverified package')
        c.verify_opp(opp);host.dsos.verify_manifest(c.read(c.verify(build['dso_manifest'])))
        runtime=reports['aclnn']['variants'][variant];query=runtime['query']
        require(runtime['opp']==opp and runtime['dso_manifest']==build['dso_manifest'] and
                host.read_query(c.verify(query['platform_file']),c.verify(query['scheduling_file']),opp,'ACLNN',variant)==query,
                'historical query misbound')
        fixed_query(query['platform'],query['schedule'],variant)
        api=c.identity(Path(opp['install'])/'vendors/customize/op_api/lib/libcust_opapi.so')
        signatures.append((api['sha256'],package['opp_comparison']['kernels']))
        variants[variant]=dict(opp=opp,query=query,dependencies=build['dependencies'],historical_runner=build['runner'],
                               manifest=build['dso_manifest'],api=api,environment=build['runtime_environment'])
    require(signatures[0]==signatures[1] and Path(variants['A']['opp']['install']).resolve()!=Path(variants['B']['opp']['install']).resolve(),
            'need separate packages with identical API/Kernel binaries')
    require(host.compare_pair(variants['A']['query'],variants['B']['query'])==reports['compare']['routes']['ACLNN'],
            'comparison does not bind both Runtime queries')
    return dict(host_reports=refs,case=case,variants=variants,cann_root=prior['current']['cann_root'],
                tools=prior['current']['evidence']['tools'])


def difference(expected,actual):
    result=compare(expected,actual,DIMS)
    mismatch=[];nonfinite=[]
    for index,((left,), (right,)) in enumerate(zip(struct.iter_unpack('<H',expected),struct.iter_unpack('<H',actual))):
        def kind(bits):
            if bits & 0x7c00 != 0x7c00:return 'finite'
            return 'nan' if bits & 0x03ff else ('-inf' if bits & 0x8000 else '+inf')
        tags=(kind(left),kind(right));row,channel=divmod(index,DIMS.n)
        if tags!=('finite','finite'):nonfinite.append([row,channel,*tags])
        if left!=right:
            mismatch.append(dict(row=row,channel=channel,expected_bits=left,actual_bits=right,
                                 ulp=abs(ordered_half(left)-ordered_half(right)) if tags==('finite','finite') else None))
    return dict(native_comparator=result,mismatches=mismatch,nonfinite=nonfinite)


def safety(folder,repeat):
    expected=['aclInit','aclrtSetDevice','aclrtGetDevice','context','stream','GetWorkspaceSize',
              'precheck','execute_attempt','aclnnDFlashGroupQuantLinear','synchronize','readonly','complete']
    events=lines(folder/'events.jsonl')
    require([row['event'] for row in events]==expected and all(row['returncode']==0 for row in events),
            'missing or failed runtime operation')
    require(events[2]['device']==0 and events[3]['present'] is True and events[4]['present'] is True and
            events[5]['workspace']==2097152 and events[6]['poison_byte']==(0x7f,0xff)[repeat] and
            all(events[10][key]==1 for key in ('x','w','s')), 'runtime evidence failed')
    for stage in ('pre','post'):
        rows=lines(folder/(stage+'-guards.jsonl'))
        require(rows==[dict(buffer=name,side=side,bytes=512,expected=165,changes=[])
                       for name in ('x','w','s','y','workspace') for side in ('prefix','suffix')], 'guard evidence failed: '+stage)
    return dict(device_id=0,context_stable=True,stream_created=True,workspace=events[5]['workspace'],execute_calls=1,
                stream_synchronized=True,input_readonly=True,pre_poison_verified=True,guards_intact=True,cpu_fallback=False)
