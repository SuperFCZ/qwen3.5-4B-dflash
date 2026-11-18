"""Uninstrumented resident OM execution and fail-closed paired decision."""
from pathlib import Path
import struct
from gate_common import ABI,GEARS,LAYERS,MODES,SHAPE_KEY,ELIGIBLE,NO_GO,c,flow,host,resume,require,stats
from raw_protocol import verify_run
from reference import ordered_half


def direction(stage):
    return 'forward' if stage.endswith('forward') else 'reverse' if stage.endswith('reverse') else None


def schedule(stage):
    gear='16' if stage.startswith('c16-') else '64'
    order=MODES[::-1] if direction(stage)=='reverse' else MODES
    return [(gear,layer,mode,0) for layer in LAYERS for mode in order]


def full_schedule(timed):
    return [(gear,'full',mode,i) for i in range(3 if timed else 1) for gear in GEARS
            for mode in (MODES[::-1] if (i+int(gear=='64'))%2 else MODES)]


def expected_run(stage):
    return full_schedule(stage=='full-benchmark') if stage in ('full-smoke','full-benchmark') else schedule(stage)


def verify_execution(e,timed):
    want=dict(status='PASS',scope='single Gate frozen Native Golden checked in process',device=0,guards=True,readonly=True,
              repeat=True,cpu_fallback=False,golden_match=True,poison_verified=True,sync_completed=True,timed_log_stable=True,
              verification_calls=4 if timed else 2,row_permutation_checked=False)
    require(all(type(e.get(k))==type(v) and e[k]==v for k,v in want.items()),'Gate execution failed numerical/safety protocol')
    t=e['timing'];want=dict(protocol='continuous-v1' if timed else 'checked-v1',status='MEASURED' if timed else 'NOT_RUN',warmup=5 if timed else 0,
        repetitions=30 if timed else 0,correctness_before_calls=2,correctness_after_calls=int(timed),timed_tail_checked=timed,
        per_timed_call_readback=not timed,per_timed_call_poison=not timed)
    require(all(type(t.get(k))==type(v) and t[k]==v for k,v in want.items()) and t['prepare']['samples_ms']==[],'unbracketed or wrong timing protocol')
    require(all(type(e[k]) is int and e[k]>0 for k in ('workspace_bytes','model_weight_bytes')),'missing guarded OM allocations')
    samples=t['execute_sync']['samples_ms']
    if not timed:require(samples==[],'timing in Smoke')
    return dict(status='GATE_ALL_311296_BITS_AND_SAFETY_MATCH',workspace_bytes=e['workspace_bytes'],model_weight_bytes=e['model_weight_bytes'],
                timing=stats(samples,30) if timed else None,profiled=False)


def use_specs(src,gear,layer):
    if layer=='full':return src['full'][gear]['io']
    case=src['cases'][gear][layer]
    return dict(abi=ABI,inputs={'x':case['inputs']['x.bin']},golden={'y':case['golden']})


def verify_outputs(row,spec,timed):
    if row['layer']=='full':return verify_run(row,spec,0,timed)
    result=verify_execution(c.read(c.verify(row['execution'])),timed)
    require(set(row['outputs'])=={'y'} and len(row['outputs']['y'])== (4 if timed else 2),'missing Gate snapshots')
    expected=c.verify(spec['golden']['y']).read_bytes()
    require(len(expected)==622592 and all((u & 0x7c00)!=0x7c00 for u, in struct.iter_unpack('<H',expected)),'nonfinite or short frozen Native output')
    for item in row['outputs']['y']:
        require(c.verify(item).read_bytes()==expected,'Native bitwise/Repeat mismatch')
    return result


def mismatch_evidence(folder,spec):
    result=[]
    for name,ref in spec['golden'].items():
        for suffix in ('actual-0.bin','actual-1.bin','benchmark-last.bin','postcheck.bin'):
            path=folder/(name+'-'+suffix)
            if not path.is_file():continue
            actual=c.identity(path)
            if actual['bytes']==ref['bytes'] and actual['sha256']==ref['sha256']:continue
            a=path.read_bytes();b=c.verify(ref).read_bytes()
            row=dict(output=actual,golden=ref,first_byte=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b))))
            if name=='y' and len(a)==len(b)==622592:
                words=struct.iter_unpack('<H',a);expected=struct.iter_unpack('<H',b)
                count=max_ulp=nonfinite=poison_words=0;first=None
                for i,((x,),(y,)) in enumerate(zip(words,expected)):
                    if x==y:continue
                    count+=1;ulp=abs(ordered_half(x)-ordered_half(y)) if x&0x7c00!=0x7c00 and y&0x7c00!=0x7c00 else None
                    nonfinite+=int((x & 0x7c00)==0x7c00);poison_words+=int(x in (0x7f7f,0xffff))
                    if first is None:first=dict(row=i//19456,channel=i%19456,actual_bits=x,expected_bits=y,ulp=ulp)
                    if ulp is not None:max_ulp=max(max_ulp,ulp)
                row.update(bit_mismatches=count,first=first,max_finite_ulp=max_ulp if count>nonfinite else None,
                           nonfinite_actual=nonfinite,poison_pattern_words=poison_words)
            elif len(a)==len(b):
                first=row['first_byte'];row.update(mismatching_bytes=sum(x!=y for x,y in zip(a,b)),actual_byte=a[first],expected_byte=b[first])
            result.append(row)
    return result


def verify_maps(path,selected,src):
    records=resume.combined(selected,src['cann'],src['extras'],c.read(resume.POLICY))
    groups=host.dsos.maps_rows(Path(path).read_text());resume.check_maps(groups,records)
    return sorted(groups)


def one_run(folder,src,selected,model,key,timed,r,previous):
    gear,layer,mode,round_id=key;full=layer=='full';spec=use_specs(src,gear,layer)
    require(model['mode']==mode and model['key']==(gear if full else layer) and model['abi']==spec['abi'] and
            model['opp_root']==selected['opp']['install'] and model['private_kernel']==selected['kernel'],'model/OPP/ABI cross-wiring')
    folder.mkdir(parents=True);binary=selected['om' if full else 'runner']
    prior=c.read(c.verify(src['host_source']))['binding']['prior']
    env=flow.build_env(prior,selected);c.verify_opp(selected['opp'])
    require(c.dependencies(binary['runner'],folder/'ldd.log',env)==binary['deps'],'OM runner ldd drift')
    for side in ('inputs','outputs'):
        (folder/(side+'.tsv')).write_text(''.join(t['name']+' '+t['dtype']+' '+' '.join(map(str,t['shape']))+'\n' for t in spec['abi'][side]))
    for side in ('inputs','golden'):
        data=folder/side;data.mkdir()
        for name,item in spec[side].items():(data/(name+'.bin')).symlink_to(c.verify(item))
    row=dict(gear=gear,layer=layer,mode=mode,round=round_id,folder=str(folder),profiled=False,device=0,
             runner=binary['runner'],om=model['om'],opp_root=model['opp_root'],opp_kernel=selected['kernel']['elf'],
             selected_kernels=model['atc_kernels'],inputs=spec['inputs'],golden=spec['golden'],atc_gate_plan=model['plans'][SHAPE_KEY]['contract'],
             launch_authorized=False,maps={})
    r['runs'].append(row)
    def admit():
        row['maps']['pre']=verify_maps(folder/'pre-maps.txt',selected,src);row['launch_authorized']=True
    try:
        flow.invoke_om([c.verify(binary['runner']),0,c.verify(model['om']),folder,folder/'inputs',folder/'golden','benchmark' if timed else 'smoke'],
                       folder/'runner.log',env,admit)
        row['maps']['post']=verify_maps(folder/'post-maps.txt',selected,src)
        row['execution']=c.identity(folder/'execution.json')
        suffixes=['actual-0.bin','actual-1.bin']+(['benchmark-last.bin','postcheck.bin'] if timed else [])
        row['outputs']={n:[c.identity(folder/(n+'-'+s)) for s in suffixes] for n in spec['golden']}
        row['validation']=verify_outputs(row,spec,timed)
        if timed:
            require(previous is not None and all(row['validation'][k]==previous['validation'][k] for k in ('workspace_bytes','model_weight_bytes')),
                    'OM memory drift since Smoke')
        c.verify_opp(selected['opp']);c.verify_tree(spec)
    except Exception as primary:
        row['error']=repr(primary)
        try:row['differences']=mismatch_evidence(folder,spec)
        except (OSError,ValueError,IndexError) as diagnostic:row['difference_read_error']=repr(diagnostic)
        raise
    finally:
        row['artifacts']=[c.identity(p) for p in sorted(folder.iterdir()) if p.is_file()]
        if row['launch_authorized']:r['device_execution']='LAUNCH_AUTHORIZED; SEE_RUN_REPORT'


def key(row):return row['gear'],row['layer'],row['mode'],row['round']


def inventory(rows,stage):
    require([key(row) for row in rows]==expected_run(stage),'incomplete, duplicated, or reordered runs: '+stage)


def pair_memory(rows):
    groups={}
    for row in rows:
        k=(row['gear'],row['layer'],row['round'])
        memory=tuple(row['validation'][key] for key in ('workspace_bytes','model_weight_bytes'))
        require(groups.setdefault(k,memory)==memory,'Off/Raw actual model Workspace/weight allocation differs: '+str(k))


def replay_runs(report,src,built,exported):
    stage=report['phase'];timed=not stage.endswith('smoke');inventory(report['runs'],stage)
    # Reverify shared records once per archived phase; compare every historical VMA
    # separately. Live pre/post admission above always uses the original verifier.
    manifests={}
    for row in report['runs']:
        gear,layer,mode,_=key(row);selected=built['variants'][mode];spec=use_specs(src,gear,layer);full=layer=='full'
        model=exported['models'][mode][gear if full else layer];folder=Path(row['folder'])
        require(row['profiled'] is False and row['device']==0 and row['launch_authorized'] is True and
                row['om']==model['om'] and row['opp_root']==selected['opp']['install'] and row['runner']==selected['om' if full else 'runner']['runner'] and
                row['opp_kernel']==selected['kernel']['elf'] and row['selected_kernels']==model['atc_kernels'] and
                row['atc_gate_plan']==model['plans'][SHAPE_KEY]['contract'] and row['atc_gate_plan']['block_dim']==8 and
                row['inputs']==spec['inputs'] and row['golden']==spec['golden'],'cross-wired run identity')
        invocation=c.read(folder/'runner.command.json')
        require(invocation['argv'][:-1]==[row['runner']['path'],'0',row['om']['path'],str(folder),str(folder/'inputs'),str(folder/'golden'),'benchmark' if timed else 'smoke'] and
                invocation['argv'][-1].isdigit() and invocation['env']==selected['env']['values'],'runner invocation/child environment drift')
        for side in ('inputs','golden'):
            for name,item in spec[side].items():require(c.identity(folder/side/(name+'.bin'))==item,'case alias drift')
        require(row['validation']==verify_outputs(row,spec,timed),'numeric/guard/timing evidence drift')
        if mode not in manifests:
            records=resume.combined(selected,src['cann'],src['extras'],c.read(resume.POLICY))
            for item in src['extras'].values():host.dsos.verify_record(item)
            manifests[mode]=records
        for side in ('pre','post'):
            groups=host.dsos.maps_rows((folder/(side+'-maps.txt')).read_text())
            require(groups and row['maps'][side]==sorted(groups),'mapped library inventory drift')
            for alias,segs in groups.items():
                require(alias in manifests[mode],'unexpected mapped DSO: '+alias)
                backing=manifests[mode][alias]['backing']
                require(all(seg['device']==backing['device'] and seg['inode']==backing['inode'] for seg in segs),'mapped backing object drift')
    pair_memory(report['runs'])


def perform(folder,src,built,models,stage,r,smoke=None):
    timed=not stage.endswith('smoke');r['runs']=[];prior={key(x)[:3]:x for x in smoke['runs']} if smoke else {}
    for run in expected_run(stage):
        gear,layer,mode,round_id=run;r['active_case']=list(run)
        dest=folder/str(round_id)/('c'+gear)/layer/mode
        one_run(dest,src,built['variants'][mode],models[mode][gear if layer=='full' else layer],run,timed,r,prior.get(run[:3]))
        if len(r['runs'])%2==0:pair_memory(r['runs'][-2:])
        last=r['runs'][-1]['validation'];timing=last['timing']
        measured=f" median_ms={timing['median_ms']:.6f} p95_ms={timing['p95_ms']:.6f}" if timing else ''
        print(f"CASE_MATCH C{gear} {layer} {mode} round={round_id} model_workspace={last['workspace_bytes']}"+measured,flush=True)
    inventory(r['runs'],stage);r.pop('active_case');r['device_execution']='COMPLETED';r['benchmark']='MEASURED' if timed else 'NOT_RUN'


def compare_gear(gear,forward,reverse):
    orders={};good=True
    for direction_name,report in [('forward',forward),('reverse',reverse)]:
        phase='c'+gear+'-'+direction_name;require(report['phase']==phase,'wrong gear/order');inventory(report['runs'],phase)
        pairs={};lookup={key(x)[:3]:x for x in report['runs']};totals={m:0. for m in MODES}
        for layer in LAYERS:
            values={}
            for mode in MODES:
                row=lookup[gear,layer,mode]
                require(row['profiled'] is False and row['atc_gate_plan']['block_dim']==8 and row['validation']['status']=='GATE_ALL_311296_BITS_AND_SAFETY_MATCH','not an admitted Gate8 timing')
                series=stats(row['validation']['timing']['samples_ms'],30);totals[mode]+=series['median_ms']
                values[mode]=dict(statistics=series,actual_om_workspace=row['validation']['workspace_bytes'],gate_plan=row['atc_gate_plan'],
                    kernel=row['selected_kernels'],opp_elf=row['opp_kernel'],om=row['om'],runner=row['runner'],golden=row['golden'],
                    correctness=row['validation']['status'])
            a,b=[values[m]['statistics'] for m in MODES];ratio=a['median_ms']/b['median_ms']
            regression=b['p95_ms']>a['p95_ms']
            require(values['off']['gate_plan']['cube_sha256']==values['raw']['gate_plan']['cube_sha256'] and
                    values['off']['gate_plan']['workspace']==values['raw']['gate_plan']['workspace'] and
                    values['off']['golden']==values['raw']['golden'],'incomparable pairs')
            pairs[layer]=dict(variants=values,off_over_raw=ratio,p95_regression=regression,median_improved=ratio>1)
            good &= ratio>1 and not regression
        aggregate=totals['off']/totals['raw'];good &= aggregate>=1.03
        orders[direction_name]=dict(layers=pairs,sum_isolated_medians_ms=totals,ratio_of_sums=aggregate)
    return dict(gear=gear,orders=orders,decision=ELIGIBLE if good else NO_GO,
                scope='INDEPENDENT_GATE_PROJECTIONS; SUMS_ARE_NOT_DRAFT_LATENCY',historical_samples_used=False)


def full_stats(rows):
    inventory(rows,'full-benchmark');result={}
    for gear in GEARS:
        per={mode:[row['validation']['timing']['samples_ms'] for row in rows if (row['gear'],row['mode'])==(gear,mode)] for mode in MODES}
        result[gear]=dict(variants={mode:stats(sum(per[mode],[]),90) for mode in MODES},
                          round_off_over_raw=[stats(a,30)['median_ms']/stats(b,30)['median_ms'] for a,b in zip(per['off'],per['raw'])])
        val=result[gear]['variants'];result[gear]['off_over_raw']=val['off']['median_ms']/val['raw']['median_ms']
    return dict(gears=result,historical_samples_used=False,scope='FULL_DRAFT_FRESH_EXECUTE_SYNC',production='NOT_APPROVED')


def comparison_text(result):
    lines=['# C'+result['gear']+' Gate Raw 8-block',result['decision'],'',
           '|Order|Layer|Off median / P95 (ms)|Raw median / P95 (ms)|Off/Raw|P95 regression|',
           '|---|---|---:|---:|---:|---|']
    for name,order in result['orders'].items():
        for layer,pair in order['layers'].items():
            a,b=[pair['variants'][v]['statistics'] for v in MODES]
            lines.append(f"|{name}|{layer}|{a['median_ms']:.6f} / {a['p95_ms']:.6f}|{b['median_ms']:.6f} / {b['p95_ms']:.6f}|{pair['off_over_raw']:.6f}|{pair['p95_regression']}|")
    lines+=['']+[f"{k}: ratio of sums of five isolated medians = {o['ratio_of_sums']:.6f}" for k,o in result['orders'].items()]
    lines+=['Not five-layer Draft latency. No historical samples, profiling, or production decision.']
    return '\n'.join(lines)+'\n'
