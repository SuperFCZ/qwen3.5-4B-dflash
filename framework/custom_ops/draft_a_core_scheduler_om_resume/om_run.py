from run_flow import OUTPUTS,build_env,c,invoke_om,require,stats,verify_run


def full(root,prior,built,exported,r,smoke=None,*,mapped):
    r['runs']=[];timed=smoke is not None;r['attribution']=exported['attribution']
    for iteration in range(3 if timed else 1):
        for gear in ('16','64'):
            for variant in ('AB' if (iteration+(gear=='64'))%2==0 else 'BA'):
                selected=built['variants'][variant];key=variant+'/'+gear
                spec=prior['current']['evidence']['H']['gears'][gear];folder=root/str(iteration)/key;folder.mkdir(parents=True)
                for side in ('inputs','outputs'):
                    (folder/(side+'.tsv')).write_text(''.join(x['name']+' '+x['dtype']+' '+' '.join(map(str,x['shape']))+'\n' for x in spec['abi'][side]))
                for side in ('inputs','golden'):
                    dest=folder/side;dest.mkdir()
                    for name,item in spec[side].items():(dest/(name+'.bin')).symlink_to(c.verify(item))
                model=exported['models'][key]['om'];binary=selected['om'];env=build_env(prior,selected)
                require(c.dependencies(binary['runner'],folder/'ldd.log',env)==binary['deps'],'OM loader drift')
                row=dict(variant=variant,gear=gear,round=iteration,om=model,runner=binary['runner'],profiled=False);r['runs'].append(row)
                def admit_maps():mapped(folder/'pre-maps.txt',selected,prior['current']['cann_root'],om=True)
                try:invoke_om([c.verify(binary['runner']),0,c.verify(model),folder,folder/'inputs',folder/'golden','benchmark' if timed else 'smoke'],folder/'runner.log',env,admit_maps)
                finally:row['files']=[c.identity(p) for p in sorted(folder.iterdir()) if p.is_file()]
                for when in ('pre','post'):mapped(folder/(when+'-maps.txt'),selected,prior['current']['cann_root'],om=True)
                row['execution']=c.identity(folder/'execution.json')
                suffixes=['actual-0.bin','actual-1.bin']+(['benchmark-last.bin','postcheck.bin'] if timed else [])
                row['outputs']={name:[c.identity(folder/(name+'-'+suffix)) for suffix in suffixes] for name in OUTPUTS}
                row['validation']=verify_run(row,spec,0,timed)
                if timed:
                    old=next(x for x in smoke['runs'] if x['variant']==variant and x['gear']==gear)
                    require(all(old['validation'][k]==row['validation'][k] for k in ('workspace_bytes','model_weight_bytes')),'OM allocation drift')
    if timed:
        r['statistics']={}
        for gear in ('16','64'):
            data={v:[x['validation']['timing']['samples_ms'] for x in r['runs'] if x['gear']==gear and x['variant']==v] for v in 'AB'}
            result={v:stats(sum(data[v],[]),90) for v in 'AB'}
            result['round_A_over_B']=[stats(a,30)['median_ms']/stats(b,30)['median_ms'] for a,b in zip(data['A'],data['B'])]
            result['A_over_B']=result['A']['median_ms']/result['B']['median_ms'];r['statistics'][gear]=result
