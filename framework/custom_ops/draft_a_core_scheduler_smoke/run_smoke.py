"""Stage-gated untimed single-FC device smoke. Only the smoke subcommand executes."""
import argparse
import json
import socket
import subprocess
import traceback
from pathlib import Path
from fc_contract import HERE, bind_history, c, difference, host, live_callback, require, safety, sources

STATES = dict(source='FC_SMOKE_SOURCE_BOUND',build='FC_SMOKE_RUNNERS_BUILT',
              smoke='FC_AB_FOUR_EXECUTIONS_MATCH',compare='FC_SCHEDULER_SINGLE_SHAPE_NPU_MATCH')
ORDER = tuple((variant,repeat) for variant in ('A','B') for repeat in range(2))


def predecessor(path,phase):
    report=c.read(path);c.verify_tree(report)
    require(report.get('status')==STATES[phase] and report['phase']==phase and not report.get('error') and
            not report.get('integrity_error') and report['sources']==sources(), 'invalid predecessor: '+phase)
    if phase=='source':require(c.read(c.verify(report['source_manifest']))==report['sources'], 'source inventory mismatch')
    return report


def environment(binding,selected):
    host.envs.verify_binding(selected['environment']);c.verify_opp(selected['opp'])
    env,identity=host.envs.make_environment(binding['cann_root'],selected['opp']['install'],tools=binding['tools'])
    require(identity['values']==selected['environment']['values'], 'changed environment since Host acceptance')
    host.dsos.verify_manifest(c.read(c.verify(selected['manifest'])))
    return env


def mappings(folder,name,selected,cann):
    records=host.loaded_libraries(folder/(name+'-maps.txt'),selected['opp'],Path(cann),c.read(c.verify(selected['manifest'])))
    private=[row['file'] for row in records if row['category']=='PRIVATE_DIAGNOSTIC_OPP']
    require(selected['api'] in private and selected['query']['host_dso'] in private, 'private API or selected Host DSO absent')
    return records


def invoke(runner,folder,repeat,env,workspace,admit):
    parent,child=socket.socketpair()
    with parent,child,(folder/'runner.log').open('wb') as log:
        argv=[str(runner),str(folder),str(repeat),str(child.fileno()),str(workspace)]
        c.write(folder/'command.json',dict(argv=argv,env=env))
        with subprocess.Popen(argv,pass_fds=(child.fileno(),),env=env,stdout=log,stderr=subprocess.STDOUT) as proc:
            child.close(); parent.settimeout(180)
            try:
                require(parent.recv(1)==b'?', 'Runner exited before query admission; see runner.log')
                admit();parent.sendall(b'G');parent.close()
                require(proc.wait(timeout=180)==0, 'Runner failed; see runner.log and events.jsonl')
            finally:
                parent.close()
                if proc.poll() is None:
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()


def compare_bytes(folder,label,expected,actual):
    result=difference(expected,actual);path=folder/(label+'.json');c.write(path,result)
    return c.identity(path),result['native_comparator']['finite'] and result['native_comparator']['bitwise_equal']


def run_device(folder,repeat,binding,selected,binary,row,prior_outputs):
    env=environment(binding,selected);case=binding['case'];query=selected['query'];variant=query['schedule']['variant']
    for name,item in case['inputs'].items():(folder/name).symlink_to(c.verify(item))
    for name in ('platform.jsonl','scheduling.jsonl'):(folder/name).touch(exist_ok=False)
    env.update(DFLASH_PLATFORM_DIAG_PATH=str(folder/'platform.jsonl'),DFLASH_PLATFORM_DIAG_ROUTE='ACLNN',
               DFLASH_SCHEDULER_EVIDENCE_PATH=str(folder/'scheduling.jsonl'))
    host.verify_environment(env,selected['environment'])
    require(c.dependencies(binary['runner'],folder/'ldd.log',env)==binary['dependencies']==selected['dependencies'], 'loader drift')
    def admit():
        live_callback(folder,query,variant)
        row['pre_loaded']=mappings(folder,'pre',selected,binding['cann_root'])
        row['launch_authorized']=True
    try:
        invoke(c.verify(binary['runner']),folder,repeat,env,query['platform']['workspace'],admit)
    finally:
        actual=folder/'actual.bin'
        if actual.is_file() and actual.stat().st_size==81920:
            row['output']=c.identity(actual);data=actual.read_bytes()
            checks=[('native',case['golden'])]
            if repeat:checks.append(('repeat',prior_outputs[(variant,0)]))
            if variant=='B':checks.append(('A_vs_B',prior_outputs[('A',repeat)]))
            row['comparisons']={}
            for label,item in checks:
                identity,match=compare_bytes(folder,label,c.verify(item).read_bytes(),data)
                row['comparisons'][label]=dict(report=identity,match=match)
    live_callback(folder,query,variant);row['post_loaded']=mappings(folder,'post',selected,binding['cann_root'])
    row['safety']=safety(folder,repeat)
    require(row.get('output') and all(item['match'] for item in row['comparisons'].values()), 'numerics/Repeat mismatch')
    c.verify_opp(selected['opp']);c.verify_tree(case)


def build(root,binding,report):
    report['variants']={}
    for variant,selected in binding['variants'].items():
        folder=root/variant;folder.mkdir();env=environment(binding,selected)
        c.command(['cmake','-S',HERE/'runner','-B',folder/'build','-DCANN_ROOT='+binding['cann_root'],
                   '-DPRIVATE_OPP='+selected['opp']['install'],'-DCOMMON_INCLUDE='+str(c.QUANT/'test')],folder/'configure.log',env)
        c.command(['cmake','--build',folder/'build','--target','scheduler_fc_smoke'],folder/'compile.log',env)
        runner=c.identity(folder/'build/scheduler_fc_smoke');deps=c.dependencies(runner,folder/'ldd.log',env)
        require(deps==selected['dependencies'] and selected['api'] in [item['file'] for item in deps], 'Smoke/Host Runner dependencies differ')
        report['variants'][variant]=dict(runner=runner,dependencies=deps,opp_root=selected['opp']['install'],
                                         manifest=selected['manifest'],cache=c.identity(folder/'build/CMakeCache.txt'))


def replay(binding,rows,built):
    require([(row['variant'],row['repeat']) for row in rows]==list(ORDER),'need exactly four ordered executions')
    outputs={};summary={}
    for row in rows:
        variant,repeat=row['variant'],row['repeat']; selected=binding['variants'][variant];query=selected['query']
        folder=Path(row['folder']);require(row['runner']==built['variants'][variant]['runner'],'Runner differs')
        require(row['launch_authorized'] is True and safety(folder,repeat)==row['safety'],'runtime safety mismatch')
        for name,item in binding['case']['inputs'].items():require(c.identity(folder/name)==item,'input link drift')
        live_callback(folder,query,variant)
        for when in ('pre','post'):
            require(mappings(folder,when,selected,binding['cann_root'])==row[when+'_loaded'],'DSO evidence differs')
        require(c.identity(folder/'actual.bin')==row['output'],'output identity differs')
        data=c.verify(row['output']).read_bytes();checks={'native':c.verify(binding['case']['golden']).read_bytes()}
        if repeat:checks['repeat']=outputs[(variant,0)]
        if variant=='B':checks['A_vs_B']=outputs[('A',repeat)]
        require(set(checks)==set(row['comparisons']),'comparison inventory differs')
        for name,expected in checks.items():
            result=difference(expected,data);item=row['comparisons'][name]
            require(result==c.read(c.verify(item['report'])) and result['native_comparator']['bitwise_equal'] and
                    result['native_comparator']['finite'] and item['match'] is True, 'comparison replay failed')
        outputs[(variant,repeat)]=data
        summary[variant]=dict(blockDim=query['platform']['blockDim'],core_getters=[query['platform'][key] for key in
                              ('GetCoreNum','GetCoreNumAic','GetCoreNumAiv')],calc=query['schedule']['calc_tsch_block_dim'],
                              workspace=query['platform']['workspace'],outputs=40960,execute_calls=2,bitwise_mismatches=0,
                              nonfinite=0,repeat='MATCH',guards='INTACT',readonly=True)
    return summary


def main(args):
    phase=args.phase
    if phase!='source':
        require(args.output_dir==args.source_report.parent.parent/phase,'phase output must be a fresh sibling of Source')
    root=c.fresh(args.output_dir,[args.host_root] if phase=='source' else [args.source_report.parent])
    report=dict(phase=phase,status='BLOCKED',sources=sources(),npu='NOT_RUN',benchmark='NOT_RUN',full_draft='NOT_RUN')
    try:
        c.clean()
        if phase=='source':
            manifest=root/'source-manifest.json';c.write(manifest,report['sources']);report['source_manifest']=c.identity(manifest)
            report['binding']=bind_history(args.host_root)
        else:
            src=predecessor(args.source_report,'source');binding=src['binding'];report['source_report']=c.identity(args.source_report)
            if phase=='build':build(root,binding,report)
            else:
                built=predecessor(args.build_report,'build');report['build_report']=c.identity(args.build_report)
                require(built['source_report']==report['source_report'],'mixed Source and Build')
                if phase=='smoke':
                    outputs={};report['runs']=[]
                    for variant,repeat in ORDER:
                        selected=binding['variants'][variant];binary=built['variants'][variant]
                        folder=root/(variant+'-'+str(repeat));folder.mkdir()
                        row=dict(variant=variant,repeat=repeat,folder=str(folder),runner=binary['runner'],launch_authorized=False)
                        report['runs'].append(row)
                        try:run_device(folder,repeat,binding,selected,binary,row,outputs)
                        finally:
                            row['artifacts']=[c.identity(path) for path in sorted(folder.iterdir()) if path.is_file() and not path.is_symlink()]
                            if row['launch_authorized']:report['npu']='LAUNCH_AUTHORIZED_SEE_RUNTIME_JOURNAL'
                        outputs[(variant,repeat)]=row['output']
                        print('MATCH:',variant,repeat,flush=True)
                    report['npu']='FOUR_SYNCHRONIZED_EXECUTIONS'
                else:
                    smoke=predecessor(args.smoke_report,'smoke');report['smoke_report']=c.identity(args.smoke_report)
                    require(all(smoke[key]==report[key] for key in ('source_report','build_report')), 'mixed Smoke chain')
                    report['comparison']=replay(binding,smoke['runs'],built)
                    report.update(npu='VERIFIED_FROM_SMOKE_NO_NEW_EXECUTE',A_B='BITWISE_EQUAL',
                                  decision='ELIGIBLE_FOR_SEPARATE_26_PROJECTION_REVIEW_ONLY')
        report['status']=STATES[phase]
    except Exception as err:
        report['error']=f'{type(err).__name__}: {err}';(root/'traceback.txt').write_text(traceback.format_exc())
    finally:
        try:
            c.verify_tree(report);require(sources()==report['sources'],'Smoke sources changed');c.clean()
        except Exception as err:report.update(status='BLOCKED',integrity_error=str(err))
        c.write(root/'report.json',report)
    print(json.dumps({key:report.get(key) for key in ('phase','status','npu','comparison','error','integrity_error')},indent=2));print('REPORT:',root/'report.json')
    return int(report['status']=='BLOCKED')


def parse(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);subs=parser.add_subparsers(dest='phase',required=True)
    for phase,flags in [('source',['host-root']),('build',['source-report']),('smoke',['source-report','build-report']),
                        ('compare',['source-report','build-report','smoke-report'])]:
        sub=subs.add_parser(phase,allow_abbrev=False)
        for flag in flags+['output-dir']:sub.add_argument('--'+flag,type=lambda raw:Path(raw).resolve(),required=True)
    return parser.parse_args(argv)


if __name__=='__main__':raise SystemExit(main(parse()))
