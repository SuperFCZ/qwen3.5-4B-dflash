import copy
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import gate_common as common
import gate_plan as plan
import run_gate8 as driver
import measurement as measure
import export_gate
import source_gate
from a3_launch import configure_build,load_build_config

G=common.GATE


def cases():
    return {g:{layer:dict(name=layer,shape=list(common.SHAPE),
                inputs={key:dict(bytes=size,sha256=common.sha((layer+key+(g if key=='x.bin' else '')).encode()))
                        for key,size in [('x.bin',81920),('w_nz.bin',49807360),('s_gn.bin',778240)]},
                golden=dict(bytes=622592,sha256=common.sha((layer+g).encode()))) for layer in common.LAYERS} for g in common.GEARS}


def graph(case):
    row=dict(name='name_unrelated_to_layer_number',ge_type=G,k=2560,n=19456,nz_bytes=49807360,roundtrip='BIT_EXACT',
             weight_sha256='logical',nz_sha256=case['inputs']['w_nz.bin']['sha256'],scale_gn_sha256=case['inputs']['s_gn.bin']['sha256'],
             storage_shape=[80,1216,16,32],scale_shape=[20,19456])
    return dict(role='diagnostic',metadata=dict(draft_ac=dict(scope='single-op-A',a_count=1,c_count=0,audit_count=0,mode='a'),tensor_abi=common.ABI),
                input_names=['x'],output_names=['y'],runtime_input_abi=dict(status='PASS',draft_ac=dict(status='PASS',counts=dict(zip(common.COUNT_KEYS,(1,0,0))),nodes=[row])),
                custom_op_audit=[dict(ge_op_type=G,ge_node_occurrences=1)])


def plans(mode):
    raw=mode=='raw';scratch=185600 if raw else 202240
    cube=bytearray(200)
    for f in plan.schema():struct.pack_into('<I',cube,f['offset'],plan.CUBE.get(f['name'],1))
    data=struct.pack('<IIIIQ',*common.SHAPE,256,scratch)+cube
    resources=dict(user_ub=76544 if raw else 59904,scratch=scratch,raw_banks=2 if raw else 1,tile_k=256,codes_bytes=32768 if raw else 16384,
        scale_bytes=512 if raw else 256,weight_bytes=32768,l1_used=40960,l0a_used=16384,l0b_used=65536,l0c_used=4096,workspace=2097152)
    return dict(schedule=dict(route='ATC',variant='A',m=16,k=2560,n=19456,n_tiles=304,tile_n=64,core_limit=0,baseline_block_dim=8,
                             selected_block_dim=8,generic=8,baseline_assignments=plan.assignments(304,8),resources=resources,tiling_hex=data.hex()),
                platform=dict(route='ATC',coreLimit=0,blockDim=8,GetCoreNum=8,availableCores=8,workspace=2097152,capacity=dict(UB=262144,L0B=65536)),
                tiling_sha256=common.sha(data))


def elf(code,symbol):
    data=code+b'\x00'+symbol.encode()+b'\x00'
    hdr=struct.pack('<16sHHIQQQIHHHHHH',b'\x7fELF\x02\x01\x01',2,183,1,0,0,64+len(data),0,64,0,0,64,2,0)
    return hdr+data+bytes(64)+struct.pack('<IIQQQQIIQQ',0,1,6,0,64,len(code),0,0,1,0)


def execution(timed):
    return dict(status='PASS',scope='single Gate frozen Native Golden checked in process',device=0,guards=True,readonly=True,
        repeat=True,cpu_fallback=False,golden_match=True,poison_verified=True,sync_completed=True,timed_log_stable=True,
        verification_calls=4 if timed else 2,row_permutation_checked=False,workspace_bytes=512,model_weight_bytes=49807360,
        timing=dict(protocol='continuous-v1' if timed else 'checked-v1',status='MEASURED' if timed else 'NOT_RUN',
                    warmup=5 if timed else 0,repetitions=30 if timed else 0,correctness_before_calls=2,correctness_after_calls=int(timed),
                    timed_tail_checked=timed,per_timed_call_readback=not timed,per_timed_call_poison=not timed,
                    prepare={'samples_ms':[]},execute_sync={'samples_ms':[1.0]*30 if timed else []}))


def timing_report(gear,order):
    stage='c'+gear+'-'+order;rows=[]
    for g,layer,mode,round_id in measure.schedule(stage):
        valid=execution(True);valid['timing']['execute_sync']['samples_ms']=[2.0 if mode=='off' else 1.9]*30
        rows.append(dict(gear=g,layer=layer,mode=mode,round=round_id,profiled=False,validation=measure.verify_execution(valid,True),
                         selected_kernels={},om={},runner={},opp_kernel={},golden={},atc_gate_plan=dict(block_dim=8,cube_sha256='same',workspace=2097152)))
    return dict(phase=stage,runs=rows)


class Gate8(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.tmp=Path(t.name).resolve()

    def test_generated_build_identity(self):
        for mode in common.MODES:
            cfg=self.tmp/(mode+'.json');header=self.tmp/(mode+'.h')
            configure_build(header,cfg,0,'batched',None,'prefetch','a-ub','broadcast',256,'loop',mode)
            value=load_build_config(cfg);common.check_config(value,mode)
            for key,bad in [('gate_copy_mode','2d'),('pipeline_mode','serial'),('dequant_mode','legacy'),('k_tile',128),('core_limit',8),('scale_mode','scalar')]:
                with self.subTest(key=key),self.assertRaises(ValueError):common.check_config(dict(value,**{key:bad}),mode)
        self.assertEqual(common.derive_header((self.tmp/'off.h').read_bytes()),(self.tmp/'raw.h').read_bytes())
        with self.assertRaises(ValueError):common.derive_header((self.tmp/'off.h').read_bytes()+b' ')

    def test_ten_cases_and_graph_binding(self):
        valid=cases();common.case_contract(valid)
        a=valid['16']['layer-3-gate_up'];g=graph(a)
        matched=common.config_graph(g,{'layer-3-gate_up':a})
        self.assertEqual(matched['layer-3-gate_up']['name'],'name_unrelated_to_layer_number')
        wrong=valid['16']['layer-1-gate_up']
        with self.assertRaises(ValueError):common.config_graph(g,{'layer-1-gate_up':wrong})
        for issue in ('layer','weight','scale','x','golden','dup','bytes'):
            rows=copy.deepcopy(valid);row=rows['64'][common.LAYERS[0]]
            if issue=='layer':rows['64'].pop(common.LAYERS[-1])
            elif issue in ('weight','scale'):row['inputs'][{'weight':'w_nz.bin','scale':'s_gn.bin'}[issue]]['sha256']='drift'
            elif issue=='x':row['inputs']['x.bin']=rows['16'][common.LAYERS[0]]['inputs']['x.bin']
            elif issue=='golden':row['golden']=rows['16'][common.LAYERS[0]]['golden']
            elif issue=='dup':rows['64'][common.LAYERS[-1]]=row
            else:row['golden']['bytes']-=2
            with self.subTest(issue=issue),self.assertRaises(ValueError):common.case_contract(rows)

    def test_graph_absent_no_aclnn_fallback(self):
        with self.assertRaises(source_gate.UnprovenGraph):source_gate.graph_source({'cases':[]},{},common.LAYERS[0],{}, {},{})

    def test_8block_real_cube_and_parent_delta(self):
        off=plans('off');raw=plans('raw')
        with patch.object(plan,'OFF_PARENT',off['tiling_sha256']):
            ctrl=plan.gate_plan(off,'off',off);new=plan.gate_plan(raw,'raw',off)
            self.assertNotEqual(new['parent_sha256'],ctrl['parent_sha256']);self.assertEqual(new['cube_sha256'],ctrl['cube_sha256'])
            for bad in ('7cores','duplicate','cube','ub','l0b','workspace','policy'):
                value=copy.deepcopy(raw);s=value['schedule']
                if bad=='7cores':value['platform']['blockDim']=7
                if bad=='duplicate':s['baseline_assignments'][7][0]=0
                if bad=='cube':
                    data=bytearray.fromhex(s['tiling_hex']);data[24+160]^=1
                    s['tiling_hex']=data.hex();value['tiling_sha256']=common.sha(data)
                if bad=='ub':s['resources']['scratch']+=32
                if bad=='l0b':s['resources']['l0b_used']-=32
                if bad=='workspace':s['resources']['workspace']+=1
                if bad=='policy':s['variant']='B'
                with self.subTest(bad=bad),self.assertRaises(ValueError):plan.gate_plan(value,'raw',off)

    def test_atc_elf_not_op_registry_object(self):
        prefix='te_dflashgroupquantlinear_'+'a'*64;sym=prefix+'__kernel0'
        real=elf(b'raw executable',sym);fake=elf(b'wrong',sym)
        attrs=[dict(key=k,value={t:v}) for k,t,v in [('_kernelname','s',sym),('_kernel_bin_id','s',prefix+common.sha(real)),('_tbeKernelSize','i',str(len(real))),('tvm_blockdim','i',8)]]
        attrs[:0]=[dict(key='_ge_attr_op_kernel_lib_name',value={'s':'AIcoreEngine'}),dict(key='_op_no_tiling',value={'b':False})]
        b=plan.node_binding(dict(name='Gate',type=G,attributes=attrs));path=self.tmp/'model.om';path.write_bytes(b'IMOD'+fake+b'x'*13+real)
        bound=plan.embedded(path,[b])['Gate'];self.assertEqual(bound['offset'],4+len(fake)+13);self.assertEqual(bound['sha256'],common.sha(real))
        plan.recheck_embedded(path,[b],{'Gate':bound})
        with self.assertRaises(ValueError):plan.recheck_embedded(path,[b],{'Gate':dict(bound,offset=4)})
        for bytes_ in (fake,real+real,real[:-1]):
            path.write_bytes(bytes_)
            with self.assertRaises(ValueError):plan.embedded(path,[b])
        attrs[-1]['value']['i']=7
        with self.assertRaises(ValueError):plan.node_binding(dict(name='Gate',type=G,attributes=attrs))

    def test_paired_machine_code_must_differ_only_on_target(self):
        gate='G';kernel=lambda ch:dict(executable_sections=[dict(bytes=11,sha256=ch)])
        def model(mode):
            return dict(mode=mode,key='16',air={},abi={},gate_nodes={str(n):gate+str(n) for n in range(5)},scope='full',opp_root=mode,
                om={'sha256':mode},private_kernel={'elf':{'sha256':mode}},plans={common.SHAPE_KEY:{'contract':dict(block_dim=8,cube_sha256='s',cube_fields={},assignments=[],workspace=123)}},
                atc_kernels={gate+str(i):kernel(mode if i<5 else 'unchanged') for i in range(26)})
        off,raw=map(model,common.MODES);export_gate.compare_models(off,raw,True)
        for failure in ('OPP','identical-target','non-target'):
            bad=copy.deepcopy(raw)
            if failure=='OPP':bad['private_kernel']=off['private_kernel']
            else:bad['atc_kernels']['G'+('0' if failure=='identical-target' else '6')]=kernel('off' if failure=='identical-target' else 'changed')
            with self.subTest(failure=failure),self.assertRaises(ValueError):export_gate.compare_models(off,bad,True)

    def test_guarded_bits_and_brackets(self):
        expected=self.tmp/'golden';expected.write_bytes(bytes(622592));good=common.c.identity(expected)
        path=self.tmp/'execution.json';common.c.write(path,execution(True))
        row=dict(layer=common.LAYERS[0],execution=common.c.identity(path),outputs={'y':[good]*4})
        self.assertTrue(measure.verify_outputs(row,{'golden':{'y':good}},True)['timing'])
        actual=self.tmp/'y-actual-0.bin';actual.write_bytes(bytes(622590)+b'\x01\x00')
        row['outputs']['y'][0]=common.c.identity(actual)
        with self.assertRaises(ValueError):measure.verify_outputs(row,{'golden':{'y':good}},True)
        d=measure.mismatch_evidence(self.tmp,{'golden':{'y':good}})[0]
        self.assertEqual((d['bit_mismatches'],d['first']['row'],d['first']['channel'],d['first']['ulp']),(1,15,19455,1))
        for key in ('poison_verified','guards','readonly','timed_log_stable','sync_completed','repeat','cpu_fallback'):
            bad=execution(True);bad[key]=not bad[key]
            with self.subTest(key=key),self.assertRaises(ValueError):measure.verify_execution(bad,True)

    def test_unknown_or_retargeted_dso_refused(self):
        manifest={'private_opp':'off','cann_root':'/cann','records':[]}
        selected={'opp':'raw','manifest':{}}
        with patch.object(common.c,'read',return_value=manifest),patch.object(common.c,'verify',return_value='unused'),patch.object(common.host.dsos,'verify_manifest'):
            with self.assertRaises(ValueError):common.resume.combined(selected,'/cann',{}, {'cann_root':'/cann','records':[]})
        with self.assertRaisesRegex(ValueError,'unbound OM DSO'):common.resume.check_maps({'/foreign/libcust_opapi.so':[{}]},{})

    def test_decision_fresh_both_orders(self):
        f=timing_report('16','forward');r=timing_report('16','reverse')
        self.assertEqual(measure.compare_gear('16',f,r)['decision'],common.ELIGIBLE)
        test_cases=[('p95',[1.9]*28+[2.2]*2),('insufficient',[1.99]*30),('median',[2.0]*30)]
        for reason,samples in test_cases:
            bad=copy.deepcopy(r)
            for row in bad['runs']:
                if row['mode']=='raw' and (reason=='insufficient' or row['layer']==common.LAYERS[4]):row['validation']['timing']['samples_ms']=samples
            with self.subTest(reason=reason):self.assertEqual(measure.compare_gear('16',f,bad)['decision'],common.NO_GO)
        for edits in ('omitted','duplicate','swapped'):
            bad=copy.deepcopy(r)
            if edits=='omitted':bad['runs'].pop()
            elif edits=='duplicate':bad['runs'][-1]=bad['runs'][0]
            else:bad['runs'][0],bad['runs'][1]=bad['runs'][1],bad['runs'][0]
            with self.assertRaises(ValueError):measure.compare_gear('16',f,bad)
        with self.assertRaises(ValueError):measure.compare_gear('64',f,r)

    def test_phase_entry_no_implicit_execution(self):
        for phase in common.STAGES[1:]:
            root=self.tmp/phase;session=root/'new'
            roots=[root/p for p in ['history','resume','profile']]
            for p in roots:p.mkdir(parents=True)
            args=SimpleNamespace(root=session,phase=phase,flow_root=roots[0],resume_root=roots[1],profile_root=roots[2])
            with patch('builtins.print'),patch.object(driver,'perform') as run,patch.object(driver,'export_stage') as export,patch.object(driver,'build') as build:
                self.assertEqual(driver.main(args),1)
                run.assert_not_called();export.assert_not_called();build.assert_not_called()
            result=common.c.read(session/phase/'report.json');self.assertEqual(result['device_execution'],'NOT_RUN');self.assertEqual(result['status'],'BLOCKED')
        with self.assertRaisesRegex(ValueError,'NOT_ELIGIBLE'):driver.stage_status('compare',dict(status=common.NO_GO,decision=common.NO_GO))
        self.assertFalse(driver.eligible({'16':dict(decision=common.ELIGIBLE),'64':dict(decision=common.NO_GO)}))

    def test_each_link_required_including_full_opt_in(self):
        root=self.tmp/'flow';root.mkdir();refs={};history=['one','two','three']
        context=lambda:dict(predecessors={},sources=[],history_roots=history)
        with patch.object(driver,'validate_stage'):
            for stage in common.STAGES[:-1]:
                nxt=common.STAGES[common.STAGES.index(stage)+1]
                with self.assertRaises(FileNotFoundError):driver.load_chain(root,nxt,context())
                folder=root/stage;folder.mkdir()
                passed=common.ELIGIBLE if stage=='compare' else common.STATE[stage]
                record=dict(phase=stage,root=str(root),status=passed,decision=passed,predecessors=refs.copy(),history_roots=history,sources=[])
                path=folder/'report.json'
                if stage=='compare':
                    common.c.write(path,dict(record,status=common.NO_GO,decision=common.NO_GO))
                    with self.assertRaisesRegex(ValueError,'NOT_ELIGIBLE'):driver.load_chain(root,'full-export',context())
                common.c.write(path,record);refs[stage]=common.c.identity(path)
                result=driver.load_chain(root,nxt,context());self.assertEqual(list(result),list(refs))
            p=root/'export/report.json';value=common.c.read(p);value['changed_after_acceptance']=True;common.c.write(p,value)
            with self.assertRaises(ValueError):driver.load_chain(root,'full-benchmark',context())

    def test_real_single_air_provenance_contract(self):
        layer=common.LAYERS[3];case=cases()['16'][layer];g=graph(case)
        directory=self.tmp/('a-c16-'+layer);directory.mkdir()
        model=directory/'model.air';model.write_bytes(b'frozen fixture')
        def relative(p):return dict(common.c.identity(p),path=p.name)
        g.update(air=relative(model),payload_files=[relative(model)])
        node=g['runtime_input_abi']['draft_ac']['nodes'][0]
        g['metadata']['draft_ac']['projections']=[{k:node[k] for k in ('n','k','weight_sha256','scale_gn_sha256')}]
        air=directory/'air.json';common.c.write(air,dict(status='PASS',artifact_kind='qwen35-dflash-torchair-bundle',graphs=[g]))
        atc=self.tmp/'atc';atc.write_text('cpu fixture');om=directory/'gate.om';om.write_bytes(b'model fixture')
        dep=directory/'deployment.json';cmd=[str(atc),'--mode=0','--framework=1','--soc_version=Ascend310P3','--precision_mode=must_keep_origin_dtype',
             '--deterministic=0','--model='+str(model),'--output='+str(om.with_suffix(''))]
        common.c.write(dep,dict(status='PASS',graphs=[dict(g,om=relative(om),atc_command=cmd)],air_manifest=relative(air),compiler={'identity':'cpu mock'}))
        agg=dict(cases=[dict(name='c16-'+layer,status='PASS',artifacts={'om':relative(om),'deployment':relative(dep),'air':relative(air)})])
        suite=self.tmp/'a-probes.json';common.c.write(suite,agg);ref=common.c.identity(suite)
        args=(agg,ref,layer,case,node,common.c.identity(atc))
        self.assertEqual(source_gate.graph_source(*args)['air'],common.c.identity(air))
        model.write_bytes(b'changed payload')
        with self.assertRaises(source_gate.UnprovenGraph):source_gate.graph_source(*args)

    def test_actual_scope_selects_current_gear_and_stops_on_raw_fault(self):
        src={'cases':cases(),'host_source':None,'cann':'/unused'}
        history=self.tmp/'history';common.c.write(history,{'binding':{'prior':{}}});src['host_source']=common.c.identity(history)
        layer=common.LAYERS[0];g='64';spec=src['cases'][g][layer]
        for name,ref in spec['inputs'].copy().items():
            path=self.tmp/name;path.write_bytes(b'cpu-fixture-'+name.encode());spec['inputs'][name]=common.c.identity(path)
        golden=self.tmp/'y';golden.write_bytes(bytes(622592));spec['golden']=common.c.identity(golden)
        runner=self.tmp/'runner';runner.touch();exe=common.c.identity(runner)
        built={'variants':{}};models={mode:{} for mode in common.MODES}
        calls=[]
        def invoke(argv,log,env,admit):
            folder=argv[3];admit();calls.append(folder.name)
            self.assertEqual((folder/'inputs/x.bin').resolve(),Path(spec['inputs']['x.bin']['path']))
            out=(folder/'golden/y.bin').read_bytes()
            if folder.name=='raw':out=b'\x01'+out[1:]
            for suffix in ('actual-0','actual-1'):(folder/('y-'+suffix+'.bin')).write_bytes(out)
            common.c.write(folder/'execution.json',execution(False))
        for mode in common.MODES:
            selected=dict(opp={'install':mode},kernel={'elf':exe},runner={'runner':exe,'deps':[]},env={})
            built['variants'][mode]=selected
            models[mode][layer]=dict(mode=mode,key=layer,abi=common.ABI,opp_root=mode,private_kernel=selected['kernel'],
                                     atc_kernels={},om=exe,plans={common.SHAPE_KEY:{'contract':{'block_dim':8}}})
        report={}
        with patch('builtins.print'),patch.object(measure.flow,'build_env',return_value={}),patch.object(measure.c,'verify_opp'),patch.object(measure.c,'dependencies',return_value=[]),patch.object(measure,'verify_maps',return_value=['private']),patch.object(measure.flow,'invoke_om',side_effect=invoke):
            with self.assertRaisesRegex(ValueError,'Native bitwise'):measure.perform(self.tmp/'runs',src,built,models,'c64-smoke',report)
        self.assertEqual(calls,['off','raw']);self.assertEqual(report['active_case'],['64',layer,'raw',0])
        self.assertEqual(report['runs'][-1]['differences'][0]['bit_mismatches'],1)

    def test_wrong_sha_rejected_even_when_already_seen(self):
        p=self.tmp/'identity';p.write_bytes(b'original');a=common.c.identity(p);seen={}
        driver.verify_tree(a,seen)
        with self.assertRaises(ValueError):driver.verify_tree(dict(a,sha256='0'*64),seen)
        p.write_bytes(b'replaced')
        with self.assertRaises(ValueError):driver.verify_tree(a,{})

    def test_optional_full_smoke_and_new_paired_samples(self):
        from raw_protocol import OUTPUTS
        pin=self.tmp/'prior';common.c.write(pin,dict(binding={'prior':{}}))
        base=self.tmp/'model';base.write_bytes(b'CPU fixture');obj=common.c.identity(base)
        abi=dict(inputs=[],outputs=[]);goldens={}
        for name in OUTPUTS:
            path=self.tmp/name;path.write_bytes(bytes(120 if name=='draft_top1' else 16));goldens[name]=common.c.identity(path)
        spec=dict(abi=abi,inputs={},golden=goldens)
        src=dict(cann='/mock',host_source=common.c.identity(pin),full={g:{'io':spec} for g in common.GEARS})
        built={'variants':{}};models={mode:{} for mode in common.MODES}
        for mode in common.MODES:
            built['variants'][mode]=dict(opp={'install':mode},om=dict(runner=obj,deps=[]),kernel={'elf':obj},env={})
            for g in common.GEARS:
                models[mode][g]=dict(mode=mode,key=g,abi=abi,opp_root=mode,private_kernel={'elf':obj},
                    om=obj,atc_kernels={},plans={common.SHAPE_KEY:{'contract':{'block_dim':8}}})
        def invoke(argv,log,env,admit):
            folder=argv[3];admit();timed=argv[-1]=='benchmark'
            result=execution(timed);result['scope']='full Draft frozen Golden checked in process'
            samples=result['timing']['execute_sync']['samples_ms']
            if timed:samples[:]=[3.0 if folder.name=='off' else 2.0]*30
            common.c.write(folder/'execution.json',result)
            for name,pin in goldens.items():
                for suffix in ['actual-0','actual-1']+(['benchmark-last','postcheck'] if timed else []):
                    (folder/(name+'-'+suffix+'.bin')).write_bytes(common.c.verify(pin).read_bytes())
        with patch('builtins.print'),patch.object(measure.flow,'build_env',return_value={}),patch.object(measure.c,'verify_opp'),patch.object(measure.c,'dependencies',return_value=[]),patch.object(measure,'verify_maps',return_value=['private']),patch.object(measure.flow,'invoke_om',side_effect=invoke):
            smoke={};bench={}
            measure.perform(self.tmp/'smoke',src,built,models,'full-smoke',smoke)
            measure.perform(self.tmp/'bench',src,built,models,'full-benchmark',bench,smoke)
            for row in smoke['runs']:self.assertEqual(set(row['outputs']),set(OUTPUTS))
            result=measure.full_stats(bench['runs'])
            self.assertFalse(result['historical_samples_used'])
            for g in common.GEARS:
                self.assertEqual(result['gears'][g]['round_off_over_raw'],[1.5]*3)
                self.assertEqual(result['gears'][g]['variants']['raw']['count'],90)
            bench['runs'][1]['validation']['workspace_bytes']+=32
            with self.assertRaises(ValueError):measure.pair_memory(bench['runs'])

    def test_gate_runner_uses_same_measurement_body(self):
        template=(common.HERE.parent/'draft_a_core_scheduler_flow/om/runner.cpp').read_text();gate=(common.HERE/'runner/runner.cpp').read_text()
        self.assertEqual(template.split('auto verify=')[1].split('const bool permutable')[0],gate.split('auto verify=')[1].split('const bool permutable')[0])
        self.assertIn('aclmdlExecuteAsync',gate);self.assertNotIn('aclnn',gate)
        self.assertIn('std::vector<int64_t>({16,19456})',gate)
