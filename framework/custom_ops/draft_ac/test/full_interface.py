"""Read-only full-factory interface gate. No Meta registration, model or NPU calls."""
import inspect
import json
import os
from pathlib import Path
import sys
from common import sha256,write_json


def module_identity(module):
    path=getattr(module,'__file__',None)
    return dict(version=str(getattr(module,'__version__','unknown')),path=path,
                sha256=sha256(path) if path and Path(path).is_file() else None)


def check_gdr_schema(operation):
    from qwen35_dflash.ascend310p.custom_op_export import _ADAPTERS,_validate_schema,NPU_CHUNK_GATED_DELTA_RULE_TORCH_OP
    # Reuse the locked validation itself; do not install a fake ten-argument op.
    return _validate_schema(operation,_ADAPTERS[NPU_CHUNK_GATED_DELTA_RULE_TORCH_OP])


def inspect_interfaces(torch_module,npu_module,custom,config):
    report=dict(status='BLOCKED',scope='dispatcher/prototype read-only; no kernel execution',checks={},
                python=dict(executable=sys.executable,version=sys.version,sys_path=list(sys.path)),torch=module_identity(torch_module),
                torch_npu=module_identity(npu_module),environment={k:os.environ.get(k,'') for k in
                    ('CONDA_PREFIX','PYTHONPATH','LD_LIBRARY_PATH','LD_PRELOAD','ASCEND_CUSTOM_OPP_PATH','ASCEND_OPP_PATH','ASCEND_HOME_PATH')})
    name=custom.NPU_CHUNK_GATED_DELTA_RULE_TORCH_OP
    try:
        operation=torch_module.ops.npu.npu_chunk_gated_delta_rule.default
        schema=operation._schema
        report['dispatcher']=dict(schema=str(schema),arguments=[dict(name=a.name,type=str(a.type),kwarg_only=a.kwarg_only) for a in schema.arguments])
        dump=getattr(torch_module._C,'_dispatch_dump',None)
        if callable(dump):
            try: report['dispatcher']['registration_dump']=dump(name)
            except Exception as error: report['dispatcher']['registration_dump_error']=str(error)
        check_gdr_schema(operation)
        report['checks']['gdr_schema']=dict(status='PASS')
    except Exception as error: report['checks']['gdr_schema']=dict(status='FAIL',error=str(error))
    wrapper=getattr(npu_module,'npu_chunk_gated_delta_rule',None)
    report['python_frontend']=dict(callable=callable(wrapper),module=getattr(wrapper,'__module__',None))
    for field,reader in (('signature',lambda: str(inspect.signature(wrapper))),('source_file',lambda: inspect.getsourcefile(wrapper))):
        try: report['python_frontend'][field]=reader()
        except (TypeError,ValueError) as error: report['python_frontend'][field+'_error']=str(error)
    report['checks']['python_frontend']=dict(status='PASS' if callable(wrapper) else 'FAIL')
    validators=[('gdr_ge',custom.validate_gdr_ge_prototype_environment),('adn_ge',custom.validate_adn_attention_ge_prototype_environment)]
    if config.get('verify_gdr','chunk')=='mtp': validators.append(('gdr_mtp_ge',custom.validate_gdr_mtp_ge_prototype_environment))
    for label,validate in validators:
        try: report['checks'][label]=validate()
        except Exception as error: report['checks'][label]=dict(status='FAIL',error=str(error))
    report['loaded_torch_libraries']=sorted(map(str,getattr(torch_module.ops,'loaded_libraries',set())))
    core=sys.modules.get('torch_npu._C')
    if core is not None: report['torch_npu_C']=module_identity(core)
    maps=Path('/proc/self/maps')
    if maps.is_file():
        try:
            report['mapped_libraries']=sorted({line.split()[-1] for line in maps.read_text().splitlines()
                if '/' in line and '.so' in line and any(k in line.lower() for k in ('torch_npu','gdr','cust_op','adn'))})
        except OSError as error: report['mapped_libraries_error']=str(error)
    if all(c.get('status')=='PASS' for c in report['checks'].values()): report['status']='PASS'
    return report


def require_full_interfaces(args,path):
    report=dict(status='BLOCKED',scope='read-only interface preflight; NPU correctness NOT_RUN')
    try:
        from resolve import linked
        config=json.loads(args.factory_config.read_text())
        deployment=json.loads(args.native_deployment.read_text())
        air_path=linked(args.native_deployment.parent,deployment['air_manifest'])
        air=json.loads(air_path.read_text())
        if config!=air.get('factory_config'): raise ValueError('factory differs from hash-bound Native AIR')
        historical=dict(air=dict(path=str(air_path),sha256=sha256(air_path)),environment=air.get('environment'),
                        operator_preflight=air.get('operator_preflight'),
                        gdr_graph_audits=[dict(graph=g['name'],audit=a) for g in air.get('graphs',[]) for a in g.get('custom_op_audit',[])
                                         if a.get('ge_op_type') in ('ChunkGatedDeltaRule','GatedDeltaRuleMTP')])
        report['native_export']=historical
        import torch
        import torch_npu
        import torchair
        from qwen35_dflash.ascend310p import custom_op_export
        report=inspect_interfaces(torch,torch_npu,custom_op_export,config)
        report['torchair']=module_identity(torchair)
        report['native_export']=historical
    except Exception as error: report.update(status='BLOCKED',error=f'{type(error).__name__}: {error}')
    write_json(path,report)
    if report['status']!='PASS':
        raise RuntimeError(f'full Draft frontend/OPP incompatible or unverified; see {path}; no schema relaxation, model load or device tests')
    return report


if __name__=='__main__':
    import argparse
    from process_boundary import decode_args
    parser=argparse.ArgumentParser(description='Read-only GDR/ADN diagnosis using a prior device worker environment')
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--device-environment',action='store_true',help=argparse.SUPPRESS)
    options=parser.parse_args(); root=options.evidence.resolve(); output=options.report.resolve()
    if output.exists(): raise SystemExit('report already exists; use a new path')
    if not options.device_environment:
        environment=json.loads((root/'device-process.json').read_text()); env=os.environ.copy()
        for key in ('ASCEND_CUSTOM_OPP_PATH','LD_LIBRARY_PATH'):
            if not isinstance(environment.get(key),str): raise SystemExit('missing saved worker environment: '+key)
            env[key]=environment[key]
        env['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
        os.execve(sys.executable,[sys.executable,'-B',str(Path(__file__).resolve()),'--evidence',str(root),
                                 '--report',str(output),'--device-environment'],env)
    arguments=decode_args(json.loads((root/'worker-input.json').read_text()))
    try:
        require_full_interfaces(arguments,output)
    except Exception as error:
        print(str(error),file=sys.stderr);raise SystemExit(1)
    print('Interface checks PASS (NPU correctness NOT_RUN): '+str(output))
