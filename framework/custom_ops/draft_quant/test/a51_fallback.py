"""Read-only validation of the six targeted synthetic fallback workloads."""
import json
from types import SimpleNamespace
import reference as ref
from a2_common import sha256,validate_execution
from a3_launch import K_TILE_POLICY,launch_evidence,k_tile_reason

TARGETS=tuple(s for s in ref.A4_WORKLOADS if s.m==80 and s.k in (512,1024) and s.n in (64,128,256))
TARGET_NAMES=tuple(s.name for s in TARGETS)


def collect(root,config,device):
    report=json.loads((root/'suite.json').read_text())
    if (report.get('status')!='PASS' or report.get('selection')!='SUBSET' or report.get('suite')!='a4' or
        report.get('requested_k_tile')!=config['k_tile'] or report.get('device_id')!=device or
        report.get('timing_protocol')!='continuous-v1' or
        tuple(w['name'] for w in report.get('workloads',[]))!=TARGET_NAMES or
        config.get('k_tile_policy')!=K_TILE_POLICY):
        raise ValueError('need six explicit fallback workloads with current policy and continuous guards')
    plans=[]
    for shape,workload in zip(TARGETS,report['workloads']):
        directory=root/shape.name
        manifest=ref.load_cases(directory)
        names=tuple(c[0] for c in ref.cases(shape))
        native=json.loads((directory/'native-eager.json').read_text())
        comparison=json.loads((directory/'comparison.json').read_text())
        if (workload.get('status')!='PASS' or workload.get('mkn')!=[shape.m,shape.k,shape.n] or
            tuple(c['name'] for c in workload.get('custom_cases',[]))!=names or
            tuple(c['name'] for c in manifest['cases'])!=names or
            native.get('status')!='PASS' or native.get('cpu_fallback') is not False or
            native.get('mkn')!=[80,shape.k,shape.n] or native.get('group_size')!=128 or native.get('inner_precise')!=0 or
            native.get('environment',{}).get('device_id')!=device or
            native.get('reference_policy')!=ref.NATIVE_POLICY or native.get('manifest_sha256')!=sha256(directory/'manifest.json') or
            tuple(c['name'] for c in native.get('cases',[]))!=names or
            comparison.get('status')!='PASS' or tuple(c['name'] for c in comparison.get('cases',[]))!=names or
            any(c.get('status')!='PASS' for c in comparison['cases'])):
            raise ValueError('incomplete targeted synthetic/native correctness')
        for name,native_case in zip(names,native['cases']):
            case=directory/name
            execution=json.loads((case/'execution.json').read_text())
            validate_execution(execution,(shape.m,shape.k,shape.n),device,'AscendCL ACLNN',5,30,'continuous-v1',128)
            if (execution.get('requested_k_tile')!=config['k_tile'] or execution.get('tile_m')!=80 or
                execution.get('weight_reuse_rows')!=80 or execution.get('k_tile_selection_reason')!=k_tile_reason((80,shape.k,shape.n),config['k_tile'])):
                raise ValueError('runner does not identify the exact requested/selected fallback')
            golden=(case/'native-0.bin').read_bytes()
            for i in range(2):
                if sha256(case/f'native-{i}.bin')!=native_case['output_sha256'][i]: raise ValueError('native output changed')
            for file in ('native-1.bin','actual-0.bin','actual-1.bin','benchmark-last.bin','postcheck.bin'):
                diff=ref.compare(golden,(case/file).read_bytes(),SimpleNamespace(m=80,n=shape.n))
                if not diff['finite'] or not diff['bitwise_equal'] or diff['max_ulp']!=0:
                    raise ValueError('targeted native/repeat/timed-tail/postcheck bitwise mismatch')
            plan=launch_evidence(case/'runner.log',(80,shape.k,shape.n),config['core_limit'],execution['workspace_bytes'],
                config['dequant_mode'],config['pipeline_mode'],5,config['kv_m80_mode'],config['scale_mode'],
                config['k_tile'],config['k_tile_policy'])
            plans.append(dict(name=f'{shape.name}/{name}',launch=plan))
    return plans
