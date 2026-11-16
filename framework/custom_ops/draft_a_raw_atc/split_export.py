"""A-only copy of export_one with explicit Runner injection; no shared changes."""
import gc
from pathlib import Path
from common import write_json,record,checked_file

def export_one(spec,folder,atc,*,runner,draft_only=False,air_only=False):
    import torch
    import torchair
    from qwen35_dflash.ascend310p.draft_ac_export import GE,converter_capabilities
    capabilities=[converter_capabilities(op,torchair) for op in spec.custom_ops if op.torch_op in GE]
    if capabilities:
        write_json(folder/'converter-capabilities.json',capabilities)
        print('A/C converter capabilities: '+str(folder/'converter-capabilities.json'),flush=True)
    from qwen35_dflash.ascend310p.draft_ac_export_pipeline import export_air_bundle
    from qwen35_dflash.ascend310p.draft_ac_export_pipeline import compile_air_bundle
    stages=dict(air='RUNNING',atc='NOT_RUN')
    write_json(folder/'export-stages.json',stages)
    air=export_air_bundle(lambda _:(spec,),{},folder/'artifacts')
    stages['air']='PASS';write_json(folder/'export-stages.json',stages)
    if draft_only:
        from draft_only import audit_draft_air
        audit_draft_air(air,folder)
    if air_only:
        return None,dict(air=record(Path(air['manifest_path']),folder),atc='NOT_RUN',npu='NOT_RUN')
    # A real custom Runner must not enable the compiler's test-double ABI escape.
    from qwen35_dflash.ascend310p.runtime_input_export import validated_runtime_input_abi
    for graph in air['graphs']:
        validated_runtime_input_abi(graph,required=True,allow_test_double=False)
    stages['atc']='RUNNING';write_json(folder/'export-stages.json',stages)
    compiled=compile_air_bundle(air['manifest_path'],atc_bin=atc,soc_version='Ascend310P3',
        extra_args=['--precision_mode=must_keep_origin_dtype','--deterministic=0'],runner=runner,atc_identity=runner.identity_text)
    stages['atc']='PASS';write_json(folder/'export-stages.json',stages)
    graph=compiled['graphs'][0]
    om=checked_file(Path(compiled['manifest_path']).parent,graph['om'])
    torch._dynamo.reset(); gc.collect()
    return om,dict(air=record(Path(air['manifest_path']),folder),deployment=record(Path(compiled['manifest_path']),folder),om=record(om,folder))
