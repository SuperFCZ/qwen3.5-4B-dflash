"""Opt-in Draft replay scope. Never prepares, exports or executes Target graphs.

The unchanged factory still loads the verified Target for embedding/head and
contract construction. Its public prepare_export hook is deliberately NOT this
entry point: that hook remains responsible for the complete Target ABI.
"""
import json
from pathlib import Path
from common import record, write_json

DRAFT_OPS = frozenset(('npu::adn_rms_norm', 'npu::npu_scatter_nd_update',
                       'npu::npu_weight_quant_batchmatmul'))
TARGET_GE = frozenset(('ChunkGatedDeltaRule', 'GatedDeltaRuleMTP', 'AdnFusedInferAttention',
                       'CacheUpdate', 'DynamicQuant', 'QuantBatchMatmulV4444'))
IDENTITIES = ('quant_source_lock', 'target_input_identity', 'target_checkpoint_manifest_sha256',
              'draft_checkpoint_manifest_sha256', 'quant_input_manifest_sha256',
              'quant_linear_manifest_sha256', 'quant_embedding_manifest_sha256',
              'target_lm_head_source')


def native_history(args):
    from resolve import linked
    from qwen35_dflash.ascend310p.incremental_plan import validate_incremental_bundle
    deployment = json.loads(args.native_deployment.read_text())
    air_path = linked(args.native_deployment.parent, deployment['air_manifest'])
    air = json.loads(air_path.read_text())
    config = json.loads(args.factory_config.read_text())
    if config != air.get('factory_config'):
        raise ValueError('Draft-only factory differs from hash-bound Native AIR')
    # Validate the COMPLETE historical bundle, including Target/Verify contract,
    # without registering or running their dispatcher operations.
    contract = validate_incremental_bundle(air['graphs'])
    if contract is None or validate_incremental_bundle(deployment['graphs']) != contract:
        raise ValueError('Draft-only historical incremental contract differs')
    if config.get('verify_gdr', 'chunk') != 'chunk':
        raise ValueError('Draft-only diagnostic is scoped to the frozen chunk-v2 Native bundle')
    native = next(g for g in deployment['graphs'] if g['name'] == 'draft')
    exported = next(g for g in air['graphs'] if g['name'] == 'draft')
    for key in ('metadata', 'runtime_input_abi', 'custom_op_audit', 'input_names', 'output_names'):
        if native.get(key) != exported.get(key):
            raise ValueError('Draft-only Native AIR/Deployment differs: ' + key)
    if len(native['input_names']) != 15 or len(native['output_names']) != 11:
        raise ValueError('Draft-only requires frozen 15-input/11-output ABI')
    if any(not native['metadata'].get(key) for key in IDENTITIES):
        raise ValueError('Draft-only missing Target/checkpoint identity metadata')
    return config, native, air_path


def draft_operators(config):
    from qwen35_dflash.ascend310p.quant_factory import _target_custom_op_exports
    from qwen35_dflash.ascend310p.contracts import CustomOpExportSpec
    from models.dflash_v1.weight_quant_matmul import TORCH_OP, GE_OP
    if config.get('draft_quantization') != 'w8a16' or config.get('draft_quant_matmul') != 'weight_quant':
        raise ValueError('Draft-only requires the original W8 WeightQuant factory')
    # Pure declarations only. No Target schema inspection/Meta/converter setup.
    specs = tuple(op for op in _target_custom_op_exports(config, incremental=True)
                  if op.torch_op in DRAFT_OPS) + (CustomOpExportSpec(TORCH_OP, GE_OP),)
    if len(specs) != 3 or {s.torch_op for s in specs} != DRAFT_OPS:
        raise ValueError('Draft-only operator scope drifted')
    return specs


def prepare_draft_interfaces(config, torchair):
    from qwen35_dflash.ascend310p.custom_op_export import prepare_custom_op_export
    sessions = [prepare_custom_op_export(op, torchair) for op in draft_operators(config)]
    return [dict(torch_target=s.spec.torch_target, ge_op_type=s.spec.ge_op_type,
                 torch_schema=s.schema, fake_kernel=s.fake_kernel,
                 converter_policy=s.converter_policy) for s in sessions]


def require_draft_interfaces(args, path):
    report = dict(status='BLOCKED', scope='Draft-only interfaces; no model execution',
                  draft_only='NOT_RUN', target_interface='NOT_RUN', end_to_end='NOT_RUN')
    try:
        config, _, air_path = native_history(args)
        report['native_air'] = record(air_path, air_path.parent)
        import torch
        import torch_npu
        import torchair
        from full_interface import check_gdr_schema, module_identity
        report['torch_npu'] = module_identity(torch_npu)
        report['torchair'] = module_identity(torchair)
        # Evidence only, NOT a registration or a required Draft capability.
        try:
            operation = torch.ops.npu.npu_chunk_gated_delta_rule.default
            report['target_gdr_schema'] = str(operation._schema)
            check_gdr_schema(operation)
            report['target_interface'] = 'SCHEMA_COMPATIBLE_NOT_VALIDATED'
        except Exception as error:
            report.update(target_interface='BLOCKED', target_error=str(error))
        report['operators'] = prepare_draft_interfaces(config, torchair)
        report['status'] = 'PASS'
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    write_json(path, report)
    if report['status'] != 'PASS':
        raise RuntimeError(f'Draft-only interface preflight blocked; see {path}')
    return report


def select_draft(specs, native):
    from qwen35_dflash.ascend310p.incremental import DraftGraph, TargetRowsGraph
    candidates = [s for s in specs if s.name == 'draft']
    if len(candidates) != 1:
        raise ValueError('Draft-only requires exactly one original Draft spec')
    source = candidates[0]
    if type(source.model) is not DraftGraph or any(isinstance(m, TargetRowsGraph) for m in source.model.modules()):
        raise ValueError('Draft-only model scope contains Target or a replacement graph')
    expected = dict((s.torch_op, s.ge_op_type) for s in draft_operators(dict(
        draft_quantization='w8a16', draft_quant_matmul='weight_quant',
        adn_rms_norm_ge_op_type=native['metadata']['adn_rms_norm_ge_op_type'])))
    if len(source.custom_ops) != 3 or {s.torch_op: s.ge_op_type for s in source.custom_ops} != expected:
        raise ValueError('Draft-only spec includes non-Draft operator or misses a required operator')
    for key in ('incremental_contract', 'tensor_abi', *IDENTITIES):
        if key not in native['metadata'] or source.metadata.get(key) != native['metadata'][key]:
            raise ValueError('reloaded Draft/Target identity differs: ' + key)
    for key in ('input_names', 'output_names'):
        if list(getattr(source, key)) != native[key]:
            raise ValueError('Draft-only ordered ABI differs: ' + key)
    return source


def load_draft(config, native):
    from qwen35_dflash.ascend310p.quant_factory import create_quant_incremental_graphs
    # The original loader verifies source lock, ALL checkpoint/quant manifests,
    # original embedding/head and constructs the COMPLETE incremental contract.
    # No .prepare_export call, no Target forward, no fabricated GDR operator.
    import torch
    def no_forward(module, inputs):
        raise RuntimeError('Draft-only factory must not execute a model forward: ' + type(module).__name__)
    guard = torch.nn.modules.module.register_module_forward_pre_hook(no_forward)
    try:
        specs = create_quant_incremental_graphs(config)
    finally:
        guard.remove()
    return select_draft(specs, native)  # Other specs are released, never exported.


def audit_draft_air(air, folder):
    from qwen35_dflash.ascend310p.utils import count_ge_ir_nodes
    from qwen35_dflash.ascend310p.draft_ac_constants import validate_ac_graph
    graphs = air['graphs']
    if len(graphs) != 1 or graphs[0]['role'] != 'diagnostic':
        raise ValueError('Draft-only export must contain exactly one diagnostic Draft')
    graph = graphs[0]
    control = graph.get('metadata', {}).get('draft_ac', {})
    if not control.get('scope', '').startswith('static real Draft'):
        raise ValueError('Draft-only AIR scope missing')
    validate_ac_graph(graph)  # Exact 26 projections, Head, Const/NZ and 15/11 ABI.
    paths = sorted(Path(folder).rglob('dynamo.pbtxt'))
    if len(paths) != 1:
        raise ValueError('Draft-only requires exactly one GE IR for its sole graph')
    counts = count_ge_ir_nodes(paths)
    forbidden = {k: v for k, v in counts.items() if k in TARGET_GE and v}
    if forbidden:
        raise ValueError('Target operators reached Draft-only AIR: ' + str(forbidden))
    report = dict(status='PASS', scope='Draft-only AIR; ATC/NPU NOT_RUN',
                  ge_counts=counts, target_ge_counts=forbidden, ir=record(paths[0], Path(folder)))
    write_json(Path(folder)/'draft-scope.json', report)
    return report
