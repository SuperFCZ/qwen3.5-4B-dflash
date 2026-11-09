"""Read-only imports. All diagnostic module names are stage-specific."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4v_raw_int32_probe'))
import raw_probe_common as raw
import run_raw_probe as raw_run
import stage4v_rounding_models as models
__all__=['raw_run','models']
identity,verify,read,write,fresh=raw.identity,raw.verify,raw.read,raw.write,raw.fresh
verify_tree=raw.v.verify_tree
BLOCKERS=[
 'UB_INT32_IS_NOT_A_VALID_SOURCE_FOR_VDEQ16_L0C_TO_UB_OVERLOAD',
 'UB_TO_L0C_WRAPPER_EXISTS_BUT_TRANSFER_EXTENT_AND_PUBLIC_SUPPORTED_PATH_CONTRACT_UNRESOLVED',
 'SDK_OVERFLOW_CHECK_EXPLICITLY_SKIPS_UB_TO_L0C',
 'FP16_MATMUL_GETTENSORC_DOES_NOT_EXPOSE_THE_SAME_L0C_AS_INT32_VIA_ITS_PUBLIC_DST_TYPED_API',
 'EXTERNAL_CO1_ITERATE_EXISTS_BUT_REQUIRES_NEW_HOST_TILING_AND_COMPLETION_CONTRACT',
]
KEY_DOTS={(0,84):-75057,(1,53):-83821,(1,201):-67885,(5,138):-89917,(7,235):-70957,(15,210):-73525}


def sources():return [identity(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]


def admission():
 return dict(status='BLOCKED',decision='BLOCKED_L0C_INJECTION_CONTRACT_UNRESOLVED',blockers=BLOCKERS,
             api_host_feasible='NOT_RUN',build='NOT_RUN',npu='NOT_RUN',opp='NOT_CREATED',runner='NOT_CREATED',
             direct_ub_vdeq16='UNSUPPORTED_BY_REVIEWED_OVERLOAD',l0c_injection='UNRESOLVED_NOT_PROVEN_UNSUPPORTED',
             external_co1_matmul='LOCAL_SYNTAX_ONLY_HOST_AND_COMPLETION_CONTRACT_PENDING',
             hardware_precision='UNRESOLVED',cause='UNRESOLVED',benchmark='PROHIBITED',golden='UNCHANGED')
