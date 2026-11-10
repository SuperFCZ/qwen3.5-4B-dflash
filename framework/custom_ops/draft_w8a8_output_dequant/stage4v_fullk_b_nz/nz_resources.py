"""Same conservative envelope, applied to actual B-NZ-transposed SDK output."""
from resources_v import validate as old_validate

def validate(platform,row):
 r=old_validate(platform,row)
 r['input_contract']='A_GM_ND_INT8_B_GM_NZ_INT8_TRANSPOSED_STORAGE_C_VECIN_NZ_FP16'
 r['details'].update(b_runtime_transdata='BYPASSED_SELECTED_IS_TRANS_TRUE_BRANCH',
  b_two_conversion_tiles_role='RETAINED_CONSERVATIVE_ENVELOPE_NOT_ALLOCATED_BY_NZ_DIRECT_COPY',
  b_physical_padding_bytes=0,b_direct_l1_block_bytes=32768,scope='Standalone resource candidate; Formal TilingContext still required')
 return r
