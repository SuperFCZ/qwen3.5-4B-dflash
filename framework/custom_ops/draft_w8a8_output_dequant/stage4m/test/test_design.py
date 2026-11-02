import sys
from pathlib import Path
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from design import GUARD,regions,equality_mask,check_group,budget

def test_every_word_and_bit_at_every_group():
 # Every one of 640 positions × 32 bits × 20 boundaries is covered in the CPU model.
 for group in range(20):
  for bit in range(32):
   batch=np.full((640,640),GUARD,dtype='<u4')
   batch[np.arange(640),np.arange(640)]^=np.uint32(1<<bit)
   mask=equality_mask(batch)
   assert np.all(np.count_nonzero(~mask,axis=1)==1)
   assert not np.any(mask.all(axis=1))

def test_multibit_special_values_and_numeric_cast_counterexample():
 variants=np.array([0,0x80000000,0x7f800000,0xff800000,0x7fc00001,0x7f800001,1,0xffffffff,GUARD],'<u4')
 assert equality_mask(variants).tolist()==[False]*8+[True]
 assert np.float32(GUARD)==np.float32(GUARD^np.uint32(1)) # lossy numeric cast would miss corruption
 rng=np.random.default_rng(17);words=rng.integers(0,2**32,size=100000,dtype=np.uint32)
 assert np.array_equal(equality_mask(words),words==GUARD)

def test_transient_requires_each_group_check():
 guards=np.full((10,64),GUARD,'<u4');assert check_group(guards)
 guards[9,63]^=np.uint32(0xffffffff);assert not check_group(guards)
 guards[9,63]=GUARD;assert check_group(guards) # tile-only check misses restored damage
 with pytest.raises(ValueError):check_group(guards[:9])

def test_storage_map_no_reserve_or_guard_overlap():
 spans=regions();assert len(spans)==10
 assert {x['mask_slot'] for x in spans}==set(range(0,320,32))
 assert all(x['byte_offset']%32==0 for x in spans)
 b=budget();assert sum(b[k] for k in ('scratch','local_fp16','cast_temporary','sum','guards','reserve'))==262144
 assert b['extra_allocation']==0 and b['proposed_mask_view'][1]<b['cast_temporary']
 assert b['lifetime_status'].startswith('UNRESOLVED')
