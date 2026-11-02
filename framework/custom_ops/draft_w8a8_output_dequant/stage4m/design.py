"""CPU bit-coverage model only. No Ascend Kernel or hardware support claim."""
import numpy as np
GUARD=np.uint32(0x5a39c671)
BODY_BYTES=(226816,4096,4096,8192,8192)

def regions():
 return [dict(buffer=b,side=side,byte_offset=0 if side==0 else 256+size,words=64,mask_slot=32*(2*b+side)) for b,size in enumerate(BODY_BYTES) for side in (0,1)]

def equality_mask(words):
 a=np.asarray(words,dtype='<u4')
 # Bit reinterpretation, NOT numeric uint32 -> float conversion.
 return a.view('<f4')==np.asarray([GUARD],dtype='<u4').view('<f4')[0]

def check_group(words):
 a=np.asarray(words,dtype='<u4')
 if a.shape!=(10,64):raise ValueError('all ten prefix/suffix regions required')
 return bool(equality_mask(a).all())

def budget():
 return dict(scratch=226816,local_fp16=8192,cast_temporary=8192,sum=8192,guards=2560,reserve=8192,total=262144,
  proposed_mask_view=[0,320],owner='existing converted body AFTER Add completion',extra_allocation=0,
  lifetime_status='UNRESOLVED_DEVICE_COMPARE_WRITE_FOOTPRINT_AND_EVENT_PROOF',kernel='NOT_CREATED')
