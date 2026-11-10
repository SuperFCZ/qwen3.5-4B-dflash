"""B-only transposed-storage NZ. Logical computation remains A[M,K] @ B[K,N]."""
import hashlib,math,time
import numpy as np
T,K,N=76,2560,256
SHAPE=(T,K//32,N//16,16,32)
BYTES=T*K*N

def pack_tile(b):
 if b.dtype!=np.int8 or b.shape!=(K,N):raise ValueError('exact INT8 KN Tile required')
 return np.ascontiguousarray(b.reshape(K//32,32,N//16,16).transpose(0,2,3,1))

def unpack_tile(z):
 if z.dtype!=np.int8 or z.shape!=SHAPE[1:]:raise ValueError('exact NZ physical Tile required')
 return np.ascontiguousarray(z.transpose(0,3,1,2)).reshape(K,N)

def offset(tile,k,n):
 if not(0<=tile<T and 0<=k<K and 0<=n<N):raise ValueError('coordinate outside logical B')
 return ((((tile*(K//32)+k//32)*(N//16)+n//16)*16+n%16)*32+k%32)

def prepack(source,dest):
 if source.stat().st_size!=BYTES or dest.exists():raise ValueError('source size or output exists')
 started=time.perf_counter_ns();copy_ns=verify_ns=read_ns=write_ns=0;rows=[]
 with source.open('rb') as inp,dest.open('xb') as out:
  for t in range(T):
   s=time.perf_counter_ns();raw=inp.read(K*N);read_ns+=time.perf_counter_ns()-s
   b=np.frombuffer(raw,'i1').reshape(K,N)
   s=time.perf_counter_ns();z=pack_tile(b);copy_ns+=time.perf_counter_ns()-s
   s=time.perf_counter_ns();back=unpack_tile(z)
   if back.tobytes()!=raw:raise ValueError('lossless inverse failed at Tile '+str(t))
   verify_ns+=time.perf_counter_ns()-s
   data=z.tobytes();s=time.perf_counter_ns();out.write(data);write_ns+=time.perf_counter_ns()-s
   rows.append(dict(tile=t,logical_sha256=hashlib.sha256(raw).hexdigest(),physical_sha256=hashlib.sha256(data).hexdigest()))
  out.flush()
 return dict(logical_shape=[T,K,N],stored_matrix_shape=[T,N,K],physical_shape=list(SHAPE),logical_elements=BYTES,physical_bytes=BYTES,padding_bytes=0,
  packing='B[t,k,n] -> NZ[t,k//32,n//16,n%16,k%32]; SetTensorB transpose=true',
  permutation_ms=copy_ns/1e6,read_ms=read_ns/1e6,write_ms=write_ns/1e6,inverse_verification_ms=verify_ns/1e6,
  prepare_wall_ms=(time.perf_counter_ns()-started)/1e6,io_note='flush completed; no fsync durability timing; SHA/verification distinct from permutation',tiles=rows,
  transfer_bytes_per_upload=BYTES,retained_old_plus_new_disk_bytes=2*BYTES,device='CPU_PREPACK_NOT_KERNEL_TIME',cache_note='Evidence hashing may warm OS cache; not a cold-disk benchmark')

def verify_pack(source,packed,record):
 if source.stat().st_size!=BYTES or packed.stat().st_size!=BYTES or record['physical_shape']!=list(SHAPE) or record['logical_shape']!=[T,K,N] or record['stored_matrix_shape']!=[T,N,K] or record['logical_elements']!=BYTES or record['physical_bytes']!=BYTES or len(record['tiles'])!=T or record['padding_bytes']!=0:raise ValueError('packing identity/shape differs')
 if any(type(record.get(k)) not in (int,float) or not math.isfinite(record[k]) or record[k]<0 for k in ('permutation_ms','read_ms','write_ms','inverse_verification_ms','prepare_wall_ms')):raise ValueError('invalid packing cost record')
 with source.open('rb') as inp,packed.open('rb') as out:
  for t in range(T):
   raw=inp.read(K*N);encoded=out.read(K*N);z=np.frombuffer(encoded,'i1').reshape(SHAPE[1:])
   want=dict(tile=t,logical_sha256=hashlib.sha256(raw).hexdigest(),physical_sha256=hashlib.sha256(encoded).hexdigest())
   if record['tiles'][t]!=want or unpack_tile(z).tobytes()!=raw:raise ValueError('B packing differs at Tile '+str(t))

def amortization(pack_ms,baseline_ms,candidate_ms):
 if any(not math.isfinite(x) or x<0 for x in (pack_ms,baseline_ms,candidate_ms)) or min(baseline_ms,candidate_ms)<=0:raise ValueError('invalid cost')
 saving=baseline_ms-candidate_ms
 return dict(per_call_saving_ms=saving,break_even_calls=math.ceil(pack_ms/saving) if saving>0 else None,
  status='MODEL_ONLY_EXCLUDES_UPLOAD_ONLINE_QUANT_AND_APPLICATION_COSTS' if saving>0 else 'NO_FINITE_AMORTIZATION_WITHOUT_PER_CALL_SAVING')
