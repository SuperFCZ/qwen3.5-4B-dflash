"""All serialized fields, SDK-derived offsets, and explicit formal coverage limits."""
import struct
from vdeq_common import verify,read


def decode(raw,schema):
 if len(raw)!=schema['bytes']:raise ValueError('serialized Tiling size mismatch')
 used=set();result={}
 for field in schema['fields']:
  n,o=field['name'],field['offset']
  if n in result or type(o)is not int or o<0 or o%4 or o+4>len(raw) or used.intersection(range(o,o+4)):raise ValueError('invalid Tiling field offsets')
  result[n]=struct.unpack_from('<i',raw,o)[0];used.update(range(o,o+4))
 padding=[i for i in range(len(raw)) if i not in used]
 if any(raw[i] for i in padding):raise ValueError('unexplained nonzero serialized padding')
 return result


def compare_fields(fp,raw,formal_fp,formal_raw):
 if set(fp)!=set(raw):raise ValueError('different schemas')
 for full,printed in ((fp,formal_fp),(raw,formal_raw)):
  if any(k not in full or full[k]!=v for k,v in printed.items()):raise ValueError('formal printed field differs from standalone serialization')
 out=[]
 for name in fp:
  same=fp[name]==raw[name]
  status='IDENTICAL' if same else ('CTYPE_WORKSPACE_RELATED_EXACT_DELTA_UNRESOLVED' if name in ('transLength','shareUbSize') else 'UNEXPLAINED_DIFFERENCE')
  out.append(dict(field=name,fp16=fp[name],raw_int32=raw[name],classification=status,
                  formal_field_recorded=name in formal_fp and name in formal_raw,
                  scope='STANDALONE_SERIALIZED_BOTH; formal equality only for emitted fields'))
 return dict(status='TILING_COMPARED_SCOPE_LIMITED',fields=out,field_count=len(out),
             formal_fields_missing=sorted(set(fp)-set(formal_fp).intersection(formal_raw)),
             formal_all_serialized_fields='UNRESOLVED_NOT_SAVED_BY_EXISTING_FORMAL_OP_REPORTS',
             interpretation='Equal scheduling fields do not prove equal CType, QuantProcessor or CopyCubeOut behavior')


def compare_evidence(data,schema):
 a=data['fp_host'];b=data['raw_standalone']
 fp=decode(verify(a['tiling_binary']).read_bytes(),schema);ri=decode(verify(b['tiling_binary']).read_bytes(),schema)
 fp_print=read(verify(a['host_query']))['cases'][0]['fields'];raw_print=read(verify(b['host_query']))['cases'][0]['fields']
 for decoded,printed in ((fp,fp_print),(ri,raw_print)):
  if any(k!='schema' and (k not in decoded or decoded[k]!=v) for k,v in printed.items()):raise ValueError('decoded query field mismatch')
 formal_fp=data['original']['frozen']['t1']['fields'];formal_raw=data['raw_host']['evidence']['t1']['fields']
 result=compare_fields(fp,ri,formal_fp,formal_raw)
 result['serialized_files']=dict(fp16=a['tiling_binary'],raw=b['tiling_binary'])
 result['formal_reports']=dict(fp16=data['original']['identities']['bounded'],raw=data['identities']['raw_host'])
 fp_admission=read(verify(data['original']['identities']['admission']))
 result['launch_contexts']=dict(fp16=dict(blocks=fp_admission['blocks'],workspace_bytes=fp_admission['workspace_bytes'],core_policy=fp_admission['core_policy']),raw=dict(blocks=data['raw_host']['formal_plan']['blocks'],workspace_bytes=data['raw_host']['workspace_bytes'],core_policy=data['raw_host']['evidence']['policy']))
 result['ctype']=dict(fp16='VECIN_NZ_half_with_SetQuantVector',raw='VECIN_NZ_int32_without_dequant')
 return result
