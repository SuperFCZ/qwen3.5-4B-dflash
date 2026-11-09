"""Identity-bound parent serialization slice; child SaveToBuffer is diagnostic only."""
import hashlib,struct
SCHEMA_SHA='f1688b61034f91c7518f65dcf73382ee9b007ab2c461fc8244cb6b883bd1d944'
HEADER_SHA='242c15b887d92f3c1d36bb927bb7540d1a9c4e626fdd3c5db1ca202590da57ab'
SOURCE='FORMAL_RAW_TILING_DATA_PARENT_SLICE'
METHOD='SDK_PARENT_AND_STANDALONE_MARKERS'
PREFIX_FORMAT='<QIiI4xQQQ'
OUTER_OFFSETS=dict(localMemSize=0,outputMode=8,tileId=12,launchBlocks=16,systemBytes=24,arenaBytes=32,arenaStride=40)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def bind_layout(proof):
 from echo_common import HERE,identity,verify,read
 schema_file=proof['schema_identity']
 if schema_file['sha256']!=SCHEMA_SHA or read(verify(schema_file))!=proof['schema']:raise ValueError('reviewed Schema identity mismatch')
 sdk={x['id']:x for x in proof['codec_and_schema_sdk']['files'] if x['id'] in ('schema','serialization')}
 if set(sdk)!={'schema','serialization'} or sdk['serialization']['file']['sha256']!=HEADER_SHA:raise ValueError('reviewed serializer SDK header mismatch')
 for x in sdk.values():verify(x['file'])
 return dict(method=METHOD,cube_payload_offset=48,cube_payload_bytes=200,op_serialization_bytes=248,
  cube_payload_source=SOURCE,schema_sha256=SCHEMA_SHA,schema_identity=schema_file,outer_field_offsets=OUTER_OFFSETS,
  sdk_sources=[dict(id=k,file=sdk[k]['file'],lines=[[75,103],[151,216],[240,249]] if k=='serialization' else sdk[k]['lines']) for k in sorted(sdk)],
  callback_sources=[identity(HERE/'op_host'/name) for name in ('host_common.h','nested_serialization.h','d_flash_external_co1_echo_probe.cpp','d_flash_external_co1_echo_probe_tiling.h')],
  scope='SDK parent marker self-check must run in each real callback; expected layout alone is insufficient')


def outer_bytes(row):
 return struct.pack(PREFIX_FORMAT,202240,0,0,1,row['system_workspace_bytes'],row['arena_bytes'],row['arena_bytes']+1024)


def verify_sdk_sources(entries):
 from echo_common import verify
 for entry in entries:verify(entry['file'])


def validate_serialization(row,op_binary,proof):
 from echo_common import identity,verify,read
 from tiling_decode import decode
 binding=proof['serialization_binding']
 required=dict(method=METHOD,cube_payload_offset=48,cube_payload_bytes=200,op_serialization_bytes=248,cube_payload_source=SOURCE,schema_sha256=SCHEMA_SHA,outer_field_offsets=OUTER_OFFSETS)
 if any(binding.get(k)!=v for k,v in required.items()):raise ValueError('serialization layout binding changed')
 if binding['schema_identity']!=proof['schema_identity'] or proof['schema_identity']['sha256']!=SCHEMA_SHA or read(verify(proof['schema_identity']))!=proof['schema']:raise ValueError('Schema SHA256/content mismatch')
 if row.get('schema_sha256')!=SCHEMA_SHA or row.get('layout_proof')!=METHOD or row.get('cube_payload_source')!=SOURCE or row.get('standalone_cube_role')!='DIAGNOSTIC_ONLY':raise ValueError('callback serialization provenance mismatch')
 for key,value in [('cube_payload_offset',48),('cube_payload_bytes',200),('op_serialization_bytes',248)]:
  if type(row.get(key))is not int or row[key]!=value:raise ValueError('callback serialization extent differs: '+key)
 if len(op_binary)!=248 or row['cube_payload_offset']+row['cube_payload_bytes']!=len(op_binary):raise ValueError('complete parent size/range mismatch')
 if op_binary[:48]!=outer_bytes(row):raise ValueError('outer48B does not match actual configured fields/padding')
 payload=op_binary[48:248]
 if decode(payload,proof['schema'])!=row['fields']:raise ValueError('nested parent TCube disagrees with50 getters')
 # Filesystem verification of SDK identities also runs through the stage's full
 # proof/integrity gate; this links callback bytes to that exact recorded source set.
 sdk={x['id']:x for x in proof['codec_and_schema_sdk']['files']}
 if {x['id'] for x in binding['sdk_sources']}!={'schema','serialization'}:raise ValueError('missing serializer SDK provenance')
 for entry in binding['sdk_sources']:
  if entry['file']!=sdk[entry['id']]['file']:raise ValueError('serializer SDK identity binding differs')
 if sdk['serialization']['file']['sha256']!=HEADER_SHA:raise ValueError('serializer header differs')
 verify_sdk_sources(binding['sdk_sources'])
 for f in binding['callback_sources']:
  if identity(f['path'])!=f:raise ValueError('callback source changed')
 return dict(cube_payload_offset=48,cube_payload_bytes=200,cube_payload_source=SOURCE,
             cube_payload_sha256=sha(payload),op_serialization_sha256=sha(op_binary),schema_sha256=SCHEMA_SHA,
             source_binding=binding)


def bind_capture(row,proof,execution):
 from echo_common import verify,identity,validate_callback
 if row.get('op_serialization') is None or row.get('cube_payload') is None or row.get('cube') is None:raise ValueError('capture file set incomplete')
 op=verify(row['op_serialization']).read_bytes();payload=verify(row['cube_payload']).read_bytes()
 diagnostic=verify(row['cube']).read_bytes()
 if len(diagnostic)!=200:raise ValueError('child diagnostic size changed')
 resources=validate_callback(row['data'],op,proof,execution)
 result=validate_serialization(row['data'],op,proof)
 if payload!=op[result['cube_payload_offset']:result['cube_payload_offset']+result['cube_payload_bytes']]:raise ValueError('captured payload differs from parent slice')
 result.update(op_serialization=row['op_serialization'],cube_payload=row['cube_payload'],callback_json=row['json'],
               standalone_cube=dict(role='DIAGNOSTIC_ONLY',file=row['cube'],all_zero=not any(diagnostic)))
 if identity(row['op_serialization']['path'])!=row['op_serialization']:raise ValueError('parent changed during validation')
 return resources,result


def compare_callbacks(rows):
 if not rows:raise ValueError('no callback capture')
 first=rows[0]
 for row in rows[1:]:
  if row['data']!=first['data']:raise ValueError('formal callback Getter/configuration drift')
  for key in ('op_serialization','cube_payload'):
   if row.get(key) is None or first.get(key) is None or (row[key]['bytes'],row[key]['sha256'])!=(first[key]['bytes'],first[key]['sha256']):raise ValueError('formal callback serialization drift')


def verify_serialization_record(record,proof):
 from echo_common import verify,read
 if record['source_binding']!=proof['serialization_binding']:raise ValueError('serialization source binding drift')
 op=verify(record['op_serialization']).read_bytes();payload=verify(record['cube_payload']).read_bytes()
 current=validate_serialization(read(verify(record['callback_json'])),op,proof)
 for key in ('cube_payload_offset','cube_payload_bytes','cube_payload_source','cube_payload_sha256','op_serialization_sha256','schema_sha256'):
  if record[key]!=current[key]:raise ValueError('serialization evidence field drift: '+key)
 if payload!=op[current['cube_payload_offset']:current['cube_payload_offset']+current['cube_payload_bytes']]:raise ValueError('serialization payload identity differs')
