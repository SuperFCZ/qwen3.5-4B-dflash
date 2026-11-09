"""Fixed-file audit and prior evidence consumption; no device or library loading."""
from pathlib import Path
from co1_contract import HERE,prev,identity,verify,read


def sdk_binding(cann):
 cann=cann.resolve(strict=True);manifest=read(HERE/'sdk-review.json');rows=[]
 release=prev.raw.v.p.host_o.bind(cann)
 for item in manifest['sources']:
  p=cann/item['relative_path'];f=identity(p)
  if cann not in Path(f['path']).parents or f['sha256']!=item['sha256'] or f['bytes']!=item['bytes']:raise ValueError('SDK mismatch '+item['id'])
  lines=p.read_text().splitlines()
  if not all(1<=lo<=hi<=len(lines) for lo,hi in item['lines']):raise ValueError('SDK range mismatch')
  rows.append(dict(id=item['id'],file=f,lines=item['lines']))
 return dict(status='CURRENT_INSTALL_IDENTITIES_MATCH_AUDITED_CANN9',files=rows,release=release,manifest=identity(HERE/'sdk-review.json'),
             limitation='Local match is not server verification; run this same fixed-file check on server')


def formal_capability():
 return dict(status='BLOCKED',scope='COMPLETE_FORMAL_TCUBE_SERIALIZATION',
  current_public_boundary='aclnn GetWorkspaceSize exposes workspaceSize and opaque aclOpExecutor; reviewed acl_meta does not expose a TilingContext/serialized-TCube getter',
  callback_boundary='GetRawTilingData/GetPlatformInfo belong to the runtime-created TilingContext inside the registered callback',
  rejected_substitutes=['Standalone Manager values','ContextFaker/manual context with guessed resources','reading undocumented executor offsets','rebuilding or patching historical OPP'],
  possible_observer='A version-bound host-only serialization observer/interposer may be investigated separately; ABI, interception completeness, buffer lifetime and original callback return preservation are NOT verified here',
  verdict='No trusted full-context observation entry is currently implemented; not a proof that all possible observers are impossible',
  kernel_execution='NOT_RUN')


def prior_evidence(root,binding):
 review=read(root/'review/report.json');stage=read(root/'tiling/report.json')
 if review['status']!='SOURCE_REVIEWED' or stage['status']!='SAVED_TILING_COMPARED_FORMAL_SCOPE_PARTIAL':raise ValueError('prior review/Tiling evidence missing')
 if stage['review_report']!=identity(root/'review/report.json') or stage['sources']!=review['sources'] or stage['sdk']!=review['sdk']:raise ValueError('prior stage binding mismatch')
 previous={x['file']['path']:x['file'] for x in review['sdk']['files']}
 for x in binding['files']:
  if x['file']['path'] in previous and previous[x['file']['path']]!=x['file']:raise ValueError('prior/current SDK drift')
 result=read(verify(stage['tiling_comparison']));rows=result['fields']
 if len(rows)!=50 or len({r['field'] for r in rows})!=50 or result['field_count']!=50:raise ValueError('full saved schema required')
 a=prev.raw.diag.read(verify(stage['evidence']['frozen']['host']))
 b=prev.raw.diag.read(verify(stage['evidence']['raw_standalone']))
 # Re-decode the bound saved blobs. No values copied into missing formal fields.
 import tiling_decode
 schema=read(verify(review['serializer']['schema']))
 fa=tiling_decode.decode(verify(a['tiling_binary']).read_bytes(),schema)
 fb=tiling_decode.decode(verify(b['tiling_binary']).read_bytes(),schema)
 if any(r['fp16']!=fa[r['field']] or r['raw_int32']!=fb[r['field']] for r in rows):raise ValueError('saved-field report differs from binaries')
 differences=[dict(field=r['field'],fp16=r['fp16'],raw_int32=r['raw_int32']) for r in rows if r['fp16']!=r['raw_int32']]
 return dict(status='PRIOR_SAVED_TILING_IDENTITIES_RECHECKED',reports=dict(review=identity(root/'review/report.json'),tiling=identity(root/'tiling/report.json'),comparison=stage['tiling_comparison']),
  saved_differences=differences,user_reported_only_share_ub_confirmed=differences==[dict(field='shareUbSize',fp16=2048,raw_int32=0)],
  formal_missing=result['formal_fields_missing'],formal_50='BLOCKED_NOT_RECORDED',
  memory_interpretation=dict(reported_share_ub=dict(fp16=fa['shareUbSize'],raw=fb['shareUbSize']),effective_share_ub_dav2002=dict(fp16=0,raw=0),
   trans_length=dict(fp16=fa['transLength'],raw=fb['transLength'],meaning='UB scratch offset/extent, not GM Workspace'),
   share_l1=dict(fp16=fa['shareL1Size'],raw=fb['shareL1Size'],meaning='L1 input allocation depth footprint, not an additional copy of the same bytes'),
   formal_launch_contexts=result['launch_contexts'],scope='These VECIN figures are not CO1 resource admission'))
