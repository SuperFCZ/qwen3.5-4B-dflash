"""Pinned CANN9.0.0 source identities; no snapshots and no SDK tree scan."""
import re
from pathlib import Path
from vdeq_common import HERE,raw,identity,verify,read,admission


def bind(cann):
 cann=cann.resolve(strict=True);spec=read(HERE/'sdk-review.json');rows=[]
 release=raw.v.p.host_o.bind(cann)
 for item in spec['sources']:
  path=cann/item['relative_path'];file=identity(path)
  if cann not in Path(file['path']).parents or file['sha256']!=item['sha256'] or file['bytes']!=item['bytes']:raise ValueError('SDK source mismatch '+item['id'])
  lines=path.read_text().splitlines()
  if not all(1<=lo<=hi<=len(lines) for lo,hi in item['lines']):raise ValueError('invalid source ranges')
  rows.append(dict(id=item['id'],file=file,lines=item['lines'],identity_scope='CURRENT_INSTALL_MATCHES_AUDITED_9_0_0_HASH',previous_server_reviewed_sha256=item['previous_server_reviewed_sha256']))
 return dict(status='SOURCE_REVIEWED',files=rows,release=release,manifest=identity(HERE/'sdk-review.json'),decision=admission(),
             limitation='Source inspection only; no claim about undocumented VDEQ intermediate arithmetic or device results')


def schema_names(binding):
 path=verify(next(x['file'] for x in binding['files'] if x['id']=='schema'))
 text=path.read_text();body=text.split('BEGIN_TILING_DATA_DEF(TCubeTiling)',1)[1].split('END_TILING_DATA_DEF',1)[0]
 declarations=re.findall(r'TILING_DATA_FIELD_DEF\((\w+),\s*(\w+)\);',body)
 if len(declarations)!=50 or len({n for _,n in declarations})!=50 or any(t!='int32_t' for t,_ in declarations):raise ValueError('unreviewed TCubeTiling schema')
 return [name for _,name in declarations]


def source_links(binding):
 return {x['id']:dict(path=x['file']['path'],sha256=x['file']['sha256'],lines=x['lines']) for x in binding['files']}
