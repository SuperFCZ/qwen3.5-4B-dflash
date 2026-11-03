"""Bounded native header selection. Aliases are not independent SDK versions."""
from pathlib import Path
from reference_r import q
SERVER_MM_SHA256='212c8d5565c54e0d78c13cff27f1150d13a5500b1859fd38a4319c96b8bb7f81'
LOCATIONS=('include/aclnnop/aclnn_mm.h','aarch64-linux/include/aclnnop/aclnn_mm.h',
 'opp/built-in/op_impl/ai_core/tbe/op_api/include/aclnn_mm.h')

def describe(path,cann):
 path=path.absolute();real=path.resolve(strict=True)
 if cann not in real.parents or not real.is_file() or real.name!='aclnn_mm.h':raise ValueError('native header outside CANN_ROOT or invalid file')
 st=real.stat()
 if not 0<st.st_size<=1024*1024:raise ValueError('invalid native header size')
 item=q.identity(real);q.verify(item)
 return dict(alias=str(path),resolved=str(real),is_symlink=path.is_symlink(),resolves_differently=path!=real,
             device=st.st_dev,inode=st.st_ino,file=item)

def select_header(cann,explicit=None):
 cann=Path(cann).resolve(strict=True);rows=[];missing=[]
 paths=[cann/rel for rel in LOCATIONS]
 if explicit is not None:
  explicit=Path(explicit).absolute()
  if explicit not in paths:paths.append(explicit)
 for path in paths:
  if not path.exists():
   if path.is_symlink() or explicit is not None and path==explicit:raise ValueError('missing/broken explicitly selected or linked header: '+str(path))
   missing.append(str(path));continue
  rows.append(describe(path,cann))
 if not rows:raise ValueError('native aclnn_mm.h missing in fixed public locations')
 groups={(row['device'],row['inode']) for row in rows}
 if explicit is None:
  if len(groups)!=1:raise ValueError('independent native headers found; use --native-mm-header explicitly; candidates='+repr(rows))
  chosen=min(rows,key=lambda row:row['resolved'])
  selection='ONE_FILE_WITH_ALIASES'
 else:
  chosen=next(row for row in rows if row['alias']==str(explicit));selection='EXPLICIT_FILE'
 if chosen['file']['sha256']!=SERVER_MM_SHA256:raise ValueError('selected native header SHA256 differs from reviewed server header')
 result=dict(status='REVIEWED_NATIVE_HEADER_BOUND',selection=selection,selected=chosen['file'],selected_alias=chosen['alias'],
             reviewed_server_sha256=SERVER_MM_SHA256,candidates=rows,missing=missing,distinct_files=len(groups),
             all_candidate_contents_identical=len({row['file']['sha256'] for row in rows})==1)
 verify_binding(result,cann);return result

def verify_binding(binding,cann):
 cann=Path(cann).resolve(strict=True)
 if binding.get('status')!='REVIEWED_NATIVE_HEADER_BOUND' or not binding.get('candidates'):raise ValueError('incomplete header binding')
 if binding['selected'] not in [row['file'] for row in binding['candidates']] or not any(row['alias']==binding['selected_alias'] and row['file']==binding['selected'] for row in binding['candidates']):raise ValueError('selected header not among recorded aliases')
 if cann not in Path(binding['selected']['path']).resolve(strict=True).parents:raise ValueError('selected header escaped CANN_ROOT')
 if binding['reviewed_server_sha256']!=SERVER_MM_SHA256 or binding['selected']['sha256']!=SERVER_MM_SHA256:raise ValueError('header review binding changed')
 for row in binding['candidates']:
  if describe(Path(row['alias']),cann)!=row:raise ValueError('native header alias/identity changed')
 q.verify(binding['selected'])
