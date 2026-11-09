"""Read-only Host ldd classification. Driver identities are never SDK identities."""
import os,re
from pathlib import Path
from raw_probe_common import identity,verify,read,v
from artifact_recheck import host_elf

DRIVER_ROOT=Path('/usr/local/Ascend/driver')
SYSTEM_ROOTS=(Path('/lib'),Path('/lib64'),Path('/usr/lib'),Path('/usr/lib64'))
DRIVER_SONAME='libascend_hal.so'
VERSION_FILES=('version.info','driver/version.info')


def inside(path,roots):return any(path==r or r in path.parents for r in roots)


def trace(path):
 """Resolve all components, including directory symlinks, without loading a DSO."""
 path=Path(os.path.abspath(path));pending=list(path.parts[1:]);current=Path('/');links=[]
 while pending:
  part=pending.pop(0)
  if part=='.':continue
  if part=='..':current=current.parent;continue
  current=current/part
  if current.is_symlink():
   if len(links)>=40:raise ValueError('runtime symlink cycle/too many links')
   text=os.readlink(current);target=Path(text) if Path(text).is_absolute() else current.parent/text
   links.append(dict(path=str(current),target=text,absolute_target=os.path.abspath(target)))
   pending=list(target.parts[1:])+pending;current=Path('/')
  elif not current.exists():raise ValueError('missing runtime path: '+str(current))
 return dict(requested_path=str(path),real_path=str(current),symlink_chain=links)


def confined(path,roots):
 result=trace(path);real=Path(result['real_path'])
 if not inside(real,roots):raise ValueError('runtime symlink escapes allowed root: '+str(path))
 for link in result['symlink_chain']:
  if inside(Path(link['path']),roots) and not inside(Path(link['absolute_target']),roots):
   raise ValueError('runtime symlink hop escapes allowed root: '+str(path))
 if not real.is_file() or not real.stat().st_size:raise ValueError('empty/non-file runtime dependency')
 result['file']=identity(real);return result


def dependencies(text):
 rows=[]
 for line in text.splitlines():
  line=line.strip()
  if not line:continue
  # vDSO has no on-disk identity; only the conventional address-only record is skipped.
  if re.fullmatch(r'linux-vdso\.so\.\d+\s+\(0x[0-9a-fA-F]+\)',line):continue
  match=re.fullmatch(r'(\S+)\s+=>\s+(/\S+)\s+\(0x[0-9a-fA-F]+\)',line)
  if match:rows.append((match[1],Path(match[2])));continue
  match=re.fullmatch(r'(/\S+)\s+\(0x[0-9a-fA-F]+\)',line)
  if match:rows.append((Path(match[1]).name,Path(match[1])));continue
  raise ValueError('unresolved/unrecognized Host ldd record: '+line)
 if not rows:raise ValueError('empty Host ldd dependencies')
 return rows


def versions(root):
 rows=[]
 for name in VERSION_FILES:
  path=root/name
  if not os.path.lexists(path):continue
  entry=confined(path,(root,))
  if entry['file']['bytes']>65536:raise ValueError('Driver version record exceeds 64KiB')
  text=Path(entry['real_path']).read_text(encoding='utf-8')
  entry['version_fields']={}
  for line in text.splitlines():
   match=re.fullmatch(r'\s*([A-Za-z0-9_.-]*[Vv]ersion[A-Za-z0-9_.-]*|[Vv]ersion)\s*[:=]\s*(.*?)\s*',line)
   if match:
    key,value=match.groups()
    if key in entry['version_fields'] and entry['version_fields'][key]!=value:raise ValueError('conflicting Driver version field')
    entry['version_fields'][key]=value
  verify(entry['file']);rows.append(entry)
 return dict(status='RECORDED' if any(x['version_fields'] for x in rows) else 'UNRESOLVED',files=rows,
             reason=None if any(x['version_fields'] for x in rows) else 'No parseable version field in fixed Driver version locations; no inferred version')


def history_records(proof):
 """Only two already-bound reports; no filesystem/log search or recursive report discovery."""
 result=[]
 for label,item in [('preflight',proof['preflight']),('frozen_machine',proof['frozen']['bounded'])]:
  report=read(verify(item));result.append(dict(label=label,report=item,content=report))
 return result


def historical_comparison(history,driver_root,driver_rows,version):
 matches=[];binary_found=False;version_found=False
 current={x['file']['path']:x['file'] for x in driver_rows+version['files']}
 # Recognize direct or {file: identity, text: ...} records by identity shape,
 # bounded to each selected report (no following arbitrary linked reports).
 def walk(value,where,depth=0):
  if depth>64:raise ValueError('historical machine nesting too deep')
  if isinstance(value,dict):
   if {'path','bytes','sha256'}<=set(value):
    path=Path(value['path'])
    if inside(path,(driver_root,)):
     old={k:value[k] for k in ('path','bytes','sha256')}
     verify(old);now=identity(path)
     if now!=old:raise ValueError('historical Driver identity changed')
     if path.name==DRIVER_SONAME or str(path) in current:
      if now not in current.values():raise ValueError('historical Driver does not match currently loaded binding')
      matches.append(dict(location=where,file=old,kind='binary' if path.name==DRIVER_SONAME or now in [x['file'] for x in driver_rows] else 'version'))
   else:
    for key,item in value.items():walk(item,where+'/'+key,depth+1)
  elif isinstance(value,list):
   for i,item in enumerate(value):walk(item,where+'/'+str(i),depth+1)
 for record in history:walk(record['content'],record['label'])
 binary_found=any(x['kind']=='binary' for x in matches);version_found=any(x['kind']=='version' for x in matches)
 return dict(binary_status='MATCHED' if binary_found else 'UNRESOLVED_NO_HISTORICAL_DRIVER_BINARY_IDENTITY',
             version_status='MATCHED' if version_found else 'UNRESOLVED_NO_HISTORICAL_DRIVER_VERSION_IDENTITY',matches=matches)


def collect(log,cann,history,driver_root=DRIVER_ROOT,system_roots=SYSTEM_ROOTS):
 cann=Path(cann).resolve(strict=True);driver_root=Path(driver_root).absolute()
 # The confirmed root itself must not redirect to a different installation.
 root_trace=trace(driver_root)
 if Path(root_trace['real_path'])!=driver_root:raise ValueError('confirmed Driver root redirects elsewhere')
 if inside(driver_root,(cann,)) or inside(cann,(driver_root,)):raise ValueError('Driver and Toolkit roots overlap')
 system_roots=tuple(Path(x).absolute() for x in system_roots)
 system_allowed=tuple(set(system_roots+tuple(x.resolve() for x in system_roots)))
 log_id=identity(log);rows=dependencies(Path(log).read_text());verify(log_id)
 sdk=[];driver=[];system=[]
 for needed,path in rows:
  if needed==DRIVER_SONAME and not inside(path,(driver_root,)):raise ValueError('HAL must bind the confirmed Driver root, not Toolkit/system')
  if inside(path,(cann,)):dest=sdk;entry=confined(path,(cann,))
  elif inside(path,(driver_root,)):
   if needed!=DRIVER_SONAME or path.name!=DRIVER_SONAME:raise ValueError('unknown Driver library: '+needed)
   entry=confined(path,(driver_root,));info,_=host_elf(Path(entry['real_path']))
   names=[s['value'] for s in info['dynamic_strings'] if s['name']=='SONAME']
   if names!=[DRIVER_SONAME]:raise ValueError('unknown/missing Driver ELF SONAME')
   entry.update(soname=names[0],elf=info['elf']);dest=driver
  elif inside(path,system_roots):dest=system;entry=confined(path,system_allowed)
  else:raise ValueError('runtime library outside selected Toolkit/Driver/system roots: '+str(path))
  entry['needed']=needed;dest.append(entry)
 if len(driver)>1:raise ValueError('ambiguous multiple HAL resolutions')
 ver=versions(driver_root);comparison=historical_comparison(history,driver_root,driver,ver)
 binding=dict(status='BOUND_INDEPENDENT_DRIVER_IDENTITY' if driver else 'NO_DRIVER_LIBRARY_IN_HOST_LDD',root=root_trace,
              allowlist=[DRIVER_SONAME],version=ver,historical_comparison=comparison,
              history=[dict(label=x['label'],report=x['report']) for x in history],
              meaning='Separate Driver binding; neither CANN SDK identity nor NPU acceptance')
 result=dict(sdk_libraries=sdk,driver_libraries=driver,system_libraries=system,driver_binding=binding,
             runtime_ldd=log_id,runtime_cann_root=str(cann),runtime_system_roots=list(map(str,system_roots)))
 verify_runtime(result);return result


def verify_runtime(record):
 v.verify_tree(record)
 root=record['driver_binding']['root']
 if trace(root['requested_path'])!=root:raise ValueError('Driver root identity/alias changed')
 for category in ('sdk_libraries','driver_libraries','system_libraries'):
  for entry in record[category]:
   actual=trace(entry['requested_path'])
   if any(actual[k]!=entry[k] for k in ('requested_path','real_path','symlink_chain')) or identity(actual['real_path'])!=entry['file']:raise ValueError('runtime library identity/alias changed')
 if versions(Path(root['real_path']))!=record['driver_binding']['version']:raise ValueError('Driver version identity changed')
