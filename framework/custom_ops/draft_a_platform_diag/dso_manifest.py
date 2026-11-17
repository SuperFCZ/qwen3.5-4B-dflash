"""Exact DSO pins and Source/Build snapshots; no dlopen or NPU operations."""
import hashlib
import mmap
import os
from pathlib import Path
import re
import stat
from platform_common import c,HERE

POLICY=HERE/'cann_opp_dso_policy.json'
KINDS={'PRIVATE_DIAGNOSTIC_OPP','PINNED_CANN_STANDARD_OPP','STANDARD_CANN_RUNTIME','ASCEND_DRIVER','SYSTEM'}


def maps_rows(text):
    rows={}
    for line in text.splitlines():
        p=line.split(None,5)
        if len(p)!=6 or not p[5].startswith('/') or '.so' not in Path(p[5]).name:continue
        if p[5].endswith(' (deleted)'):raise ValueError('deleted loaded DSO: '+p[5])
        if not re.fullmatch('[0-9a-f]+-[0-9a-f]+',p[0]):raise ValueError('invalid DSO maps address')
        major,minor=(int(x,16) for x in p[3].split(':'))
        rows.setdefault(p[5],[]).append(dict(address=p[0],permissions=p[1],offset=int(p[2],16),device=[major,minor],inode=int(p[4])))
    return rows


def stat_record(st):
    return dict(device=[os.major(st.st_dev),os.minor(st.st_dev)],inode=st.st_ino,mode=stat.S_IMODE(st.st_mode))


def mount_namespace():return os.readlink('/proc/self/ns/mnt')


def readonly_backing(real,expected):
    """Reproduce this file's stat->VMA identity with PROT_READ only, never dlopen."""
    proc=Path('/proc/self');before={r['address'] for rows in maps_rows((proc/'maps').read_text()).values() for r in rows}
    with Path(real).open('rb') as f:
        st=os.fstat(f.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_size<=0:raise ValueError('invalid DSO file')
        with mmap.mmap(f.fileno(),0,flags=mmap.MAP_PRIVATE,prot=mmap.PROT_READ) as view:
            rows=maps_rows((proc/'maps').read_text());found=[]
            for path,segments in rows.items():
                if Path(path).resolve(strict=True)!=Path(real):continue
                for row in segments:
                    a,b=(int(x,16) for x in row['address'].split('-'))
                    if row['address'] not in before and row['offset']==0 and row['permissions']=='r--p' and b-a==(len(view)+mmap.PAGESIZE-1)//mmap.PAGESIZE*mmap.PAGESIZE:found.append(row)
            if len(found)!=1:raise ValueError('cannot identify unique readonly DSO mapping: '+str(real))
            if len(view)!=expected['bytes'] or hashlib.sha256(view).hexdigest()!=expected['sha256']:raise ValueError('readonly mapped bytes changed')
            row=found[0]
            if row['inode']!=st.st_ino:raise ValueError('unexplained mapped inode translation')
            result=dict(device=row['device'],inode=row['inode'],method='FD_MAP_PRIVATE_PROT_READ_SHA256',namespace=mount_namespace(),
                        scope='Controlled file VMA identity; not a digest of the prior ACLNN process memory')
        if stat_record(st)!=stat_record(os.fstat(f.fileno())):raise ValueError('DSO fd identity changed')
    return result


def file_snapshot(alias,owner):
    alias=Path(alias);owner_alias=Path(owner).absolute();owner=owner_alias.resolve(strict=True)
    if not alias.is_absolute() or (alias!=owner_alias and owner_alias not in alias.parents):raise ValueError('DSO alias outside allowed root: '+str(alias))
    real,links=c.alias_target(alias)
    if owner not in real.parents:raise ValueError('DSO realpath escapes allowed root: '+str(alias))
    st=real.stat();item=c.identity(real)
    if not stat.S_ISREG(st.st_mode):raise ValueError('DSO is not a regular file')
    with real.open('rb') as f:
        if f.read(4)!=b'\x7fELF':raise ValueError('DSO is not ELF: '+str(real))
    if stat_record(st)!=stat_record(real.stat()):raise ValueError('DSO changed during snapshot')
    return dict(alias=str(alias),links=links,file=item,stat=stat_record(st))


def record(alias,owner,kind):
    if kind not in KINDS:raise ValueError('unknown DSO category')
    row=file_snapshot(alias,owner);row.update(owner=str(Path(owner).absolute()),category=kind)
    row['backing']=readonly_backing(Path(row['file']['path']),row['file']);verify_record(row)
    return row


def verify_record(row):
    actual=file_snapshot(row['alias'],row['owner'])
    if any(actual[k]!=row[k] for k in ('alias','links','file','stat')):raise ValueError('DSO path/link/bytes/SHA/permissions/device/inode drift: '+row['alias'])
    if row['backing']['inode']!=row['stat']['inode']:raise ValueError('invalid backing inode')
    if mount_namespace()!=row['backing']['namespace']:raise ValueError('DSO mount namespace changed')


def source_manifest(cann,driver,reference):
    cann=Path(cann).resolve();driver=Path(driver).resolve();policy=c.read(POLICY);rows=[]
    for item in policy['standard_opp']:
        row=record(cann/item['relative'],cann,'PINNED_CANN_STANDARD_OPP')
        if row['file']['path']!=str(cann/item['relative']) or any(row['file'][k]!=item[k] for k in ('bytes','sha256')) or row['stat']['mode']!=item['mode']:
            raise ValueError('reviewed standard CANN OPP pin differs: '+item['relative'])
        rows.append(row)
    for name in policy['runtime_lib64']:
        # Two explicitly selected aliases, not a wildcard CANN-root admission.
        for rel in ('aarch64-linux/lib64','lib64'):rows.append(record(cann/rel/name,cann,'STANDARD_CANN_RUNTIME'))
    for name in policy['builtin_host']:
        rows.append(record(cann/'opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/aarch64'/name,cann,'STANDARD_CANN_RUNTIME'))
    for rel in policy['driver']:rows.append(record(driver/rel,driver,'ASCEND_DRIVER'))
    # Bind only OS DSOs actually recorded in the hash-checked old maps, not arbitrary user paths.
    old_private=Path(reference['private_opp']['install']).resolve()
    for alias in maps_rows(c.verify(reference['files']['maps.txt']).read_text()):
        p=Path(alias);real=p.resolve(strict=True)
        if any(root in real.parents for root in (cann,driver,old_private)):continue
        if 'libcust_op' in p.name or 'libcust_op' in real.name:raise ValueError('FOREIGN_PRIVATE_OPP: '+alias)
        owners=[Path(x).resolve() for x in ('/lib','/usr/lib') if Path(x).exists()]
        owner=next((x for x in owners if x in real.parents and (p==x or x in p.resolve().parents)),None)
        # Preserve absolute aliases such as /lib -> /usr/lib in the selected OS root.
        alias_owner=next((Path(x) for x in ('/lib','/usr/lib') if Path(x) in p.parents),None)
        if owner is None or alias_owner is None:raise ValueError('UNKNOWN non-system DSO: '+alias)
        rows.append(record(p,alias_owner,'SYSTEM'))
    result=dict(schema=1,kind='SOURCE_DSO_MANIFEST',policy=c.identity(POLICY),cann_root=str(cann),driver_root=str(driver),
                records=rows,reference_report=reference['report'],reference_maps=reference['files']['maps.txt'],hash_basis='Explicit reviewed CANN OPP pin; named runtime/Driver/observed OS DSOs sealed at new Source, not historical SDK hashes')
    verify_manifest(result);return result


def with_private(source,opp,source_identity):
    c.verify_opp(opp);result=dict(source,kind='BUILD_DSO_MANIFEST',source_manifest=source_identity,private_opp=opp,records=list(source['records']))
    root=Path(opp['install']).resolve();seen=set()
    for item in opp['preflight']['files']:
        if '.so' not in Path(item['path']).name:continue
        for alias in (item['path'],str(Path(item['path']).resolve())):
            if alias in seen:continue
            seen.add(alias);row=record(alias,root,'PRIVATE_DIAGNOSTIC_OPP')
            if row['file']['sha256']!=item['sha256']:raise ValueError('private Build DSO differs from OPP binding')
            result['records'].append(row)
    if not seen:raise ValueError('no Diagnostic private DSO')
    verify_manifest(result);return result


def verify_manifest(manifest):
    if manifest.get('schema')!=1 or manifest.get('kind') not in ('SOURCE_DSO_MANIFEST','BUILD_DSO_MANIFEST'):raise ValueError('DSO manifest schema')
    if manifest['policy']!=c.identity(POLICY):raise ValueError('DSO policy identity changed')
    c.verify_tree(manifest);aliases=set();policy=c.read(POLICY)
    cann=Path(manifest['cann_root']);driver=Path(manifest['driver_root'])
    approved={str(cann/x['relative']):'PINNED_CANN_STANDARD_OPP' for x in policy['standard_opp']}
    for name in policy['runtime_lib64']:
        for rel in ('aarch64-linux/lib64','lib64'):approved[str(cann/rel/name)]='STANDARD_CANN_RUNTIME'
    for name in policy['builtin_host']:approved[str(cann/'opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/aarch64'/name)]='STANDARD_CANN_RUNTIME'
    for rel in policy['driver']:approved[str(driver/rel)]='ASCEND_DRIVER'
    if any(manifest['reference_maps'][k]!=policy['reference_files']['maps.txt'][k] for k in ('bytes','sha256')):raise ValueError('DSO manifest reference maps pin changed')
    for path in maps_rows(c.verify(manifest['reference_maps']).read_text()):
        if any(Path(x) in Path(path).parents for x in ('/lib','/usr/lib')):approved[path]='SYSTEM'
    nonprivate={x['alias']:x['category'] for x in manifest['records'] if x['category']!='PRIVATE_DIAGNOSTIC_OPP'}
    if nonprivate!=approved:raise ValueError('DSO manifest adds or omits unreviewed paths')
    for row in manifest['records']:
        if row['alias'] in aliases:raise ValueError('duplicate DSO alias binding')
        aliases.add(row['alias']);verify_record(row)
        if row['category'] in ('PINNED_CANN_STANDARD_OPP','STANDARD_CANN_RUNTIME'):
            if row['owner']!=str(cann) or cann not in Path(row['file']['path']).parents or 'devlib' in Path(row['file']['path']).relative_to(cann).parts:raise ValueError('CANN DSO escapes selected runtime scope')
        elif row['category']=='ASCEND_DRIVER':
            if row['owner']!=str(driver) or driver not in Path(row['file']['path']).parents:raise ValueError('Driver DSO scope changed')
        elif row['category']=='SYSTEM':
            if row['owner'] not in ('/lib','/usr/lib') or 'libcust_op' in Path(row['file']['path']).name:raise ValueError('foreign private DSO classified as system')
        if row['category']=='PINNED_CANN_STANDARD_OPP':
            pin=next(x for x in policy['standard_opp'] if str(cann/x['relative'])==row['alias'])
            if any(row['file'][k]!=pin[k] for k in ('bytes','sha256')) or row['stat']['mode']!=pin['mode'] or row['file']['path']!=row['alias']:raise ValueError('standard OPP explicit identity pin changed')
        if row['category'] not in KINDS:raise ValueError('unknown DSO binding category')
    private_rows=[x for x in manifest['records'] if x['category']=='PRIVATE_DIAGNOSTIC_OPP']
    if manifest['kind']=='SOURCE_DSO_MANIFEST' and private_rows:raise ValueError('private DSO cannot enter Source standard manifest')
    if manifest['kind']=='BUILD_DSO_MANIFEST':
        root=Path(manifest['private_opp']['install']).resolve();allowed_private={}
        for item in manifest['private_opp']['preflight']['files']:
            if '.so' not in Path(item['path']).name:continue
            for alias in (item['path'],str(Path(item['path']).resolve())):allowed_private[alias]=item['sha256']
        if {x['alias'] for x in private_rows}!=set(allowed_private):raise ValueError('private DSO inventory differs from Build')
        for row in private_rows:
            if root not in Path(row['alias']).parents or root not in Path(row['file']['path']).parents or row['owner']!=str(root) or row['file']['sha256']!=allowed_private[row['alias']]:raise ValueError('private DSO identity outside Build binding')
        original=c.read(c.verify(manifest['source_manifest']))
        if [x for x in manifest['records'] if x['category']!='PRIVATE_DIAGNOSTIC_OPP']!=original['records']:raise ValueError('Source DSO manifest changed during Build')
        c.verify_opp(manifest['private_opp'])


def loaded(maps,opp,cann,manifest):
    verify_manifest(manifest);c.verify_opp(opp)
    if manifest.get('private_opp')!=opp or manifest['cann_root']!=str(Path(cann).resolve()):raise ValueError('loaded DSO manifest/OPP/CANN binding differs')
    inventory={x['alias']:x for x in manifest['records']};rows=[];private=False
    for alias,segments in sorted(maps_rows(Path(maps).read_text()).items()):
        row=inventory.get(alias)
        if row is None:raise ValueError('FOREIGN_PRIVATE_OPP / UNKNOWN unbound mapped DSO: '+alias)
        verify_record(row)
        ids={(tuple(x['device']),x['inode']) for x in segments}
        expected=(tuple(row['backing']['device']),row['backing']['inode'])
        if ids!={expected}:raise ValueError('mapped device/inode differs from bound file VMA: '+alias)
        private |= row['category']=='PRIVATE_DIAGNOSTIC_OPP'
        rows.append(dict(row,segments=segments,mapping_scope='Bound Source/Build file SHA and controlled fd VMA object identity; not runtime-memory SHA'))
    if not rows or not private:raise ValueError('Diagnostic OPP not present in mappings')
    return rows
