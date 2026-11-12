"""Exact selected-CANN vendor admission; frozen B-NZ private libraries stay forbidden."""
import os
import errno
import hashlib
import json
import mmap
import re
from pathlib import Path
import real_mlp_common as c
from raw_probe_runtime_identity import confined, trace

VENDOR_PATHS = (
    'opp/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so',
    'opp/vendors/customize/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so',
    'opp/vendors/customize/op_api/lib/libcust_opapi.so',
)
POLICY = 'EXACT_THREE_SELECTED_CANN_VENDOR_DSOS_MOUNT_PROOF_V2'


def under(path, root):
    return path == root or root in path.parents


def bind_native_runtime(cann, normal_build, proc=Path('/proc/self')):
    """Called only after historical Build/runtime verification, before Torch import.

    These vendor hashes are newly observed in the selected installation. They
    are not asserted to be historical SDK source-lock hashes. Freeze them for
    this Smoke and require the identical binding for Benchmark.
    """
    cann = Path(cann).resolve(strict=True)
    opp = trace(normal_build['opp']); root = Path(opp['real_path'])
    if under(root, cann) or under(cann, root):
        raise ValueError('private B-NZ OPP overlaps selected CANN: ' + str(root))
    installed = normal_build['installed']
    items = [installed['api'], installed['host']]
    items += [f for f in installed['all_files'] if '.so' in Path(f['path']).name]
    libs = {str(c.verify(f)): f for f in items}
    if any(not under(Path(p), root) for p in libs):
        raise ValueError('historical B-NZ private library escapes OPP: ' + repr(sorted(libs)))
    host_alias = trace(installed['host_alias'])
    if host_alias['real_path'] != installed['host']['path']:
        raise ValueError('historical B-NZ Host alias differs: ' + installed['host_alias'])
    c.verify(normal_build['runner'])
    binding = dict(policy=POLICY, cann_root=str(cann), mount_environment=mount_snapshot(proc)[0],
        vendor_hash_basis='OBSERVED_SELECTED_CANN_BEFORE_TORCH_IMPORT_NOT_HISTORICAL_SDK_HASH',
        private_bnz=dict(opp=opp, host_alias=host_alias, runner=normal_build['runner'],
                         libraries=[libs[p] for p in sorted(libs)]), vendor_libraries=[])
    for rel in VENDOR_PATHS:
        entry = confined(cann / rel, (cann,))
        entry.update(relative_path=rel, name=Path(rel).name)
        reject_private(Path(entry['requested_path']), entry['file'], binding)
        binding['vendor_libraries'].append(entry)
    return binding


def reject_private(alias, item, binding):
    private = binding['private_bnz']; real = Path(item['path'])
    roots = [Path(private['opp'][k]) for k in ('requested_path', 'real_path')]
    if any(under(alias, root) or under(real, root) for root in roots):
        raise ValueError(f'private B-NZ OPP loaded: alias={alias} realpath={real}')
    for frozen in private['libraries'] + [private['runner']]:
        if (str(alias) == frozen['path'] or str(real) == frozen['path'] or
                (item['bytes'], item['sha256']) == (frozen['bytes'], frozen['sha256'])):
            raise ValueError(f'private B-NZ identity/copy loaded: alias={alias} realpath={real} frozen={frozen["path"]}')


def verify_native_binding(binding, cann, proc=Path('/proc/self')):
    cann = Path(cann).resolve(strict=True)
    if binding['policy'] != POLICY or binding['cann_root'] != str(cann):
        raise ValueError('native DSO policy/CANN root changed: ' + str(cann))
    current_environment=mount_snapshot(proc)[0]
    if binding['mount_environment']!=current_environment:
        raise ValueError('frozen mount environment changed: '+json.dumps(dict(expected=binding['mount_environment'],actual=current_environment),sort_keys=True))
    if [v['relative_path'] for v in binding['vendor_libraries']] != list(VENDOR_PATHS):
        raise ValueError('native vendor exact-path inventory changed')
    private = binding['private_bnz']
    for key in ('opp', 'host_alias'):
        if trace(private[key]['requested_path']) != private[key]:
            raise ValueError('private B-NZ path/alias drift: ' + private[key]['requested_path'])
    c.verify(private['runner'])
    for f in private['libraries']:c.verify(f)
    for expected in binding['vendor_libraries']:
        alias = cann / expected['relative_path']
        current = confined(alias, (cann,))
        current.update(relative_path=expected['relative_path'], name=alias.name)
        if current != expected:
            raise ValueError('selected CANN vendor identity/alias drift: ' + str(alias))
        reject_private(alias, current['file'], binding)


def stat_id(st):
    return (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino)


def stable_stat(st):
    return (*stat_id(st), st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def mount_snapshot(proc):
    raw = (proc/'mountinfo').read_bytes()
    namespace = os.readlink(proc/'ns/mnt')
    records = []
    def unescape(s):return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), s)
    for line in raw.decode().splitlines():
        left, right = line.split(' - ', 1); a = left.split(); b = right.split()
        if len(a) < 6 or len(b) != 3:raise ValueError('invalid mountinfo record: ' + line)
        major, minor = map(int, a[2].split(':'))  # mountinfo decimal; maps hexadecimal
        records.append(dict(id=int(a[0]), parent=int(a[1]), device=[major,minor],
            root=unescape(a[3]), point=unescape(a[4]), options=a[5], optional=a[6:],
            fstype=b[0], source=unescape(b[1]), super_options=b[2]))
    if not records:raise ValueError('empty mountinfo')
    return dict(namespace=namespace, kernel_release=os.uname().release,
                mountinfo_sha256=hashlib.sha256(raw).hexdigest()), records


def covering_mount(real, records):
    eligible = [r for r in records if under(real, Path(r['point']))]
    if not eligible:raise ValueError('no covering mount: ' + str(real))
    depth = max(len(Path(r['point']).parts) for r in eligible)
    top = [r for r in eligible if len(Path(r['point']).parts) == depth]
    if len(top) != 1:raise ValueError('ambiguous stacked covering mounts: ' + str(real))
    return top[0]


def map_rows(text):
    rows = {}
    for line in text.splitlines():
        p = line.split(maxsplit=5)
        if len(p) != 6 or not p[5].startswith('/'):continue
        if not re.fullmatch(r'[0-9a-f]+-[0-9a-f]+',p[0]):raise ValueError('invalid maps range')
        major,minor=(int(v,16) for v in p[3].split(':'))
        row=dict(range=p[0],permissions=p[1],offset=int(p[2],16),loaded=(major,minor,int(p[4])),path=p[5])
        if p[0] in rows and rows[p[0]] != row:raise ValueError('DSO mapping identity changed during scan: '+p[5])
        rows[p[0]]=row
    return rows


def fd_digest(fd):
    h=hashlib.sha256();offset=0
    while True:
        block=os.pread(fd,1024*1024,offset)
        if not block:break
        h.update(block);offset+=len(block)
    return dict(bytes=offset,sha256=h.hexdigest())


def probe_readonly_mapping(fd, real, proc):
    """Measure this fd's VMA backing identity; never PROT_EXEC, writable or NPU."""
    before=map_rows((proc/'maps').read_text())
    size=os.fstat(fd).st_size
    if size<=0:raise ValueError('empty mapped DSO')
    with mmap.mmap(fd,0,flags=mmap.MAP_PRIVATE,prot=mmap.PROT_READ) as view:
        after=map_rows((proc/'maps').read_text())
        found=[]
        for key,row in after.items():
            if key in before or row['offset'] or row['permissions']!='r--p':continue
            start,end=(int(x,16) for x in key.split('-'))
            if end-start != (size+mmap.PAGESIZE-1)//mmap.PAGESIZE*mmap.PAGESIZE:continue
            if row['path'].endswith(' (deleted)'):continue
            if Path(row['path']).resolve(strict=True)==real:found.append(row)
        if len(found)!=1:raise ValueError('cannot uniquely identify read-only mmap probe: '+str(real))
        return dict(loaded=list(found[0]['loaded']),bytes=len(view),sha256=hashlib.sha256(view).hexdigest(),
                    flags='MAP_PRIVATE_PROT_READ_NO_EXEC_NO_WRITE')


def mismatch_proof(real, item, loaded, segments, proc, mount):
    """No device-number allowlist: each discrepant file needs a live witness."""
    path_stat=real.stat()
    proof=dict(mapped=list(loaded),path_stat=list(stat_id(path_stat)),mount=mount,
               map_files_readlink='NOT_RUN',map_files_open='NOT_RUN',mapped_content='NOT_VERIFIED')
    try:
        if path_stat.st_ino!=loaded[2]:raise ValueError('mapped device/inode differs (inode mismatch)')
        targets=[]
        for segment in segments:
            target=os.readlink(proc/'map_files'/segment['range'])
            if target.endswith(' (deleted)'):raise ValueError('deleted map_files target')
            if not Path(target).is_absolute() or Path(target).resolve(strict=True)!=real:
                raise ValueError('map_files target differs: '+target)
            targets.append(target)
        proof.update(map_files_readlink='MATCHED_ALL_SEGMENTS',targets=sorted(set(targets)))
        link=proc/'map_files'/segments[0]['range']
        try:fd=os.open(link,os.O_RDONLY|os.O_CLOEXEC)
        except OSError as exc:
            proof.update(map_files_open='UNAVAILABLE',map_files_errno=exc.errno)
            if exc.errno not in (errno.EPERM,errno.EACCES):raise
        else:
            try:
                first=os.fstat(fd);content=fd_digest(fd)
                proof.update(map_files_open='OPENED',map_files_fstat=list(stat_id(first)),map_files_content=content)
                if content!={k:item[k] for k in ('bytes','sha256')}:
                    raise ValueError('map_files content differs from bound pathname')
                if stable_stat(first)!=stable_stat(os.fstat(fd)):raise ValueError('map_files changed during hashing')
                if stat_id(first)==loaded:
                    proof.update(status='MAP_FILES_FD_CONTENT_MATCH',mapped_content='FD_SHA256_MATCH')
                elif stat_id(first)==stat_id(path_stat):
                    proof['map_files_open']='PATH_VIEW_ONLY_NOT_BACKING_IDENTITY'
                else:raise ValueError('unexplained map_files fstat identity')
            finally:os.close(fd)
        if 'status' not in proof:
            # A mount label and matching inode are NOT proof. Demonstrate the
            # per-file stat->VMA conversion on an fd whose complete bytes match.
            if mount['fstype']!='overlay' or mount['device']!=list(stat_id(path_stat)[:2]):
                raise ValueError('unexplained device translation; no eligible overlay mount')
            fd=os.open(real,os.O_RDONLY|os.O_CLOEXEC)
            try:
                if stable_stat(os.fstat(fd))!=stable_stat(path_stat):raise ValueError('pathname fd changed')
                witness=probe_readonly_mapping(fd,real,proc);proof['readonly_probe']=witness
                if tuple(witness['loaded'])!=loaded:raise ValueError('probe backing device/inode differs')
                if any(witness[k]!=item[k] for k in ('bytes','sha256')):raise ValueError('probe mapped content differs')
                if stable_stat(os.fstat(fd))!=stable_stat(path_stat):raise ValueError('probe fd changed')
            finally:os.close(fd)
            proof.update(status='OVERLAY_TRANSLATION_PROVEN_BY_READONLY_MMAP',
                         mapped_content='READONLY_PROBE_SHA256_MATCH_ORIGINAL_VMA_OBJECT_ID')
        if stable_stat(real.stat())!=stable_stat(path_stat):raise ValueError('DSO pathname changed during proof')
        current=map_rows((proc/'maps').read_text())
        if any(current.get(s['range'])!=s for s in segments):raise ValueError('original VMA changed during proof')
        return proof
    except (OSError,ValueError) as exc:
        proof.update(status='BLOCKED',reason=str(exc),path=str(real),errno=getattr(exc,'errno',None))
        raise ValueError('DSO mount identity BLOCKED: '+json.dumps(proof,sort_keys=True)) from exc


def mapped_runtime(cann, binding, maps=Path('/proc/self/maps')):
    """Bind file-backed loaded DSOs, excluding ASLR addresses from cross-run identity."""
    cann = Path(cann).resolve(strict=True)
    verify_native_binding(binding, cann, maps.parent)
    driver = Path('/usr/local/Ascend/driver').resolve()
    proc=maps.parent;environment,mounts=mount_snapshot(proc)
    rows = {}; mappings = {}; segments = {}
    initial=map_rows(maps.read_text())
    for segment in initial.values():
        raw = Path(segment['path'])
        # Also forbid mapping the historical Runner itself, even without a .so suffix.
        private = binding['private_bnz']
        if '.so' not in raw.name and str(raw) != private['runner']['path']:
            continue
        if segment['path'].endswith(' (deleted)'):raise ValueError('deleted DSO mapping: '+str(raw))
        loaded = segment['loaded']
        if raw in mappings and mappings[raw] != loaded:
            raise ValueError('DSO mapping identity changed during scan: ' + str(raw))
        mappings[raw] = loaded
        segments.setdefault(raw,[]).append(segment)
    # Multiple mappings of one DSO are normal; hash each pathname only once.
    for raw, loaded in mappings.items():
        resolved = trace(raw); real = Path(resolved['real_path'])
        item = c.identity(real)
        reject_private(raw, item, binding)
        if under(raw, cann) or under(real, cann) or str(raw).startswith('/usr/local/Ascend/') or str(real).startswith('/usr/local/Ascend/'):
            if not under(real, cann) and not under(real, driver):
                raise ValueError(f'native process resolved another Ascend installation: alias={raw} realpath={real}')
        matches = [v for v in binding['vendor_libraries']
                   if str(raw) in (v['requested_path'], v['real_path'])]
        role = 'RECORDED_NON_VENDOR_DSO'
        if matches:
            if len(matches) != 1 or item != matches[0]['file']:
                raise ValueError(f'loaded CANN vendor identity differs: alias={raw} realpath={real}')
            # Includes all symlink hops, not just the final destination.
            confined(raw, (cann,))
            role = 'EXACT_SELECTED_CANN_VENDOR'
        elif 'vendors' in raw.parts or 'vendors' in real.parts:
            raise ValueError(f'unknown vendor DSO: alias={raw} realpath={real}')
        # /proc maps device+inode identifies the mapped file, rather than merely
        # hashing a replacement that now happens to occupy the pathname.
        stat = real.stat(); major, minor, inode = loaded
        mount=covering_mount(real,mounts)
        if stat.st_ino!=inode:raise ValueError(f'mapped device/inode differs: alias={raw} realpath={real}')
        if loaded != stat_id(stat):
            proof=mismatch_proof(real,item,loaded,segments[raw],proc,mount)
        else:proof=dict(status='DIRECT_DEVICE_INODE_MATCH',path_stat=list(stat_id(stat)))
        row = dict(alias=str(raw), realpath=str(real), name=raw.name, real_name=real.name,
                   file=item, symlink_chain=resolved['symlink_chain'], classification=role,
                   loading_identity=dict(device_major=major, device_minor=minor, inode=inode),
                   mount_environment=dict(environment,covering_mount=mount),mount_proof=proof)
        rows[str(raw)] = row
    if not rows:raise ValueError('native shared-library map unavailable')
    if mount_snapshot(proc)[0]!=environment:raise ValueError('mount namespace/table changed during DSO scan')
    return [rows[k] for k in sorted(rows)]


def compare_loaded(current, admitted, complete=False):
    """Before timing allow only the already-loaded subset of Smoke; finally exact set."""
    expected = {r['alias']: r for r in admitted}
    for row in current:
        if expected.get(row['alias']) != row:
            raise ValueError('native loaded DSO differs from Smoke: ' + row['alias'])
    if complete and {r['alias'] for r in current} != set(expected):
        missing = sorted(set(expected) - {r['alias'] for r in current})
        raise ValueError('native loaded DSO inventory differs from Smoke; missing=' + repr(missing))
