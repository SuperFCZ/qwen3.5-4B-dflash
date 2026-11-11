"""Exact selected-CANN vendor admission; frozen B-NZ private libraries stay forbidden."""
import os
from pathlib import Path
import real_mlp_common as c
from raw_probe_runtime_identity import confined, trace

VENDOR_PATHS = (
    'opp/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so',
    'opp/vendors/customize/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so',
    'opp/vendors/customize/op_api/lib/libcust_opapi.so',
)
POLICY = 'EXACT_THREE_SELECTED_CANN_VENDOR_DSOS_NO_PRIVATE_BNZ_V1'


def under(path, root):
    return path == root or root in path.parents


def bind_native_runtime(cann, normal_build):
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
    binding = dict(policy=POLICY, cann_root=str(cann),
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


def verify_native_binding(binding, cann):
    cann = Path(cann).resolve(strict=True)
    if binding['policy'] != POLICY or binding['cann_root'] != str(cann):
        raise ValueError('native DSO policy/CANN root changed: ' + str(cann))
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


def mapped_runtime(cann, binding, maps=Path('/proc/self/maps')):
    """Bind file-backed loaded DSOs, excluding ASLR addresses from cross-run identity."""
    cann = Path(cann).resolve(strict=True)
    verify_native_binding(binding, cann)
    driver = Path('/usr/local/Ascend/driver').resolve()
    rows = {}; mappings = {}
    for line in maps.read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6 or not parts[5].startswith('/'):
            continue
        raw = Path(parts[5])
        # Also forbid mapping the historical Runner itself, even without a .so suffix.
        private = binding['private_bnz']
        if '.so' not in raw.name and str(raw) != private['runner']['path']:
            continue
        try:
            major, minor = (int(v, 16) for v in parts[3].split(':'))
            inode = int(parts[4])
        except ValueError as exc:
            raise ValueError('invalid DSO loading identity: ' + str(raw)) from exc
        loaded = (major, minor, inode)
        if raw in mappings and mappings[raw] != loaded:
            raise ValueError('DSO mapping identity changed during scan: ' + str(raw))
        mappings[raw] = loaded
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
        if loaded != (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino):
            raise ValueError(f'mapped device/inode differs: alias={raw} realpath={real}')
        row = dict(alias=str(raw), realpath=str(real), name=raw.name, real_name=real.name,
                   file=item, symlink_chain=resolved['symlink_chain'], classification=role,
                   loading_identity=dict(device_major=major, device_minor=minor, inode=inode))
        rows[str(raw)] = row
    if not rows:raise ValueError('native shared-library map unavailable')
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
