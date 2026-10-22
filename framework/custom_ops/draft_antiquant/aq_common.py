"""Independent AntiQuant experiment; no device imports in CPU stages."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def verify(item):
    if identity(item['path']) != item:
        raise ValueError('identity changed: ' + item['path'])
    return Path(item['path'])


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def fresh(path, protected=()):
    path = Path(path).absolute()
    if path.resolve() != path:
        raise ValueError('output must not traverse symlinks')
    for item in (HERE.parent.parent.parent, *protected):
        p = Path(item).resolve()
        if path == p or p in path.parents or path in p.parents:
            raise ValueError('output overlaps read-only source/evidence: ' + str(p))
    path.mkdir(parents=True, exist_ok=False)
    path.chmod(0o700)
    return path


def experiment_sources():
    return [identity(p) for p in sorted(HERE.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
