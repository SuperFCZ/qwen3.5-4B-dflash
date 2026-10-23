"""Shared identities for this independent W8A8 experiment."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
QUANT=HERE.parent/'draft_quant'
OPS={'raw':('DFlashW8A8Accumulator','d_flash_w8_a8_accumulator'),
     'quant':('DFlashW8A8Fixpipe','d_flash_w8_a8_fixpipe')}

def identity(path):
    p=Path(path).resolve(strict=True);h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())

def verify(item):
    if identity(item['path'])!=item:raise ValueError('identity changed: '+item['path'])
    return Path(item['path'])

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def fresh(path,protected=()):
    p=Path(path).absolute()
    if p.resolve()!=p:raise ValueError('output traverses symlink')
    for item in (HERE.parent.parent.parent,*protected):
        r=Path(item).resolve()
        if p==r or p in r.parents or r in p.parents:raise ValueError('output overlaps protected path: '+str(r))
    p.mkdir(parents=True,exist_ok=False);p.chmod(0o700);return p

def sources():
    return [identity(p) for p in sorted(HERE.rglob('*')) if p.is_file() and
            '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
