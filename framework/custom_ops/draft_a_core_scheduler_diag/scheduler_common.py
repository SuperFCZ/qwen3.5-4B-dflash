from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
PD=HERE.parent/'draft_a_platform_diag';sys.path.insert(0,str(PD))
from platform_common import c


def sources():
    return [c.identity(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]


def borrowed_sources():
    from platform_common import sources as old_sources
    return old_sources()
