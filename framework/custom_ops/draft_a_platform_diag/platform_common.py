from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
SIDECAR=HERE.parent/'draft_a_raw_sidecar';sys.path.insert(0,str(SIDECAR))
from sidecar_common import c,ec
__all__=['HERE','SIDECAR','c','ec','sources']


def sources():return [c.identity(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
