"""Fixed-file identity checks; no SDK source snapshots or whole-tree scanning."""
import json
from pathlib import Path
from evidence import HERE,identity

def bind(cann):
 cann=Path(cann).resolve(strict=True);spec=json.loads((HERE/'sdk-review.json').read_text());rows=[]
 for row in spec['sources']:
  path=cann/'aarch64-linux/asc'/row['relative_path'];item=identity(path)
  if cann not in Path(item['path']).parents or item['sha256']!=row['sha256']:raise ValueError('SDK identity mismatch '+row['relative_path'])
  lines=Path(item['path']).read_text().splitlines()
  if not all(1<=lo<=hi<=len(lines) for lo,hi in row['lines']):raise ValueError('SDK line range mismatch')
  rows.append(dict(file=item,id=row['id'],lines=row['lines']))
 return dict(status='SOURCE_REVIEWED_FOR_SELECTED_SDK',files=rows,contract=identity(HERE/'sdk-review.json'),
  hardware_rounding='UNRESOLVED',host_feasible='NOT_RUN',npu='NOT_RUN')
