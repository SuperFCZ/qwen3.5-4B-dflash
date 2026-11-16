import sys
import traceback
from pathlib import Path
from sidecar_common import c
from host_build import build


def main(path):
    req=c.read(path);root=Path(req['output']);root.mkdir(parents=True,exist_ok=False)
    r=dict(status='BLOCKED',npu='NOT_RUN')
    try:
        c.verify_tree(req);r=build(root,req['prior'],req['variant'])
    except Exception as e:
        r.update(status='BLOCKED',error=f'{type(e).__name__}: {e}');(root/'traceback.txt').write_text(traceback.format_exc())
    finally:
        try:c.verify_tree(req);c.verify_tree(r)
        except Exception as e:r.update(status='BLOCKED',integrity_error=str(e))
        c.write(root/'report.json',r)
    return int(r['status']=='BLOCKED')


if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
