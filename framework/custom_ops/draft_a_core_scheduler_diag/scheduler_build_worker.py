import sys,traceback
from pathlib import Path
from scheduler_common import c
from scheduler_build import build


def main(path):
    q=c.read(path);out=Path(q['output']);out.mkdir(parents=True,exist_ok=False);r={'status':'BLOCKED','npu':'NOT_RUN'}
    try:c.verify_tree(q);r=build(out,q['prior'],q['variant'])
    except Exception as e:r.update(status='BLOCKED',error=str(e));(out/'traceback.txt').write_text(traceback.format_exc())
    finally:
        try:c.verify_tree(q);c.verify_tree(r)
        except Exception as e:r.update(status='BLOCKED',integrity_error=str(e))
        c.write(out/'report.json',r)
    return int(r['status']=='BLOCKED')


if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
