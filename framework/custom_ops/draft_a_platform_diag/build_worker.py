import sys,traceback
from pathlib import Path
from platform_common import c
from build_host import build


def main(path):
    q=c.read(path);root=Path(q['output']);root.mkdir(parents=True,exist_ok=False);r={'status':'BLOCKED','npu':'NOT_RUN'}
    try:c.verify_tree(q);r=build(root,q['prior'],q['variant'])
    except Exception as e:r.update(status='BLOCKED',error=str(e));(root/'traceback.txt').write_text(traceback.format_exc())
    finally:
        try:c.verify_tree(q);c.verify_tree(r)
        except Exception as e:r.update(status='BLOCKED',integrity_error=str(e))
        c.write(root/'report.json',r)
    return int(r['status']=='BLOCKED')


if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
