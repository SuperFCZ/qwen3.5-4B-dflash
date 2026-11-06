#!/usr/bin/env python3
"""Analyze all4096 Tile0 elements without changing Golden or granting admission."""
import argparse
from pathlib import Path
from evidence import load,identity,verify,write,fresh,sources
from stage4v_rounding_models import analyze


def main(args):
 root=fresh(args.output_dir,[args.frozen_root]);r=dict(status='NOT_RUN',cause='UNRESOLVED',sources=sources(),npu='NOT_RUN',benchmark_admission='PROHIBITED',replacement_golden='NOT_CREATED')
 try:
  data=load(args.frozen_root);r['evidence']=data['identities']
  r['frozen_state']=dict(status=data['frozen']['status'],npu=data['frozen'].get('npu'),error=data['frozen'].get('error'),checks=data['readonly_and_ub_guard'],repeat='NO_NEW_REPEAT_EVIDENCE')
  summary,all_rows,bad=analyze(data['dots'],data['words'],data['actual'],data['expected'])
  r.update(summary)
  r['dot_oracle']='CPU_INT64_RECOMPUTED_ALL4096_MATCH_STORED_I32'
  write(root/'elements.json',all_rows);write(root/'original-mismatches.json',bad)
  write(root/'model-summary.json',{k:v for k,v in summary.items() if k!='scale_fields'})
  r['details']={n:identity(root/n) for n in ('elements.json','original-mismatches.json','model-summary.json')}
 except Exception as e:r.update(status='BLOCKED_EVIDENCE_OR_MODEL_SCOPE',error=f'{type(e).__name__}: {e}')
 finally:
  try:
   for item in r['sources']:verify(item)
   if 'evidence' in r:
    from evidence import v
    v.verify_tree(r['evidence'])
  except Exception as e:r.update(status='BLOCKED_INTEGRITY',integrity_error=str(e))
  write(root/'analysis-report.json',r)
 print(r['status']+': '+str(root/'analysis-report.json'))
 return int(r['status'].startswith('BLOCKED'))

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--frozen-root',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True)
 raise SystemExit(main(ap.parse_args()))
