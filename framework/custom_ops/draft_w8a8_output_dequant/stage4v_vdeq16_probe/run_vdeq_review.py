#!/usr/bin/env python3
"""Bounded source review, complete saved-Tiling comparison and frozen numeric analysis."""
import argparse,os
from pathlib import Path
import vdeq_common as c
from vdeq_sdk import bind


def main(a):
 root=c.fresh(a.output_dir,[a.cann_root]+[p for p in (a.frozen_root,a.raw_root) if p is not None])
 r=dict(status='NOT_RUN',stage=a.stage,sources=c.sources(),npu='NOT_RUN',cause='UNRESOLVED',benchmark='PROHIBITED')
 try:
  if a.stage in ('host','build','bounded'):
   r.update(c.admission());r['reason']='Execution gates are intentionally closed; no OPP, device Runner or launch code exists'
  else:
   r['sdk']=bind(a.cann_root)
   if a.stage=='review':
    from schema_tool import build
    r['serializer']=build(root/'serializer',a.cann_root,r['sdk'])
    r.update(status='SOURCE_REVIEWED',api_admission=c.admission())
   else:
    previous=c.read(a.review_report)
    if previous['status']!='SOURCE_REVIEWED' or previous['sources']!=r['sources'] or previous['sdk']!=r['sdk']:raise ValueError('source review identity changed')
    r['review_report']=c.identity(a.review_report)
    from vdeq_evidence import load
    data=load(a.frozen_root,a.raw_root,a.cann_root);r['evidence']=data['identities']
    schema=c.read(c.verify(previous['serializer']['schema']))
    from tiling_decode import compare_evidence
    tiling=compare_evidence(data,schema);c.write(root/'tiling-comparison.json',tiling)
    r['tiling_comparison']=c.identity(root/'tiling-comparison.json')
    if a.stage=='analyze':
     from vdeq_analysis import analyze
     original=data['original'];summary,elements,bad=analyze(data['dots'],original['words'],original['actual'],original['expected'])
     for name,value in [('numeric-summary.json',summary),('elements.json',elements),('six-positions.json',summary['key_positions']),('original-mismatches.json',bad)]:c.write(root/name,value)
     r['analysis']={n:c.identity(root/n) for n in ('numeric-summary.json','elements.json','six-positions.json','original-mismatches.json')}
     r.update(status='FROZEN_RAW_AND_FP16_ANALYZED_CAUSE_UNRESOLVED',api_admission=c.admission())
    else:r.update(status='SAVED_TILING_COMPARED_FORMAL_SCOPE_PARTIAL',api_admission=c.admission())
 except Exception as e:r.update(status='BLOCKED',error=f'{type(e).__name__}: {e}')
 finally:
  try:
   c.verify_tree(r)
   if r['sources']!=c.sources():raise ValueError('source inventory changed')
   if 'sdk' in r and bind(a.cann_root)!=r['sdk']:raise ValueError('SDK identity changed')
  except Exception as e:r.update(status='BLOCKED',integrity_error=str(e))
  c.write(root/'report.json',r)
 print(r['status']+': '+str(root/'report.json'));return int(r['status']=='BLOCKED')


def parse(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('review','tiling','analyze','host','build','bounded'))
 p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
 for flag in ('review-report','frozen-root','raw-root'):p.add_argument('--'+flag,type=Path)
 a=p.parse_args(argv)
 if a.stage in ('tiling','analyze') and any(getattr(a,x) is None for x in ('review_report','frozen_root','raw_root')):p.error('tiling/analyze require --review-report --frozen-root --raw-root')
 for key in ('output_dir','cann_root','review_report','frozen_root','raw_root'):
  if getattr(a,key) is not None:setattr(a,key,getattr(a,key).resolve())
 return a

if __name__=='__main__':raise SystemExit(main(parse()))
