#!/usr/bin/env python3
"""Read-only Stage4L Profile provenance; no mixed medians or Pipe sums."""
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4l'))
from profile_rows import review
from l_common import NAMES,read,identity,write,fresh

def analyze(path):
 r=read(path)
 if r.get('status')!='STAGE4L_PROFILE_COMPLETED' or set(r.get('raw_profile',{}))!=set(NAMES):raise ValueError('complete four-variant Stage4L Profile required')
 variants={}
 for name,raw in r['raw_profile'].items():
  data=review(raw);groups={}
  for row in data['rows']:
   key=json.dumps([row['file']['path'],row['row_type'],row['units'],sorted(row['raw_record'].keys())],sort_keys=True)
   group=groups.setdefault(key,dict(file=row['file'],row_type=row['row_type'],units=row['units'],count=0,duplicate_count=0,records=[],examples=[]))
   group['count']+=1;group['duplicate_count']+=int(row['duplicate_of'] is not None);group['records'].append(row['csv_record_number'])
   if len(group['examples'])<2:group['examples'].append(row['raw_record'])
  variants[name]=dict(groups=list(groups.values()),availability=raw['availability'],rows=data['rows'])
 return dict(status='PROVENANCE_ONLY_NO_CAUSAL_AGGREGATION',source=identity(path),variants=variants,
  profile_support='UNRESOLVED_UNTIL_ACTUAL_ROWS_REVIEWED',critical_path='UNRESOLVED',pipe_time_sum='FORBIDDEN')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 out=fresh(a.output_dir,[a.report.parent.parent]);r=analyze(a.report);write(out/'profile-review.json',r)
 summary=dict(r);summary['variants']={k:{'groups':v['groups'],'availability':v['availability']} for k,v in r['variants'].items()}
 write(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
