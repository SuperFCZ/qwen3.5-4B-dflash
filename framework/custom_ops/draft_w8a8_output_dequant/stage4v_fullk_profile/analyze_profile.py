#!/usr/bin/env python3
"""Offline reparse of a sealed, safely completed capture; no profiler invocation."""
import argparse
from pathlib import Path
import profile_evidence as e
from profile_parse import collect
from profile_analysis import analyze,markdown
from run_profile import HERE


def main(a):
 r=e.read(a.report);e.verify_tree(r)
 if r.get('status') not in ('FULLK_PROFILE_COMPLETED','FULLK_PROFILE_COMPLETED_METRICS_PARTIAL') or r.get('capture_status')!='TWO_SAFE_FULL_EXECUTIONS_COMPLETED' or r.get('error') or r.get('integrity_error'):raise ValueError('only completed safe captures may be reanalyzed')
 if r['sources']!=e.sources():raise ValueError('analysis/source revision differs; do not silently rewrite old evidence')
 e.verify_bound(r['evidence'])
 if e.validate(r['run'],r['evidence']['proof'],r['device'])!=r['run']['validation']:raise ValueError('profile safety/numeric record changed')
 anchor=e.verify(r['evidence']['full_anchor']).read_bytes()
 if any(e.verify(x).read_bytes()!=anchor for x in r['run']['outputs']):raise ValueError('profile output drift')
 raw_root=Path(r['output_dir'])/'profiler'
 out=e.fresh(a.output_dir,[HERE,Path(r['output_dir'])])
 raw=collect(raw_root)
 if raw['files']!=r['raw_profile']['files']:raise ValueError('profile artifact inventory changed')
 report=analyze(raw,r['source_counts'],r['evidence']['benchmark_statistics']);report['profile_report']=e.identity(a.report)
 e.write(out/'metrics-report.json',raw);e.write(out/'analysis-report.json',report)
 (out/'BOTTLENECK_REPORT.md').write_text(markdown(report,raw))
 return 0

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
 raise SystemExit(main(p.parse_args()))
