"""Compact terminal handoff; reads only the selected new report."""
import json,sys
from pathlib import Path


def main(path):
    r=json.loads(Path(path).read_text())
    result={k:r.get(k) for k in ('status','phase','error','integrity_error','npu','compile','benchmark','profile','adoption')}
    e=r.get('evidence')
    if e:
        result['identities']={
            'H_OM':{g:x['om'] for g,x in e['H']['gears'].items()},
            'H_legacy_raw_flag':e['historical_a_raw_flag'],
            'B_R':{v:{'header':e[v]['header'],'kernel_files':e[v]['kernel_files'],
                'configuration':{k:e[v]['configuration'].get(k) for k in ('gate_prefetch_mode','gate_copy_mode','k_tile','scale_mode','core_limit')}} for v in ('B','R')},
            'projection_cases':{g:len(x) for g,x in e['projections'].items()},
            'source_lock':e['H']['source_lock']['status']}
    if 'comparison' in r:result['comparison']=r['comparison']
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main(sys.argv[1])
