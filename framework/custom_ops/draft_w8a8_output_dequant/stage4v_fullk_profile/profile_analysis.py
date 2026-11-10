"""Evidence-qualified bottleneck report; ranking is not a causal time decomposition."""
import json


def recommendations(metrics):
 available=metrics['availability']
 specs=[
 ('B_ND_TO_NZ','B input conversion',('mte2','vec','memory'),
  'GM ND INT8 input uses SDK conversion path; Full-K makes B layout a distinct experiment variable.',
  'Investigate offline B-only NZ packing with independently checked layout and SDK Host Tiling. Keep A/Scale/FP16 conversion semantics.',
  'May remove input conversion/control work; magnitude UNRESOLVED.',
  'Changed layout/Tiling may alter data mapping, workspace or accumulation order. Require INT32 oracle and existing output/safety evidence.',
  'One independently built B-layout candidate versus unchanged normal Full-K; new 3x30 paired samples, all mismatches preserved.'),
 ('LIFECYCLE','Matmul lifecycle and synchronization',('scalar','duration'),
  '76 Iterate/GetTensorC/End calls and one Matmul object per launched Core; counter activity is not an API cost measurement.',
  'Identify a specific redundant state setup only after selected SDK lifetime proof; do not remove End, barriers or events.',
  'Potential control/wait reduction; magnitude UNRESOLVED.',
  'Scratch, L0 and coefficient lifetime; a larger scalar_time is not proof of removable wait.',
  'One proven state-management change; preserve all sync and Guard coverage, strict new paired measurement.'),
 ('GUARD_ARENA','Guard and Arena overhead',('scalar',),
  'Source has87552 UB Guard GetValue words plus21248 Arena GetValue words for7 Blocks; source counts are not instruction counts.',
  'Investigate an equivalent-coverage check implementation only after interface/coverage proof. Do not revive rejected bitmap APIs.',
  'May reduce control/checking work; magnitude UNRESOLVED.',
  'Transient corruption coverage must remain; weaker diagnostics cannot become production candidates.',
  'Independent equivalent-coverage candidate, CPU/device fault coverage before paired latency; no simultaneous layout changes.'),
 ('NZ_TO_ND','Final output writes',('mte3','scalar'),
  '19456 source-level32B calls write622592B. Historical Stage4S grouped-K strided-write test did not meet its material-gain threshold; different compute path.',
  'A one-variable strided writeback study is only justified by new evidence; preserve exact NZ/ND geometry and synchronization.',
  'Might reduce issue overhead; no claimed MTE3 or wall-time saving.',
  'Stride units, per-row output coverage, cross-Tile overlap. MTE3 includes other data movement.',
  'Only final writeback changed, all elements/poison/guards checked, new Full-K paired3x30 samples. Do not reuse Stage4S timing.'),
 ('SCHEDULING','Core/Tile balance',('duration',),
  'Static ownership is11/11/11/11/11/11/10 Tiles. Per-Core completion times may be unavailable.',
  'Change assignment only if same-task Core records show a repeatable tail; preserve the admitted7-Block policy.',
  'Potential tail reduction UNRESOLVED; tile-count imbalance is not measured tail.',
  'Output overlap, missed Tiles, different memory locality; no extra Core assumption.',
  'One assignment change with full coverage proof and paired timing, no prefetch/overlap changes.')]
 result=[]
 for i,(key,name,counters,evidence,principle,impact,risk,ab) in enumerate(specs,1):
  counter_evidence=[dict(file=g['file'],row_type=g['row_type'],task_type=g['task_type'],names=g['names'],identifiers=g['identifiers'],field=k,**v) for g in metrics.get('groups',[]) for k,v in g['metrics'].items() if v['category'] in counters]
  result.append(dict(investigation_rank=i,id=key,name=name,counter_availability={k:available[k] for k in counters},counter_evidence=counter_evidence,evidence=evidence,principle=principle,expected_impact=impact,risk=risk,minimal_ab=ab,
   implementation_go='NOT_GRANTED',ranking_basis='PROVISIONAL_SOURCE_AND_HISTORICAL_EVIDENCE_NOT_MEASURED_BENEFIT'))
 return result


def analyze(metrics,counts,benchmark):
 return dict(status='EVIDENCE_REPORT_GENERATED',numeric_status='NUMERIC_MISMATCH_UNRESOLVED',production='NOT_APPROVED',
  unprofiled_benchmark=benchmark,profile_availability=metrics['availability'],source_counts=counts,
  bottleneck='UNRESOLVED_NO_CAUSAL_PHASE_DECOMPOSITION',
  first_investigation='B_ND_TO_NZ',first_implementation='UNRESOLVED_PENDING_REAL_PROFILE_AND_SDK_LAYOUT_RESOURCE_PROOF',
  recommendations=recommendations(metrics),
  boundaries=['No Pipe busy-time sum','No Host Execute+Sync subtraction to infer device phase time','No mixed CSV/task/core/unit medians','No counter ranking claimed as critical-path ranking','No improvement claim or production approval'])


def markdown(report,metrics):
 lines=['# Full-K profile evidence and optimization review','',
  'Numerics: **NUMERIC_MISMATCH_UNRESOLVED**. No Kernel change. Profile times are excluded from the uninstrumented Benchmark.','',
  '## Actual device evidence','',f'Task coverage: `{metrics["task_coverage"]}`. Per-Core evidence: `{metrics["core_tail"]}`.','',
  '| Counter category | Availability |','|---|---|']
 lines.extend('|'+k+'|'+v+'|' for k,v in metrics['availability'].items())
 lines+=['','## Separate CSV groups','']
 for i,g in enumerate(metrics['groups']):
  lines += [f'Group{i}: `{g["file"]["path"]}`; SHA256 `{g["file"]["sha256"]}`.',
   f'Kernel/Op `{json.dumps(g["names"],sort_keys=True)}`; type `{g["file_kind"]}/{g["row_type"]}/{g["task_type"]}`; identifiers `{json.dumps(g["identifiers"],sort_keys=True)}`.',
   f'Records `{g["records"]}`; duplicate records excluded `{g["duplicate_records"]}`.','',
   '| Original field | Unit | Count | Median | Min | Max |','|---|---|---:|---:|---:|---:|']
  for k,v in g['metrics'].items():lines.append(f'|{k}|{v["unit"]}|{v["count"]}|{v["median"]}|{v["min"]}|{v["max"]}|')
  if not g['metrics']:lines.append('|No safely summarized metrics|UNAVAILABLE|0||||')
  lines.append('')
 lines += ['## Interpretation','',
  'Raw busy counters support activity observations only. ND→NZ, VDEQ16, final NZ→ND writes, End/sync and Guard/Arena costs are not independently measured by PipeUtilization. Per-phase attribution remains UNAVAILABLE without PC/source or equivalent controlled evidence.',
  'Static Tile assignment and19456 DataCopy source calls are not dynamic instruction counts or measured bottlenecks. Timeline records and units are preserved separately; displayTimeUnit is not used as timestamp-unit proof.','',
  '## Provisional investigation priority, not implementation approval','']
 for x in report['recommendations']:
  lines += [f'{x["investigation_rank"]}. **{x["name"]}**',f'   Evidence: {x["evidence"]}',f'   Principle: {x["principle"]}',f'   Expected effect: {x["expected_impact"]}',f'   Risk: {x["risk"]}',f'   Minimal A/B: {x["minimal_ab"]}','']
 lines += ['First bounded investigation: **B-only ND→NZ input conversion**. The first implementation is **UNRESOLVED** until actual Profile and SDK layout/resource evidence support a safe single-variable candidate. No ranking of measured gains is claimed.','']
 return '\n'.join(lines)
