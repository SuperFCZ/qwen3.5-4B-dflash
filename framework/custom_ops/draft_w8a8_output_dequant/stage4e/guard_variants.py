"""Exact single-variable deletions relative to the accepted Stage4D B source."""
import difflib,hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4d'))
import guard_variant as d_guard
FROZEN=d_guard.FROZEN;KERNEL=d_guard.KERNEL
TILE_SCAN=d_guard.SCAN.replace('        if','    if').replace('           !','       !')
ARENA_CALL='    if(!CheckArena(pipe,converted.ReinterpretCast<uint32_t>(),coreArena,td.arenaBytes))return;\n'
# Keep the identical per-side transfer, PIPE_ALL, event fetch and Set/Wait operations.
# The original CheckArena helper remains unchanged for the pre-loop admission check.
ARENA_NO_SCAN='''    { // DIAGNOSTIC ONLY: retain Arena transfers/synchronization, omit Scalar validation.
        auto temp=converted.ReinterpretCast<uint32_t>();auto arena=coreArena;const uint64_t bytes=td.arenaBytes;
        for(uint32_t side=0;side<2;++side) {
            DataCopy(temp,arena[side?(512+bytes)/4:0],128);PipeBarrier<PIPE_ALL>();
            const auto e=pipe.FetchEventID(HardEvent::MTE2_S);
            SetFlag<HardEvent::MTE2_S>(e);WaitFlag<HardEvent::MTE2_S>(e);
        }
    }
'''
POLICIES={
 'B0':'DIAGNOSTIC_ONLY_WEAKER_GUARD_COVERAGE',
 'B1':'DIAGNOSTIC_ONLY_NO_TILE_FIVE_UB_SCANS',
 'B2':'DIAGNOSTIC_ONLY_NO_TILE_ARENA_SCALAR_SCAN',
}
COVERAGE={v:dict(group_five_ub_scans=False,tile_five_ub_scans=v!='B1',
 tile_arena_scalar_scan=v!='B2',tile_arena_transfer_and_sync=True,initial_arena_scan=True,
 final_p0_cast_scans=True,guard_initialization=True,runner_gm_guards=True,readonly_poison_repeat=True,
 missed_errors=('Transient group UB corruption restored before tile end can escape.' +
 (' Scratch/p1/sum corruption is no longer checked at tile end; final p0/cast checks remain.' if v=='B1' else '')+
 (' Internal Arena corruption after the initial check can escape; external allocation guards do not replace it.' if v=='B2' else '')))
 for v in POLICIES}
REMOVED={'B0':0,'B1':304*5*64*2,'B2':304*2*128}

def digest(data):return hashlib.sha256(data).hexdigest()

def transform(b0,variant):
 if variant not in POLICIES:raise ValueError('unknown variant')
 s=b0.decode()
 if s.count(TILE_SCAN)!=1 or s.count(ARENA_CALL)!=2:raise ValueError('unexpected tile guard source span')
 anchor='SetFlag<HardEvent::V_S>(ready);WaitFlag<HardEvent::V_S>(ready);\n'
 if anchor+TILE_SCAN+ARENA_CALL not in s:raise ValueError('not audited tile-tail guard location')
 if variant=='B1':return s.replace(TILE_SCAN,'',1).encode()
 if variant=='B2':return s.replace(anchor+TILE_SCAN+ARENA_CALL,anchor+TILE_SCAN+ARENA_NO_SCAN,1).encode()
 return b0

def expected_originals():
 return [FROZEN/'operator.json']+[p for sub in ('op_host','op_kernel') for p in sorted((FROZEN/sub).iterdir()) if p.is_file()]

def diff(a,b,v):
 return ''.join(difflib.unified_diff(a.decode().splitlines(True),b.decode().splitlines(True),fromfile='B0/'+KERNEL,tofile=v+'/'+KERNEL))

def materialize(root,identity,write,accepted_b):
 root.mkdir();original=(FROZEN/KERNEL).read_bytes();b0=d_guard.transform(original,'B')
 if Path(accepted_b['path']).read_bytes()!=b0 or identity(accepted_b['path'])!=accepted_b:raise ValueError('Stage4D B source identity differs')
 variants={};diffs={}
 for v in POLICIES:
  target=root/v;target.mkdir();files=[]
  for p in expected_originals():
   rel=p.relative_to(FROZEN);out=target/rel;out.parent.mkdir(parents=True,exist_ok=True)
   out.write_bytes(transform(b0,v) if str(rel)==KERNEL else p.read_bytes());files.append(identity(out))
  variants[v]=dict(source_dir=str(target),files=files,kernel=identity(target/KERNEL),safety_protocol=POLICIES[v])
  path=root/(v+'-only.diff');path.write_text(diff(b0,transform(b0,v),v));diffs[v]=identity(path)
 audit=dict(status='EXACT_TILE_GUARD_SOURCE_DIFFS',originals=[identity(p) for p in expected_originals()],
  accepted_stage4d_b=accepted_b,variants=variants,diffs=diffs,coverage=COVERAGE,
  removed_success_path_getvalue_calls=REMOVED,host_and_operator_json_identical=True,
  interpretation='Source call counts only. B2 retains Arena DataCopy/PIPE_ALL/MTE2_S; no pure Scalar-to-wall-time claim.',
  adoption='ALL_VARIANTS_DIAGNOSTIC_ONLY_NOT_PRODUCTION_ELIGIBLE')
 write(root/'source-audit.json',audit);return audit

def verify_audit(audit,verify):
 expected=expected_originals()
 if audit.get('status')!='EXACT_TILE_GUARD_SOURCE_DIFFS' or set(audit.get('variants',{}))!=set(POLICIES):raise ValueError('audit variant/status differs')
 if len(audit['originals'])!=len(expected) or {p['path'] for p in audit['originals']}!={str(p.resolve()) for p in expected}:raise ValueError('original inventory differs')
 for p in audit['originals']:verify(p)
 b0=verify(audit['accepted_stage4d_b']).read_bytes()
 if b0!=d_guard.transform((FROZEN/KERNEL).read_bytes(),'B'):raise ValueError('B0 not accepted Stage4D B')
 if audit['coverage']!=COVERAGE or audit['removed_success_path_getvalue_calls']!=REMOVED:raise ValueError('coverage/counts changed')
 for v,entry in audit['variants'].items():
  if entry['safety_protocol']!=POLICIES[v]:raise ValueError('coverage label differs')
  target=Path(entry['source_dir'])
  wanted={str((target/p.relative_to(FROZEN)).resolve()) for p in expected}
  if len(entry['files'])!=len(wanted) or {p['path'] for p in entry['files']}!=wanted:raise ValueError('source inventory differs')
  for item in entry['files']:verify(item)
  for p in expected:
   rel=p.relative_to(FROZEN);payload=transform(b0,v) if str(rel)==KERNEL else p.read_bytes()
   if (target/rel).read_bytes()!=payload:raise ValueError('unauthorized source change')
  if verify(entry['kernel']).read_bytes()!=transform(b0,v):raise ValueError('kernel binding differs')
  if verify(audit['diffs'][v]).read_text()!=diff(b0,transform(b0,v),v):raise ValueError('audit diff differs')
