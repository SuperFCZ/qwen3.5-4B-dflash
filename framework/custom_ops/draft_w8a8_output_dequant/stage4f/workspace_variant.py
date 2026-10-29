"""Only move one existing statement from the group loop to the tile loop."""
import difflib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4d'))
import guard_variant as d
FROZEN=d.FROZEN;KERNEL=d.KERNEL
CALL='        mm.SetLocalWorkspace(scratch);\n'
OLD_COMMENT='Rebind all inputs/Scale/workspace after End; false starts a fresh INT32 C.'
NEW_COMMENT='Rebind inputs/Scale after End; workspace remains bound for this tile; false starts a fresh INT32 C.'
ANCHOR='    for(uint32_t g=0;g<20;++g) {\n'
POLICIES={v:('FULL_STAGE4B_GUARD_PROTOCOL' if v[0]=='A' else 'DIAGNOSTIC_ONLY_WEAKER_GUARD_COVERAGE') for v in ('A0','A1','B0','B1')}
COVERAGE={v:dict(group_five_ub_scans=v[0]=='A',tile_five_ub_scans=True,tile_arena_scalar_scan=True,
 final_p0_cast_scans=True,initial_arena_scan=True,all_sync_retained=True,
 workspace_binding='PER_TILE' if v[1]=='1' else 'PER_GROUP',workspace_calls=304 if v[1]=='1' else 6080,
 safety_equivalence=('SAME_SOURCE_GUARD_CHECKS_RUNTIME_NOT_PROVEN' if v[0]=='A' else 'NOT_EQUIVALENT_TO_STAGE4B')) for v in POLICIES}

def hoist(raw):
 s=raw.decode()
 if s.count(CALL)!=1 or s.count(ANCHOR)!=1:raise ValueError('workspace call/group anchor not unique')
 if 'Duplicate(sum,0.0f,1024);PipeBarrier<PIPE_ALL>();\n'+ANCHOR not in s:raise ValueError('tile reset/synchronization anchor drift')
 if ANCHOR+'        // Rebind all inputs/Scale/workspace after End; false starts a fresh INT32 C.\n'+CALL not in s:raise ValueError('workspace call not at audited group entry')
 # Only move the call and correct its accompanying comment.
 return s.replace(CALL,'',1).replace(OLD_COMMENT,NEW_COMMENT,1).replace(ANCHOR,'    mm.SetLocalWorkspace(scratch);\n'+ANCHOR,1).encode()

def transform(original,v):
 if v not in POLICIES:raise ValueError('unknown variant')
 base=original if v[0]=='A' else d.transform(original,'B')
 return hoist(base) if v[1]=='1' else base

def originals():return [FROZEN/'operator.json']+[p for sub in ('op_host','op_kernel') for p in sorted((FROZEN/sub).iterdir()) if p.is_file()]

def pair_diff(a,b,pair):return ''.join(difflib.unified_diff(a.decode().splitlines(True),b.decode().splitlines(True),fromfile=pair+'0/'+KERNEL,tofile=pair+'1/'+KERNEL))

def materialize(root,identity,write,accepted_b):
 root.mkdir();raw=(FROZEN/KERNEL).read_bytes()
 if identity(accepted_b['path'])!=accepted_b or Path(accepted_b['path']).read_bytes()!=transform(raw,'B0'):raise ValueError('accepted Stage4D B source differs')
 variants={}
 for v in POLICIES:
  folder=root/v;folder.mkdir();files=[]
  for p in originals():
   rel=p.relative_to(FROZEN);out=folder/rel;out.parent.mkdir(exist_ok=True,parents=True)
   out.write_bytes(transform(raw,v) if str(rel)==KERNEL else p.read_bytes());files.append(identity(out))
  variants[v]=dict(source_dir=str(folder),files=files,kernel=identity(folder/KERNEL),safety_protocol=POLICIES[v])
 diffs={}
 for pair in ('A','B'):
  p=root/(pair+'-workspace-only.diff');p.write_text(pair_diff(transform(raw,pair+'0'),transform(raw,pair+'1'),pair));diffs[pair]=identity(p)
 audit=dict(status='EXACT_WORKSPACE_MOVE_ONLY',originals=[identity(p) for p in originals()],accepted_stage4d_b=accepted_b,
  variants=variants,diffs=diffs,coverage=COVERAGE,removed_calls=5776,resource_change='NONE',sync_change='NONE',
  interpretation='Source-level call reduction only; compiler may already eliminate repeated state updates.')
 write(root/'source-audit.json',audit);return audit

def verify_audit(audit,verify):
 paths=originals();raw=(FROZEN/KERNEL).read_bytes()
 if audit.get('status')!='EXACT_WORKSPACE_MOVE_ONLY' or set(audit['variants'])!=set(POLICIES) or audit['coverage']!=COVERAGE:raise ValueError('audit scope/coverage differs')
 if len(audit['originals'])!=len(paths) or {i['path'] for i in audit['originals']}!={str(p.resolve()) for p in paths}:raise ValueError('original source inventory differs')
 for i in audit['originals']:verify(i)
 if verify(audit['accepted_stage4d_b']).read_bytes()!=transform(raw,'B0'):raise ValueError('baseline B changed')
 for v,entry in audit['variants'].items():
  folder=Path(entry['source_dir']);wanted={str((folder/p.relative_to(FROZEN)).resolve()) for p in paths}
  if entry['safety_protocol']!=POLICIES[v] or len(entry['files'])!=len(wanted) or {i['path'] for i in entry['files']}!=wanted:raise ValueError('variant inventory/label differs')
  for i in entry['files']:verify(i)
  for p in paths:
   rel=p.relative_to(FROZEN)
   if (folder/rel).read_bytes()!=(transform(raw,v) if str(rel)==KERNEL else p.read_bytes()):raise ValueError('unexpected Kernel/Host/ABI edit')
  if verify(entry['kernel']).read_bytes()!=transform(raw,v):raise ValueError('kernel binding differs')
 for pair in ('A','B'):
  if verify(audit['diffs'][pair]).read_text()!=pair_diff(transform(raw,pair+'0'),transform(raw,pair+'1'),pair):raise ValueError('pair diff differs')
 if audit['removed_calls']!=5776 or audit['resource_change']!='NONE' or audit['sync_change']!='NONE':raise ValueError('audit claim differs')
