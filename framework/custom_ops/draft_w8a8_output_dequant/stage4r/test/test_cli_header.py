"""CLI regressions and native file identities; no device or SDK query."""
import hashlib,os,subprocess,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import run_stage4r as run
import header_binding as hb
REQUIRED={'prepare':(), 'build':('prepare-report',), 'smoke':('prepare-report','build-report'),
          'benchmark':('prepare-report','build-report','smoke-report')}

def cli(stage,candidate='N256'):
 out=[stage,'--output-dir','/tmp/new-stage4r','--candidate',candidate]
 for flag in REQUIRED[stage]:out += ['--'+flag,'/tmp/stage4r/'+flag+'.json']
 return out

@pytest.mark.parametrize('stage',REQUIRED)
@pytest.mark.parametrize('candidate',['N128','N256'])
def test_all_stages_parse_without_conflicts_or_history_override(stage,candidate):
 a=run.parse(cli(stage,candidate)+['--benchmark-report','/tmp/stage4r/benchmark.json'])
 assert a.stage==stage and a.candidate==candidate and a.benchmark_report==Path('/tmp/stage4r/benchmark.json')
 assert a.build_report==(Path('/tmp/stage4r/build-report.json') if 'build-report' in REQUIRED[stage] else None)
 assert a.prepare_report==(Path('/tmp/stage4r/prepare-report.json') if 'prepare-report' in REQUIRED[stage] else None)
 for unused in ('stage2b_root','stage2b_build_report','basic_report','zero_report','nonpower_report','stage2c_report','stage4j_root','stage4i_root','stage4c_report','stage4b_root','msprof'):
  assert not hasattr(a,unused)

@pytest.mark.parametrize('stage,missing',[(s,k) for s,keys in REQUIRED.items() for k in keys])
def test_each_missing_required_stage_report_is_cli_error(stage,missing,capsys):
 argv=cli(stage);i=argv.index('--'+missing);del argv[i:i+2]
 with pytest.raises(SystemExit) as e:run.parse(argv)
 assert e.value.code==2 and '--'+missing in capsys.readouterr().err

@pytest.mark.parametrize('flag',['--stage2b-build-report','--basic-report','--zero-report','--nonpower-report'])
def test_unused_inherited_flags_are_rejected(flag):
 with pytest.raises(SystemExit) as e:run.parse(cli('prepare')+[flag,'/tmp/old.json'])
 assert e.value.code==2

@pytest.mark.parametrize('stage',['prepare','build','smoke'])
@pytest.mark.parametrize('candidate',['N128','N256'])
def test_prior_stage_candidate_must_match(stage,candidate,tmp_path):
 r=dict(sources=[],evidence={},device=0,candidate=candidate,policy=run.POLICY)
 p=tmp_path/'prior.json';record=dict(r,status=run.STATES[stage],phase='complete');run.q.write(p,record)
 assert run.prior(p,stage,r)['candidate']==candidate
 record['candidate']='N256' if candidate=='N128' else 'N128';run.q.write(p,record)
 with pytest.raises(ValueError,match='admission'):run.prior(p,stage,r)

def test_entrypoint_help_constructs_the_whole_parser():
 p=subprocess.run([sys.executable,'-B',str(HERE/'run_stage4r.py'),'--help'],capture_output=True,text=True)
 assert p.returncode==0,p.stderr
 assert '--build-report' in p.stdout and '--stage2b-root' not in p.stdout

def test_header_selection_option_only_affects_build():
 assert run.parse(cli('build')+['--native-mm-header','/tmp/sdk/aclnn_mm.h']).native_mm_header==Path('/tmp/sdk/aclnn_mm.h')
 with pytest.raises(SystemExit):run.parse(cli('smoke')+['--native-mm-header','/tmp/sdk/aclnn_mm.h'])

@pytest.fixture
def sdk(tmp_path,monkeypatch):
 root=tmp_path/'sdk';target=root/hb.LOCATIONS[1];target.parent.mkdir(parents=True);target.write_bytes(b'CPU header fixture; not SDK source')
 monkeypatch.setattr(hb,'SERVER_MM_SHA256',hashlib.sha256(target.read_bytes()).hexdigest())
 return root,target

def test_server_style_directory_symlink_is_one_identity(sdk):
 root,target=sdk;(root/'include').symlink_to(root/'aarch64-linux/include',target_is_directory=True)
 b=hb.select_header(root)
 assert b['distinct_files']==1 and b['selection']=='ONE_FILE_WITH_ALIASES'
 assert b['selected']['path']==str(target) and b['all_candidate_contents_identical']
 assert len(b['candidates'])==2
 assert {(r['device'],r['inode']) for r in b['candidates']}=={(target.stat().st_dev,target.stat().st_ino)}
 assert sum(r['resolves_differently'] for r in b['candidates'])==1
 hb.verify_binding(b,root)

def test_hardlinks_are_same_file_not_ambiguous(sdk):
 root,target=sdk;alias=root/hb.LOCATIONS[0];alias.parent.mkdir(parents=True);os.link(target,alias)
 b=hb.select_header(root);assert b['distinct_files']==1 and len(b['candidates'])==2

@pytest.mark.parametrize('identical',[True,False])
def test_independent_files_require_explicit_choice(sdk,identical):
 root,target=sdk;second=root/hb.LOCATIONS[0];second.parent.mkdir(parents=True);second.write_bytes(target.read_bytes() if identical else b'unreviewed variant')
 with pytest.raises(ValueError,match='--native-mm-header'):hb.select_header(root)
 b=hb.select_header(root,target)
 assert b['selection']=='EXPLICIT_FILE' and b['distinct_files']==2
 assert b['all_candidate_contents_identical']==identical and len(b['candidates'])==2
 if not identical:
  with pytest.raises(ValueError,match='SHA256'):hb.select_header(root,second)

def test_tampered_or_retargeted_alias_rejected(sdk):
 root,target=sdk;alias=root/hb.LOCATIONS[0];alias.parent.mkdir(parents=True);alias.symlink_to(target)
 b=hb.select_header(root);other=root/'other/aclnn_mm.h';other.parent.mkdir();other.write_bytes(target.read_bytes())
 alias.unlink();alias.symlink_to(other)
 with pytest.raises(ValueError,match='alias/identity'):hb.verify_binding(b,root)
 target.write_bytes(b'tampered')
 with pytest.raises(ValueError,match='SHA256'):hb.select_header(root,target)

def test_escape_empty_missing_and_dangling_rejected(sdk,tmp_path):
 root,target=sdk;outside=tmp_path/'outside/aclnn_mm.h';outside.parent.mkdir();outside.write_bytes(target.read_bytes())
 with pytest.raises(ValueError,match='CANN_ROOT'):hb.select_header(root,outside)
 target.write_bytes(b'')
 with pytest.raises(ValueError,match='size'):hb.select_header(root)
 target.unlink()
 with pytest.raises(ValueError,match='missing'):hb.select_header(root)
 target.symlink_to(root/'missing/aclnn_mm.h')
 with pytest.raises(ValueError,match='broken'):hb.select_header(root)

def test_reviewed_server_hash_is_the_supplied_identity():
 assert hb.SERVER_MM_SHA256=='212c8d5565c54e0d78c13cff27f1150d13a5500b1859fd38a4319c96b8bb7f81'
