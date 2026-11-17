import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from platform_common import c
import old_runtime_reference as old
from test_platform_diag import runtime


class OldReference(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        for p in ('source','build','atc','aclnn/fc'):(self.root/p).mkdir(parents=True)
        src={'status':'PLATFORM_DIAG_SOURCE_BOUND','sources':[],'binding':{'prior':{'current':{'cann_root':'/fixture-cann'}}}}
        c.write(self.root/'source/report.json',src);sid=c.identity(self.root/'source/report.json')
        c.write(self.root/'build/report.json',dict(status='PLATFORM_DIAG_OPP_RUNNER_BUILT',source_report=sid,package={'opp':{}}));bid=c.identity(self.root/'build/report.json')
        execution,events=runtime();fc=self.root/'aclnn/fc'
        c.write(fc/'execution.json',execution);(fc/'runtime-events.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))
        (fc/'platform.jsonl').write_text('{}\n{}\n');(fc/'runner.log').write_text('old host-only fixture\n')
        (fc/'maps.txt').write_text('1000-2000 r-xp 0 08:11 1 '+old.FOREIGN+'\n')
        self.ids=[c.identity(fc/name) for name in old.PINS]
        self.pins={Path(x['path']).name:(x['bytes'],x['sha256']) for x in self.ids}
        c.write(self.root/'aclnn/report.json',dict(status='BLOCKED',error='ValueError: different Ascend installation/private OPP: '+old.FOREIGN,integrity_error=None,retained_runtime=self.ids,build_report=bid,source_report=sid,environment_contract={'ASCEND_CUSTOM_OPP_PATH':'old:foreign'}))
        atc=self.root/'atc/platform.jsonl';atc.write_text('{}\n')
        self.atc_platform={'source':c.identity(atc),'observation':{'GetCoreNum':8}}
        c.write(self.root/'atc/report.json',dict(status='PLATFORM_DIAG_ATC_OBSERVED',source_report=sid,build_report=bid,platform=self.atc_platform,environment_contract={'ASCEND_CUSTOM_OPP_PATH':'old:foreign'}))
    def audit(self):
        def read(path,route,opp):
            return self.atc_platform if route=='ATC' else {'source':c.identity(path),'observation':{'GetCoreNum':7,'GetCoreNumAic':8,'GetCoreNumAiv':8}}
        with patch.dict(old.PINS,self.pins,clear=True),patch.object(old.c,'verify_opp'),patch.object(old,'read_platform',side_effect=read):return old.audit(self.root)
    def test_complete_reference_is_never_formal_pass_or_reused_atc(self):
        before=(self.root/'aclnn/report.json').read_bytes();r=self.audit()
        self.assertEqual(r['status'],'CONTAMINATED_RUNTIME_REFERENCE_REVERIFIED_NOT_ADMITTED')
        self.assertEqual(r['scope'],'REFERENCE_ONLY_NOT_FORMAL_PASS');self.assertEqual(r['execution']['execute_calls'],0)
        self.assertTrue(r['atc_reuse'].startswith('NOT_ADMITTED'));self.assertFalse(r['current_source_identity_equal'])
        self.assertEqual((self.root/'aclnn/report.json').read_bytes(),before)
    def test_missing_or_modified_artifact_blocks(self):
        p=self.root/'aclnn/fc/maps.txt';b=p.read_bytes();p.unlink()
        with self.assertRaises(FileNotFoundError):self.audit()
        p.write_bytes(b+b'changed\n')
        with self.assertRaises(ValueError):self.audit()
    def test_old_claimed_pass_is_rejected(self):
        p=self.root/'aclnn/report.json';r=c.read(p);r['status']='PLATFORM_DIAG_ACLNN_HOST_OBSERVED';c.write(p,r)
        with self.assertRaisesRegex(ValueError,'contaminated'):self.audit()
    def test_missing_bound_hash_is_not_reaccepted(self):
        p=self.root/'aclnn/report.json';r=c.read(p);r['retained_runtime'].pop();c.write(p,r)
        with self.assertRaisesRegex(ValueError,'identity/pin'):self.audit()


if __name__=='__main__':unittest.main()
