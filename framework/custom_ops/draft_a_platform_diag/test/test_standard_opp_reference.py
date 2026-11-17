import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from platform_common import c
import standard_opp_reference as ref
from test_platform_diag import runtime


class StandardReference(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.cann=self.root/'cann';self.rel='opp/vendors/customize/libcust_opmaster_rt2.0.so';self.std=self.cann/self.rel
        for x in ('source','build','atc','aclnn/fc'):(self.root/x).mkdir(parents=True)
        source=dict(status='PLATFORM_DIAG_SOURCE_BOUND',binding={'prior':{'current':{'cann_root':str(self.cann)}}});c.write(self.root/'source/report.json',source)
        sid=c.identity(self.root/'source/report.json');opp={'install':str(self.root/'diag')}
        c.write(self.root/'build/report.json',dict(status='PLATFORM_DIAG_OPP_RUNNER_BUILT',source_report=sid,package={'opp':opp}));bid=c.identity(self.root/'build/report.json')
        run,j=runtime();fc=self.root/'aclnn/fc';c.write(fc/'execution.json',run)
        (fc/'runtime-events.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in j));(fc/'platform.jsonl').write_text('{}\n{}\n')
        (fc/'maps.txt').write_text('1000-2000 r-xp 0 08:11 10 '+str(self.std)+'\n')
        ids=[c.identity(fc/x) for x in ('execution.json','runtime-events.jsonl','platform.jsonl','maps.txt')]
        c.write(self.root/'aclnn/report.json',dict(status='BLOCKED',error='different Ascend installation/private OPP: '+str(self.std),integrity_error=None,source_report=sid,build_report=bid,retained_runtime=ids))
        (self.root/'atc/platform.jsonl').write_text('{}\n');self.atc={'source':c.identity(self.root/'atc/platform.jsonl'),'observation':{'GetCoreNum':8}}
        c.write(self.root/'atc/report.json',dict(status='PLATFORM_DIAG_ATC_OBSERVED',source_report=sid,build_report=bid,platform=self.atc))
        self.policy=self.root/'policy.json';c.write(self.policy,dict(standard_opp=[{'relative':self.rel}],reference_files={Path(x['path']).name:{k:x[k] for k in ('bytes','sha256')} for x in ids}))
    def audit(self):
        def platform(path,route,opp):return self.atc if route=='ATC' else {'source':c.identity(path),'observation':{'GetCoreNum':7}}
        with patch.object(ref,'POLICY',self.policy),patch.object(ref.c,'verify_opp'),patch.object(ref,'read_platform',side_effect=platform):return ref.audit(self.root)
    def test_old_blocked_kept_readonly_and_not_promoted(self):
        p=self.root/'aclnn/report.json';before=p.read_bytes();r=self.audit()
        self.assertEqual(r['old_status'],'BLOCKED');self.assertEqual(r['scope'],'REFERENCE_ONLY_NO_RETROACTIVE_PASS')
        self.assertEqual(r['execution']['execute_calls'],0);self.assertEqual(p.read_bytes(),before)
    def test_old_file_missing_or_tampered_blocked(self):
        p=self.root/'aclnn/fc/platform.jsonl';data=p.read_bytes();p.unlink()
        with self.assertRaises(FileNotFoundError):self.audit()
        p.write_bytes(data+b'changed')
        with self.assertRaises(ValueError):self.audit()
    def test_retained_pin_cannot_be_replaced(self):
        p=self.root/'aclnn/report.json';r=c.read(p);r['retained_runtime'][0]['sha256']='0'*64;c.write(p,r)
        with self.assertRaisesRegex(ValueError,'pin differs'):self.audit()


if __name__=='__main__':unittest.main()
