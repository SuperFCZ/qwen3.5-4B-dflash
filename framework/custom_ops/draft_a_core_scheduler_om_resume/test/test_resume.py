import copy
import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import resume


class Contract(unittest.TestCase):
    def setUp(self):
        policy=resume.c.read(resume.POLICY)
        self.assertEqual(len({x['alias'] for x in policy['records']}),22)
        self.pin=policy['records'][-1];p=self.pin
        self.row=dict(alias=p['alias'],file=dict(path=p['alias'],sha256=p['sha256'],bytes=p['bytes']),links=[],
                      stat=dict(inode=p['inode'],mode=p['mode']),backing=dict(inode=p['inode'],device=p['maps_device']))

    def test_byte_link_and_mode_pins(self):
        resume.pin_check(self.pin,self.row)
        for key,field,value in [('file','bytes',0),('file','sha256','wrong'),('file','path','/foreign'),('stat','mode',511),('backing','inode',0),('backing','device',[0,69])]:
            changed=copy.deepcopy(self.row);changed[key][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):resume.pin_check(self.pin,changed)
        self.row['links']=[{'target':'other.so'}]
        with self.assertRaises(ValueError):resume.pin_check(self.pin,self.row)

    def test_mapping(self):
        alias=self.pin['alias'];records={alias:self.row};segments=[self.row['backing'].copy()]
        with patch.object(resume.host.dsos,'verify_record') as verify:
            resume.check_maps({alias:segments},records);verify.assert_called_once_with(self.row)
            with self.assertRaisesRegex(ValueError,'unbound OM'):resume.check_maps({'/other/'+Path(alias).name:segments},records)
            segments[0]['inode']+=1
            with self.assertRaisesRegex(ValueError,'backing'):resume.check_maps({alias:segments},records)

    def test_original_protocol(self):
        self.assertEqual(inspect.getsource(resume.full),inspect.getsource(resume.flow.full).replace('r,smoke=None):','r,smoke=None,*,mapped):'))
