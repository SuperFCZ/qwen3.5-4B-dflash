import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_smoke as smoke
from fc_contract import c, difference, fixed_query, live_callback, safety


class Contract(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.folder=Path(temp.name)

    def records(self,name,rows):
        (self.folder/name).write_text(''.join(json.dumps(row)+'\n' for row in rows))

    def test_numeric_and_poison(self):
        expected=bytes(81920);actual=bytearray(expected)
        for idx,value in ((0,0x8000),(2560,1),(40958,0x7f7f),(40959,0xfc00)):
            struct.pack_into('<H',actual,2*idx,value)
        result=difference(expected,actual)
        self.assertEqual(result['native_comparator']['bit_mismatches'],4)
        self.assertEqual([(row['row'],row['channel'],row['ulp']) for row in result['mismatches']],
                         [(0,0,1),(1,0,1),(15,2558,None),(15,2559,None)])
        self.assertEqual(result['nonfinite'],[[15,2558,'finite','nan'],[15,2559,'finite','-inf']])
        self.assertTrue(difference(expected,expected)['native_comparator']['bitwise_equal'])
        with self.assertRaises(ValueError):difference(expected,actual[:-2])

    def test_actual_callback_required(self):
        platform=dict(GetCoreNum=7,GetCoreNumAic=8,GetCoreNumAiv=8,blockDim=8,coreLimit=0,workspace=2097152)
        schedule=dict(variant='B',slice_num=8,calc_tsch_block_dim=8,selected_block_dim=8,tiling_hex='12')
        query=dict(platform=platform,schedule=schedule)
        self.records('platform.jsonl',[platform]);self.records('scheduling.jsonl',[schedule]);live_callback(self.folder,query,'B')
        for changes in (dict(blockDim=7),dict(workspace=0),dict(GetCoreNum=8)):
            with self.assertRaises(ValueError):fixed_query(dict(platform,**changes),schedule,'B')
        for bad in ([],[schedule,schedule],[dict(schedule,tiling_hex='21')],[dict(schedule,calc_tsch_block_dim=7)]):
            self.records('scheduling.jsonl',bad)
            with self.assertRaises(ValueError):live_callback(self.folder,query,'B')

    def test_prelaunch_interlock(self):
        runner=self.folder/'fixture';runner.write_text('#!'+sys.executable+'\n'+
            'import os,sys\nfrom pathlib import Path\nfd=int(sys.argv[3]);os.write(fd,b"?")\n'+
            'if os.read(fd,1)!=b"G":sys.exit(6)\n(Path(sys.argv[1])/"released").touch()\n')
        runner.chmod(0o700)
        def reject():raise ValueError('query changed')
        with self.assertRaisesRegex(ValueError,'query changed'):smoke.invoke(runner,self.folder,0,dict(os.environ),2097152,reject)
        self.assertFalse((self.folder/'released').exists())
        smoke.invoke(runner,self.folder,0,dict(os.environ),2097152,lambda:None)
        self.assertTrue((self.folder/'released').exists())

    def test_safety_evidence(self):
        events=[dict(event=name,returncode=0) for name in ('aclInit','aclrtSetDevice','aclrtGetDevice','context','stream','GetWorkspaceSize','precheck','execute_attempt','aclnnDFlashGroupQuantLinear','synchronize','readonly','complete')]
        events[2]['device']=0;events[3]['present']=True;events[4]['present']=True;events[5]['workspace']=2097152
        events[6]['poison_byte']=127;events[10].update(x=1,w=1,s=1)
        guards=[dict(buffer=name,side=side,bytes=512,expected=165,changes=[]) for name in ('x','w','s','y','workspace') for side in ('prefix','suffix')]
        for name in ('pre','post'):self.records(name+'-guards.jsonl',guards)
        self.records('events.jsonl',events);self.assertEqual(safety(self.folder,0)['execute_calls'],1)
        with self.assertRaises(ValueError):safety(self.folder,1)
        for idx,key,bad in ((9,'returncode',1),(10,'w',0),(5,'workspace',0),(4,'present',False),(8,'event','execute_attempt')):
            corrupted=[row.copy() for row in events];corrupted[idx][key]=bad;self.records('events.jsonl',corrupted)
            with self.assertRaises(ValueError):safety(self.folder,0)
        self.records('events.jsonl',events);guards[-1]['changes']=[[511,0]];self.records('post-guards.jsonl',guards)
        with self.assertRaises(ValueError):safety(self.folder,0)

    def test_blocked_or_drifted_predecessor(self):
        path=self.folder/'report.json';data=self.folder/'data';data.write_text('original');identity=c.identity(data)
        with patch.object(smoke,'sources',return_value=[]):
            report=dict(status=smoke.STATES['build'],phase='build',sources=[],artifact=identity);c.write(path,report)
            smoke.predecessor(path,'build');c.write(path,dict(report,status='BLOCKED'))
            with self.assertRaises(ValueError):smoke.predecessor(path,'build')
            c.write(path,report);data.write_text('replaced')
            with self.assertRaises(ValueError):smoke.predecessor(path,'build')

    def test_four_runs_no_partial_compare(self):
        self.assertEqual(smoke.ORDER,(('A',0),('A',1),('B',0),('B',1)))
        with self.assertRaisesRegex(ValueError,'four ordered'):smoke.replay({},[dict(variant='A',repeat=0)],{})

    def test_replay_recomputes_numeric_evidence(self):
        golden=self.folder/'golden';golden.write_bytes(bytes(81920));runner=c.identity(golden)
        binding=dict(case=dict(inputs={},golden=c.identity(golden)),cann_root='/fixture',variants={})
        built={'variants':{}};rows=[]
        for variant,repeat in smoke.ORDER:
            folder=self.folder/(variant+str(repeat));folder.mkdir();output=folder/'actual.bin';output.write_bytes(golden.read_bytes())
            platform=dict(GetCoreNum=7,GetCoreNumAic=8,GetCoreNumAiv=8,blockDim=7+(variant=='B'),workspace=2097152)
            binding['variants'][variant]=dict(query=dict(platform=platform,schedule=dict(calc_tsch_block_dim=8)))
            built['variants'][variant]=dict(runner=runner)
            checks={}
            for label in ['native']+(['repeat'] if repeat else [])+(['A_vs_B'] if variant=='B' else []):
                identity,match=smoke.compare_bytes(folder,label,golden.read_bytes(),output.read_bytes());checks[label]=dict(report=identity,match=match)
            rows.append(dict(variant=variant,repeat=repeat,folder=str(folder),runner=runner,launch_authorized=True,
                             safety={},pre_loaded=[],post_loaded=[],output=c.identity(output),comparisons=checks))
        with patch.object(smoke,'mappings',return_value=[]),patch.object(smoke,'safety',return_value={}),patch.object(smoke,'live_callback'):
            self.assertEqual(smoke.replay(binding,rows,built)['B']['blockDim'],8)
            output.write_bytes(bytes(81918)+b'\x01\x00');rows[-1]['output']=c.identity(output)
            with self.assertRaisesRegex(ValueError,'comparison replay'):smoke.replay(binding,rows,built)


if __name__=='__main__':unittest.main()
