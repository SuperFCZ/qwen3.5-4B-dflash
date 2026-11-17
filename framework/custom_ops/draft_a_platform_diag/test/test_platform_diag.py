import copy,json,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from platform_common import c
from platform_contracts import read_platform,compare,validate_runtime,loaded_libraries,verify_loaded,environment
import derive,run_diag


def runtime():
    r=dict(status='HOST_TILING_ONLY',execute_calls=0,m=16,k=12800,n=2560,device=0,workspace=2097152,explicit_context_created=False)
    r.update({x:True for x in ('acl_init','set_device','context_present','context_stable','stream_created','guards','readonly','poison_unchanged')})
    names=['aclInit','aclrtSetDevice','aclrtGetDevice','context_after_set_device','aclrtCreateStream','context_after_stream','GetWorkspaceSize','GetWorkspaceSize','aclrtSynchronizeStream']
    j=[dict(event=n,returncode=0) for n in names]
    j[1]['requested_device']=0;j[2]['device']=0;j[3]['present']=True;j[4]['present']=True;j[5]['same_context']=True
    for i in (0,1):j[6+i].update(call=i,workspace=r['workspace'],executor_present=True)
    return r,j


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.dso=self.root/'opp/vendors/customize/libcust_opmaster_rt2.0.so';self.dso.parent.mkdir(parents=True);self.dso.write_bytes(b'fixture host')
        self.opp=dict(install=str(self.root/'opp'),preflight=dict(files=[c.identity(self.dso)]))
        self.path=self.root/'platform.jsonl'
        self.row=dict(schema=1,route='ACLNN',m=16,k=12800,n=2560,platform_info_present=True,GetCoreNum=7,GetCoreNumAic=8,GetCoreNumAiv=8,soc_enum=2,arch=2002,platform_raw_core=7,platform_raw_core_locked=7,availableCores=7,blockDim=7,coreLimit=0,workspace=2097152,capacity=dict(UB=262144,L1=1048576,L0A=65536,L0B=65536,L0C=262144),host_dso=str(self.dso),resource_map_ok=True,resource_map={'SocInfo':{'ai_core_cnt':'8'}})
    def read(self,rows=None):
        self.path.write_text(''.join(json.dumps(x)+'\n' for x in (rows or [self.row])))
        return read_platform(self.path,'ACLNN',self.opp)
    def test_seven_aggregate_eight_aic_is_observation_only(self):
        b=self.read();a=copy.deepcopy(b)
        for x in ('GetCoreNum','platform_raw_core','platform_raw_core_locked','availableCores','blockDim'):a['observation'][x]=8
        a['observation']['resource_map']['Runtime']={'available':'8'}
        result=compare(a,b)
        self.assertEqual(result['status'],'PLATFORM_VIEWS_DIFFER');self.assertEqual(result['selected_runtime_cores'],7)
        self.assertEqual(result['core8_admission'],'NOT_GRANTED');self.assertTrue(result['virtualization'].startswith('UNRESOLVED'))
        self.assertNotIn('GetCoreNumAic',result['field_differences'])
        self.assertIn('Runtime.available',result['resource_map_differences'])
    def test_matching_views_not_numeric_pass(self):
        x=self.read();self.assertEqual(compare(x,x)['status'],'PLATFORM_VIEWS_MATCH')
        self.assertEqual(compare(x,x)['core8_admission'],'NOT_GRANTED')
    def test_changed_or_invalid_records_block(self):
        for key,value in [('coreLimit',8),('blockDim',8),('m',64),('route','MANAGER'),('resource_map_ok',False),('GetCoreNum',True),('capacity',{'UB':0}),('host_dso',str(HERE/'run_diag.py'))]:
            with self.subTest(key=key):
                row=dict(self.row);row[key]=value
                with self.assertRaises(ValueError):self.read([row])
    def test_route_loader_and_environment_contract(self):
        x=self.read();y=copy.deepcopy(x);y['host_dso']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'different Host DSO'):compare(x,y)
        env={'ASCEND_CUSTOM_OPP_PATH':'/private','ASCEND_RT_VISIBLE_DEVICES':'0'}
        self.assertNotEqual(environment(env),environment(dict(env,ASCEND_RT_VISIBLE_DEVICES='1')))
    def test_callback_drift_block(self):
        x=dict(self.row);x['GetCoreNumAic']=7
        with self.assertRaisesRegex(ValueError,'drift'):self.read([self.row,x])
    def test_empty_and_truncated_block(self):
        for text in ('','{"schema":1'):
            self.path.write_text(text)
            with self.assertRaises(ValueError):read_platform(self.path,'ACLNN',self.opp)
    def test_duplicate_or_unknown_json_field_block(self):
        self.path.write_text(json.dumps(self.row)[:-1]+',"GetCoreNum":8}\n')
        with self.assertRaisesRegex(ValueError,'duplicate'):read_platform(self.path,'ACLNN',self.opp)
        changed=dict(self.row);changed['UNKNOWN']=1
        with self.assertRaisesRegex(ValueError,'field set'):self.read([changed])
    def test_dso_sha_drift_block(self):
        self.dso.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'unbound'):self.read()
    def test_runtime_journal(self):
        r,j=runtime();validate_runtime(r,j)
        for key in ('guards','readonly','context_stable','poison_unchanged'):
            bad=dict(r);bad[key]=False
            with self.assertRaises(ValueError):validate_runtime(bad,j)
        for key,value in [('status','HOST_ONLY'),('execute_calls',1),('device',1),('explicit_context_created',True)]:
            bad=dict(r);bad[key]=value
            with self.assertRaises(ValueError):validate_runtime(bad,j)
        for pos,key,val in [(0,'returncode',1),(1,'requested_device',1),(3,'present',False),(5,'same_context',False),(7,'workspace',1),(6,'executor_present',False)]:
            bad=copy.deepcopy(j);bad[pos][key]=val
            with self.assertRaises(ValueError):validate_runtime(r,bad)
    def test_actual_mapping_alias_and_identity(self):
        alias=self.dso.parent/'liboptiling.so';alias.symlink_to(self.dso.name)
        maps=self.root/'maps';maps.write_text('1000-2000 r-xp 00000000 08:11 123 '+str(alias)+'\n')
        rows=loaded_libraries(maps,self.opp,self.root/'cann');verify_loaded(rows)
        self.assertEqual(rows[0]['category'],'PRIVATE_DIAGNOSTIC_OPP');self.assertTrue(rows[0]['links'])
        # Device/inode values are recorded by raw maps, never assumed equal to overlay stat.
        alias.unlink();alias.symlink_to(HERE/'run_diag.py')
        with self.assertRaises(ValueError):verify_loaded(rows)
    def test_deleted_and_unbound_private_mapping_block(self):
        maps=self.root/'maps';maps.write_text('1000-2000 r-xp 0 08:11 123 '+str(self.dso)+' (deleted)\n')
        with self.assertRaisesRegex(ValueError,'deleted'):loaded_libraries(maps,self.opp,self.root/'cann')
        rogue=self.root/'libcust_opapi.so';rogue.write_bytes(b'other private')
        maps.write_text('1000-2000 r-xp 0 08:11 123 '+str(rogue)+'\n')
        with self.assertRaisesRegex(ValueError,'different Ascend'):loaded_libraries(maps,self.opp,self.root/'cann')
    def test_source_transform_keeps_scheduler_and_math(self):
        derive.verify();s=(HERE/'op_host/d_flash_group_quant_linear.cpp').read_text()
        self.assertEqual(s,derive.host_text());self.assertIn('availableCores = platform.GetCoreNum()',s)
        self.assertNotIn('SetBlockDim(8)',s);self.assertNotIn('SetCoreNum(',s)
        runner=(HERE/'runner/main.cpp').read_text()
        self.assertNotIn('aclnnDFlashGroupQuantLinear(',runner)
        self.assertIn('aclnnDFlashGroupQuantLinearGetWorkspaceSize(',runner)
        self.assertNotIn('aclrtCreateContext(',runner)
        self.assertIn('HOST_TILING_ONLY',runner)
    def test_cli_gates_and_import_only(self):
        needs=dict(source=['admission'],build=['source'],atc=['source','build'],aclnn=['source','build'],compare=['source','build','atc','aclnn'])
        for phase,keys in needs.items():
            argv=[phase,'--output-dir',str(self.root/'out')]
            with patch('sys.stderr'):
                with self.assertRaises(SystemExit):run_diag.parse(argv)
            for key in keys:argv += ['--'+key+'-report',str(self.root/(key+'.json'))]
            self.assertEqual(run_diag.parse(argv).phase,phase)
        x=subprocess.run([sys.executable,'-B',str(HERE/'run_diag.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(x.returncode,0,x.stderr);self.assertIn('aclnn',x.stdout)
        c.clean()


class Writer(unittest.TestCase):
    @unittest.skipUnless(shutil.which('c++'),'C++ compiler unavailable')
    def test_cpp_queries_read_only_and_concurrent_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'platform').mkdir()
            (root/'platform/platform_infos_def.h').write_text('''#pragma once
#include <map>
#include <string>
namespace fe {struct PlatFormInfos {unsigned GetCoreNum()const{return 7;}unsigned GetCoreNumWithLock()const{return 7;}bool GetPlatformResWithLock(std::map<std::string,std::map<std::string,std::string>>&m){m["SocInfo"]["ai_core_cnt"]="8";return true;}};}
''')
            src=root/'test.cpp';src.write_text('''#include <cstdint>
#include <thread>
#include <vector>
namespace platform_ascendc {enum class CoreMemType {UB,L1,L0_A,L0_B,L0_C};struct PlatformAscendC {unsigned GetCoreNum()const{return 7;}unsigned GetCoreNumAic()const{return 8;}unsigned GetCoreNumAiv()const{return 8;}unsigned GetSocVersion()const{return 2;}unsigned GetCurNpuArch()const{return 2002;}void GetCoreMemSize(CoreMemType,uint64_t&v)const{v=65536;}};}
#include "platform_diag.h"
int main(int argc,char**argv){fe::PlatFormInfos info;platform_ascendc::PlatformAscendC p;if(argc!=2)return 2;setenv("DFLASH_PLATFORM_DIAG_PATH",argv[1],1);setenv("DFLASH_PLATFORM_DIAG_ROUTE","ACLNN",1);std::vector<std::thread>threads;for(int i=0;i<16;i++)threads.emplace_back([&](){if(!dflash_platform_diag::Write(&info,p,16,12800,2560,7,7,0,2097152))std::abort();});for(auto&t:threads)t.join();return 0;}
''')
            cmd=['c++','-std=c++11','-pthread','-I'+str(root),'-I'+str(HERE/'op_host'),str(src),'-o',str(root/'writer')]
            if sys.platform!='darwin':cmd.append('-ldl')
            subprocess.run(cmd,check=True,capture_output=True)
            out=root/'platform.jsonl';out.touch();subprocess.run([str(root/'writer'),str(out)],check=True)
            rows=[json.loads(x) for x in out.read_text().splitlines()];self.assertEqual(len(rows),16)
            self.assertTrue(all(x==rows[0] for x in rows));self.assertEqual(rows[0]['GetCoreNum'],7);self.assertEqual(rows[0]['GetCoreNumAic'],8)
            self.assertEqual(rows[0]['resource_map']['SocInfo']['ai_core_cnt'],'8')
            fail=subprocess.run([str(root/'writer'),str(root/'missing')],capture_output=True)
            self.assertNotEqual(fail.returncode,0)


if __name__=='__main__':unittest.main()
