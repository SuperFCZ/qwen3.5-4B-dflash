import copy,json,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from scheduler_common import c,PD
from scheduler_contracts import assignments,cpu_plan,read_query,compare_pair
from scheduler_evidence import prefix_review,verify_prefix_review
import scheduler_derive as derive
import run_scheduler


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.dso=self.root/'opp/libcust_opmaster_rt2.0.so';self.dso.parent.mkdir();self.dso.write_bytes(b'fixture')
        self.opp={'install':str(self.dso.parent),'preflight':{'files':[c.identity(self.dso)]}}
    def query(self,variant,cores=7):
        p=dict(schema=1,route='ACLNN',m=16,k=12800,n=2560,platform_info_present=True,resource_map_ok=True,
               GetCoreNum=cores,GetCoreNumAic=8,GetCoreNumAiv=8,platform_raw_core=cores,platform_raw_core_locked=cores,
               availableCores=cores,blockDim=cores if variant=='A' else 8,coreLimit=0,soc_enum=2,arch=2002,workspace=2097152,
               capacity={'UB':262144,'L1':1048576,'L0A':65536,'L0B':65536,'L0C':262144},host_dso=str(self.dso),resource_map={'SoCInfo':{'ai_core_cnt':'8','vector_core_cnt':'7','core_type_list':'AiCore'}})
        s=dict(schema=1,variant=variant,route='ACLNN',m=16,k=12800,n=2560,tile_n=64,n_tiles=40,core_limit=0,
               generic=cores,aic=8,aiv=8,independent_vector=7,slice_num=8,calc_tsch_block_dim=8,baseline_block_dim=cores,
               selected_block_dim=p['blockDim'],baseline_assignments=assignments(40,cores),candidate_assignments=assignments(40,8),
               resources=dict(user_ub=59904,scratch=202240,l1_used=40960,l0a_used=16384,l0b_used=65536,l0c_used=4096,codes_bytes=16384,scale_bytes=256,weight_bytes=32768,raw_banks=1,tile_k=256,workspace=2097152),
               tiling_bytes=224,tiling_hex=(b'\x10'+bytes(223)).hex(),approval='HOST_ONLY_NOT_DEVICE_ADMITTED')
        return p,s
    def read(self,p,s):
        a=self.root/'platform.jsonl';b=self.root/'scheduling.jsonl';a.write_text((json.dumps(p)+'\n')*2);b.write_text((json.dumps(s)+'\n')*2)
        return read_query(a,b,self.opp,'ACLNN',s['variant'])
    def test_cyclic_coverage_and_worker_slice_semantics(self):
        p=cpu_plan(40,7,8,8);self.assertEqual(p,dict(baseline=7,slice_num=8,expected_calc=8))
        self.assertEqual(list(map(len,assignments(40,7))),[6,6,6,6,6,5,5])
        self.assertEqual(list(map(len,assignments(40,8))),[5]*8)
        for t in (1,3,40):
            for cores in (1,3,7,8,80):
                plan=cpu_plan(t,cores,cores,cores);self.assertEqual(sorted(x for row in assignments(t,plan['expected_calc']) for x in row),list(range(t)))
        self.assertEqual(cpu_plan(3,7,8,8)['slice_num'],3)
    def test_invalid_or_separated_geometry_blocked(self):
        for values in ((0,7,8,8),(40,0,8,8),(40,7,8,16),(40,7,0,0),(True,7,8,8)):
            with self.assertRaises(ValueError):cpu_plan(*values)
    def test_full_ab_contract(self):
        a=self.read(*self.query('A'));b=self.read(*self.query('B'));r=compare_pair(a,b)
        self.assertEqual((r['A_block_dim'],r['B_block_dim']),(7,8));self.assertEqual(r['npu'],'NOT_RUN')
    def test_wrong_api_return_or_actual_launch_blocked(self):
        for key,val in [('slice_num',40),('calc_tsch_block_dim',0),('calc_tsch_block_dim',7),('selected_block_dim',7),('core_limit',8),('slice_num',True)]:
            p,s=self.query('B');s[key]=val
            with self.subTest(key=key,val=val),self.assertRaises(ValueError):self.read(p,s)
        p,s=self.query('B');p['blockDim']=7
        with self.assertRaisesRegex(ValueError,'actual SetBlockDim'):self.read(p,s)
    def test_resources_or_tiling_changed_rejected(self):
        a=self.read(*self.query('A'));b=copy.deepcopy(a)
        b['schedule']['resources']['scratch']+=32
        with self.assertRaises(ValueError):compare_pair(a,b)
        b=copy.deepcopy(a);b['tiling_sha256']='0'*64
        with self.assertRaises(ValueError):compare_pair(a,b)
        for key,val in [('l0b_used',65537),('scratch',262144)]:
            p,s=self.query('B');s['resources'][key]=val
            with self.assertRaises(ValueError):self.read(p,s)
    def test_duplicate_missing_tile_and_unbound_dso(self):
        p,s=self.query('B');s['candidate_assignments'][0].append(1)
        with self.assertRaises(ValueError):self.read(p,s)
        p,s=self.query('B');self.dso.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'unbound'):self.read(p,s)
    def test_architecture_and_vector_view_not_guessed(self):
        p,s=self.query('B');p['arch']=2201
        with self.assertRaises(ValueError):self.read(p,s)
        p,s=self.query('B');p['resource_map']['SoCInfo']['core_type_list']='CubeCore,VectorCore'
        with self.assertRaises(ValueError):self.read(p,s)
        p,s=self.query('B');s['independent_vector']=8
        with self.assertRaises(ValueError):self.read(p,s)
    def test_derivation_and_no_execute(self):
        derive.verify();text=(HERE/'op_host/d_flash_group_quant_linear.cpp').read_text()
        self.assertNotIn('SetBlockDim(8)',text);self.assertIn('scheduling.selected',text)
        runner=(HERE/'runner/main.cpp').read_text();self.assertNotIn('aclnnDFlashGroupQuantLinear(',runner)
        self.assertEqual((HERE/'runner/main.cpp').read_bytes(),(PD/'runner/main.cpp').read_bytes())
    def test_cli_gates(self):
        needs={'source':['baseline-root'],'build':['source-report'],'atc':['source-report','build-report'],'aclnn':['source-report','build-report'],'compare':['source-report','build-report','atc-report','aclnn-report']}
        for phase,flags in needs.items():
            args=[phase,'--output-dir',str(self.root/'new')]
            with patch('sys.stderr'),self.assertRaises(SystemExit):run_scheduler.parse(args)
            for key in flags:args+=['--'+key,str(self.root/key)]
            self.assertEqual(run_scheduler.parse(args).phase,phase)
        proc=subprocess.run([sys.executable,'-B',str(HERE/'run_scheduler.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stderr)
    def test_environment_alias_review_preserves_unknown_scope(self):
        root=self.root/'cann';(root/'aarch64-linux/lib64').mkdir(parents=True);(root/'lib64').symlink_to('aarch64-linux/lib64')
        lib=root/'aarch64-linux/lib64/libplatform.so';lib.write_bytes(b'library')
        a={'LD_LIBRARY_PATH':str(root/'aarch64-linux/lib64')};b={'LD_LIBRARY_PATH':str(root/'lib64')}
        r=prefix_review(root,a,b,{'records':[{'file':c.identity(lib)}]})
        self.assertTrue(r['library_resolution']['libplatform.so']['equal'])
        self.assertEqual(r['actual_atc_full_maps'],'NOT_CAPTURED');self.assertFalse(r['directories']['aarch64-linux/python/site-packages']['exists'])
        verify_prefix_review(r)
        (root/'aarch64-linux/python/site-packages').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError,'prefix directory'):verify_prefix_review(r)


class CppContract(unittest.TestCase):
    @unittest.skipUnless(shutil.which('c++'),'C++ compiler unavailable')
    def test_actual_helper_selection_and_invalid_sdk_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'platform').mkdir()
            (root/'platform/platform_infos_def.h').write_text('''#pragma once
#include <map>
#include <string>
namespace fe {struct PlatFormInfos {unsigned GetCoreNum()const{return 7;}unsigned GetCoreNumWithLock()const{return 7;}bool GetPlatformResWithLock(std::map<std::string,std::map<std::string,std::string>>&){return true;}};}
''')
            code='''#include <cassert>
#include <cstdint>
namespace platform_ascendc {enum class CoreMemType {UB,L1,L0_A,L0_B,L0_C};enum class SocVersion {ASCEND310P=2};
struct PlatformAscendC {unsigned lastSlice=0,arch=2002;bool bad=false;unsigned GetCoreNum()const{return 7;}unsigned GetCoreNumAic()const{return 8;}unsigned GetCoreNumAiv()const{return 8;}unsigned GetCoreNumVector()const{return 7;}SocVersion GetSocVersion()const{return SocVersion::ASCEND310P;}unsigned GetCurNpuArch()const{return arch;}void GetCoreMemSize(CoreMemType,uint64_t&v)const{v=65536;}unsigned CalcTschBlockDim(unsigned s,unsigned a,unsigned v){lastSlice=s;assert(a==8&&v==8);return bad?0:s;}};}
#include "scheduler_evidence.h"
int main(){platform_ascendc::PlatformAscendC p;dflash_scheduler::Plan r;assert(dflash_scheduler::Make(p,7,7,40,0,r));assert(p.lastSlice==8);assert(r.selected==(DFLASH_SCHEDULER_POLICY?8:7));p.bad=true;assert(!dflash_scheduler::Make(p,7,7,40,0,r));p.bad=false;p.arch=2201;assert(!dflash_scheduler::Make(p,7,7,40,0,r));}
'''
            cpp=root/'test.cpp';cpp.write_text(code)
            for variant in (0,1):
                out=root/('test'+str(variant));args=['c++','-std=c++11','-DDFLASH_SCHEDULER_POLICY='+str(variant),'-I'+str(root),'-I'+str(HERE/'op_host'),'-I'+str(PD/'op_host'),str(cpp),'-o',str(out)]
                if sys.platform!='darwin':args.append('-ldl')
                subprocess.run(args,check=True,capture_output=True);subprocess.run([str(out)],check=True)


if __name__=='__main__':unittest.main()
