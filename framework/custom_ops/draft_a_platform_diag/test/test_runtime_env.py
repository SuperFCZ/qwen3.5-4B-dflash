import copy,json,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from platform_common import c
import platform_runtime_env as e
from platform_contracts import loaded_libraries,validate_runtime
from isolated_atc import IsolatedAtcRunner
from test_platform_diag import runtime


class RuntimeEnv(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.cann=self.root/'cann';self.driver=self.root/'driver';self.opp=self.root/'diag-opp'
        for rel in (*e.CANN_LD,'opp','python/site-packages','opp/built-in/op_impl/ai_core/tbe','bin'):(self.cann/rel).mkdir(parents=True,exist_ok=True)
        for rel in e.DRIVER_LD:(self.driver/rel).mkdir(parents=True,exist_ok=True)
        (self.opp/'vendors/customize/op_api/lib').mkdir(parents=True)
        for name in e.CANN_LIBS:(self.cann/'lib64'/name).write_bytes(b'CPU SDK fixture '+name.encode())
        (self.driver/'lib64/driver/libascend_hal.so').write_bytes(b'CPU Driver fixture')
        self.api=self.opp/'vendors/customize/op_api/lib/libcust_opapi.so';self.api.write_bytes(b'private fixture')
        self.base=dict(HOME=str(self.root),LANG='C.UTF-8',PATH='/poison/bin',ASCEND_CUSTOM_OPP_PATH='/home/other/vendors/customize_dynamic:/home/another/customize_grouped',LD_LIBRARY_PATH='/home/other/vendors/customize_quantMatmul/op_api/lib:/wrong/cann/lib64',PYTHONPATH='/other/custom/op',LD_PRELOAD='/other/evil.so',BASH_ENV='/other/startup',ASCEND_RT_VISIBLE_DEVICES='0')
    def make(self,ready=True,opp=True):
        return e.make_environment(self.cann,self.opp if opp else None,base=self.base,driver_root=self.driver,ready=ready)
    def test_minimal_environment_does_not_mutate_parent(self):
        original=dict(self.base);env,binding=self.make()
        self.assertEqual(self.base,original);e.verify_binding(binding);e.assert_invocation(env,binding)
        self.assertEqual(env['ASCEND_CUSTOM_OPP_PATH'],str(self.opp/'vendors/customize'))
        self.assertEqual(env['ASCEND_OPP_PATH'],str(self.cann/'opp'))
        self.assertEqual(env['ASCEND_RT_VISIBLE_DEVICES'],'0')
        self.assertNotIn('/home/other',json.dumps(env));self.assertNotIn('/wrong',json.dumps(env))
        self.assertNotIn('/poison',env['PATH']);self.assertNotIn('LD_PRELOAD',env);self.assertNotIn('BASH_ENV',env)
        self.assertEqual(binding['replaced_parent_paths']['LD_LIBRARY_PATH'],original['LD_LIBRARY_PATH'])
        self.assertTrue(env['LD_LIBRARY_PATH'].startswith(str(self.opp/'vendors/customize/op_api/lib')+':'))
    def test_source_has_no_private_opp_search(self):
        env,binding=self.make(opp=False);self.assertEqual(env['ASCEND_CUSTOM_OPP_PATH'],'');e.verify_binding(binding)
    def test_missing_cann_and_driver_library_block(self):
        for path in (self.cann/'lib64/libnnopbase.so',self.driver/'lib64/driver/libascend_hal.so'):
            b=path.read_bytes();path.unlink()
            with self.assertRaises(FileNotFoundError):self.make()
            path.write_bytes(b)
    def test_missing_private_api_blocks_ready_runtime(self):
        self.api.unlink()
        with self.assertRaises(FileNotFoundError):self.make()
        env,_=self.make(ready=False);self.assertEqual(env['ASCEND_CUSTOM_OPP_PATH'],str(self.opp/'vendors/customize'))
    def test_ld_directory_symlink_escape_blocks(self):
        d=self.cann/'lib64/plugin/nnengine';d.rmdir();out=self.root/'foreign';out.mkdir();d.symlink_to(out)
        with self.assertRaisesRegex(ValueError,'escapes'):self.make()
    def test_library_sha_or_alias_drift_blocks(self):
        _,b=self.make();self.api.write_bytes(b'new API')
        with self.assertRaises(ValueError):e.verify_binding(b)
    def test_cannot_append_unknown_vendor_or_library_path(self):
        env,b=self.make()
        for key,value in [('ASCEND_CUSTOM_OPP_PATH',env['ASCEND_CUSTOM_OPP_PATH']+':/other/private'),('LD_LIBRARY_PATH',env['LD_LIBRARY_PATH']+':/other/lib')]:
            bad=copy.deepcopy(b);bad['values'][key]=value
            with self.assertRaises(ValueError):e.verify_binding(bad)
        with self.assertRaises(ValueError):e.assert_invocation(dict(env,LD_PRELOAD='/evil.so'),b)
    def test_phase_environment_difference_is_explicit(self):
        env,b=self.make();a=dict(env,DFLASH_PLATFORM_DIAG_ROUTE='ATC',AI_RUN_DIR='/out/atc');r=dict(env,DFLASH_PLATFORM_DIAG_ROUTE='ACLNN')
        e.assert_invocation(a,b);e.assert_invocation(r,b);diff=e.compare_invocations(a,r)
        self.assertTrue(diff['stable_environment_equal']);self.assertEqual(set(diff['launch_differences']),{'DFLASH_PLATFORM_DIAG_ROUTE','AI_RUN_DIR'})
        with self.assertRaises(ValueError):e.compare_invocations(a,dict(r,ASCEND_RT_VISIBLE_DEVICES='1'))
    def test_unknown_libcust_op_inside_cann_still_blocked(self):
        p=self.cann/'lib64/libcust_opmaster_rt2.0.so';p.write_bytes(b'unknown vendor')
        maps=self.root/'maps';maps.write_text('1000-2000 r-xp 0 08:11 12 '+str(p)+'\n')
        opp=dict(install=str(self.opp),preflight=dict(files=[c.identity(self.api)]))
        with self.assertRaisesRegex(ValueError,'different Ascend installation/private OPP'):loaded_libraries(maps,opp,self.cann)
    def test_external_private_mapping_still_blocked(self):
        p=self.root/'other/libcust_opmaster_rt2.0.so';p.parent.mkdir();p.write_bytes(b'other')
        maps=self.root/'maps';maps.write_text('1000-2000 r-xp 0 08:11 12 '+str(p)+'\n')
        opp=dict(install=str(self.opp),preflight=dict(files=[c.identity(self.api)]))
        with self.assertRaises(ValueError):loaded_libraries(maps,opp,self.cann)
    def test_runtime_still_two_queries_zero_execute(self):
        r,j=runtime();validate_runtime(r,j)
        with self.assertRaises(ValueError):validate_runtime(dict(r,execute_calls=1),j)
        with self.assertRaises(ValueError):validate_runtime(r,j[:-1])
        text=(HERE/'runner/main.cpp').read_text();self.assertIn('i<2',text);self.assertNotIn('aclnnDFlashGroupQuantLinear(',text)
    def test_atc_adapter_passes_explicit_sealed_env(self):
        env,b=self.make();out=self.root/'out';out.mkdir();tool=self.cann/'bin/atc';tool.write_bytes(b'fake, never executed')
        args=[str(tool),'--output='+str(out/'fc')]
        def fake(argv,**kw):
            self.assertEqual(kw['env'],env);self.assertNotIn('/home/other',json.dumps(kw['env']))
            kw['stdout'].write(b'complete output\n');return subprocess.CompletedProcess(argv,0)
        runner=IsolatedAtcRunner(out/'atc',c.identity(tool),out,env=env,binding=b,expected_command=args)
        with patch('isolated_atc.subprocess.run',side_effect=fake):self.assertEqual(runner(args,out).returncode,0)
        self.assertEqual(runner.records[0]['environment'],env)
    @unittest.skipUnless(sys.platform.startswith('linux') and shutil.which('cc'),'requires Linux ELF loader')
    def test_real_cpu_loader_selects_only_diagnostic_private_api(self):
        src=self.root/'library.c';src.write_text('int platform_fixture_value(void){return 42;}\n')
        subprocess.run(['cc','-shared','-fPIC',str(src),'-o',str(self.api)],check=True)
        foreign=self.root/'foreign';foreign.mkdir();src.write_text('int platform_fixture_value(void){return 99;}\n')
        subprocess.run(['cc','-shared','-fPIC',str(src),'-o',str(foreign/'libcust_opapi.so')],check=True)
        self.base['LD_LIBRARY_PATH']=str(foreign);env,b=self.make()
        # This helper only loads a tiny CPU fixture DSO, never CANN/ACLNN/NPU.
        script='import ctypes,pathlib; x=ctypes.CDLL("libcust_opapi.so"); print(x.platform_fixture_value()); print(pathlib.Path("/proc/self/maps").read_text())'
        p=subprocess.run([sys.executable,'-c',script],env=env,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);self.assertTrue(p.stdout.startswith('42\n'));self.assertNotIn(str(foreign),p.stdout)
        e.verify_binding(b)


if __name__=='__main__':unittest.main()
