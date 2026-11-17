import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import atc_dependency_identity as d
from platform_common import c


class AtcDependencies(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.cann=self.root/'cann';self.bin=self.cann/'aarch64-linux/bin';self.bin.mkdir(parents=True)
        self.tools=self.root/'tools';self.tools.mkdir();self.calls=[]
        for name in ('bash','realpath','dirname','ldd'):self.file(self.tools/name,b'\x7fELFfixture '+name.encode(),0o755)
        self.main=self.file(self.bin/'atc.bin',b'\x7fELFfixture ATC',0o555)
        self.wrapper=self.file(self.bin/'atc',b'#!/bin/bash\n# Authored CPU fixture, not SDK source.\n./atc.bin "$@"\n',0o555)
        self.expected=c.identity(self.wrapper)
        # The profile registry is mocked only in tests. Production has one full-file SDK SHA.
        self.profile=dict(name='CPU_FIXTURE',bytes=self.expected['bytes'],mode=0o555,shebang='#!/bin/bash',interpreter=str(self.tools/'bash'),
            target_bytes=self.main.stat().st_size,target_sha256=c.sha(self.main),target_mode=0o555)
        self.registry=patch.dict(d.REVIEWED_WRAPPERS,{self.expected['sha256']:self.profile});self.registry.start();self.addCleanup(self.registry.stop)
        self.lib=self.file(self.cann/'aarch64-linux/lib64/libtest.so',b'fixture dependency',0o444)
        self.env=dict(PATH=str(self.tools),LD_LIBRARY_PATH=str(self.cann/'lib64'),PYTHONPATH='/python-before')
    def file(self,p,data,mode=0o555):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);p.chmod(mode);return p
    def fake_ldd(self,argv,stdout,stderr,env,check):
        self.assertEqual(argv[0],str(self.tools/'ldd'));self.assertEqual(len(argv),2)
        self.assertNotEqual(argv[1],str(self.wrapper));self.assertTrue(Path(argv[1]).read_bytes().startswith(b'\x7fELF'))
        self.assertFalse(check);self.calls.append(dict(argv=argv,env=dict(env)))
        stdout.write(('linux-vdso.so.1 (0x000001)\nlibtest.so => '+str(self.lib)+' (0x000002)\n').encode())
        return subprocess.CompletedProcess(argv,0)
    def audit(self,entry=None,env=None,callback=None,name='audit'):
        entry=entry or self.wrapper
        with patch.object(d.subprocess,'run',side_effect=callback or self.fake_ldd):
            return d.audit(c.identity(entry),str(entry),self.cann,env or self.env,self.root/name)
    def test_direct_elf_checks_one_real_executable(self):
        r=self.audit(self.main);self.assertEqual(r['chain']['kind'],'DIRECT_ELF');self.assertEqual(len(self.calls),1)
        self.assertEqual(r['executables'][0]['role'],'atc');d.verify(r)
    def test_wrapper_full_chain_and_original_identity(self):
        before=d.snapshot(self.wrapper,True);r=self.audit();d.verify(r)
        self.assertEqual(d.snapshot(self.wrapper,True),before)
        self.assertEqual(r['chain']['entry']['file'],self.expected)
        self.assertEqual([x['role'] for x in r['executables']],['bash_interpreter','realpath','dirname','atc.bin'])
        self.assertEqual(len(self.calls),4)
        for row in self.calls[:3]:self.assertEqual(row['env']['LD_LIBRARY_PATH'],self.env['LD_LIBRARY_PATH'])
        self.assertEqual(self.calls[-1]['env']['LD_LIBRARY_PATH'],str(self.cann/'aarch64-linux/lib64')+':'+self.env['LD_LIBRARY_PATH'])
        self.assertEqual(self.calls[-1]['env']['PYTHONPATH'],str(self.cann/'aarch64-linux/python/site-packages')+':/python-before')
        self.assertEqual(r['atc'],'NOT_RUN');self.assertEqual(r['npu'],'NOT_RUN')
    def test_entry_alias_and_symlink_chain(self):
        alias=self.cann/'bin/atc';alias.parent.mkdir();alias.symlink_to('../aarch64-linux/bin/atc')
        r=self.audit(alias);self.assertTrue(r['chain']['entry']['links']);self.assertEqual(r['chain']['entry']['alias'],str(alias))
        alias.unlink();alias.symlink_to('../aarch64-linux/bin/atc.bin')
        with self.assertRaises(ValueError):d.verify(r)
    def test_target_content_drift_rejected(self):
        r=self.audit();self.main.chmod(0o755);self.main.write_bytes(b'\x7fELFchanged')
        with self.assertRaises(ValueError):d.verify(r)
        self.main.chmod(0o555)
        with self.assertRaisesRegex(ValueError,'server-confirmed ELF'):self.audit(name='second')
    def test_wrapper_sha_unchanged_enforced(self):
        self.wrapper.chmod(0o755);self.wrapper.write_bytes(b'#!/bin/bash\necho replaced\n');self.wrapper.chmod(0o555)
        with patch.object(d.subprocess,'run') as run:
            with self.assertRaisesRegex(ValueError,'identity changed'):d.audit(self.expected,str(self.wrapper),self.cann,self.env,self.root/'failure')
            run.assert_not_called()
    def test_wrapper_mode_drift_rejected(self):
        r=self.audit();self.wrapper.chmod(0o755)
        with self.assertRaises(ValueError):d.verify(r)
        with self.assertRaisesRegex(ValueError,'mode/shebang'):self.audit(name='mode-fail')
    def test_elf_permission_drift_rejected(self):
        r=self.audit();self.main.chmod(0o444)
        with self.assertRaises(ValueError):d.verify(r)
    def test_missing_dynamic_library_fails_even_rc_zero(self):
        def bad(argv,stdout,stderr,**kw):
            stdout.write(b'libmissing.so => not found\n');return subprocess.CompletedProcess(argv,0)
        with self.assertRaisesRegex(ValueError,'not found'):self.audit(callback=bad)
        report=c.read(self.root/'audit/report.json');self.assertEqual(report['dependency_audit'],'INCOMPLETE')
        self.assertEqual(report['executables'][0]['returncode'],0);c.verify(report['executables'][0]['stdout'])
    def test_ldd_nonzero_fails_with_raw_logs(self):
        def bad(argv,stdout,stderr,**kw):
            stderr.write(b'not a dynamic executable\n');return subprocess.CompletedProcess(argv,1)
        with self.assertRaisesRegex(ValueError,'ldd returned 1'):self.audit(callback=bad)
        r=c.read(self.root/'audit/report.json');self.assertEqual(r['status'],'BLOCKED')
        self.assertEqual(c.verify(r['executables'][0]['stderr']).read_bytes(),b'not a dynamic executable\n')
    def test_unknown_wrapper_never_reaches_ldd(self):
        p=self.file(self.bin/'other-atc',b'#!/bin/bash\nexec "$CHOSEN_COMPILER" "$@"\n')
        with patch.object(d.subprocess,'run') as run:
            with self.assertRaisesRegex(ValueError,'unreviewed ATC wrapper'):d.audit(c.identity(p),str(p),self.cann,self.env,self.root/'unknown')
            run.assert_not_called()
    def test_non_elf_helper_blocks_complete_chain(self):
        self.file(self.tools/'realpath',b'#!/bin/bash\nexec unknown\n',0o755)
        with self.assertRaisesRegex(ValueError,'non-ELF command realpath'):self.audit()
        self.assertFalse(self.calls)
    def test_missing_helper_blocks(self):
        (self.tools/'dirname').unlink()
        with self.assertRaisesRegex(ValueError,'not found.*dirname'):self.audit()
        self.assertFalse(self.calls)
    def test_library_identity_drift_rejected(self):
        r=self.audit();self.lib.chmod(0o644);self.lib.write_bytes(b'changed library');self.lib.chmod(0o444)
        with self.assertRaises(ValueError):d.verify(r)
    def test_unknown_ldd_output_is_not_ignored(self):
        for text in ('statically linked\n','',str(self.lib)+' (0x123)\nunknown dependency line\n'):
            with self.subTest(text=text),self.assertRaises(ValueError):d.parse_ldd(text)
    def test_loader_line_without_soname_supported(self):
        rows=d.parse_ldd(str(self.lib)+' (0x1000)\n');self.assertEqual(rows[0]['file'],c.identity(self.lib))
    def test_shell_or_loader_injection_blocks_static_proof(self):
        for key,value in [('BASH_ENV','/injected'),('BASH_FUNC_realpath%%','() { :; }'),('CDPATH','/other'),('LD_PRELOAD','/other.so'),('PATH','.:/usr/bin')]:
            with self.subTest(key=key),self.assertRaises(ValueError):self.audit(env=dict(self.env,**{key:value}),name='env-'+str(len(self.calls))+'-'+key.replace('/','_'))
        self.assertFalse(self.calls)
    def test_target_symlink_cannot_escape_cann(self):
        self.main.unlink();outside=self.file(self.root/'outside',b'\x7fELFfixture ATC');self.main.symlink_to(outside)
        with self.assertRaisesRegex(ValueError,'escapes selected CANN'):self.audit()
    def test_missing_chain_member_cannot_verify_pass(self):
        r=self.audit();bad=copy.deepcopy(r);bad['executables'].pop()
        with self.assertRaisesRegex(ValueError,'incomplete ELF'):d.verify(bad)
    def test_stderr_requires_review(self):
        def bad(argv,stdout,stderr,**kw):
            result=self.fake_ldd(argv,stdout,stderr,**kw);stderr.write(b'loader warning\n');return result
        with self.assertRaisesRegex(ValueError,'stderr requires review'):self.audit(callback=bad)


if __name__=='__main__':unittest.main()
