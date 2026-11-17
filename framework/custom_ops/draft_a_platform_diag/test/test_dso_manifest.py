import os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from platform_common import c
import dso_manifest as d
from platform_contracts import loaded_libraries,verify_loaded


class Manifest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.cann=self.root/'cann';self.driver=self.root/'driver';self.opp=self.root/'diag';self.old=self.root/'old'
        self.rel='opp/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so'
        self.standard=self.file(self.cann/self.rel,b'\x7fELF standard fixture')
        self.runtime=self.file(self.cann/'aarch64-linux/lib64/libruntime.so',b'\x7fELF runtime')
        (self.cann/'lib64').symlink_to('aarch64-linux/lib64')
        self.hal=self.file(self.driver/'lib64/driver/libascend_hal.so',b'\x7fELF driver')
        self.private=self.file(self.opp/'vendors/customize/libcust_opapi.so',b'\x7fELF private')
        self.old_private=self.file(self.old/'vendors/customize/libcust_opapi.so',b'\x7fELF old private')
        self.maps=self.root/'maps.txt';self.maps.write_text(''.join(self.line(p) for p in (self.standard,self.runtime,self.hal,self.old_private)))
        self.report=self.root/'old-report.json';self.report.write_text('{}\n')
        self.policy=dict(schema=1,standard_opp=[dict(relative=self.rel,bytes=self.standard.stat().st_size,sha256=c.sha(self.standard),mode=0o644)],runtime_lib64=['libruntime.so'],builtin_host=[],driver=['lib64/driver/libascend_hal.so'],reference_files={'maps.txt':{k:c.identity(self.maps)[k] for k in ('bytes','sha256')}})
        pp=self.root/'policy.json';c.write(pp,self.policy)
        self.patch_policy=patch.object(d,'POLICY',pp);self.patch_policy.start();self.addCleanup(self.patch_policy.stop)
        self.ns=patch.object(d,'mount_namespace',return_value='mnt:[fixture]');self.ns.start();self.addCleanup(self.ns.stop)
        def backing(real,expected):
            st=Path(real).stat();return dict(device=[os.major(st.st_dev),os.minor(st.st_dev)],inode=st.st_ino,method='CPU_FIXTURE',namespace='mnt:[fixture]',scope='NOT_HARDWARE_PROOF')
        self.back=patch.object(d,'readonly_backing',side_effect=backing);self.back.start();self.addCleanup(self.back.stop)
        self.popp=patch.object(c,'verify_opp');self.popp.start();self.addCleanup(self.popp.stop)
        self.opp_data={'install':str(self.opp),'preflight':{'files':[c.identity(self.private)]}}
    def file(self,p,data):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);p.chmod(0o644);return p
    def line(self,p,dev=None,inode=None):
        st=p.stat();dev=dev or [os.major(st.st_dev),os.minor(st.st_dev)]
        return '1000-2000 r-xp 00000000 %02x:%02x %d %s\n'%(dev[0],dev[1],inode if inode is not None else st.st_ino,p)
    def source(self):
        return d.source_manifest(self.cann,self.driver,dict(private_opp={'install':str(self.old)},files={'maps.txt':c.identity(self.maps)},report=c.identity(self.report)))
    def build(self):
        src=self.source();path=self.root/'source-manifest.json';c.write(path,src)
        return d.with_private(src,self.opp_data,c.identity(path))
    def query(self,m,extra=()):
        maps=self.root/'new-maps.txt';maps.write_text(''.join(self.line(p) for p in (self.standard,self.runtime,self.hal,self.private,*extra)))
        return loaded_libraries(maps,self.opp_data,self.cann,m)
    def test_exact_source_and_build_categories(self):
        m=self.build();rows=self.query(m);verify_loaded(rows)
        self.assertEqual({r['category'] for r in rows},{'PINNED_CANN_STANDARD_OPP','STANDARD_CANN_RUNTIME','ASCEND_DRIVER','PRIVATE_DIAGNOSTIC_OPP'})
        row=next(r for r in rows if r['category']=='PINNED_CANN_STANDARD_OPP')
        self.assertEqual(row['stat']['mode'],0o644);self.assertEqual(row['segments'][0]['inode'],row['stat']['inode'])
    def test_pin_sha_and_mode_not_path_only(self):
        self.standard.write_bytes(b'\x7fELF changed')
        with self.assertRaisesRegex(ValueError,'pin differs'):self.source()
    def test_permission_drift_blocks(self):
        m=self.build();self.standard.chmod(0o600)
        with self.assertRaises(ValueError):self.query(m)
    def test_source_to_build_identity_drift_blocks(self):
        m=self.build();self.runtime.write_bytes(b'\x7fELF modified runtime')
        with self.assertRaises(ValueError):d.verify_manifest(m)
    def test_unknown_cann_libcust_and_runtime_block(self):
        m=self.build()
        for name in ('libcust_opapi.so','libunexpected.so'):
            p=self.file(self.cann/'lib64'/name,b'\x7fELF unknown')
            with self.assertRaisesRegex(ValueError,'UNKNOWN'):self.query(m,[p])
    def test_other_user_private_is_not_admitted(self):
        m=self.build();foreign=self.file(self.root/'foreign/libcust_opmaster_rt2.0.so',b'\x7fELF foreign')
        with self.assertRaisesRegex(ValueError,'FOREIGN_PRIVATE_OPP'):self.query(m,[foreign])
    def test_unbound_alias_to_same_bytes_is_rejected(self):
        m=self.build();alias=self.cann/'alias.so';alias.symlink_to(self.standard)
        with self.assertRaisesRegex(ValueError,'unbound'):self.query(m,[alias])
    def test_bound_alias_retarget_and_escape_block(self):
        m=self.build();original=self.cann/'lib64';original.unlink();original.symlink_to(self.root/'foreign')
        with self.assertRaises((ValueError,FileNotFoundError)):d.verify_manifest(m)
    def test_device_and_inode_are_checked(self):
        m=self.build();maps=self.root/'badmaps'
        for dev,ino in (([99,99],self.private.stat().st_ino),(None,self.private.stat().st_ino+1)):
            maps.write_text(self.line(self.private,dev,ino))
            with self.assertRaisesRegex(ValueError,'device/inode'):d.loaded(maps,self.opp_data,self.cann,m)
    def test_per_file_translation_requires_matching_bound_witness(self):
        # Abstract model only: a real Linux read-only mmap test is separate below.
        src=self.source();p=self.root/'source-manifest.json'
        for r in src['records']:r['backing']['device']=[8,17]
        c.write(p,src);m=d.with_private(src,self.opp_data,c.identity(p))
        maps=self.root/'translated';maps.write_text(self.line(self.standard,[8,17])+self.line(self.private))
        rows=d.loaded(maps,self.opp_data,self.cann,m);self.assertEqual(next(x for x in rows if x['category']=='PINNED_CANN_STANDARD_OPP')['backing']['device'],[8,17])
        maps.write_text(self.line(self.standard,[8,18])+self.line(self.private))
        with self.assertRaises(ValueError):d.loaded(maps,self.opp_data,self.cann,m)
    def test_deleted_maps_rejected(self):
        m=self.build();maps=self.root/'deleted';maps.write_text(self.line(self.private).rstrip()+' (deleted)\n')
        with self.assertRaisesRegex(ValueError,'deleted'):d.loaded(maps,self.opp_data,self.cann,m)
    def test_forged_source_inventory_extension_rejected(self):
        src=self.source();p=self.file(self.cann/'lib64/extra.so',b'\x7fELF extra')
        src['records'].append(d.record(p,self.cann,'STANDARD_CANN_RUNTIME'))
        with self.assertRaisesRegex(ValueError,'unreviewed'):d.verify_manifest(src)
    def test_foreign_library_cannot_claim_private_build_category(self):
        m=self.build();p=self.file(self.root/'foreign/libcust_opapi.so',b'\x7fELF foreign')
        m['records'].append(d.record(p,p.parent,'PRIVATE_DIAGNOSTIC_OPP'))
        with self.assertRaisesRegex(ValueError,'private DSO inventory'):d.verify_manifest(m)
    def test_source_manifest_missing_or_namespace_changed(self):
        m=self.build()
        with patch.object(d,'mount_namespace',return_value='mnt:[changed]'):
            with self.assertRaisesRegex(ValueError,'namespace'):d.verify_manifest(m)
        Path(m['source_manifest']['path']).unlink()
        with self.assertRaises(FileNotFoundError):d.verify_manifest(m)


class LinuxMapping(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith('linux'),'Linux proc maps required')
    def test_readonly_file_mapping_binds_bytes_and_backing_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();p=root/'fixture.so';p.write_bytes(b'\x7fELF CPU-only data, not executable\n'*1000)
            before=p.read_bytes();row=d.record(p,root,'SYSTEM');d.verify_record(row)
            self.assertEqual(row['backing']['inode'],p.stat().st_ino);self.assertEqual(p.read_bytes(),before)
            self.assertEqual(row['backing']['method'],'FD_MAP_PRIVATE_PROT_READ_SHA256')


if __name__=='__main__':unittest.main()
