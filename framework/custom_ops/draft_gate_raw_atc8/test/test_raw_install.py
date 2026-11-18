"""Exercise the packaging step after compilation, including the real Host loop."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import build_gate as build


class RawInstall(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve()
        self.old=self.root/'off'; self.install=self.root/'raw'; self.build=self.root/'device-build'
        self.targets=[Path('host/tiling')/build.HOST_LIBS[0],Path('host/proto')/build.HOST_LIBS[1],
                      Path('kernel')/build.STEM/'Gate.o',Path('kernel')/build.STEM/'Gate.json']

    def prepare(self,products=None):
        for index,target in enumerate(self.targets):
            dest=self.old/target; dest.parent.mkdir(parents=True,exist_ok=True)
            if products:
                shutil.copyfile(products['off'][index],dest)
            else:
                dest.write_bytes(f'control-{index}'.encode())
                source=self.build/target; source.parent.mkdir(parents=True,exist_ok=True)
                source.write_bytes(f'candidate-{index}'.encode())
        (self.old/'host'/'liboptiling.so').symlink_to(Path('tiling')/build.HOST_LIBS[0])
        (self.old/'unchanged').write_bytes(b'api')
        shutil.copytree(self.old,self.install,symlinks=True)
        self.before=build.tree(self.old)
        self.kernel={key:build.c.identity(self.old/path) for key,path in zip(('elf','metadata'),self.targets[-2:])}
        self.sources=products['raw'] if products else [self.build/path for path in self.targets]
        self.compiler={key:build.c.identity(path) for key,path in zip(('binary','metadata'),self.sources[-2:])}

    def install_raw(self):
        return build.install_raw_outputs(self.old,self.install,self.build,self.before,self.kernel,self.compiler,self.root,dict(os.environ))

    def assert_install(self):
        allowed=self.install_raw()
        self.assertEqual(allowed,{str(path) for path in self.targets})
        self.assertEqual(build.compare_tree(self.before,build.tree(self.install),allowed),allowed)
        for target,source in zip(self.targets,self.sources):
            self.assertEqual((self.install/target).read_bytes(),source.read_bytes())
        self.assertEqual(build.tree(self.old),self.before)
        self.assertEqual((self.install/'host/liboptiling.so').resolve(),self.install/self.targets[0])

    def test_four_distinct_destinations(self):
        self.prepare()
        with patch.object(build.c,'dependencies',return_value=[]) as deps:self.assert_install()
        self.assertEqual([Path(call.args[0]['path']) for call in deps.call_args_list],
                         [base/path for path in self.targets[:2] for base in (self.install,self.old)])

    def test_wrong_metadata_rejected_before_copy(self):
        self.prepare()
        self.compiler['metadata']=build.c.identity(self.sources[1])
        with self.assertRaisesRegex(ValueError,'wrong Raw artifact'):self.install_raw()
        self.assertEqual(build.compare_tree(self.before,build.tree(self.install),set()),set())

    @unittest.skipUnless(os.getenv('GATE8_LOCAL_BUILD_REPORT'),'optional real Linux CANN packaging fixture')
    def test_real_compiled_artifacts_and_ldd(self):
        report=Path(os.environ['GATE8_LOCAL_BUILD_REPORT']).resolve()
        compilers=json.loads(report.read_text())['compilers']; products={}
        for mode in ('off','raw'):
            output=report.parent/mode/'device-build'
            if mode=='raw':self.build=output
            products[mode]=[output/'op_host'/name for name in build.HOST_LIBS]+[
                build.c.verify(compilers[mode][key]) for key in ('binary','metadata')]
        self.targets[-2:]=[path.with_name(source.name) for path,source in zip(self.targets[-2:],products['raw'][-2:])]
        self.prepare(products); self.assert_install()
