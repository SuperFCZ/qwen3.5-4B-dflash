import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recover_inputs import recover, GATE
from restore_python import restore, snapshot


class Recovery(unittest.TestCase):
    def test_restore_content_and_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "surviving"
            source.mkdir()
            dest = root / "new/inputs"
            (source / "x.bin").write_bytes(b"bad")
            (source / "renamed.bin").write_bytes(b"yes")
            (source / "link.bin").symlink_to(source / "gone")
            expected = {"x.bin": (3, hashlib.sha256(b"yes").hexdigest()), "golden.bin": (4,"0"*64)}
            result = recover([root],dest,expected)
            self.assertEqual(result["recovered"]["x.bin"],str(source/"renamed.bin"))
            self.assertEqual(result["missing"],["golden.bin"])
            self.assertEqual((dest/"x.bin").read_bytes(),b"yes")
            self.assertEqual(recover([root],dest,expected)["recovered"]["x.bin"],str(dest/"x.bin"))
            (dest/"x.bin").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError,"preserved"):
                recover([root],dest,expected)
            self.assertEqual((dest/"x.bin").read_bytes(),b"bad")

    def test_torchair_can_be_an_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp).resolve()/"conda"
            venv=Path(tmp).resolve()/"overlay"
            packages={}
            for name,folder in (("torch",base),("torch_npu",base),("torchair",base),("transformers",venv)):
                module=ModuleType("torch_npu.dynamo.torchair" if name=="torchair" else name)
                p=folder/name/"__init__.py"
                p.parent.mkdir(parents=True)
                p.write_text("pass")
                module.__file__=str(p)
                module.__version__="5.13.1" if name=="transformers" else "test"
                packages[name]=module
            init=ModuleType("transformers.initialization")
            init.copy_=lambda: None
            packages[init.__name__]=init
            with patch.dict(sys.modules,packages),patch.object(sys,"prefix",str(venv)),patch.object(sys,"base_prefix",str(base)),patch.object(sys,"path",list(sys.path)):
                state=snapshot(venv,base)
            self.assertEqual(state["torchair"]["module"],"torch_npu.dynamo.torchair")
            self.assertFalse(state["historical_init_files_match"])

    def test_reject_symlink_destination_and_wrong_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"x.bin").symlink_to(root/"outside")
            with self.assertRaisesRegex(ValueError,"preserved"):
                recover([],root,GATE)
            with self.assertRaisesRegex(ValueError,"prefix"):
                snapshot(root,root)
            with patch.object(sys,"version_info",(3,8)):
                with self.assertRaisesRegex(ValueError,"ls_llm"):
                    restore(root,root)
