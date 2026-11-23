import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

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
