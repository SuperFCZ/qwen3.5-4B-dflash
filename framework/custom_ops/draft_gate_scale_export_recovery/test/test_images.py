import tempfile
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from archive import c
from evidence import NODES
from elf_identity import sha
from elf_layout import table
from full_proof import compare_images
from test_elf import elf, SYMBOL
from test_native_dfx import obj, CAST, CAST_NOTES


class EmbeddedImages(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.names = ["DFlashGroupQuantLinear" + (("_" + str(i)) if i else "") for i in range(26)]
        self.targets = set(NODES.values())

    def models(self, other_delta=False):
        ms = []
        images = []
        for mode in (0, 1):
            chunks = []
            gql = {}
            native = {}

            def add(n, symbol, code, custom):
                offset = sum(map(len, chunks))
                chunks.append(code)
                texts = [dict(bytes=s["bytes"], sha256=s["sha256"]) for s in table(code)[1] if s["flags"] & 4]
                rec = dict(
                    node=n,
                    sha256=sha(code),
                    bytes=len(code),
                    symbol=symbol,
                    offset=offset,
                    executable_sections=texts,
                )
                if custom:
                    rec.update(block_dim=8, engine="AIcoreEngine", id=symbol.split("__kernel")[0] + sha(code))
                return rec

            for n in self.names:
                code = bytearray(elf("629406" if mode else "607982"))
                if mode and (n in self.targets or other_delta and n == self.names[0]):
                    at = next(s["offset"] for s in table(code)[1] if s["name"] == ".text")
                    code[at] ^= 1
                gql[n] = add(n, SYMBOL, bytes(code), True)
            native["MatMul"] = add("MatMul", SYMBOL, elf(), False)
            native["Cast_114Mul_34"] = add("Cast_114Mul_34", CAST, obj(CAST, CAST_NOTES[mode]), False)
            p = self.root / str(mode)
            p.write_bytes(b"".join(chunks))
            ms.append(dict(om=c.identity(p), atc_kernels=gql))
            images.append(native)
        return ms, images

    def test_actual_selected_bytes_and_scoped_exceptions(self):
        m, n = self.models()
        l, r, result = compare_images(*m, self.targets, n)
        self.assertEqual(set(result["targets"]), self.targets)
        self.assertEqual(result["counts"]["non_gate_projections"], {"LOCAL_STT_FILE_TAG_ONLY": 21})
        self.assertEqual(
            result["counts"]["native"], {"EXACT_ELF": 1, "ET_REL_DFX_ARGUMENT_DESCRIPTORS_ONLY": 1}
        )
        with self.assertRaisesRegex(ValueError, "protected entry"):
            compare_images(*self.models(True)[0], self.targets, n)

    def test_binding_key_and_offset_fail_closed(self):
        m, n = self.models()
        key = self.names[0]
        m[1]["atc_kernels"][key]["id"] = "other" + m[1]["atc_kernels"][key]["sha256"]
        with self.assertRaisesRegex(ValueError, "compiler key"):
            compare_images(*m, self.targets, n)
        m, n = self.models()
        m[1]["atc_kernels"][key]["offset"] += 1
        with self.assertRaisesRegex(ValueError, "embedded ELF"):
            compare_images(*m, self.targets, n)
