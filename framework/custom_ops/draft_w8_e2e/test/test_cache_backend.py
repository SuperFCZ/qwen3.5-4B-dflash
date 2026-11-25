import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cache_backend import provider, verify_nodes
from winners import c


class CacheBackend(unittest.TestCase):
    def test_provider_requires_tensor_indices_and_310p(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inc = root / "op_api/include"
            lib = root / "op_api/lib"
            dev = root / "op_impl/ai_core/tbe/kernel/ascend310p/cache_update"
            for d in (inc, lib, dev):
                d.mkdir(parents=True)
            h = inc / "aclnn_cache_update.h"
            text = "aclnnStatus aclnnCacheUpdateGetWorkspaceSize(const aclTensor *x, const aclTensor *updates, const aclTensor *targetBlock, const aclTensor *offsetInBlock, uint64_t *workspace, aclOpExecutor **executor);"
            h.write_text(text)
            binary = dev / "CacheUpdate_test.o"
            binary.write_bytes(b"\x7fELF\x02\x01fixture")
            binary.with_suffix(".json").write_text(json.dumps({"sha256": c.sha(binary)}))
            (lib / "libcust_opapi.so").write_bytes(b"fixture")
            self.assertEqual(provider(root)["artifacts"][0]["binary"]["sha256"], c.sha(binary))
            h.write_text(text.replace("const aclTensor *targetBlock", "int64_t targetBlock"))
            with self.assertRaisesRegex(ValueError, "tensor-index"):
                provider(root)

    def test_compiled_engine_not_air_name(self):
        def node(name, engine):
            attrs = {
                "_ge_attr_op_kernel_lib_name": {"s": engine},
                "_kernelname": {"s": "te_cacheupdate__kernel0"},
                "tvm_blockdim": {"i": 8},
            }
            return {
                "name": name,
                "type": "CacheUpdate",
                "inputs": ["x:0", "v:0", "block:0", "row:0"],
                "attributes": [dict(key=k, value=v) for k, v in attrs.items()],
            }

        ns = [node("CacheUpdate", "AIcoreEngine"), node("CacheUpdate_1", "AIcoreEngine")]
        rows, refs = verify_nodes(ns, Path("/private"), 2)
        self.assertEqual(len(rows), 2)
        with self.assertRaisesRegex(ValueError, "engine="):
            verify_nodes([ns[0], node("CacheUpdate_1", "aicpu_ascend_kernel")], Path("/private"), 2)
        with self.assertRaisesRegex(ValueError, "inventory"):
            verify_nodes(ns[:1], Path("/private"), 2)


if __name__ == "__main__":
    unittest.main()
