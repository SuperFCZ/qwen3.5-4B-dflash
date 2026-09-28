"""Compile the shared shape predicates with CPU shape objects, without CANN."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class HostContractTests(unittest.TestCase):
    def test_ge_and_aclnn_metadata_contract(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("CPU metadata regression requires a C++17 compiler")
        source = Path(__file__).with_name("host_contract.cpp")
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "host-contract-test"
            compiled = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                                       str(source), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            executed = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)


if __name__ == "__main__":
    unittest.main()
