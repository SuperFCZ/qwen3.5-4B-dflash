"""The real kernel body under a CPU API model; no Ascend SDK/device evidence."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class KernelModelTests(unittest.TestCase):
    def test_all_n_k_tiles_against_dense_oracle(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("CPU kernel control-flow model requires C++17/_Float16 support")
        here = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "kernel-model"
            command = [compiler, "-std=c++17", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                       "-Wno-unused-parameter", "-I", str(here / "cpu_kernel_include"),
                       str(here / "kernel_model.cpp"), "-o", str(binary)]
            compiled = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            executed = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)
            self.assertIn("NPU NOT_RUN", executed.stdout)


if __name__ == "__main__":
    unittest.main()
