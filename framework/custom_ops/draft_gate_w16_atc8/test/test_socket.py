"""Linux-only CPU test of the original '?'/'G' gate with real SCM_CREDENTIALS and /proc."""

import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import w16_runtime as run

PROGRAM = r"""
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <unistd.h>
#include <sys/wait.h>
int main(int argc,char **argv) {
    if(argc!=8) return 2;
    std::string dir=argv[3];
    std::ifstream maps("/proc/self/maps");
    std::ofstream out(dir+"/pre-maps.txt");out<<maps.rdbuf();out.close();
    if(std::getenv("TEST_OTHER_PID") && fork()!=0){int s=0;wait(&s);return 0;}
    int fd=std::atoi(argv[7]);
    char reply=0;
    if(write(fd,"?",1)!=1||read(fd,&reply,1)!=1||reply!='G')return 3;
    std::ofstream(dir+"/cpu-released")<<"CPU fixture only";
    return 0;
}
"""


@unittest.skipUnless(
    sys.platform.startswith("linux") and hasattr(socket, "SO_PASSCRED"), "Linux credentials and procfs"
)
class GateSocketTests(unittest.TestCase):
    def test_real_sender_endpoint_and_prelaunch_denials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            cpp = root / "fixture.cpp"
            cpp.write_text(PROGRAM)
            binary = root / "fixture"
            subprocess.run(["c++", "-std=c++17", cpp, "-o", binary], check=True)
            runner = run.c.identity(binary)
            om = root / "not-a-device-model"
            om.write_bytes(b"CPU fixture")
            model = dict(om=run.c.identity(om))
            for mode in ("pass", "foreign-pid", "rejected-dso"):
                folder = root / mode
                folder.mkdir()
                row = dict(maps={}, launch_authorized=False)
                env = dict(os.environ)
                if mode == "foreign-pid":
                    env["TEST_OTHER_PID"] = "yes"

                def verify(*args):
                    if mode == "rejected-dso":
                        raise ValueError("unbound fixture DSO")
                    return ["CPU fixture"]

                with patch.object(run, "verify_maps", verify):
                    if mode == "pass":
                        run.invocation(folder, runner, model, "smoke", env, {}, {}, row)
                        self.assertEqual(row["gate"]["pid"], row["gate"]["child_pid"])
                        self.assertTrue((folder / "cpu-released").is_file())
                    else:
                        with self.assertRaises(ValueError):
                            run.invocation(folder, runner, model, "smoke", env, {}, {}, row)
                        self.assertFalse(row["launch_authorized"])
                        self.assertFalse((folder / "cpu-released").exists())
