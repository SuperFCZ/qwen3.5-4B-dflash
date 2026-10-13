"""CPU filesystem fixtures only; no CANN binary or NPU execution."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import head_opp_preflight as head
import opp_preflight as quant


def installed(root,names=head.OP_KERNELS):
    binaries=[]
    for name in names:
        directory=root/'vendors/customize/op_impl/ai_core/tbe/kernel/ascend310p'/name
        directory.mkdir(parents=True)
        binary=directory/'kernel_test.o'
        binary.write_bytes(b'CPU fixture, not a compiled kernel')
        binary.with_suffix('.json').write_text('{"binFileName":"kernel_test"}\n')
        binaries.append(binary)
    return binaries


class HeadOppPreflightTests(unittest.TestCase):
    def test_both_operators_and_all_pairs_are_recorded_without_changing_a(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'opp'; binaries=installed(root)
            extra=binaries[1].with_name('kernel_second.o'); extra.write_bytes(b'second CPU fixture')
            extra.with_suffix('.json').write_text('{}')
            report=head.check_install(root)
            self.assertEqual(report['status'],'PASS'); self.assertEqual(report['npu_status'],'NOT_RUN')
            self.assertEqual(tuple(x['name'] for x in report['operators']),head.OP_KERNELS)
            self.assertEqual([len(x['kernels']) for x in report['operators']],[1,2])
            for op in report['operators']:
                for pair in op['kernels']:
                    for kind in ('binary','json'):
                        raw=Path(pair[kind]).read_bytes()
                        self.assertEqual(pair[kind+'_bytes'],len(raw))
                        self.assertEqual(pair[kind+'_sha256'],hashlib.sha256(raw).hexdigest())
            self.assertIs(head.check_binary_path,quant.check_binary_path)
            self.assertEqual(quant.OP_KERNEL,'d_flash_group_quant_linear')
            with self.assertRaisesRegex(ValueError,'no installed d_flash_group_quant_linear'):
                quant.check_install(root)

    def test_either_missing_operator_is_rejected(self):
        for missing in head.OP_KERNELS:
            with self.subTest(missing=missing),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/'opp'; installed(root,[n for n in head.OP_KERNELS if n!=missing])
                with self.assertRaisesRegex(ValueError,missing): head.check_install(root)

    def test_missing_or_empty_object_in_either_operator_is_rejected(self):
        for index in (0,1):
            for empty in (False,True):
                with self.subTest(index=index,empty=empty),tempfile.TemporaryDirectory() as directory:
                    root=Path(directory)/'opp'; binary=installed(root)[index]
                    if empty: binary.write_bytes(b'')
                    else: binary.unlink()
                    with self.assertRaisesRegex(ValueError,'empty kernel|no installed'): head.check_install(root)

    def test_missing_malformed_empty_and_non_object_json(self):
        for index in (0,1):
            for data in (None,b'',b'not json',b'[]',b'null',b'\xff'):
                with self.subTest(index=index,data=data),tempfile.TemporaryDirectory() as directory:
                    root=Path(directory)/'opp'; metadata=installed(root)[index].with_suffix('.json')
                    if data is None: metadata.unlink()
                    else: metadata.write_bytes(data)
                    with self.assertRaises((OSError,ValueError)): head.check_install(root)

    def test_unsafe_root_filename_and_symlink_prefixes(self):
        for name in ('head.older','head.json-cache'):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/name; installed(root)
                with self.assertRaisesRegex(ValueError,'path collision'): head.check_install(root)
                alias=Path(directory)/'head-safe'; alias.symlink_to(root,target_is_directory=True)
                with self.assertRaisesRegex(ValueError,'path collision'): head.check_install(alias)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'opp'; binary=installed(root)[1]
            binary.rename(binary.with_name('kernel.old.o'))
            with self.assertRaisesRegex(ValueError,'path collision'): head.check_install(root)

    def test_cli_saves_pass_and_failure_reports(self):
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            root=Path(directory)/'opp'; binaries=installed(root); evidence=Path(directory)/'evidence.json'
            with patch('sys.argv',['head_opp_preflight.py','--install-root',str(root),'--report',str(evidence)]):
                self.assertEqual(head.main(),0)
                self.assertEqual(len(json.loads(evidence.read_text())['operators']),2)
                binaries[1].with_suffix('.json').unlink()
                self.assertEqual(head.main(),1)
                report=json.loads(evidence.read_text())
                self.assertEqual(report['status'],'FAIL'); self.assertEqual(report['npu_status'],'NOT_RUN')
                self.assertIn(head.OP_KERNELS[1],report['error'])


if __name__=='__main__': unittest.main()
