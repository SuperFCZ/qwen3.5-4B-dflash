"""Every import-order/admission scenario runs in a clean independent interpreter."""
import subprocess,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1]
HARNESS=HERE/'test/import_dependency_case.py'

@pytest.mark.parametrize('order',['raw_first','history_first','validator_first','models_first'])
@pytest.mark.parametrize('action',['imports','good','wrong_target','tampered_header'])
def test_import_order_and_target_identity(order,action,tmp_path):
 cp=subprocess.run([sys.executable,'-B',str(HARNESS),order,action,str(tmp_path/'fixture')],capture_output=True,text=True,timeout=60)
 assert cp.returncode==0,cp.stdout+cp.stderr

@pytest.mark.parametrize('action',['depfile','disagree','missing','escaped_header','wrong_mapping','tampered_after','retarget_after','wrong_function','wrong_module'])
def test_dependency_and_validator_rejections(action,tmp_path):
 cp=subprocess.run([sys.executable,'-B',str(HARNESS),'history_first',action,str(tmp_path/'fixture')],capture_output=True,text=True,timeout=60)
 assert cp.returncode==0,cp.stdout+cp.stderr

@pytest.mark.parametrize('entry',['raw_host.py','analyze.py'])
def test_cli_help(entry):
 cp=subprocess.run([sys.executable,'-B',str(HERE/entry),'--help'],capture_output=True,text=True,timeout=60)
 assert cp.returncode==0,cp.stderr
 assert '--frozen-root' in cp.stdout and '--output-dir' in cp.stdout


def test_renames_are_not_shims_or_parser_changes():
 assert not (HERE/'dependency_identity.py').exists()
 assert not (HERE/'models.py').exists()
 code=(HERE/'raw_host.py').read_text()
 assert 'from raw_host_dependency_identity import compiled_dependencies,verify_dependency_binding' in code
 assert 'sys.modules' not in code.replace('# Also protect direct imports and --help; do not alter sys.modules or search paths.','')
