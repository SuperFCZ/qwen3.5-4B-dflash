"""Filesystem race regressions; no CANN, Simulator or Kernel execution."""
import errno
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

SIM=Path(__file__).resolve().parents[1]/'simulator'
sys.path.insert(0,str(SIM))
import bounded_budget as budget


def test_stable_files_one_snapshot_includes_max_and_allocated_size(tmp_path):
    (tmp_path/'small').write_bytes(b'a'*7)
    directory=tmp_path/'tmp_dump';directory.mkdir()
    (directory/'large').write_bytes(b'b'*8000)
    (tmp_path/'link').symlink_to(directory/'large')
    infos=[(tmp_path/'small').lstat(),(directory/'large').lstat()]
    with patch.object(budget,'usage',wraps=budget.usage) as scan:
        result=budget.limits(tmp_path,1024*1024,8192)
        scan.assert_called_once()
    assert result['logical_bytes']==8007 and result['files']==2
    assert result['accounted_bytes']==sum(max(s.st_size,s.st_blocks*512) for s in infos)
    assert result['max_file_bytes']==8000 and result['max_file_path'] in (str(directory/'large'),str(tmp_path/'link'))
    assert result['symlink_targets_checked']==1
    assert result['disappeared_count']==0


def test_file_removed_between_enumeration_and_lstat_is_recorded(tmp_path):
    gone=tmp_path/'core14_ub_log.dump';gone.write_bytes(b'temporary')
    (tmp_path/'stable').write_bytes(b'abc')
    real=Path.lstat;diagnostics={}
    def lstat(path,*args,**kwargs):
        if path==gone:gone.unlink()
        return real(path,*args,**kwargs)
    # The old is_file()/stat() sequence must not be used at all.
    with patch.object(Path,'lstat',lstat),patch.object(Path,'is_file',side_effect=AssertionError('TOCTOU check')):
        result=budget.limits(tmp_path,1024*1024,1024,diagnostics)
    assert result['logical_bytes']==3 and result['files']==1
    assert result['disappeared_count']==diagnostics['disappeared_count']==1
    assert diagnostics['disappeared_samples']==[dict(path=str(gone),operation='lstat',errno=errno.ENOENT)]


def test_parent_removed_before_scandir_is_recorded(tmp_path):
    gone=tmp_path/'tmp_dump';gone.mkdir();child=gone/'core14_ub_log.dump';child.write_bytes(b'x')
    real=os.scandir
    def scandir(path):
        if Path(path)==gone:child.unlink();gone.rmdir()
        return real(path)
    with patch.object(budget.os,'scandir',scandir):result=budget.usage(tmp_path)
    assert result['files']==0 and result['disappeared_count']==1
    assert result['disappeared_samples'][0]['operation']=='scandir'


@pytest.mark.parametrize('where',['lstat','scandir'])
@pytest.mark.parametrize('code',[errno.EACCES,errno.EIO,errno.ENOTDIR])
def test_only_enoent_is_tolerated(tmp_path,where,code):
    child=tmp_path/'tmp_dump';child.mkdir()
    error=OSError(code,'injected real I/O error',str(child))
    real=Path.lstat if where=='lstat' else os.scandir
    def fail(path,*a,**kw):
        if Path(path)==child:raise error
        return real(path,*a,**kw)
    target=Path if where=='lstat' else budget.os
    with patch.object(target,where,fail),pytest.raises(OSError) as caught:budget.limits(tmp_path,1024*1024,1024)
    assert caught.value.errno==code


def test_missing_root_is_not_an_empty_successful_scan(tmp_path):
    with pytest.raises(FileNotFoundError):budget.usage(tmp_path/'missing')


@pytest.mark.parametrize('total,per_file,reason',[(4096,8192,'OUTPUT_BUDGET_EXCEEDED'),(1024*1024,4096,'FILE_BUDGET_EXCEEDED')])
def test_persistent_oversize_is_rejected_even_with_disappearing_neighbor(tmp_path,total,per_file,reason):
    (tmp_path/'persistent').write_bytes(b'x'*8192)
    gone=tmp_path/'temporary';gone.write_bytes(b'x');real=Path.lstat
    def lstat(path,*a,**kw):
        if path==gone:gone.unlink()
        return real(path,*a,**kw)
    with patch.object(Path,'lstat',lstat),pytest.raises(RuntimeError,match=reason):budget.limits(tmp_path,total,per_file)
    with pytest.raises(RuntimeError,match=reason):budget.limits(tmp_path,total,per_file)


class Done:
    returncode=0
    def poll(self):return 0


def test_normal_child_exit_still_rejects_final_oversize(tmp_path):
    def spawn(*a,**kw):
        (tmp_path/'persistent').write_bytes(b'x'*16384)
        return Done()
    command=budget.BudgetCommands(tmp_path,1024*1024,8192)
    with patch.object(budget.subprocess,'Popen',spawn),pytest.raises(RuntimeError,match='FILE_BUDGET_EXCEEDED'):
        command(['fake'],tmp_path/'run.log',{},tmp_path)
    report=json.loads((tmp_path/'run.budget.json').read_text())
    assert report['returncode']==0 and report['status']=='STOPPED'
    assert report['final_budget_error'].startswith('RuntimeError: FILE_BUDGET_EXCEEDED')
    assert report['scan_diagnostics']['scans']==2
    assert report['truncated_output_accepted'] is False


def test_poll_disappearance_is_retained_in_final_audit(tmp_path):
    class Running(Done):
        calls=0
        def poll(self):
            self.calls+=1
            return None if self.calls==1 else 0
    gone=tmp_path/'core14_ub_log.dump';real=Path.lstat
    def spawn(*a,**kw):gone.write_bytes(b'tmp');return Running()
    def lstat(path,*a,**kw):
        if path==gone:gone.unlink()
        return real(path,*a,**kw)
    with patch.object(budget.subprocess,'Popen',spawn),patch.object(Path,'lstat',lstat),patch.object(budget.time,'sleep') as sleep:
        budget.BudgetCommands(tmp_path,1024*1024,8192)(['fake'],tmp_path/'run.log',{},tmp_path)
    sleep.assert_called_once_with(0.25)
    report=json.loads((tmp_path/'run.budget.json').read_text())
    assert report['status']=='COMMAND_COMPLETED'  # Not artifact or numerical PASS.
    assert report['usage']['disappeared_count']==0
    assert report['scan_diagnostics']['disappeared_count']==1
    assert report['scan_diagnostics']['scans']==3


def test_final_scan_permission_error_is_reported_and_raised(tmp_path):
    real=budget.usage;calls=0
    def scan(*a,**kw):
        nonlocal calls
        calls+=1
        if calls==2:raise PermissionError(errno.EACCES,'final scan denied')
        return real(*a,**kw)
    with patch.object(budget.subprocess,'Popen',return_value=Done()),patch.object(budget,'usage',scan),pytest.raises(PermissionError):
        budget.BudgetCommands(tmp_path,1024*1024,8192)(['fake'],tmp_path/'run.log',{},tmp_path)
    report=json.loads((tmp_path/'run.budget.json').read_text())
    assert report['status']=='STOPPED' and report['usage'] is None
    assert report['final_budget_error'].startswith('PermissionError:')


def test_linked_file_still_obeys_single_file_limit_without_double_counting(tmp_path):
    root=tmp_path/'out';root.mkdir()
    external=tmp_path/'large';external.write_bytes(b'x'*8192)
    (root/'link').symlink_to(external)
    result=budget.usage(root)
    assert result['files']==result['logical_bytes']==result['accounted_bytes']==0
    assert result['max_file_bytes']==8192
    with pytest.raises(RuntimeError,match='FILE_BUDGET_EXCEEDED'):budget.limits(root,1024*1024,4096)
