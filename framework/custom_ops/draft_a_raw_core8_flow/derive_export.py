"""Fresh export keeps the original numerical path; only ATC transport is sidecar."""
from c8_common import HERE
from worker_derivation import replaced


def text():
    value=(HERE.parent/'draft_a_raw_atc/worker.py').read_text()
    value=replaced(value,'from split_atc import SplitAtcRunner,one_capture','from sidecar_atc import SidecarAtcRunner\nfrom split_atc import one_capture')
    value=replaced(value,"capture=SplitAtcRunner(root/'atc-streams'","capture=SidecarAtcRunner(root/'atc-streams'")
    value=replaced(value,"result['atc_log']=split['stdout'];result['atc_stderr']=split['stderr']","result['atc_log']=split['stdout'];result['atc_stderr']=split['stderr'];result['host_evidence']=split['host_evidence']")
    # Ensure the explicit dependency paths are installed before historical imports.
    value=replaced(value,'import gc,os,sys','import gc,os,sys\nfrom c8_common import c')
    value=replaced(value,'import raw_common as c','# c is supplied by the isolated Core8 bootstrap')
    return value


def verify():
    if (HERE/'export_worker.py').read_text()!=text():raise ValueError('export worker changed beyond sidecar transport')
