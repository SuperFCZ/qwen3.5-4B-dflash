"""Derive only the ATC transport fields; preserve the frozen-AIR compile path."""
from sidecar_common import c,HERE
from worker_derivation import replaced


def text():
    p=HERE.parent/'draft_a_raw_atc/resume_worker.py'
    value=p.read_text()
    value=replaced(value,'from split_atc import c,SplitAtcRunner,one_capture','from sidecar_atc import c,SidecarAtcRunner\nfrom split_atc import one_capture')
    value=replaced(value,"capture=SplitAtcRunner(root/'atc-streams'","capture=SidecarAtcRunner(root/'atc-streams'")
    value=replaced(value,"atc_stderr=split['stderr'],","atc_stderr=split['stderr'],host_evidence=split['host_evidence'],")
    value=replaced(value,"status='A_ONLY_ATC_RESUMED_NOT_EXECUTED'","status='A_ONLY_SIDECAR_ATC_COMPILED_NOT_EXECUTED'")
    return value


def verify():
    if (HERE/'bc16_worker.py').read_text()!=text():raise ValueError('sidecar worker changed beyond transport')
    return c.identity(HERE.parent/'draft_a_raw_atc/resume_worker.py')


if __name__=='__main__':verify()
