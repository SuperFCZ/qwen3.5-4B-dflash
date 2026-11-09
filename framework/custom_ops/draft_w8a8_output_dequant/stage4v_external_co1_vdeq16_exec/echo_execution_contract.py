"""Exact process-report contracts; these do not replace callback/numerical checks."""

def validate_execution(report,device,workspace_bytes,host):
 expected=dict(status='HOST_TILING_ONLY',npu='NOT_RUN',workspace_bytes=workspace_bytes,device=device) if host else dict(
  status='ECHO_EXECUTION_MATCH',npu='COMPLETED',device=device,workspace_bytes=workspace_bytes,repetitions=2,
  guards=True,readonly=True,poison=True,repeat_equal=True,l0c_guard='RANGE_ONLY_NO_CANARIES',
  vdeq16='NOT_IMPLEMENTED_PENDING_ECHO_PASS',cpu_fallback=False)
 if type(device)is not int or type(workspace_bytes)is not int or device<0 or workspace_bytes<=0:raise ValueError('invalid expected execution scope')
 if type(report)is not dict or set(report)!=set(expected) or any(type(report[k])is not type(v) or report[k]!=v for k,v in expected.items()):
  raise ValueError('Host-only receiver proof differs' if host else 'Echo safety contract differs')
 return expected
