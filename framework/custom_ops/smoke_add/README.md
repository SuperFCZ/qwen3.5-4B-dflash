# AddCustom smoke test (Ascend 310P3)

This isolated FP16 AddCustom operator exercises the CANN 9.0.0 `msopgen` → source overlay → build → install → ACLNN test path. It accepts two contiguous ND tensors with 16,384 elements (the test uses `[8, 2048]`) and writes their elementwise sum. The tiling code deliberately rejects other element counts. It is not connected to the Qwen/DFlash graph.

The `AddCustom.json`, `op_host/`, and `op_kernel/` starting point follows the [official Ascend AddCustom framework sample](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/1_add_frameworklaunch), whose source copyright and disclaimer are retained in those files. The server test uses the same ACLNN API pattern as the [official minimal invocation sample](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/1_add_frameworklaunch/AclNNInvocationNaive).

On the 310P3 server, after pulling this branch:

```bash
export CANN_ROOT=/usr/local/Ascend/ascend-toolkit/latest  # adjust to the installed CANN 9.0.0 toolkit
export DEVICE_ID=0
bash framework/custom_ops/smoke_add/run_server.sh
```

The script builds under `.build/`, installs the generated `custom_opp_*.run` into the active CANN OPP vendor directory, compiles the separate ACLNN test, and checks all 16,384 output elements. Installation changes that server's `customize` vendor package; run on the intended test server/account. The script does not run on a machine without CANN and an NPU.
