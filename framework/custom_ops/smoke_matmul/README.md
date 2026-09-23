# FP16 Cube MatMul smoke test

`SmokeMatmul` is an isolated Ascend C MatMul high-level API test for Ascend 310P3 / CANN 9.0.0. It computes `Y = X @ W.T` for the fixed contiguous ND shapes:

| Tensor | Shape | Dtype |
| --- | --- | --- |
| X | `[16, 256]` | FP16 |
| W | `[64, 256]` | FP16 |
| Y | `[16, 64]` | FP16 |

The host rejects other input shapes. The kernel uses the Cube MatMul API with a transposed B operand and one core; it does not contain a scalar matrix multiplication loop. This is a correctness smoke test, not a performance implementation. It is not connected to the Qwen/DFlash graph.

The host/tiling and kernel structure follows the [official Ascend single-core MatMul sample](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/10_matmul_frameworklaunch/MatmulCustomSingleCore). The [Matmul API documentation](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0614.html) describes the operand type and B transpose parameters.

On the server, from the repository root:

```bash
export CANN_ROOT=/usr/local/Ascend/ascend-toolkit/latest  # adjust to CANN 9.0.0
export MODEL_PYTHON=/path/to/python-with-torch          # optional if python3 has torch
export DEVICE_ID=0
bash framework/custom_ops/smoke_matmul/run_server.sh
```

The script repeats the `smoke_add` workflow: `msopgen`, overlay source, build, install into the independent `.../custom_ops_runtime/smoke_matmul` directory, load that OPP, and run ACLNN. `test/reference.py` generates FP16 inputs on CPU and compares all output elements against PyTorch `X.float() @ W.float().T`, rounded to FP16. Input values are exact multiples of 1/8, so this fixed case expects bitwise equal FP16 results. The server's existing custom OPP environment is only removed from the installer subprocess.
