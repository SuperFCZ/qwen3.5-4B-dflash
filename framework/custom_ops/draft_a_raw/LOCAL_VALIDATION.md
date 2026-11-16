# 本地验证记录 — 2026-11-16

这是本地开发证据，**没有服务器 NPU 结果**，也不是私有 OPP/OM 的编译验收。
历史服务器根报告 SHA 仅来自用户终端，需运行 Precheck 重核实际产物。

| 检查 | 结果 | 边界 |
|---|---|---|
| macOS Python CPU tests | 29 passed | `/opt/anaconda3/bin/python -m pytest -q -p no:cacheprovider framework/custom_ops/draft_a_raw/test` |
| Docker Linux ARM64 CPU tests | 29 passed | 现有 `qwen35-cann9-host`，Python3.8.10；无 Torch-NPU/NPU |
| 原 Gate copy/prefetch CPU 回归 | 11 passed | 未修改历史实现；只运行已有纯 CPU 测试 |
| pyflakes | PASS | 仅新增目录 |
| bash syntax | PASS | `server_stage.sh` |
| 实际 Host C++ Runner 编译/链接 | COMPLETED | Docker CANN9.0.0 headers + libascendcl；不是 AscendC Kernel 编译 |
| Runner ldd | 无 unresolved，在 CANN 自带 set_env.sh 后 | 初始未 source 环境时 profiler/dump 两库未解析；没有删除依赖或安装组件 |
| 新 OPP / 完整 OM 编译 | NOT_RUN | 由服务器 Build/Export 门控取得 |
| 全部真实 projection / OM Smoke / Benchmark | NOT_RUN | CPU fake ACL 测试不能替代设备证据 |

真实 Host C++ 检查产物仅保存在本地 Docker：

- `/work/build/draft-a-raw-runner-Lhtpyk/raw_om_runner`
- SHA256 `7ba809c565ff3386b568ecd641847b78f38f4c320e3ed8dd709d90b8e53940e9`
- 动态链接至本地 `/usr/local/Ascend/cann-9.0.0/lib64/libascendcl.so`。
- `libascend_hal.so` 来自 SDK **devlib**，不是服务器 Driver；`LOCAL_ONLY_NOT_SERVER_VERIFIED`。
- 没有执行该可执行文件；服务器须自行重新编译和绑定自己的库/Runner。

CPU fixture 使用实际新 Runner 源码，显式链接 `test/fake_model.cpp`，与实际 CMake 完全分离。
验证 smoke 2 次、benchmark 38 次执行（2 前检+5 warmup+30 timed+1 postcheck），跨语言
execution.json 合同、15 Token / 全 KV Golden、最后 KV 字节损坏、Readonly、外侧 Guard、
Tail、Postcheck、异常计时日志、Execute 失败阻断。伪实现始终只用于 CPU 测试。

另覆盖 H/B/R 文件漂移、独立 OPP 库/ELF/库存、仅 Raw 配置差异、严格前序状态、禁止
覆盖目录、B/R 环境隔离、实际 Host 日志解析/缺失/交错拒绝、已验收 Gate Core/Tiling
漂移拒绝、非 Gate 计划漂移、完整三轮协议及禁止混入历史/Profile 样本。

无任何新的完整 Draft 加速结论。五层孤立 Gate 实测和差不转换为整图预测。
