# 本地验证 — 2026-11-16

- macOS：新增15项环境合同测试 + 原A-only Raw59项，共 **74 passed**。
- 现有Docker Linux ARM64/Python3.8.10：相同 **74 passed**。
- pyflakes与Git whitespace检查通过。
- `draft_a_raw/`、`draft_ac/`、`draft_quant/`、`draft_head/`、models和framework/python
  相对8d08f57均无修改；辅助目录不进入原Source Hash清单。

覆盖：历史版本静态读取不执行包代码、distribution/module一致性、包shadowing、混合
搜索路径、历史SHA漂移、initialization安装RECORD拒绝篡改/缺失、前序源码漂移阻断、
模块别名改指向、缺依赖时不准备接口/不启动Export、环境快照漂移和CLI阶段参数。

包与metadata fixtures只存在于测试临时目录，未安装进开发或服务器环境，不供真实导出。
Docker无Torch-NPU/真实历史venv，因此测试不能证明真实import/Interface/Export通过。
本轮没有安装或升级软件、没有CANN Build、没有NPU执行、没有Benchmark/Profile。

服务器当前已回传：原Precheck/Build/104 Probes PASS，Export在错误Python环境下BLOCKED。
新import-only检查和修复环境后的Export均为 **NOT_RUN**，需用户在服务器单独执行。
