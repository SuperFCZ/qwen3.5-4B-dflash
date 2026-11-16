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

## 子进程临时模块生命周期修复 — 2026-11-16

- macOS完整回归：**126 passed**。
- Docker Linux ARM64：**125 passed, 1 skipped**。该镜像没有Git CLI，所以仅跳过实际
  Git命令集成测试；独立源码归档合同fixture照常通过。真实Git读取/哈希验证在macOS通过。
- 新增15项CPU回归包括实际子进程退出/atexit清理、持久快照验证、生成内容/所有者/
  来源/生成器漂移拒绝、普通模块缺失和未知进程模块拒绝、B/R内容归因、不可变Git源码
  校验及保持Admission资产不变的续接。CPU fixture不冒充真实Torch/NPU模块。
- pyflakes、Git whitespace检查通过。
- 原Raw、ATC工具、Kernel/Tiling/OPP/模型相关源码均未修改；原Source Lock严格验证保留。
- 没有重建旧/tmp目录、没有修改Torch清理逻辑、没有安装组件。
- 真实服务器Import-only复测、ATC、NPU、Benchmark/Profile仍 **NOT_RUN**。
