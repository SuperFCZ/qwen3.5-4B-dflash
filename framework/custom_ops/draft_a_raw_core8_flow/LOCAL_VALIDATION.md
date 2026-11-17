# 本地验证 — 2026-11-17

- macOS：新增Core8合同测试 **22 passed**。
- Docker Linux ARM64：相同 **22 passed**。
- pyflakes、shell语法、导出worker精确派生审计通过。
- 未重跑不受影响的历史CPU套件；未编译新的计算Kernel。

覆盖六类C16/C64形状、独立B8/R8canonical合同、Gate Raw/非Gate差异、7/9 Core拒绝、
core_limit/Block错误、布尔冒充整数、assignments步长/边界/重复/遗漏拒绝、304 Gate Tile
每核38个、完整Probe sidecar及漂移、阶段前序参数、源身份变化与失败报告拒绝。
测试数据均为CPU fixture，不是服务器资源或NPU数值证明。

复用B8-C16 OM的用户声明SHA已固定在准入代码，但实际服务器文件尚未由本次工具读取。
新Source Admission、R8 Build、104项8-Core Probes、四图Export、Smoke、Benchmark均NOT_RUN。
没有自动执行NPU，没有虚报8-Core正确性或整图速度。

历史Raw/Sidecar/ATC/环境工具、Kernel、模型及导出路径没有修改。工作区原有未跟踪
`draft_a_raw_core8/`两份草稿保持原样，本流程不导入它们。
