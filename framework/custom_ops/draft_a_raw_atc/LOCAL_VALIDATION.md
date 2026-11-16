# 本地验证 — 2026-11-16

- macOS：22项新增CPU测试 + 原环境15项 + 原Raw59项，共 **96 passed**。
- 现有Docker Linux ARM64：相同 **96 passed**。
- pyflakes、Git whitespace、精确worker/helper派生审计通过。
- 原Raw、Raw环境检查、A/C共享导出/编译器、Host/Kernel、模型源文件无修改。

新增CPU测试包括：

1. stdout在CUBE JSON中途分段写，stderr插入1/5条GMM Warning；六条JSON完整，
   6/30条Warning原始字节完整保留，两个文件SHA有效。
2. 截断、未知Host记录、缺字段、重复JSON key、缺shape、交错顺序全部拒绝。
3. C16/C64的B/R Gate/非Gate原资源合同比较及变更拒绝。
4. subprocess失败仍保留两路日志，外层失败报告仍绑定原始诊断。
5. AIR/Payload复制后SHA相同、原文件不改、不使用硬链接；命令只重定位model/output。
6. ATC二进制漂移、数学flags改变、AIR/日志/ABI/原命令改变拒绝。
7. Worker/Manifest/原OM准入及未知OM二进制差异不误记数值或性能PASS。
8. 新目录不进入旧Source Hash集合；派生worker/helper只允许明确的改动。

子进程测试使用明确标记的CPU Python fixture，不是真实ATC，也没有真实NPU输入。
本轮没有编译/安装OPP、没有运行CANN ATC、没有NPU Smoke/Benchmark/Profile。
真实服务器日志分离验证、B-C16 Resume、另外三份完整OM均 **NOT_RUN**。
旧B-C16编译成功与外层资源失败是用户回传的历史事实，未被改写为本轮成功。
