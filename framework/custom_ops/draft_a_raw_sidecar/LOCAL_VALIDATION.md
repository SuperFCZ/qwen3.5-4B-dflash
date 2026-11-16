# 本地验证 — 2026-11-16

- macOS：**145 passed**。
- Docker Linux ARM64：**144 passed, 1 skipped**。唯一跳过项是旧环境续接工具的真实Git
  CLI集成测试（Docker无Git）；该测试在macOS通过，归档合同fixture在两侧均通过。
- 新增sidecar测试19项，包括实际C++ writer的多线程及多进程写入、189条stdout Warning
  全保留、36条JSON、缺失/截断/错序/重复shape/字段类型/未知记录拒绝、B/R资源合同、
  unset兼容、写失败不回退、仅Host DSO可变化与Kernel字节保持。
- pyflakes、源码变换/Worker派生审计通过；所有原实验与共享源码无改动。

## 实际 CANN Host 编译检查

现有ARM64 Docker、已安装CANN9.0.0；隔离目录：
`/work/build/a-raw-sidecar-host-le2oo1y_`。

msopgen、CMake configure、Host build均返回0。该本地项目只生成A算子接口，编译
`cust_optiling`、`cust_op_proto`，移除了Kernel构建子目录。使用N64/K256/Broadcast/
a-ub/off配置，仅作Host接口可编译性检查，不冒充服务器完整联合OPP身份。

| 本地产物 | bytes | SHA256 |
|---|---:|---|
| libcust_opmaster_rt2.0.so |862208|fbbf7f578dba10059f1f8d45d5595f70bd10c4595cf96ea46d9b8ad2e6124cd8|
| libcust_opsproto_rt2.0.so |874496|832f360507cc81b7e73f7124ea2e33fd68ddc6909804e6358f4d6db07f740808|

`LOCAL_ONLY_NOT_SERVER_VERIFIED`。未安装该包、未执行GetTiling/NPU，设备Kernel编译
NOT_RUN。生成项目与SDK源码没有提交仓库。

服务器新B Host包、Kernel复制身份检查、实际ATC sidecar及B-C16资源验收均NOT_RUN。
旧实测OM及日志只作为用户冻结输入，未被改写为本轮PASS。
