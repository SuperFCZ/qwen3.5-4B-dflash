# 本地执行记录

环境为 macOS arm64 与 Linux aarch64 容器 `qwen35-cann9-host`，CANN 根为 `/usr/local/Ascend/cann-9.0.0`。历史 Gate8/Profiling 真实服务器文件未挂载到此容器。以下报告在容器里保留；未向仓库提交二进制、模型、原始权重或 SDK 头文件。

| 本轮本地操作 | 结果 | 实际证据 |
|---|---|---|
| 双 W 的 CPU Host、proto、CANN Gate 设备目标、SDK package、分别私有安装、Runner 编译链接和 ldd | LOCAL_HOST_DEVICE_COMPILED_NPU_NOT_RUN | `/work/build/gate-w16-px0mgc_g/report.json`，SHA256 `1ed01f6fc4762d3900ad3c3cfbcc281c1bc21c4b06e4c8b4d7c64281e62d3175` |
| Down 二维搬运的独立私有安装、形状专用 Runner 编译链接和 ldd | LOCAL_HOST_DEVICE_COMPILED_NPU_NOT_RUN | `/work/build/down-dma-vbs_st0a/report.json`，SHA256 `c3d30df3d8ed03e2546335ad94df2b1c4eda8c9cab12db5ef4b1aac3779b08a8` |
| 两档 scratch 的实际 MatmulApiTiling 命名平台查询 | 两次 rc=0；TCube 同为 200 B，SHA256 `745b846a7e3b31de10827caa5e493026d3984e282dd698177d2da380b826359a` | `/work/build/w16-sdk-query-Oejkbn/result.json`，SHA256 `98ae8d5a6a87c10991e9ae831e44806321bb9c8160102f011971c3f6674facf0` |
| SDK serializer | 224 B parent、24 B prefix、200 B nested、50 个不同字段标记均相符 | 主报告的 codec；原始 check.log、markers 和 ELF 在同根 layout/ |
| 当地包内 ELF 解码 | 两份最终 W16 各有 **19599 条未解码地址行**，Down 有 13127 条 | 各变体目录 package.disassembly；llvm-objdump SHA256 `2b15d0cbe2eebdf8d230b76550c6d713e8c4855acf9cfb9f2a6089b113881b3c` |

| 私有安装的 Gate 设备对象 | bytes | ELF SHA256 | SDK 安装包 SHA256 |
|---|---:|---|---|
| double-w-serial | 86992 | `8f5ffd7f677119fb8d485a0593aafd39b68705da7c898fda485c2bbe7d31937c` | `3e09c0ce916cd35fcf97d1e9b52f9130e86d1785996a10d852ec9a29146a8d18` |
| double-w-overlap | 86992 | `0d0dc2f0baf8ea5c5aca49c984c745a4c07d20ea174f7d5fdb2ecf47eb028cb4` | `655d1c8e379ffd1ff380e338e161faa2361cb15d3cb08575b7b4da65d8511708` |
| down-2d | 57216 | `f9a57f173c161b3adf623bfba78f897a769852e628de3e69e0eb8911637ef5b3` | `968966541a2039fd2f4a9367df8b4978935eb1f15865bac8caf708a09b5eead4` |

Gate Runner 为 249912 B，SHA256 `374bcb8759e36bc4bef349c441d0e6761ef16dd7149ad4bab7e928871dc5281c`。Down Runner 为 249912 B，SHA256 `0d3f4f2279a573033fa0d548f22059161a6095979be301c67ffd3dbc2345c985`。B/C 本地 tiling Host DSO 的 SHA256 分别为 `ec10f6dfd9050585b08a33a882eb086df0f832ebd9c4d8e3ba2d7a2e36d178c3`、`fb585017b7f733952a3d8a1efaf9130b15aa9fc5f445e06cac418d09b1c2d91b`。构建目录和 DSO 文件身份独立，Host 源及资源合同相同。服务器以自己的完整联合 OPP 记录为准。

本地 package 使用已有的 **单 Gate SDK 生成工程**，实际编译 `cust_optiling`、`cust_op_proto`、`DFlashGroupQuantLinear_ascend310p` 和 Runner，再执行 SDK 的 binary、package 与指定私有根的安装器。核对安装后的 ELF/metadata、ATC 动态实现源码及 Raw 生成头。服务器 Build 从已验收的联合私有 Raw OPP 复制基线，仅替换受控的 Host、目标 Kernel 和动态源，保留两个历史 Head binary/API。只有服务器可以重验其完整冻结联合包身份。早期仅完成编译的输出和失败的 CPack 尝试均保留在原目录，上表引用最终成功安装报告。

## 复现本地编译

在上述容器从仓库挂载目录运行，每条命令均创建独立的新输出目录。

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cd /src
S=framework/custom_ops/draft_gate_w16_atc8
T=/work/build/gate8-check-t6eu9_qo/raw/CustomOp
python3 -B "$S/w16_build.py" --template "$T" --output-parent /work/build
python3 -B "$S/w16_build.py" --down-only --template "$T" --output-parent /work/build
TMPDIR=/work/build bash framework/custom_ops/draft_w8a16_deep_profile/w16-query.sh
```

CPU 合同覆盖实际 C++ 模板循环在每个 K 位置的普通失败、两版 W16 和 Down 实际 C++ 在确定性 CPU API 夹具上的 Group128 数值/地址/尾部、不同数据 gear、完整 Cube/parent 拒绝检查、配对顺序/样本数/NO_GO、完整输出末元素篡改、未解码 ELF、CFG/操作数证据、错误凭据和 DSO 拒绝。Linux 用真正的两个进程及 SCM_CREDENTIALS 测 socket 放行和 PID 冒用；该测试不链接 ACL。CPU 模拟数据仅在测试目录的临时文件中出现。

| CPU 回归套件 | macOS arm64 | Linux ARM64 CANN 容器 |
|---|---|---|
| 本目录 | 21 PASS，Linux socket 凭据测试跳过 | 22/22 PASS，含真实 PID/FD 和拒绝错误 DSO 后不发送 G |
| 已有 Gate8 | 18 PASS，容器产物项跳过 | 19/19 PASS，使用已安装 SDK 环境及真实 Gate8 构建 fixture |
| 已有 W16 源码审计 | 17 PASS，CANN header 项跳过 | 18/18 PASS，23 个文件 SHA 完整匹配 |

**当前真实服务器状态**：本轮自己的 Source、联合 OPP Build、冻结 AIR ATC、专用 ELF 静态验收、三版 Smoke、配对计时、五层、Full Draft 和 Down 各服务器阶段均为 NOT_RUN。包内反汇编不能给出设备运行准入。当前没有新的 W16 数值结果、性能样本或 GO 决策。
