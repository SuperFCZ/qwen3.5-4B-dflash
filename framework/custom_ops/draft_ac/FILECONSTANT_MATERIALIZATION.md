# A/C FileConstant 物化边界

用户提供的 `ac-ge-20261014-082228` 已完成联合 OPP 编译安装和 runner 构建，但 `a-c16-fc` 在 `_convert_data_to_const` Hook 内检查 `scale` 文件时失败。旧错误合并了缺失、越界、大小及哈希问题，不能单独据此确认服务器上的具体文件状态。

## 已核对的源码顺序

本轮读取了 [Ascend 官方 TorchAir export_utils.py](https://github.com/Ascend/torchair/blob/master/python/torchair/_utils/export_utils.py)。该上游实现的顺序是：

```text
make_export_graph
  → _convert_data_to_const
      创建 FileConstant，file_path = export_dir / weight_name.replace('.', '_')
      返回 (weight_externalized, used_weight_num)，尚未写权重文件
  → _save_weight2file（externalized 且 used_weight_num 非零）
      将原 Tensor 写入上述文件
  → dump_graph / 后续 GE 保存
```

这解释了为什么在 convert 返回前检查文件可能过早。上游 master 不是服务器安装版本的证明；新版本在服务器保存 `ac-torchair-fileconstant-api.json`，包含实际三个函数的签名、源码、源码文件路径及 SHA256，用于确认具体 SDK 的调用时机和文件命名。

## 修复范围

仅对显式 A/C 导出安装作用域内的两个 Hook：

1. convert 前保留 exporter 绑定的不可变 Tensor，并记录其原始字节 SHA256。只捕获 A/C 消费的权重和 scale；不更改 Tensor 数值。
2. 对标准 externalized 返回协议，标记 `PENDING_MATERIALIZATION`，校验路径安全，但不提前生成 NZ Const、不声称哈希已通过。
3. `_save_weight2file` 的输入 Tensor 身份、weight-name 映射、目录、数量必须与 convert 一致。每次沿用 SDK 原写入器，不主动预写或重复保存；完成后才运行原 A/C `_bound`、`_verify_payload`、NZ 打包及 bitwise 反解，随后 SDK 才继续 dump/GE 保存。
4. 内嵌 Const 或已同步物化的生产者仍立即执行真实字节验证。未知/未完成的物化顺序明确失败，不能仅凭 Tensor 哈希代替文件哈希。
5. 验证成功后将 A/C FileConstant 的路径固定为已验证的绝对路径，避免 ATC 工作目录改变文件指向。支持绝对路径、export-dir 相对路径和 SDK 的 directory-prefixed 相对路径；不按 basename 搜索或猜测其它目录，符号链接逃逸仍拒绝。
6. 导出结束后再次检查 Tensor、文件、FileConstant 属性绑定和 SHA256。后置失败会将 A/C 审计标为 FAIL，不能发布 PASS AIR manifest。两个 Hook 在成功或异常后均恢复。

文件大小必须严格等于逻辑载荷大小。当前源码每权重一文件、offset=0；非零 offset 仍明确报 `UNSUPPORTED_NONZERO_OFFSET`，不猜测切片/容器格式或放宽原规则。C 的完整 Head 始终按原始 FP16 位型验证，采用 4 MiB 分块哈希，避免校验额外复制整份约 1.27 GB Head。没有 CPU 计算 fallback。

A/C Kernel、Host Tiling、Native Golden、Target/Verify、默认生产 Draft 不变；原有 GE 节点数量、Const descriptor、NZ 反解和后续 bitwise/Guard/只读验收保留。未带 `draft_ac` 的生产导出不安装新增物化 Hook。

## 新诊断

每个 AIR 图目录保留：

- `ac-torchair-fileconstant-api.json`：实际 SDK 函数和版本源码证据。
- `ac-payloads.json`：convert/save/export 阶段、原始节点属性和 descriptor、绑定 Tensor 的 dtype/shape/bytes/SHA256、验证后文件 SHA256、CWD、解析路径及有界目录布局（不转储 Tensor 原始数据）。
- `ac-constants.json`：A/C 常量最终审计；任何后置失败必须为 FAIL。

错误区分：`MISSING_FILE_PATH`、`PATH_ESCAPE`、`NOT_REGULAR_FILE`、`MISSING_FILE`、`FILE_SIZE_MISMATCH`、`FILE_HASH_MISMATCH`、`CONST_SIZE_MISMATCH`、`CONST_HASH_MISMATCH`、`IMMUTABLE_TENSOR_CHANGED`、`FILE_BINDING_CHANGED`、`NOT_MATERIALIZED` 和物化器异常。阶段信息可区分“convert 后按协议尚未写入”和“save 已返回但文件仍不存在”。

## 服务器复测

使用原 Capture、Native OM 和新证据目录：

```bash
git pull --ff-only
export AC_OUTPUT_DIR=/home/w00949577/qwen35-dflash-run/ac-ge-$(date +%Y%m%d-%H%M%S)
MODEL_PYTHON=python bash framework/custom_ops/draft_ac/run_known_evidence.sh
```

请回传 `suite.json`、旁边的 `.server.log`，以及失败图目录下上述三个 JSON、`converter-capabilities.json` 和导出/ATC 日志。当前修复的真实服务器 AIR 导出、ATC 和 NPU 正确性在执行前为 **NOT_RUN**；CPU 文件回归不能替代它们。

本地回归：FileConstant/Const 与物化边界 52 PASS；连同 A/C、runtime-input、prepack/layout 回归合计 354 PASS、84 SKIP（既有测试需要未构建的 fake ACL runner）。包含相对/绝对路径、符号链接逃逸、文件与 Tensor 篡改、未调用/失败的写入器、同步及延迟物化、异常时 Hook 恢复，以及完整 Head 字节载荷的分块验证。Bash 语法检查通过。
