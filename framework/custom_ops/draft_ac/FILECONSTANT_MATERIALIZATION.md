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

## 091937：FileConstant 端口名称映射与快照同步

用户回传 `/home/w00949577/qwen35-dflash-run/ac-ge-20261014-091937` 的两个 JSON，确认 `arg1_1` 是 FC 的 GN scale：FP16 `[100,2560]`、512000 B。文件已写入，`after_save_before_graph_dump` 为 PASS，NZ 反解为 BIT_EXACT。最终 live 节点相对旧快照新增：

```text
_input_name_key   = []
_input_name_value = []
_output_name_key  = ["y"]
_output_name_value= [0]
```

`file_path`、空 `file_id`、dtype=1 和 shape 属性未改变。live output descriptor 已是 `DT_FLOAT16 [100,2560], ND`，旧快照仍只有 `name="y"`。旧日志没有保存写完文件时的 live 属性，所以不能进一步断言映射是在写入器还是后续 SDK 图保存过程中添加。旧检查在 scale 的导出后哈希复查之前中止；不把写入后 PASS 冒充导出后或 NPU PASS。

修复保留 `FILE_BINDING_CHANGED`，只接受**完整且内容准确**的四个端口映射属性新增：FileConstant 没有输入，只有一个已命名输出，索引为 0。缺项、错误名称/索引、混入其它字段、修改/删除已有映射均拒绝；其它属性一律仍受保护。

快照先从原始节点独立重放 `_bound()` 已验证的 dtype/shape 修复及路径规范化，将逻辑 FileConstant descriptor 明确规范为 ND，然后与 live 节点逐字段比较。通过比较后才同步快照，不能直接复制任意 live 属性来掩盖变化。导出结束继续检查完整属性、输入边、input/output descriptor、Tensor 与文件 SHA256/字节数；只有已证明被 A 的 NZ Const 替换的原权重节点可消失，scale/Head 不可消失。

`ac-payloads.json.binding_checks` 记录各阶段的 snapshot、final、逐字段 `field_diff`、允许变化及判定；失败详情同时写入 `ac-constants.json`。诊断区分经不可变 Tensor 验证的 descriptor/path 修复、已验证的 SDK 端口映射新增和受保护字段变动。回归同时模拟写入器阶段及图保存阶段添加映射，并覆盖真实 `[100,2560]` 的 descriptor、恶意路径/offset/file_id/dtype/shape/descriptor/映射/未知属性/节点删除与文件篡改。

上述证据仅支持此次元数据兼容修复。修复版本真实 AIR/ATC/NPU 验收仍为 **NOT_RUN**，使用上方新目录命令复测并回传两个 JSON 和 server.log。
