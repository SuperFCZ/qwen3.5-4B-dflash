# Head OPP 别名身份误报诊断

范围：仅 `draft_a_raw` 的证据路径处理；不改历史 C/Head/A-only、AIR、OM、OPP、Golden、
Source Lock 或已有失败报告。本地没有读取服务器文件，服务器事实来自用户回传。

## 结论和证据等级

`a-only-raw-full-fixed-BpnCJR/precheck/report.json` 的最终递归检查 BLOCKED。
两处记录来自：

- `evidence.H.state.provenance.c_installed_opp.libraries[2]`
- `evidence.H.state.provenance.c_installed_opp.libraries[3]`

两条声明均为 866256 B、SHA256
`868c7a542a0226cb9ffe299bf112a6f9585364f32263366bd11ca70a5c107916`。
用户脚本显示两条路径目前都解析到：

`draft_head/.build/head-0LRC5uaJ/opp/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so`

当前内容大小、SHA 均匹配，脚本读取期间 inode/mtime/size 稳定。报错中的
`.../op_tiling/liboptiling.so` 是别名。

确定的代码错误：`raw_common.identity()` 先 `resolve(strict=True)`，返回真实路径；
旧 `verify()` 把这个结果与保留别名路径的历史字典整体比较。即使 bytes/SHA 完全相同，
`path` 字符串不同也必然失败。CPU 测试复现了这一条件。
现有证据不支持把此次报错归因为内容修改或并发写入；它不证明所有历史时刻都没有写入。
没有触发备份搜索或替换。终端末尾在**真实目标目录**搜索 `liboptiling.so*` 为空，不是
原别名文件不存在的证据，也不能用来判断备份是否存在。

## 引用来源审查

| 源码 | 实际作用 |
|---|---|
| `draft_head/test/head_opp_preflight.py:52–58` | 遍历私有 vendor 下 `*.so*`；读取内容但保存原始路径，不 resolve。别名及其真实文件可各占一条记录。 |
| `draft_ac/test/resolve.py:128–140,199–200` | 核对已接受 C Release 的安装产物，将 `release['installed_opp']` 放入 provenance；不是 AIR payload。 |
| `draft_ac/test/run.py:386–401` | 构建新的 H 联合 OPP，选中 H/joint/opp，保存 provenance 和新 opp 到 worker-input。 |
| `draft_ac/test/c_profile.py:35–59` | 按各 gear 的 artifacts→AIR/Deployment 链核对 SHA、ABI、图审计；`payload_files` 来自该 AIR graph，逐文件 checked_file 后放入 watch。 |
| `framework/python/qwen35_dflash/ascend310p/draft_ac_export_pipeline.py:288–301` | payload 列表在导出时来自 graph_dir 文件清单并保存文件身份。不能拿 provenance 的库列表当成此清单。 |
| `raw_evidence.historical()` | 继承了原 worker state 的 provenance；两条库记录不是它再次从 AIR 收集出来的。 |

因此 `.build/head-0LRC5uaJ/opp` 是历史 **C Release 来源包**，不能称为 H A-only AIR
所选运行 OPP。H 自己的 `joint/opp.json`、`state.opp`、device-process 的选择路径仍独立
校验，A-only 图仍要求 26 A / 0 C / 0 C-Audit。这里不声称重新采集过服务器 DSO maps。

## 最小修复的严格边界

1. 先按 H 已冻结 provenance 中的 `records.c_release` SHA 核对原 Release 文件，要求
   其路径与原 args.c_release 一致，installed_opp 与原 worker state 完全相同。
2. 对该历史 libraries 清单逐条检查。原别名必须在声明 OPP 内，其真实目标也必须在
   同一 OPP 内；**历史清单必须已经有一条匹配真实目标路径、bytes、SHA 的记录**。
   不能仅凭当前目标内容一致，就接受没有历史记录的新目标。
3. 在新的 state 深拷贝中保存规范身份；bytes/SHA 使用原声明，不替换为当前值。
   原数组顺序和重复项保留，不用去重掩盖冲突。
4. 新 `head_release_binding` 单独记录每条原 alias、canonical file、安装根及逐跳链接链。
   当前链是本次观察，不冒充旧报告已经保存的链接证据。用户输出未打印两条原始 path，
   所以冻结 canonical 对照是否齐全由新服务器 Precheck 实际判断；缺失则 BLOCKED。
5. `verify_tree()` 对此显式绑定调用严格 alias 验证：重新解析目标/链接链、检查根范围、
   校验原 bytes/SHA，并复核读取前后链未改变。普通规范身份的原检查保持。
   不忽略 `.so`，不删除递归验证，不允许所有别名无条件通过。
6. 同一个声明路径存在不同身份时立即拒绝；别名与真实目标的声明不一致也拒绝。
   已建立绑定之后，即使改指向同字节复制件或改变链接链而最终内容相同，也拒绝。

不创建任何服务器软链接/复制件，不更新原清单。所有测试中的文件和软链接只在 CPU
临时 fixture 目录创建。原 `args`、Source Lock、Native Boundary 校验不变。

## 复测

旧两个 BLOCKED 目录保持不变。修复版须使用新目录只跑 Precheck；成功状态仍是
`A_RAW_IDENTITIES_VERIFIED_NO_DEVICE`，不代表 OPP Build/Export/Smoke/NPU 通过。
后续所有阶段继续 NOT_RUN。命令见 README 的 Precheck 入口。
