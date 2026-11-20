# C16 Layer0 Gate/Up 的 Scale 展开实验

`raw` 复用 `gate-raw-atc8-Np20yu` 已验收的单 W、raw 双 bank、loop copy 的 Gate OM、私有 OPP 和带 DSO 放行的 Runner。`raw-transpose-b16` 在独立私有 OPP 中仅改变该 Gate 的 Scale 展开，按原 ABI 和同一份 AIR 生成另一份 OM。两版在设备 0 的独立进程中使用各自的私有包，按相同协议采集新样本。当前用户报告的完整 Draft 基线 C16/C64 分别为 30.569870/31.426295 ms，只作背景。目标先低于 20 ms、远期达到 13.33 ms，整图收益须另行实测。

**候选的数学合同**为 `expanded[16*c+j] = s[group,c]`（16 位原样复制）。每 K 保留 8 次 Cast、2 次 FP16 Mul、6 次应用 PIPE_V、原始 raw Set/Wait、V_S 和全部 PIPE_ALL。替换后的两个 B16 转置各有 4 个 repeat。保留相同的 expanded/rear、SDK reserve 和 185600 B Matmul scratch。布局证明、可选 SDK 路径和生命周期见 [SDK 审查](SDK_REVIEW.md)。

## 冻结对象

M16/K2560/N19456，304 个 N64 tile，8 个 block 各 38 个 tile。每个 N64 tile 在同一 FP32 累加器中完成十次 K256，末尾一次 FP16 写出。

| 原始输入/产物 | 历史身份 |
|---|---|
| `x.bin` | FP16 [16,2560]，81920 B，`44fed068f079c18e95ef303c0d02cf5401470db7277ec17d9d024fbf29079256` |
| `w_nz.bin` | INT8 NZ [80,1216,16,32]，49807360 B，`c898ef7b555f6e3bcf3b183ec7742b012f0a4e40ab4a46cd34184b77218b5a2c` |
| `s_gn.bin` | FP16 GN [20,19456]，778240 B，`1365af20b6f09734c6d02f2f3d3045fb20e4af1f53d414c9b4dd9011c7ced333` |
| Native Golden | FP16 [16,19456]，622592 B，`353839a7a41bdd44930e91f72ae750adc2f7f15e14f2b679d5f0799e1b5da76c` |
| 对照 OM / 内嵌 Gate | OM 50617371 B，`938d0fa4b937f7af67f46f63d904d729957e015635c3eaab8f5d54c843c242c5`；内嵌 ELF `73e570d30e465b81aa27842f77302b1bc18aaf30a1c2794db46def5ba2a04846` |
| 实际 Raw Tiling | parent 224 B，`0bbc0ee73d673d9141065d6bb73bee22b9714cc96bfe2e03c0477dfa68eae1a3`；offset 24 的 200 B Cube `745b846a7e3b31de10827caa5e493026d3984e282dd698177d2da380b826359a` |

Source 以服务器回传的原始字节数与 SHA256 重验 Gate8 的 source、build、export、c16-smoke、full-benchmark 五份报告、前驱引用和原 Source Lock，再校验本层使用的文件、旧 Smoke、真实生成配置和图常量。新目录不写历史证据。注册 GE 名保持 `DFlashGroupQuantLinear`，由私有 vendor、独立编译缓存、编译输入、图节点和 OM 中的实际 ELF 建立各版身份。

## 服务器逐阶段命令

从原 CANN 9.0 环境和原 venv 操作。在同一个 shell 中先执行：

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only origin feature/ascend310p-custom-ops
BASE=/home/w00949577/qwen35-dflash-run
PY="$BASE/venv-hf513-probe/bin/python"
S="$PWD/framework/custom_ops/draft_gate_scale_broadcast"
R=$(mktemp -d "$BASE/gate-scale-broadcast-XXXXXX")
echo "R=$R"
scale_stage() { TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/run_scale.py" "$@" --root "$R"; }
scale_stage source
```

以下每行均为**单独一次手动调用**。确认前一份 `$R/<阶段>/report.json` 状态成功后再继续。Source 使用新的实验根，后续使用同一个 R。

| 命令 | 成功状态 / 核验 |
|---|---|
| `scale_stage build` | `SCALE_BUILD_PASS`。从冻结工程单独生成候选，编译 Gate 目标。复制已核验的联合 OPP，在副本中替换目标 .o、关联 JSON 及动态 C++。Host/Proto/API、其余算子、wrapper、生成头、注册表及 Runner 身份保持一致。 |
| `scale_stage atc` | `SCALE_ATC_PASS`。只编一份候选单投影 OM。核对独立进程环境、真实回调、源/常量/ABI、保护节点、内嵌 ELF 和可执行段确有变化，校验实际图节点 8 block、50 个 Cube 字段、完整 parent 与控制一致。比较 ELF 中声明的可写区范围，并保存设备 metadata 身份。 |
| `scale_stage review` | `SCALE_REVIEW_PASS`。安全/资源证据绑定到实际 ATC 产物。在隔离的主机 SDK CPU 进程中对照同一个 helper、实际 Brcb 和独立逐位预期。一次覆盖全 65536 种 16 位输入，一次覆盖原始 389120 个真实 Scale 字（每版展开 6225920 个 half），包括全部 group/tile。逐项检查输入只读、输出前后 canary 和 rear。仅 CPU。 |
| `scale_stage smoke --allow-device` | `SCALE_SMOKE_PASS`。两轮基线/候选和反序，四个独立进程。每进程两个不同 Poison 的未计时 Execute+Sync，共八次。每次完整输出 311296 个 FP16 均须与该层 Native Golden 逐 bit 一致。 |
| `scale_stage benchmark --allow-device` | `SCALE_PAIRED_MEASURED`。实际安全与数值准入后的无插桩 A/B，三轮控制先、候选先、控制先。每轮每版 5 次 warmup、30 次连续 resident Execute+Sync，每版 90 个正式样本，共 180 个。 |
| `scale_stage compare` | `SCALE_GATE_GO` 或 `SCALE_GATE_NO_GO`。从逐进程执行证据重放样本。输出 Median/P95/Min/Max、每轮倍率、总体倍率和全部样本。 |

每个 Run 复核私有 OPP、Runner 和依赖，保留 pre-launch socket 的 PID、FD、环境与实时 maps 检查，通过后才发 G，结束后复核 post-maps。GM 输入/输出、模型存储及 Workspace 的 Guard、Readonly、Poison、Repeat、同步、无 CPU fallback 仍由原受保护 Runner 核验；权重与 Scale 作为 AIR/OM 常量绑定。设备阶段不会由 CPU review 或其他阶段自动触发。任何身份、资源、布局、完整输出、依赖或安全条件不符即 BLOCKED，保留日志和 active_case。发生设备异常后停止该实验。Benchmark 计时区间不读回、不比较数值、不做 Host Guard 扫描或文件 IO、不跑 profiler；每个计时进程保留正确性前置、末样本与独立 postcheck。

采用**中位延迟降低至少 3%**、三轮中位延迟全部改善、总体 P95 无回退作为本次研究信号，门槛在采样前固定。Compare 的 NO_GO 退出 3，阻断退出 1。GO 只允许讨论五个 Gate 的新实验。此次没有完整 Draft 或生产部署入口。

## 从 ABnQDE 的已编译 OM 续接 CPU Review

`gate-scale-broadcast-ABnQDE` 的 Source、Build 和 ATC 通过。旧 Review 因把 Gate8 的历史编译器记录当作新包的字段格式而触发 `KeyError('tools')`。Gate8 记录在 `compiler.command_path_tools[name]`，内含 `alias` 和 `file`。修复后的 Review 验证别名的实际目标、执行权限、大小和 SHA256，直接运行所绑定的 CMake 和 C++ 工具。报告保存这两份完整记录，并复核 configure/build 的 argv 和 cwd。

服务器在仓库根目录执行：

```bash
git pull --ff-only origin feature/ascend310p-custom-ops
BASE=/home/w00949577/qwen35-dflash-run
PY="$BASE/venv-hf513-probe/bin/python"
S="$PWD/framework/custom_ops/draft_gate_scale_broadcast"
OLD="$BASE/gate-scale-broadcast-ABnQDE"
R=$(mktemp -d "$BASE/gate-scale-reviewfix-XXXXXX")
echo "R=$R"
scale_fixed() { TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/run_scale.py" "$@" --root "$R" --resume-from "$OLD"; }
scale_fixed review
```

只对这一次封存失败开放续接。程序以服务器回传的失败报告 SHA256 `a5395a1786568096ae734d2a7c4978a2da8878dc810600cd61dba372921a0c32` 及三个成功阶段的独立摘要校验原件。原锁定源码以提交 `6ddb9cbe1353b08707f7e85f0beb27f84b655291` 的 Git blob 核验本次替换的文件，其他旧依赖须与磁盘一致。三个阶段在旧路径重新执行证据校验，保留完整 OPP、OM、回调和数值资源合同。新报告的 `reuse` 记录原报告、来源提交和源文件迁移摘要，`predecessors` 直接引用原件。CPU 输出写入新根。旧 Review 继续保留 BLOCKED。

仅当新报告为 `SCALE_REVIEW_PASS`，下一条手动设备命令才是 `scale_fixed smoke --allow-device`。其后沿用同一函数和新根，分别执行 `scale_fixed benchmark --allow-device` 和 `scale_fixed compare`。

## 本地证据和复现

- 真实 ARM64 CANN 9.0 CPU 模型（tikicpulib::Ascend310P3，对应 dav_2002）对全部 65536 个位型进行了对照，控制/候选各 1048576 个展开元素以及源、前后 canary、rear，错误数为 0。`/work/build/scale-sdk-cpu-p4B7qs/probe.log` SHA256 为 `eca7cd6d6a90af9def41468bd6ddbead200625da461e3b544a44333d79f13dc5`。此文件是 CPU 数据，本地没有真实 310P3 输出或延迟。
- 本地生成工程产物在 `/work/build/gate-scale-compile-i7f_p5rf/report.json`，SHA256 为 `278bd4e9ed0d1aa80482666b6df7b34450037ab539ea5da9870c2d4cd005d99e`。同一份候选 C++ SHA256 为 `ed11b76a2a602558e0f24bd393f33d7fe4246988326e13fd57205d22284f348b`。SDK 生成并安装的 Gate 包内对象为 57216 B、SHA256 `797a7e97bb750fbb560a8482c0f502cf65e69e86df3fbc8bd1aaead5adeb6a56`。此本地模板只有 Gate，服务器会另行核验联合 OPP 及专用 OM。
- 可移植 CPU 测试用实际派生的两个 kernel 源、8 个 block、不同通道/组的确定性数据，按独立逻辑 NK 参考核对全部 311296 个位型和唯一写 owner。它另外检查 3040 次 Iterate、2736 次 partial、3344 次全流水 fence 及原 Mul、Cast 次数。这些数据与冻结 Native Golden 分开。

```bash
python3 -B -m unittest discover -s framework/custom_ops/draft_gate_scale_broadcast/test -v
# 已有无 NPU CANN ARM64 容器中
SCALE_CANN_ROOT=/usr/local/Ascend/cann-9.0.0 python3 -B -m unittest discover -s framework/custom_ops/draft_gate_scale_broadcast/test -v
# 本地同版 SDK 生成工程 fixture，只做隔离包构建
python3 -B framework/custom_ops/draft_gate_scale_broadcast/scale_build.py local \
  --template /work/build/gate8-check-t6eu9_qo/raw/CustomOp --parent /work/build
```

## 若单层收益不足

下一项优先研究**按热点 Gate 在模型加载时物化同源有效 FP16 权重并跨调用复用**，核内不再每次做整遍 Cast/Scale/Mul。仓库已有 [同源有效权重与 tiled FP16 对照](../draft_quant/fp16_tiled/README.md)，应先复核其实际 Golden 和构建记录，再将这一条数据流与本次 Raw 的静态 8-block OM、真实同层输入作新配对。首个比较保持 K 顺序和 FP16 舍入，权重文件逐元素绑定 RN16(int8×GN)。单 Gate resident 权重及 Scale payload 由 50585600 B 变为 99614720 B；保留两份时另计额外常驻占用。分别测加载/预打包/上传成本、单次延迟与权重复用达到的摊销点；一次只选一层。若带宽或常驻预算不允许，此方向也应停止。历史其他 core 策略、单算子与完整 Draft 的数据不能混为同一组收益。
