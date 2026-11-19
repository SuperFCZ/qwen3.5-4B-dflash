# C16 Layer0 Gate 双 W16 排程实验

基线审查为 `3971424e8946c13b3a757bce887f3db53cc788d2`。第一轮仅编译同一份 C16 Layer0 Gate AIR。使用真实 FP16 [16,2560]、INT8 NZ [80,1216,16,32]、GN FP16 [20,19456] 和 Native FP16 [16,19456] Golden。输入、权重、Scale、Golden 的 bytes 与 SHA256 来自 Gate8 Source，逐一重验，并核对审计记录的四个固定 SHA。完整 Draft 和五层入口均需单独调用。

| 变体 | 实现与身份 | 可回答的问题 |
|---|---|---|
| single-w-raw (A) | 复用 gate-raw-atc8-Np20yu 中已经验收的 Raw 单投影/完整 OM、私有 OPP、Guarded Runner。每次计时取新样本 | 对当前 Raw 的实际净收益 |
| double-w-serial (B) | 独立私有 OPP。先 F(k)，再 D(k+1) | 双 W 的串行对照 |
| double-w-overlap (C) | 另一独立私有 OPP。Iterate(k) 成功、raw(k+1) 两个 Wait 完成后，在另一 W 上 D(k+1)，然后 F(k) | 在相同资源下移动 D 的收益 |

B/C 共用 [同一个模板循环](kernel/tile_schedule.h)、[同一组设备操作](kernel/double_w.inc)和[Host 资源检查](host/w16_admission.inc)。在 Git SHA 锁定的 Raw 生成工程中拼装独立 Host/Kernel，仅一个模板布尔值不同。每个 block 的 W0/W1 各 32768 B，两份 raw q/s、2304 B broadcast 含 rear、152832 B Matmul scratch、顶端 SDK reserve 8192 B。代码同时检查实际 TPipe 物理地址与 8-block 执行条件。每次 K 仍有 V_S 和原 PIPE_ALL，下一次 SetTensorA/B、Iterate 在 fence 后。k=9 无 lookahead，失败时只 drain 已经发出的 raw 工作，保留未写 Poison。每个 N tile 仅一次最终 FP16 输出。B/C 都把 raw0 就绪和 D0 放在 prologue，并在当前 Iterate 前发下一 raw copy；这点与单 W 的原始发射时机不同。

保留 DFlashGroupQuantLinear 的注册名与原始 GE ABI，以便原 AIR 直接编译；每版在独立根中构建/运行，不向全局 CANN 安装。历史 7-block ACLNN 结果不参与。正式回调要求 Scheduler A、core_limit=0、实际 blockDim=8、304 个 N tile 各有唯一 owner、每 block 38 个。完整父序列化 224 B，首 24 B 仅允许 matmulUbBytes 从 185600 变成 152832。后 200 B 的 50 字段及 SHA256 必须与冻结 Raw 完全一致，L0B 保持 65536 B。单算子 Workspace 从正式回调读取；Runner 的整 OM work/weight 分配另行核对。

## 只读复核 O34Eaf 的 ATC 图比较

`gate-w16-atc8-O34Eaf` 在比较 `arg2_1` 时退出。服务器回传中两边都是 Data 节点。按键对齐后，Data 和两个 Const 节点的属性值和出现次数一致。Gate 和 NetOutput 的 TensorDesc 中有重复键。已观察到的换序是 `origin_shape_initialized` 的两个布尔值，以及 GroupQuant 第 2 个输入（NZ 权重）上 `format_for_int` 的两个整数 29、100663325。Gate 的专用 ELF 大小为 19608 / 32304 B，`custom_op_file_path` 分别指向两份私有包。

CANN 9.0 安装文件 `include/proto/ge_ir.proto` 第 339、359 行把 descriptor 和 operator 的 attr 定义为 map，第 356、375–376 行把输入边和输入/输出 descriptor 定义为 repeated。文件 SHA256 为 `20cde9c443d712d802c46de87a6c1214ad96858ac3bee9b2056eb2e29dcebb7e`。比较器按键比较 dump 中的属性条目，保留完整值、类型和重数，上述两类同键换序按实际值配对。其他同键异值仍检查出现顺序。输入边、槽位、维度、属性内部的列表保持位置比较。GroupQuant 的包路径在两侧分别核对到该侧已绑定的 `vendors/customize`。非目标节点的代码身份继续严格比较。

在仓库根目录拉取修复后，可直接读取已封存的两个 dump 和两个 OM。该命令只打印结果。

```bash
BASE=/home/w00949577/qwen35-dflash-run
PY="$BASE/venv-hf513-probe/bin/python"
S="$PWD/framework/custom_ops/draft_gate_w16_atc8"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/recheck_graph.py" --root "$BASE/gate-w16-atc8-O34Eaf"
```

预期为 `ARCHIVED_W16_GRAPH_MATCH`，并显示 `archived_atc_status=BLOCKED`。历史 Raw / serial dump SHA256 分别为 `e2993d7c751c21789d536a6794e6ff9307333790b9ac8ab5928534bd56eb1d10`、`30cdb4778c408e7f292ea3dd766449c68bacf489e9ad04721a09720e49d4d63e`。正式阶段按全模块源码清单绑定前驱，本次验证器修复改变了清单。使用下方的新实验目录逐次执行 source、build、atc，再单独进行 static。

## 服务器命令

仓库下执行以下初始化。当前 shell 已启用该机原 CANN 环境和历史 venv。

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only origin feature/ascend310p-custom-ops
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
BASE=/home/w00949577/qwen35-dflash-run
PY="$BASE/venv-hf513-probe/bin/python"
S="$PWD/framework/custom_ops/draft_gate_w16_atc8"
R=$(mktemp -d "$BASE/gate-w16-atc8-XXXXXX")
echo "$R"
w16() { TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/run_w16.py" "$@" --root "$R"; }
w16 source
```

默认历史根为 `$BASE/gate-raw-atc8-Np20yu` 和 `$BASE/w8a16-deep-profile-acJHJJ`。Source 只读重验 Gate8 全链及 Raw 配对 Profile 的绑定，不重新运行历史实验。检查每个阶段的 report.json 和退出状态；下列各行是独立的手动入口。

```bash
w16 build
w16 atc
w16 static
```

| 阶段 | 成功状态 | 主要证据 |
|---|---|---|
| source | W16_SOURCE_PASS | Gate8 各阶段原件、两档当前 Raw OM、单投影输入/Golden、SDK 23 个文件、反汇编工具 |
| build | W16_BUILD_PASS | 两个新私有 OPP、独立编译命令和产物、原 Runner/动态库重新绑定。原 API、注册索引和非目标文件核验 |
| atc | W16_ATC_PASS | 新 B/C 单投影 OM、原 ATC stdout/stderr、独立 callback、平台/资源、图、三版内嵌 ELF 和解码结果 |
| static | W16_STATIC_PASS | 两版实际指令的 UB 地址/资源所有权、SDK 路径、首中末与失败 CFG 证据 |
| smoke | W16_SMOKE_PASS | 三版各两次不同 Poison，311296/311296 位型、Repeat、Readonly、Guard、DSO 及同步 |
| schedule-benchmark / net-benchmark | W16_SCHEDULE_BENCHMARK_PASS / W16_NET_BENCHMARK_PASS | 两组各 180 个新鲜的完整样本及实际身份 |
| compare | W16_COMPARE_PASS 或 W16_NO_GO | 三轮测速、总体倍率和 P95、两次 C 锚点漂移、是否可扩层 |

**先看 ATC 的实际 ELF。** `atc` 成功表示取得编译产物，不能代替 `static`。SDK 工具未解码、无法证明地址/消费者/控制流时得到 `BLOCKED_STATIC_ADMISSION`。调度被完全串行化则为 `NO_TESTABLE_OVERLAP`。此时保留原目录，停在静态阶段。证据格式、手工审查与只读重验入口见 [Static Admission](STATIC_ADMISSION.md)。

仅在 W16_STATIC_PASS 后，才手动调用以下设备入口。每次都是独立私有环境、独立进程，强制原 Runner 发送 socket token，检查 SCM_CREDENTIALS、PID、FD 端点、可执行文件、argv、环境和 pre-maps 与实时 maps，DSO 身份通过后才发 G。每次检查 post-maps、512 B GM/Workspace 双侧 Guard、输入只读、不同 Poison、同步及完整 Golden。权重和 Scale 是 OM 常量，逐 Gate 绑定原 payload 及编译图。Runner 的设备只读扫描覆盖其输入，模型权重区保留原 Runner 的 Guard 范围。

```bash
w16 smoke --allow-device
w16 schedule-benchmark --allow-device
w16 net-benchmark --allow-device
w16 compare
```

每组控制→候选、候选→控制、控制→候选，各轮每版 5 次 warmup + 30 次 resident Execute+Sync，共 90 个有效样本/版。计时前两次 correctness，计时末尾回查及独立 postcheck，读回/比较/日志/模型加载均在计时区间外。配对报告给出每轮/总体 Median、P95、Min、Max、原始样本、Workspace、OM/ELF、Runner/DSO。两个配对都要求三轮 median 同向，总体中位倍率 ≥1.03，且总体 P95 无回退。未达标完整记录 `W16_NO_GO` 后停止扩大。

compare 中的 `serial_minus_single_anchor_adjusted_ms` 是两组以 C 为锚点的差分估计；同报 C 的漂移。若要直接测 A 与 B 的串行综合成本，可单独执行 `w16 resource-benchmark --allow-device`。A/B 的差还含 raw issue 时机变化，这三种实现不能独立测出“纯额外 UB 分配”的因果成本。

## 可选扩层和整图

仅在单层 compare GO 后逐行执行。五份 Gate 的逻辑 shape 一致，每层的 NZ/GN 在 C16/C64 相同且激活、Golden 独立绑定；其余四层各编一次 B/C 单投影 OM。C64 复用已封存的对应层 OM，另跑 C64 数据。不能用 C16 数值当 C64 验收。

```bash
w16 five-c16-atc
w16 five-c16-static
w16 five-c16-smoke --allow-device
w16 five-c16-benchmark --allow-device
w16 five-c64-smoke --allow-device
w16 five-c64-benchmark --allow-device
```

两档分别报告每层三轮结果及两种顺序的独立中位数之和，各层三轮改善且无各顺序 P95 回退，两个顺序的五层倍率均 ≥1.03，两个配对均通过才能扩大。完整 Draft 只需新编 C 两份 OM，A 两份直接复用当前单 W Raw。每档使用同一冻结 A-only AIR。26 个 GroupQuant 中只有五个 Gate 可以变，核查其余 21 个投影的实际代码与资源、Head 等原生节点绑定、图结构和 ABI。新的完整 OM 的五个 Gate ELF 必须重新绑定静态证据。

```bash
w16 full-atc
w16 full-static
w16 full-smoke --allow-device
w16 full-benchmark --allow-device
```

完整 Smoke 比较全部 15 个 token 和 10 个完整 KV buffer（含未更新区）。完整延迟来自每档每版 90 个新鲜无插桩样本。独立投影和、插桩窗口、历史 median 均不作为完整 Draft 延迟。

## 独立 Down 后备

单独使用新根和 Down 专用 Runner。控制使用原 Raw OPP，候选保留相同 Host、单 W 和 Raw 双 bank，只改 Down 的两个 GM→UB 描述符。在同一真实历史单 Down AIR 编译两份 OM，并核对控制专用执行段与已验收 Raw 完整图的对应投影。实际 8-block Tiling、Workspace 和资源须相同。按 dav_m200 的 32 B 单位，q 为 (8,64,2496,0)，Scale 为 (2,4,156,0)。40 个 N64、38 个 K256 均整除，不存在尾部补齐。CPU 测试遍历全部跨度。另外以实际候选 C++ 在确定性夹具上检查最后 N tile、Group128 舍入和越界写。真实 Down 的 x、NZ、GN 和 40960 个 FP16 Golden 从绑定的 Scheduler Flow Source 的 `16/layer-0-down` 读取。

```bash
D=$(mktemp -d "$BASE/down-raw-dma-XXXXXX")
down() { TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/down_fallback.py" "$@" --root "$D"; }
down source
down build
down atc
# 由服务器操作者分别启动
down smoke --allow-device
down benchmark --allow-device
```

下述测试只使用 CPU，Linux 项额外检验真实 socket 凭据。可在仓库根运行。

```bash
python3 -B -m unittest discover -s framework/custom_ops/draft_gate_w16_atc8/test -v
python3 -B -m unittest discover -s framework/custom_ops/draft_w8a16_overlap_review -v
```

每阶段 JSON 独占目录，失败退出非零，保留定位信息和已产生的文件。退出 3 表示 NO_GO 或 NO_TESTABLE_OVERLAP。不同 stage 的执行状态互不替代。[主流程 Schema](report.schema.json) · [Down Schema](down/report.schema.json) · [本地实证与待运行阶段](LOCAL_VALIDATION.md)
