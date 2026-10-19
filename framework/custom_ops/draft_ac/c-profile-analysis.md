# C 完整 Draft 性能归因

Profiling：NOT_RUN；端到端：NOT_RUN。
历史 median/p95 是未插桩数据；本次 profile 时间仅用于归因，不能替换历史 benchmark。

| Gear | Mode | Frozen median / p95 ms | Profile 状态 |
|---|---|---:|---|
| C16 | off | 45.7116 / 未提供 | NOT_RUN |
| C16 | a | 31.2055 / 未提供 | NOT_RUN |
| C16 | c | 55.0590 / 未提供 | NOT_RUN |
| C16 | ac | 40.5132 / 未提供 | NOT_RUN |
| C64 | off | 45.8936 / 未提供 | NOT_RUN |
| C64 | a | 32.0814 / 未提供 | NOT_RUN |
| C64 | c | 55.2393 / 未提供 | NOT_RUN |
| C64 | ac | 41.3948 / 未提供 | NOT_RUN |

## 四模式对照

- C16 c − off：历史 median 差 9.3474 ms。
  Device 时间线解释：NOT_RUN 或数据/Head 映射不足。
- C16 ac − a：历史 median 差 9.3077 ms。
  Device 时间线解释：NOT_RUN 或数据/Head 映射不足。
- C64 c − off：历史 median 差 9.3457 ms。
  Device 时间线解释：NOT_RUN 或数据/Head 映射不足。
- C64 ac − a：历史 median 差 9.3134 ms。
  Device 时间线解释：NOT_RUN 或数据/Head 映射不足。

## Head 及转换明细

### C16 / off

- pre_atc_graph：NOT_RUN；节点分类 {}。
- compiled_graph：NOT_RUN；节点分类 {}。
- 测量窗口未可靠识别：不从所有 task 总和推算单次 Head 时间。
- 描述符、权重格式、常量属性和邻接转换：见 `c-profile-report.json` cases 的图与原始 CSV fields。
### C16 / a

- pre_atc_graph：NOT_RUN；节点分类 {}。
- compiled_graph：NOT_RUN；节点分类 {}。
- 测量窗口未可靠识别：不从所有 task 总和推算单次 Head 时间。
- 描述符、权重格式、常量属性和邻接转换：见 `c-profile-report.json` cases 的图与原始 CSV fields。
### C16 / c

- pre_atc_graph：NOT_RUN；节点分类 {}。
- compiled_graph：NOT_RUN；节点分类 {}。
- 测量窗口未可靠识别：不从所有 task 总和推算单次 Head 时间。
- 描述符、权重格式、常量属性和邻接转换：见 `c-profile-report.json` cases 的图与原始 CSV fields。
### C16 / ac

- pre_atc_graph：NOT_RUN；节点分类 {}。
- compiled_graph：NOT_RUN；节点分类 {}。
- 测量窗口未可靠识别：不从所有 task 总和推算单次 Head 时间。
- 描述符、权重格式、常量属性和邻接转换：见 `c-profile-report.json` cases 的图与原始 CSV fields。
### C64 / off

- pre_atc_graph：NOT_RUN；节点分类 {}。
- compiled_graph：NOT_RUN；节点分类 {}。
- 测量窗口未可靠识别：不从所有 task 总和推算单次 Head 时间。
- 描述符、权重格式、常量属性和邻接转换：见 `c-profile-report.json` cases 的图与原始 CSV fields。
### C64 / a

- pre_atc_graph：NOT_RUN；节点分类 {}。
- compiled_graph：NOT_RUN；节点分类 {}。
- 测量窗口未可靠识别：不从所有 task 总和推算单次 Head 时间。
- 描述符、权重格式、常量属性和邻接转换：见 `c-profile-report.json` cases 的图与原始 CSV fields。
### C64 / c

- pre_atc_graph：NOT_RUN；节点分类 {}。
- compiled_graph：NOT_RUN；节点分类 {}。
- 测量窗口未可靠识别：不从所有 task 总和推算单次 Head 时间。
- 描述符、权重格式、常量属性和邻接转换：见 `c-profile-report.json` cases 的图与原始 CSV fields。
### C64 / ac

- pre_atc_graph：NOT_RUN；节点分类 {}。
- compiled_graph：NOT_RUN；节点分类 {}。
- 测量窗口未可靠识别：不从所有 task 总和推算单次 Head 时间。
- 描述符、权重格式、常量属性和邻接转换：见 `c-profile-report.json` cases 的图与原始 CSV fields。

## 已证实与未证实

- 用户报告 52+4+8 bitwise PASS；尚未在本地读取服务器原始产物。
- 给定 medians 的四个 C 增量均约 9.3 ms；这不是设备 Kernel 归因。
- 未证实：Native immutable Head 的 MatMul/Argmax 时间与权重格式尚未取得。
- 未证实：Custom C 内部 MatMul、Vector Top1、权重读取、跨核同步与图调度贡献尚未确定。
- 未证实：独立 runtime-weight Head 延迟不可直接用于分解 immutable-weight 整图时延。

## 后续候选（尚未实现或测得收益）

- 若设备明细证明 Custom C 总时间高于 Native MatMul + Argmax，且额外转换不足以解释差值：分别评估 C 的 MatMul 数据流/权重布局，以及保留 Native MatMul、仅优化 Top1。
- Native MatMul + Top1 必须消费最终 FP16 logits，保留饱和/舍入边界、NaN/±0、最小 ID tie-break；M15 全词表 logits 逻辑大小为 7,449,600 B，显式写回及读取、额外调度成本必须计入整图实测。
- 只有同一 immutable-weight 整图下消除的已测耗时才构成收益假设依据；不从独立 runtime-weight 的 17.1/23.3 ms 外推。
- Fused C 的 MatMul、Vector Top1 与跨核同步在单个 kernel 内；本工具不修改 Kernel，不能凭 Pipe 比例分割这些耗时。Pipe 计数可重叠，不能相加。
- 只做性能归因：不修改 Kernel、Tiling、模型数学、Golden 或历史 evidence。
