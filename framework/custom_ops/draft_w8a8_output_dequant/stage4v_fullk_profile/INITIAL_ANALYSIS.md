# Before the new device Profile: evidence limits

The user reports Full-K median1.641665ms, P951.64961ms (90 uninstrumented samples)
and stable201 one-ULP mismatches. Relative to historical Group1282.77505ms and
FP160.78698ms, the arithmetic ratios are about1.6904x faster and2.0860x slower,
respectively. These are historical shape-matched comparisons, not a fresh paired
hardware-throughput result. They do not identify the device critical path.

The existing normal source gives these counts for full Auto7 Blocks:

| Source operation | Count | What cannot be inferred |
|---|---:|---|
| Caller Matmul object/registration |7|SDK internal reinitialization cost per Tile|
| Iterate / GetTensorC / End |76 each|machine instruction count or waiting duration|
| Final FP16 GM DataCopy |19456 ×32B|actual MTE3 transactions or wall-time contribution|
| Output payload |622592B|total memory traffic including input transforms/workspace|
| UB Guard GetValue words |87552|pure Guard time or critical-path Scalar cycles|
| Arena GetValue words |21248|pure Arena time; underlying DMA overlaps are unknown|
| Core assignments |11/11/11/11/11/11/10|measured execution imbalance or optimal schedule|

No new raw msprof files are available locally. CUBE/MAC, MTE1/2/3, Vector, Scalar,
Memory and per-Core tail evidence are **UNAVAILABLE** for this new experiment.
The Driver will populate exact source/type/unit groups from the server capture.

Provisional investigation order: B input ND→NZ conversion; Matmul lifecycle;
equivalent-coverage Guard/Arena checks; output writeback; scheduling. This is an
investigation plan based on structure and prior negative writeback evidence,
not a ranked list of measured potential gains. Source/API proof remains required
before any implementation. Full-K VDEQ numerical mismatches remain UNRESOLVED.
