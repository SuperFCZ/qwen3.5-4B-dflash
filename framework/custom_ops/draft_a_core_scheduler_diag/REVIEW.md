# CANN 调度 API 与当前7-Core语义

## 证据等级与版本

安装根 `/usr/local/Ascend/cann-9.0.0`，精确文件/行号/SHA见 `sdk-review.json`。
平台头文件 SHA `44cf3f35…97f7f2a` 已得到服务器核验；本地两个Host版本的编译链接也接受这些签名。
新增 TilingContext getter头文件在服务器Source阶段必须再次匹配。

安装的 `platform_ascendc.h`：

| 行号 | 接口/注释所能证明的范围 |
|---|---|
| 96 | 构造参数为实际 `fe::PlatFormInfos*` |
| 97–102 | GetCoreNum按core type返回；注释另说明910B MIX特例 |
| 103–112 | GetCoreNumAic / GetCoreNumAiv存在，返回uint32 |
| 113–117 | GetCoreNumVector用于m200的独立Vector资源 |
| 118–125 | sliceNum为切分份数；使用矩阵API传Aic getter，使用矢量API传Aiv getter |
| 140–141 | SoC与Arch查询 |

实际库中确认存在这些符号；未反汇编。实现正文没有随本地安装头文件交付。
官方固定提交 `354d1153d73ec4e7c746dae0c18251d5f802dd1f` 的平台头文件与安装文件完整SHA相同，
其[对应实现](https://raw.gitcode.com/cann/asc-devkit/raw/354d1153d73ec4e7c746dae0c18251d5f802dd1f/impl/utils/tiling/platform/platform_ascendc.cpp)
SHA为 `f91e2555…3b399`。这是**匹配头文件的关联源码**，不是已证明服务器库由该源码逐字编译的证据。
不将该公开tag名称改写成服务器版本；服务器安装版本仍以冻结9.0.0身份记录为准。

特别发现：当前公开 `v9.0.0` 平台头文件SHA为 `7ba13233…d4d58`，与安装文件不同；
官方[/900/调度文档](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_1033.html)
正文也已使用 `CalcTschNumBlocks`。不能仅凭URL或版本号把它当成此安装包实现，更不能自动迁移接口。
本实验调用安装头文件真实声明的 `CalcTschBlockDim`。

## 当前7的准确解释与未闭合部分

实测已证明：同一Host DSO在ACLNN下，PlatformAscendC通用getter与PlatFormInfos两个getter都为7；
专用Aic/Aiv getter均为8。ATC的通用getter为8。SoC、Arch、容量和Workspace一致。

关联源码123–125将通用查询委托给PlatFormInfos；77–96、113–120表明专用查询根据
core_type_list选择资源字典键，在非分离视图中都取ai_core_cnt。99–110单独读取310P的独立Vector数量。
这解释了为何这些getter**不必读取同一个数据分量**，并与观测一致。

因此不能把GetCoreNum=7解释为“只有7个物理Cube”“8减去一个故障核”，也不能把
vector_core_cnt=7和ai_cpu_cnt=7的数值巧合当成赋值调用链证明。
**Runtime的PlatFormInfos通用Core值由谁选择/写入，以及是否代表资源限制，仍UNRESOLVED。**
本实验新增GetCoreNumVector和原始core_type_list记录；不会调用SetCoreNum或改写PlatformInfo。

当前正式路径继续保留历史GetCoreNum/LaunchBlockCount。独立B计划只用于Host研究，不能直接生产采用。

## CalcTschBlockDim与sliceNum

关联源码222–236的算法可概括为：有有效AIC/AIV且AIV不少于AIC时，以整数比例
`r=AIV/AIC` 将sliceNum换算为 `ceil(sliceNum/r)`；该源码还对1:2分离组合的上限作拒绝检查。
其他输入分支不提供自动物理容量裁剪。因此它不是“输入所有Tile数，自动得到可用核数”的接口。

当前安装头文件没有给出dav_2002禁用声明。相关实现也没有排除310P的SoC分支，
在1:1时只是恒等换算。公开文档主要描述分离模式；这里**不启用分离模式**，只检验实际SDK
在已确认耦合Arch=2002、SoC=310P、AIC=AIV且非零时的1:1返回。若实际库返回0、不同结果或
实际架构/比例不同，立即BLOCKED，不改用手写返回值、其他API或默认8。

本算子每个worker执行完整N64 Tile，不切分K。N2560共有40个Tile；Kernel原循环为
`t=blockIdx; t<40; t+=blockNum`。因此本实验选择：

- A：`min(40, GetCoreNum())`，沿用原LaunchBlockCount和core_limit=0。
- B：先选择至多一个worker/所报告耦合AI Core，即 `sliceNum=min(40, GetCoreNumAiv())`；
  再调用真实CalcTschBlockDim，传入真实Aic/Aiv getter；要求返回值等于该1:1切分数且不超过两种getter。
- 不把独立Vector Core数加入Cube可用worker，也不把8个Cube和8个Vector相加成16。
- core_limit非零、非2002、AIC/AIV不相等等情况不属于本次候选，直接拒绝。

按用户现有getter观测形成的**CPU计划，不是新的真机结果**：

| 路径 | A workers | B sliceNum | B预期API值 | 每worker Tile数 |
|---|---:|---:|---:|---|
| ATC | 8 | 8 | 8 | 两版均5 |
| ACLNN | 7 | 8 | 8 | A：6,6,6,6,6,5,5；B：均5 |

40个Tile在两种分配中都恰好覆盖一次。这个变化没有新增Cube/Vector重叠，也没有证明物理核占用率。
带宽、共享Workspace、Runtime配额和实际启动行为仍需后续独立设备Smoke；不能从上述计数推算加速比。

## 资源与数值保持

独立Host仅替换选择blockDim的表达式，并增加证据输出；源码逐字变换审计。
A/B都复制同一冻结的Kernel、ACLNN API和配置；只重建Host tiling/proto DSO。
两份OPP、两份Runner和所有进程互相隔离。Runner源码与已验证Host-only Runner逐字相同，没有Execute调用。

仍由原Host执行全部MatmulApiTiling、类型/Shape、UB、L1/L0和Workspace检查。
新sidecar记录实际SetBlockDim回读值、两种完整Tile分配、UB使用/容量、Scratch、L1/L0使用、
raw bank、tileK及原始父级TilingData序列化。后者取已经写入Context的完整缓冲，不调用嵌套子对象SaveToBuffer。

必须同时满足：

1. A/B每条路径的资源账完全相同；完整父级Tiling字节相同。
2. 两路容量、泛型/专用Core getter及原始资源字典保持各自基线值。
3. ACLNN两次实际Workspace返回值与callback及历史基线完全相同。
4. 40 Tile覆盖无重叠、无遗漏；A/B之间仅选择worker数不同。
5. 所有Kernel/API字节、SDK、私有及标准DSO身份、别名和环境检查保留。

## ATC环境前缀

本地实际文件系统：`lib64`与`aarch64-linux/lib64`解析到同一目录；
`aarch64-linux/python/site-packages`不存在，通用 `python/site-packages`存在。
这些本地事实不能替代服务器检查。Source在服务器重新记录四个目录的存在性/真实路径/链接，
并逐个比较已绑定库名在两套LD路径中的第一候选文件SHA。

如果新增LD前缀只是同一目录别名，且没有与私有API同名的更早库，它不会改变这些已检查文件的选择。
如果Python前缀不存在，它不能提供模块；若存在且不同，需要真实模块来源证据。
ATC完整maps未采集，故不宣称所有动态依赖相同，也不据此排除PlatformInfo生产者差异。
两条路径Host DSO相同只能约束Host实现身份，不能证明传入的PlatFormInfos相同。

## 建议边界

本实验的意义是检验显式coupled worker计划是否得到实际Host接受，**不是用CalcTschBlockDim修复或掩盖7的来源**。
若全部Host门槛通过，可提交一次独立8-worker单FC数值/Guard Smoke方案；执行仍需单独批准。
若API、比例、资源、序列化或环境不满足，停止，不强行设置8，也不自动替换正式7-Core主线。
完整Draft收益、Benchmark、Vector/Cube流水重叠和Runtime配额解释均未由本实验完成。
