# Project Guidance for Coding Agents

## 项目契约

- 原始要求与 Canvas 100 分评分表见 [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md)（原始选题留档于 [docs/archive/1.md](docs/archive/1.md)）。核心任务为基于预训练视觉语言共享表征的双向检索与对比预训练原理解释；不从零训练基础模型；正式截止时间 **2026-12-01 23:59 HKT**。
- 日常沟通使用中文；主 `README.md`、最终报告与提交材料使用英文。
- 证据优先级：用户最新要求 → 原始课程要求与 Rubric → 当前代码/运行证据 → 项目协议 → 历史归档。发现冲突须显式记录，不静默改写范围。
- 核心四文档各司其职：[docs/REQUIREMENTS.md](docs/REQUIREMENTS.md)（题目与评分）、[docs/PLAN.md](docs/PLAN.md)（架构与 7 天排期）、[docs/EVALUATION.md](docs/EVALUATION.md)（唯一实验协议 v2）、[docs/PROGRESS.md](docs/PROGRESS.md)（唯一任务状态与决策台账）。历史审核与 Proposal 存放在 `docs/archive/`。

## 当前设计决定（v2）

- **Phase 1 必做主线**：对比 `CLIP ViT-B/32`（基线）、`CLIP RN50` 与 `SigLIP`（InfoNCE vs Sigmoid 损失）；保留 `raw` vs `a photo of ` 前缀对照；执行数据探索、Modality Gap 去中心/白化新特征表示、PCA 维度压缩与 K-Means/Hubness 分析；实现 **CSLS** 与 **Dual-Softmax** 重排并通过 dev 上 **image-grouped 5-fold CV** 调参。
- **Phase 2 可选冲刺**：在 Phase 1 通过且预算落定后，可选启动 NegCLIP-lite 组合性硬负例轻量适应；不以训练提分为项目成败条件。
- **数据策略**：Day 1 可验证 Flickr30k 授权与 split 则用 Flickr30k，否则直接切换官方 COCO `val2017` 自定义划分并显式标注，不冒称 Karpathy 5K test。
- **防泄漏铁律**：所有特征变换参数（均值、白化矩阵、PCA、K-Means）、CV 超参与可选微调选择均只允许在 dev/train 完成；held-out test 与 Winoground 仅在配置冻结后运行一次。Winoground 严格 evaluation-only。
- 不因本地存在 `.agents/`、`.specify/`、`openspec/`、`.omx/` 而自动启动对应工作流框架。

## 执行边界

- 在已授权范围内，普通文档/代码编辑与测试验证直接推进，不重复请求许可；`README.md` 可直接维护。
- **大批量删除必须先说明准确范围并等待用户明确批准**；不得被自主规则覆盖。
- 下载数据集/模型权重、安装依赖或启动训练前，确认属于当前任务授权；不代用户接受数据集条款、分享账户凭证或绕过访问限制。
- 不虚构命令、指标、完成状态、资源可用性或组员分工；禁止手填数字替代运行产物。

## 完成与审查

- 状态枚举为 `todo / doing / blocked / review / done / deferred`，统一登记在 [docs/PROGRESS.md](docs/PROGRESS.md)。`done` 必须附验证命令、输出/产物路径与审查结论；写完代码不等于完成实验。
- 改动评估、特征变换或重排逻辑时，必须跑过对应合成单元测试（多正例、重复/跨划分 ID 泄漏、排序轴、平分、Winoground 严格不等式与 24 排列、5-fold image-grouped 无泄漏、特征拟合仅限 dev）；只改文档不运行无关模型实验。
- 正式运行产物须绑定 `run_id`、git commit / dirty diff hash、依赖版本、数据与 split hash、权重 hash、配置、日志及逐查询/逐组输出。
- 优先由另一 agent 或组员只读审查核心评估代码与最终报告数字；不可用时标注自审边界。

## 当前快照（2026-11-24）

已完成 T0、Round 1（T1+T2）、Round 2（T3）、Round 3（T4：Phase 1 1K dev 全量实验）及 **Round 4（T5：Phase 2 NegCLIP-lite 3K train 硬负例轻量适应，dev 选定 `cfg1_text_adapter_lr1e5_hn0.5` ep3，全配置已冻结至 `configs/frozen_all.json`，17 项单测全通过）**。下一步进入 **Round 5（T6）**：按冻结配置一次性运行 held-out 1K `test`、Winoground 诊断、1,000 次 paired bootstrap CI、分层失败案例与终稿图表。
