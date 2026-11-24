# 项目架构与 7 天执行计划（2026-11-24 — 2026-12-01）

定位：面向 2026-12-01 截止日的 CS5489 课程研究原型与提交方案。以**“理解并改进预训练对比视觉语言检索：从 Modality Gap、Hubness 到组合性鲁棒性”**为主线，兼顾学术严谨性与 100 分 Rubric 全覆盖；不做生产服务、向量数据库或复杂 Web UI。

> 本文件替代 2026-10-20 的 6 周旧规划（原审核记录与 Proposal 已归档至 [archive/](archive/)）。代码与实验尚未启动，以下目录在对应任务启动时创建。

## 1. 研究主线与分级范围

### 核心研究问题
- **RQ1（基础与表征对比）**：不同预训练视觉语言表征（CLIP ViT-B/32、CLIP RN50、SigLIP）在双向检索上的表现差异是什么？Softmax InfoNCE 与 Sigmoid 对比损失、以及固定 `a photo of ` 前缀带来什么影响？
- **RQ2（表征几何与新特征/重排方法）**：预训练跨模态嵌入是否存在显著的 **Modality Gap** 与 **Hubness**？通过去中心化/白化新特征表示、PCA 降维、以及 **CSLS / Dual-Softmax** 重排（经 5-fold CV 调参）能否改善双向检索与效率？
- **RQ3（组合性极限与轻量适应）**：在 Winoground 组合性诊断上暴露出哪些失败模式？在不接触 Winoground 的前提下，通过合成词序/属性置换硬负例做轻量适应（Phase 2 冲刺），能否改善组合性而不损害常规检索？

### 分层交付策略（控制 7 天风险）
- **Phase 1（必做核心，保底覆盖 100 分 Rubric 全部条目）**：
  1. 完整可复现双向检索流水线 + 严格合成单元测试；
  2. 多骨干/多对比损失对比（`CLIP ViT-B/32`、`CLIP RN50`、`SigLIP`）+ `raw` vs `a photo of ` 前缀对照；
  3. 数据探索 + **PCA/UMAP Modality Gap 可视化** + **PCA 压缩实验** + **K-Means 语义簇与 Hubness 分析**；
  4. **新特征表示**（模态去中心化 / 白化对齐）+ **两种无训练重排新方法**（`CSLS`、`Dual-Softmax`），在 dev 集上做 **image-grouped 5-fold CV** 调参后冻结评测；
  5. Winoground 组合性诊断（访问受限时保留已验证 scorer 并如实交代）+ paired bootstrap 置信区间 + 分层失败分析。
- **Phase 2（冲刺加分，Day 4~5 在 Phase 1 通过后启动）**：
  - **NegCLIP-lite 组合性硬负例微调**：在训练集 caption 上自动生成词序/属性置换 hard negatives，冻结图像主干，仅对文本编码器顶层 LoRA 或跨模态投影层做轻量适应（预算封顶 ≤ 2 小时），同时验证常规检索与 Winoground。若算力/时间不达标或无增益，如实报告结果或标记 `deferred`，不影响 Phase 1 交付完整性。

---

## 2. 目标代码架构

```text
.
├── README.md                     # 英文入口 & Code ZIP 复现说明
├── AGENTS.md                     # Agent 执行与审查契约
├── requirements.txt              # T1 锁定后的最小可复现依赖
├── docs/
│   ├── REQUIREMENTS.md           # 原始题目 + Canvas 100 分 Rubric 与提交规范
│   ├── PLAN.md                   # 本文件：架构与 7 天冲刺计划
│   ├── EVALUATION.md             # 唯一实验协议（v2）
│   ├── PROGRESS.md               # 唯一任务状态、决策与证据台账
│   └── archive/                  # 原始题目留档、旧审核记录与已提交 Proposal
├── paper/                        # CLIP / Winoground 参考论文
├── configs/                      # 待建：dataset / model / experiment YAML/JSON 配置
├── src/vlm_retrieval/            # 待建：单包，模块化纯函数优先
│   ├── data.py                   # manifest 构建与校验、ID/split 隔离检查
│   ├── model.py                  # 统一编码接口（OpenAI CLIP & SigLIP）、预处理、截断记录
│   ├── cache.py                  # 特征与来源 metadata 缓存、完整性校验
│   ├── features.py               # Modality Gap 去中心/白化新特征、PCA 压缩、UMAP/K-Means/Hubness 分析
│   ├── retrieval.py              # 双向精确余弦检索、分块评分
│   ├── rerank.py                 # CSLS、Dual-Softmax 重排 + dev 5-fold CV 超参选择
│   ├── adapt.py                  # Phase 2（可选）：Hard-Negative 构造与轻量 LoRA/Adapter
│   ├── metrics.py                # 纯函数 R@K、Winoground 严格计分、paired bootstrap CI
│   ├── evaluate.py               # 编排评测、写 run bundle、导出图表数据
│   └── cli.py                    # encode / search / analyze / evaluate 统一入口
├── tests/                        # 待建：纯合成 fixture 单元测试 + 小规模集成 smoke
├── data/                         # 待建（不入 Git）：raw / manifests
├── artifacts/                    # 待建（不入 Git）：特征缓存、run bundles、中间图表
└── reports/                      # 待建：最终图表、英文报告源文与 PDF、5 分钟讲稿
```

### 模块契约（核心不变量）

| 模块 | 输入 → 输出 | 关键不变量 |
|---|---|---|
| `data.py` | raw annotations → split manifests | image ID 唯一；每图全部描述同 split；记录缺失/重复/hash |
| `model.py` | images/texts → `[B, D]` embeddings | 权重冻结（Phase 1）；有限值；L2 范数 ≈ 1；记录 token 截断率 |
| `cache.py` | embeddings + IDs + metadata → `.npz/.json` | 任一模型/预处理/文本策略/split hash 变化即失效 |
| `features.py` | `[N, D]` + `[M, D]` → transformed / stats | 拟合参数（均值/白化矩阵/PCA/K-Means）**仅在 dev/train 拟合**，再应用到 test |
| `retrieval.py` | query/cand embeddings → ranked IDs/scores | 确定性平分规则（稳定 ID 升序）；分块与整矩阵完全一致 |
| `rerank.py` | similarity matrix + CV folds → tuned reranked scores | 5-fold CV 按 image ID 分组；所有超参在测试前冻结 |
| `metrics.py` | rankings + relevance / 4-tuple → metrics + CI | 纯函数、零外部依赖，可独立由合成 fixture 验证 |
| `evaluate.py` | config + manifest + cache → run bundle | 每次运行保存完整 provenance、metrics、ranks 与日志 |

---

## 3. 七天冲刺排期（2026-11-24 — 2026-12-01）

| 日期 | 任务 ID | 核心工作 | 退出证据（Gate） |
|---|---|---|---|
| **Day 1 (11/24)** | T1, T2 | 敲定数据（Flickr30k 已有/当天可得则用之，否则直接锁定 COCO val2017 官方直下）；锁定最小 Python 环境与 `requirements.txt`；100-image smoke 测试通 CLIP & SigLIP 加载与缓存 | 数据来源/条款/split hash 记录；环境锁；100 图 smoke 时长与内存实测 |
| **Day 2 (11/25)** | T3 | 实现 `data.py / retrieval.py / metrics.py / rerank.py / features.py` 核心纯函数与全套合成单元测试 | pytest 全通过（覆盖多正例、轴交换、平分、泄漏、Winoground 24 排列、CSLS/Dual-Softmax/PCA 正确性） |
| **Day 3 (11/26)** | T4 | 在 dev 集跑完 3 个模型（CLIP ViT-B/32, RN50, SigLIP）× raw/prefix 基线编码缓存 + 数据探索可视化 + Modality Gap 白化/PCA/K-Means 分析 + 5-fold CV 调参（CSLS / Dual-Softmax / Whitening） | dev 完整结果表、CV 选参记录、初步图表、冻结 Phase 1 配置 |
| **Day 4 (11/27)** | T5 | **Phase 2 决策与冲刺**：若 Phase 1 稳定且算力允许，实现并运行 NegCLIP-lite 组合性 Hard-Negative 轻量适应（仅在 dev 选参）；否则明确 `deferred` | 训练日志/dev 对比结果（或明确延期决策）；**冻结全部实验配置** |
| **Day 5 (11/28)** | T6 | 在冻结配置下一次性跑完 **held-out test** 与 **Winoground**（若拿到访问权）；计算 paired bootstrap 95% CI；按固定 seed=5489 抽样失败案例并生成全部论文级图表 | 完整 run bundles（含 commit/hash/metrics/ranks）、所有最终图表可溯源 |
| **Day 6 (11/29)** | T7 | 撰写英文最终报告 PDF（覆盖对比损失原理、方法对比、CV 实验、降维聚类、失败分析与局限）+ 整理 Code ZIP 复现文档与干净仓库 | 报告 PDF 排版核查；对照 Rubric 100 分逐项自检；数字与 run bundle 一致 |
| **Day 7 (11/30–12/01)** | T8 | 录制 ≤5 分钟 Panopto 演示视频（标题 `Group X: Y`）；独立复现跑通 CLI/评估；打包 `ProjectGroup_X_Code.zip`；提交 3 个 Canvas 入口（含组员贡献声明） | 3 个 Canvas 提交全部确认；ZIP 不含大数据集且按 README 可复现 |

---

## 4. 执行、审查与停止规则

1. **状态流转**：所有任务状态仅在 [PROGRESS.md](PROGRESS.md) 维护（`todo → doing → review → done`，可选 `blocked / deferred`）。代码写完不等于 `done`，必须附验证命令、退出码与产物路径。
2. **防泄漏铁律**：所有预处理统计量（白化均值/协方差、PCA 投影、K-Means 中心）、重排超参（CSLS $k$、Dual-Softmax $\tau$）与可选微调模型选择**一律只能在 dev/train 上完成**；held-out test 与 Winoground 仅在配置冻结后运行一次。测试后产生的新改动必须标为探索性。
3. **停止规则**：时间紧迫时优先保证 **Phase 1 全部 Rubric 项高质量闭环 + 报告图表 + 视频 + 可复现代码包**，绝不因过度调参拖延最终报告和提交。负结果或无提升均如实报告并分析原因。
4. **独立核查**：关键评估代码（`metrics.py`、`rerank.py`、`features.py`）和最终报告数字优先交由另一 agent 或组员只读核查；条件受限时标注自审边界。
