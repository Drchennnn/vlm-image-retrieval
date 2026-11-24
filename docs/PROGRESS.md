# 项目进度与决策台账

更新日期：2026-11-24。本文件是任务状态、决策与验收证据的唯一来源。正式截止时间：**2026-12-01 23:59 HKT**。

## 任务清单

| ID | 任务 | 状态 | 依赖 | 执行者 / 审查者 | 验收证据或下一步 |
|---|---|---|---|---|---|
| T0 | 初始审核、英文 Proposal、文档架构精简与 v2 协议升级 | done | 原始题目 + Canvas Rubric | Codex（10/20）/ Antigravity（11/24） | 旧文档归档至 `docs/archive/`；合并 `REQUIREMENTS.md`、`PLAN.md`、`EVALUATION.md` v2；链接与 git 状态核查通过 |
| T1 | 数据获取/条款确认、split manifest 校验与环境锁定 | done | T0 | Antigravity / 自审（含合成单测） | 官方 COCO val2017（5,000 图，0 缺失/损坏/重复内容）切分为 `3,000 train (15,007 caps) / 1,000 dev (5,005 caps) / 1,000 test (5,002 caps)`；ID 清单入库 `configs/splits/coco2017_split_ids.json`；依赖锁定至 `requirements.txt`（78 包） |
| T2 | 模型加载（CLIP ViT-B/32, RN50, SigLIP）与 100 图 smoke | done | T1 | Antigravity / 自审（含缓存往返校验） | 3 模型全部在 M1 Pro MPS 加载并完成 100 图/501 文本 smoke；`cache_roundtrip_exact=true`；单模型编码 1.1–2.4s，峰值 RSS 2060.5 MB；`pytest` 4/4 通过 |
| T3 | 核心检索、计分、新特征、重排与合成单元测试 | done | T0（可与 T1/T2 并行） | Antigravity / 自审（14 项合成单测 + CLI 端到端 smoke） | 完成 `retrieval / metrics / features / rerank / evaluate / cli` 模块；`pytest` 14/14 通过（含多正例、轴交换、平分、Winoground 24 排列=4/24、Hubness 救援、5-fold image-grouped 无泄漏、特征仅 dev 拟合）；CLI `evaluate` 与双向 `search` smoke 通过 |
| T4 | dev 集完整基线（E0–E2）、数据/降维聚类分析、新特征（E3）与 5-fold CV 重排（E4） | done | T1–T3 | Antigravity / 自审（含可视化图复核与 3 路降维对照） | 在 1,000 图 / 5,005 描述 dev 全库完成 3 模型 × 2 策略基线、5-fold CV 超参调优、Modality Gap/Hubness/PCA/K-Means 诊断；冻结配置 `configs/frozen_phase1.json`，汇总 `reports/dev_phase1_summary.json` 及 4 张图；`pytest` 14/14 通过 |
| T5 | Phase 2 可选冲刺：NegCLIP-lite 组合性硬负例轻量适应（E5） | done | T4（Phase 1 稳定 + 独立 train 划分 + ≤2h 预算） | Antigravity / 自审（3 项新增单测 + 3 组受控消融） | 独立 3K `train` 集完成 3 组受控配置 × 3 epochs（总耗时 415.9s，本地 M1 Pro MPS，仅训 4.51% 参数）；`dev` 选定 `cfg1_text_adapter_lr1e5_hn0.5`（ep3）：dev cosine Mean R@1 `60.64% → 62.74% (+2.11%)`，Dual-Softmax `67.90% → 68.55%`，dev 组合性硬负例准确率 `65.19% → 88.27% (+23.08%)`；全配置已冻结至 `configs/frozen_all.json` |
| T6 | 最终 held-out test、Winoground 诊断、bootstrap CI 与分层失败分析 | todo | T4 done 且 T5 为 done/deferred | 待认领 / 待分配 | 完整 run bundles、95% CI、全部最终图表（含缺失访问显式说明） |
| T7 | 英文最终报告 PDF 与可复现代码清理 | todo | T6 | 待认领 / 待分配 | 对照 100 分 Rubric 逐项核对；全部图表与表格可溯源至 run ID |
| T8 | ≤5 分钟 Panopto 演示视频、Code ZIP 打包与 Canvas 三项提交 | todo | T7 | 待认领 / 待分配 | 独立复现验证、`ProjectGroup_X_Code.zip`、视频嵌入、组员贡献声明 |

状态枚举：`todo / doing / blocked / review / done / deferred`。

---

## 决策记录

| ID / 日期 | 决策与原因 | 重开条件 |
|---|---|---|
| D01 / 2026-10-20 | 双向冻结预训练 VLM 检索为必做；不从零训练基础模型；微调可选 | 课程要求变更 |
| D02 / 2026-10-20 | Winoground 严格 evaluation-only，不参与训练、选参或提示选择；访问受限时如实交代 | 新研究设计且不复用作独立测试 |
| D03 / 2026-10-20 | 英文提交材料与主 README；组员署名 Benyi Chen / Muyao Yan / Jianheng Qiao | 用户确认官方拼写、组号或 EID |
| D04 / 2026-11-24 | 对齐 Canvas 100 分 Rubric 与 7 天截止日（12/01）：Phase 1 升级为“多骨干/损失对比（CLIP ViT-B/32, RN50, SigLIP）+ Modality Gap 白化/PCA/K-Means/Hubness 分析 + CSLS/Dual-Softmax 5-fold CV 重排”；Phase 2 保留 NegCLIP-lite 硬负例轻量适应为冲刺项 | Day 1–3 设备实测显存/时长受限或用户调整范围 |
| D05 / 2026-11-24 | 数据务实切换规则：若 Day 1 内无法立即获得 Flickr30k 授权与官方 split，直接启用官方 COCO val2017 自定义划分并显式标注，不等待审批阻塞进度 | Flickr30k 当日获批且校验通过 |
| D06 / 2026-11-24 | 精简仓库文档架构：删除冗余 `CHECKLIST*.md` 与 `README.zh-CN.md`，归档历史 Proposal/Review/1.md 至 `docs/archive/`，统一核心四文档（`REQUIREMENTS / PLAN / EVALUATION / PROGRESS`）；保留 `.agents / .specify / openspec / .omx` 工具目录不动；`.gitignore` 改为标准黑名单 | 后续提交 ZIP 打包前按需微调 |

---

## 执行日志

### 2026-10-20 — 初始审核与 Proposal 交付（P0）
- 完成原始题目、两篇论文与外部官方来源核查；产出 3 页英文 Proposal PDF（[Project_Proposal.pdf](archive/Project_Proposal.pdf)）与 v1 评估协议；独立审查通过。本轮无代码实现或模型实验。

### 2026-11-24 — Rubric 对齐、创新方案设计与文档重构（T0）
- 发现 Canvas 正式截止日为 **2026-12-01 23:59 HKT**，且 100 分 Rubric 重奖多方法、交叉验证、特征对比、降维聚类及新方法/新特征。
- 精简重构文档架构：归档 `1.md`、`PROJECT_PROPOSAL.md`、`Project_Proposal.pdf`、`REVIEW.md` 至 `docs/archive/`；删除冗余 `CHECKLIST*.md`、`README.zh-CN.md`；合并产出 [REQUIREMENTS.md](REQUIREMENTS.md)、7 天冲刺版 [PLAN.md](PLAN.md) 与 [EVALUATION.md](EVALUATION.md) v2；更新 [README.md](../README.md)、[AGENTS.md](../AGENTS.md) 与 `.gitignore`；按要求完整保留 4 个本地工具目录。
- 当前仍无代码实现、数据下载或实验指标。下一步：进入 **T1 + T2**（Day 1 确认数据集、锁定环境与 100-image smoke）。

### 2026-11-24 — Round 1：COCO val2017 切分校验、环境锁定与 3 模型 100 图 Smoke（T1 + T2）
- **数据策略执行（D05）**：未持有 Flickr30k 授权，直接下载官方公开 COCO `val2017`（5,000 图 + `captions_val2017.json`，`sha256=afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9`）。经 PIL 逐图解码与 SHA256 去重核查：0 缺失、0 损坏、0 内容重复。按 `PCG64(5489)`（NumPy `2.5.5`）切分为严格互斥的 `train`（3,000 图 / 15,007 描述）、`dev`（1,000 图 / 5,005 描述）、`test`（1,000 图 / 5,002 描述）与 `smoke_dev100`（100 图 / 501 描述），显式 ID 清单已追踪保存至 `configs/splits/coco2017_split_ids.json`。
- **环境与模型权重锁定**：使用 `uv` 建立 `.venv`（Python 3.12.16、`torch==2.15.0` MPS 可用、`open-clip-torch==4.2.2`、`transformers==5.24.0`、`scikit-learn==1.9.1`、`umap-learn==0.6.0`），冻结至 `requirements.txt`。3 个权重文件实测 SHA256：
  - `CLIP ViT-B/32 (openai, QuickGELU)`：`e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31`（512 维，ctx=77）
  - `CLIP RN50 (openai, QuickGELU)`：`da0baa37fb2211eee5729ab69ed587eb10d48f5b7076ceeed01552fc3d4cb4ee`（1024 维，ctx=77）
  - `SigLIP ViT-B/16 (webli)`：`81942ea8c09b9e41963357cca5d2682118bf7eb7491c2ae51a28bc4f4f95b194`（768 维，ctx=64）
- **实现与验证**（退出码均为 0）：
  - 新增 `pyproject.toml`、`configs/models.yaml`、`src/vlm_retrieval/{data,model,cache}.py`、`scripts/{prepare_coco,smoke_round1}.py`、`tests/test_data_and_cache.py`。
  - 修复 smoke 发现的两处细节：缓存加载不再二次重归一化（保证 float32 bit-exact 往返一致）；OpenAI CLIP 启用 `force_quick_gelu=True` 消除激活函数不匹配警告。
  - `.venv/bin/python scripts/smoke_round1.py`（产物 `artifacts/smoke/round1_smoke_report.json`）：3 模型 × 2 文本策略全部 `cache_roundtrip_exact=true`，截断率均为 0.0%，单模型 100 图编码耗时 1.13s–2.40s，峰值 RSS 2060.5 MB；100 图 smoke 参考 R@1（仅作健全性检查，非正式评测）：CLIP ViT-B/32 raw 79.84/93.00%、RN50 raw 77.84/91.00%、SigLIP raw 91.02/96.00%（T2I/I2T）。
  - `.venv/bin/pytest -v`：4 passed in 3.95s。
- **下一步**：Round 2（T3）实现核心检索、精确计分、Winoground 计分、特征变换/降维聚类、CSLS/Dual-Softmax 5-fold CV 与完整合成单元测试。

### 2026-11-24 — Round 2：核心检索/计分/新特征/重排/CLI 模块与合成单元测试（T3）
- **新增实现**：
  - `src/vlm_retrieval/retrieval.py`：精确 float32 余弦相似度（整矩阵/分块一致）、确定性平分排序（分数降序 + 稳定 ID 升序）。
  - `src/vlm_retrieval/metrics.py`：纯函数双向 `R@1/5/10`、`MedR`、`MeanR`、`MRR`（多正例支持）；Winoground 严格不等式计分与标签拆解；按 `image_id` 分组的 paired bootstrap 95% CI；固定 seed 分层成功/失败抽样。
  - `src/vlm_retrieval/features.py`：仅拟合 dev/train 的 `ModalityAligner`（`center`、`zca_per_modality`、`zca_shared`）、`CrossModalPCA`、Modality Gap 几何统计、Hubness（$k$-occurrence 偏度/Gini/Robin Hood/orphan 率）与 K-Means 语义聚类分析。
  - `src/vlm_retrieval/rerank.py`：`CSLS` 与 `Dual-Softmax`（含数值稳定的 `symmetric_log` 与 `prior_weighted` 两种变体）重排，以及按 `image_id` 分组的 5-fold 交叉验证调参器（`cross_validate_reranker`、`cross_validate_modality_aligner`）。
  - `src/vlm_retrieval/evaluate.py` & `cli.py`：标准化 run-bundle 写出（`config.json / provenance.json / metrics.json / ranks.jsonl / run.log`）与 `encode / evaluate / search` CLI 入口。
- **验证**（退出码均为 0）：
  - `.venv/bin/pytest -v`：**14 passed in 6.92s**（覆盖 `test_data_and_cache.py`、`test_retrieval_and_metrics.py`、`test_features_and_rerank.py`、`test_evaluate_and_cli.py`，含 Winoground 24 排列严格 25%/25%/16.67% 校验、合成 Hubness 注入下 CSLS 与 Dual-Softmax 从 16.67% 恢复至 100% R@1、防 test 泄漏断言）。
  - CLI smoke：`evaluate`（CSLS on `smoke_dev100` 生成完整 run bundle `artifacts/smoke_runs/smoke_cli_check`）与双向 `search`（`--image-id` I2T 与 `--text` T2I）均通过。
- **下一步**：Round 3（T4）在完整 1,000 图 `dev` 集（5,005 条描述）上运行 Phase 1 全量基线、数据/降维聚类分析、新特征与 5-fold CV 超参选择。

### 2026-11-24 — Round 3：Phase 1 完整 1,000 图 Dev 实验、5-Fold CV 与表征诊断（T4）
- **运行脚本**：`.venv/bin/python scripts/run_round3_dev.py`（严格仅在 `dev` 划分 1,000 图 / 5,005 描述执行，未触碰 `test`）。
- **核心 dev 全库结果（Mean R@1 = 0.5×(T2I R@1 + I2T R@1)）**：
  - `CLIP ViT-B/32`（InfoNCE / ViT）：`raw` 60.25%（T2I 50.11 / I2T 70.40）→ `photo_prefix` 60.64% → `CSLS(k=5)` **65.86% (+5.22%)** → `Dual-Softmax(τ=0.02, prior_weighted)` **67.90% (+7.26%)**。
  - `CLIP RN50`（InfoNCE / CNN）：`raw` 57.58%（46.65 / 68.50）→ `photo_prefix` 59.02% → `CSLS(k=5)` **65.93% (+6.91%)** → `Dual-Softmax(τ=0.02, prior_weighted)` **68.16% (+9.14%)**。
  - `SigLIP ViT-B/16`（Sigmoid / ViT）：`raw` 74.53%（66.75 / 82.30）→ `photo_prefix` 75.59%（截断 1/5,005 条）→ `CSLS(k=10)` **78.62% (+3.03%)** → `Dual-Softmax(τ=0.02, prior_weighted)` **79.99% (+4.40%)**。
- **关键科学发现（机理洞见）**：
  1. **Modality Gap 与方向不对称效应**：原始 `CLIP ViT-B/32` 图文质心 L2 距离高达 `0.812`（`SigLIP` 达 `1.047`）；`ModalityAligner` 将质心距离压缩至 `0.022` 后，**CLIP 的 T2I R@1 提升**（ViT-B/32: 49.47→51.29%；RN50: 47.33→49.45%），但 **I2T R@1 下降**（71.80→65.70%），导致双向均值略降，解释了全局去中心重归一化对一对多描述结构的扰动。
  2. **Hubness 显著抑制**：`CLIP ViT-B/32` T2I Top-10 候选最大出现次数由 `271`（期望 50.05，偏度 `1.59`，Gini `0.391`）经 `Dual-Softmax` 降至 `109`（偏度 `-0.15`，Gini `0.202`）。
  3. **跨模态降维几何发现（5-fold CV 3 路对比）**：朴素拼接 `Naive PCA` 因 5:1 文本占比与 Modality Gap 偏移原点，256 维时 I2T R@1 暴跌至 50.6%；平衡 `Uncentered SVD` 在 256 维保持几乎无损（Mean R@1 79.58% vs 80.24%）；而极低维（64d / 32d）时，`Modality-Centered PCA` 先消除模态偏移，反超 `Uncentered SVD` **+5.8%（64d）至 +15.4%（32d）**！
  4. **K-Means 语义聚类难度差异**：`K=8`（Silhouette=0.125）时，运动/球场类场景（Cluster 2，如棒球/网球）I2T R@1 仅 `49.2%`，而食物/室内静物类（Cluster 7/1）达 `81.2%–84.5%`。
- **产物与验证**：`configs/frozen_phase1.json`、`reports/dev_phase1_summary.json`、`reports/figures/dev_*.png`（4 张图已人工目视核查）、`artifacts/runs/dev/`（6 个基线 run bundles）；`pytest` **14/14 passed**。
- **下一步**：Round 4（T5）执行 Phase 2 轻量适应（NegCLIP-lite：在独立 3K `train` 集构造组合性硬负例，冻结图像主干、微调文本端，在 `dev` 集选参并冻结最终配置）。

### 2026-11-24 — Round 4：Phase 2 NegCLIP-lite 组合性硬负例轻量适应与全配置冻结（T5）
- **新增实现与验证**：
  - `src/vlm_retrieval/adapt.py`：4 算子组合性硬负例生成器（`relational_swap`、`np_swap`、`attribute_swap`、`content_shuffle`）、零初始化 `ResidualImageAdapter`、对称 InfoNCE + 硬负例成对对比损失 `negclip_loss`、仅解冻 CLIP 文本编码器顶层 2 层 Transformer + `ln_final` + `text_projection`。
  - `tests/test_adapt.py` + 全套 `pytest`：**17/17 passed in 7.64s**。
- **训练协议与资源实测**（`scripts/run_round4_phase2.py`，严格仅用 3,000 图 `train`、每步每图仅采 1 条正描述防假负例、零触碰 `test` 与 Winoground）：
  - 一次性预缓存 3,000 张 `train` 图像特征耗时 `31.3s`；3 组受控配置 × 3 epochs 在 M1 Pro MPS 上每 epoch 仅 `19.3–22.4s`，全流程总耗时 **`415.9s`（~6.9 分钟）**，峰值 RSS `2,147.0 MB`（仅训练 4.51% / 6.83M 参数）。
- **Dev 消融与选参结果（相对 Frozen CLIP ViT-B/32 `photo_prefix`：dev_cos 60.64%、dev_DS 67.90%、dev_comp 65.19%）**：
  - **`cfg1_text_adapter_lr1e5_hn0.5`（ep3，选中冻结）**：`dev_cos = 62.74% (+2.11%)`（T2I `49.47→53.49%`、I2T `71.80→72.00%` 双向上涨），`dev_CSLS = 67.82%`，`dev_DS = 68.55%`，`dev_comp = 88.27% (+23.08%)`。
  - `cfg2_text_adapter_lr3e5_hn1.0`（ep2/3）：强硬负例权重把 `dev_comp` 推至 `92.63%`、T2I 推至 `54.55%`、`dev_DS` 推至 `69.90%`，但导致原始余弦 I2T 回落至 `67.70%`（清晰揭示检索—组合性权衡，而 Dual-Softmax 有效抵消了文本端偏移）。
  - `cfg3_text_only_lr2e5_hn1.0`（ep2，去 Adapter 消融）：`dev_cos = 61.50%`，`dev_comp = 91.57%`，`dev_DS = 69.43%`。
- **冻结与产物**：权重保存至 `artifacts/checkpoints/negclip_lite_best.pt`（不入库）；全实验配置冻结至 `configs/frozen_all.json`；结果汇总 `reports/dev_phase2_summary.json`；训练曲线图 `reports/figures/dev_phase2_negclip_lite.png`。
- **下一步**：Round 5（T6）在冻结配置 `configs/frozen_all.json` 下一次性运行 **held-out test（1,000 图 / 5,002 描述）** 与 **Winoground**（如已授权），计算 paired bootstrap 95% CI、分层失败案例与全部终稿图表。

---

## 后续日志模板

```text
日期 / Task ID / 执行者 / 审查者：
状态变化与实际改动：
验证命令、退出码与结果摘要：
run ID / git commit 或 dirty diff hash / 产物路径：
阻塞与下一步：
```
