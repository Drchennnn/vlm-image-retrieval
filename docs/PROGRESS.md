# 项目进度与决策台账

更新日期：2026-11-24。本文件是任务状态、决策与验收证据的唯一来源。正式截止时间：**2026-12-01 23:59 HKT**。

## 任务清单

| ID | 任务 | 状态 | 依赖 | 执行者 / 审查者 | 验收证据或下一步 |
|---|---|---|---|---|---|
| T0 | 初始审核、英文 Proposal、文档架构精简与 v2 协议升级 | done | 原始题目 + Canvas Rubric | Codex（10/20）/ Antigravity（11/24） | 旧文档归档至 `docs/archive/`；合并 `REQUIREMENTS.md`、`PLAN.md`、`EVALUATION.md` v2；链接与 git 状态核查通过 |
| T1 | 数据获取/条款确认、split manifest 校验与环境锁定 | done | T0 | Antigravity / 自审（含合成单测） | 官方 COCO val2017（5,000 图，0 缺失/损坏/重复内容）切分为 `3,000 train (15,007 caps) / 1,000 dev (5,005 caps) / 1,000 test (5,002 caps)`；ID 清单入库 `configs/splits/coco2017_split_ids.json`；依赖锁定至 `requirements.txt`（78 包） |
| T2 | 模型加载（CLIP ViT-B/32, RN50, SigLIP）与 100 图 smoke | done | T1 | Antigravity / 自审（含缓存往返校验） | 3 模型全部在 M1 Pro MPS 加载并完成 100 图/501 文本 smoke；`cache_roundtrip_exact=true`；单模型编码 1.1–2.4s，峰值 RSS 2060.5 MB；`pytest` 4/4 通过 |
| T3 | 核心检索、计分、新特征、重排与合成单元测试 | todo | T0（可与 T1/T2 并行） | 待认领 / 待分配 | `pytest` 全覆盖 EVALUATION §5 单元测试清单 |
| T4 | dev 集完整基线（E0–E2）、数据/降维聚类分析、新特征（E3）与 5-fold CV 重排（E4） | todo | T1–T3 | 待认领 / 待分配 | dev 对比表、Modality Gap/PCA/K-Means/Hubness 图、CV 选参记录；冻结 Phase 1 配置 |
| T5 | Phase 2 可选冲刺：NegCLIP-lite 组合性硬负例轻量适应（E5） | deferred | T4（Phase 1 稳定 + 独立 train 划分 + ≤2h 预算） | 待认领 / 待分配 | Day 4 决策是否激活；若激活，仅在 dev 选参并冻结后进入 T6 |
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

---

## 后续日志模板

```text
日期 / Task ID / 执行者 / 审查者：
状态变化与实际改动：
验证命令、退出码与结果摘要：
run ID / git commit 或 dirty diff hash / 产物路径：
阻塞与下一步：
```
