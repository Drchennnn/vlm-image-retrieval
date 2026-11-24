# 项目进度与决策台账

更新日期：2026-11-24。本文件是任务状态、决策与验收证据的唯一来源。正式截止时间：**2026-12-01 23:59 HKT**。

## 任务清单

| ID | 任务 | 状态 | 依赖 | 执行者 / 审查者 | 验收证据或下一步 |
|---|---|---|---|---|---|
| T0 | 初始审核、英文 Proposal、文档架构精简与 v2 协议升级 | done | 原始题目 + Canvas Rubric | Codex（10/20）/ Antigravity（11/24） | 旧文档归档至 `docs/archive/`；合并 `REQUIREMENTS.md`、`PLAN.md`、`EVALUATION.md` v2；链接与 git 状态核查通过 |
| T1 | 数据获取/条款确认、split manifest 校验与环境锁定 | todo | T0 | 待认领 / 待分配 | Day 1 确认 Flickr30k 或直切 COCO val2017；保存 manifest + SHA256；锁定 `requirements.txt` |
| T2 | 模型加载（CLIP ViT-B/32, RN50, SigLIP）与 100 图 smoke | todo | T1 | 待认领 / 待分配 | 权重校验、截断统计、缓存往返一致、设备耗时与内存实测 |
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

---

## 后续日志模板

```text
日期 / Task ID / 执行者 / 审查者：
状态变化与实际改动：
验证命令、退出码与结果摘要：
run ID / git commit 或 dirty diff hash / 产物路径：
阻塞与下一步：
```
