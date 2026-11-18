# 基于预训练视觉语言模型的自然语言图像检索

[English](README.md) · CS5489 Machine Learning · 2026-10-20 审核

## 当前状态

**已完成本轮独立审核与文档规划，尚未进入实现和实验阶段。** 未验证数据获取、项目环境安装、模型运行或任何实验指标。原始要求见 [1.md](1.md)。

项目将使用预训练 CLIP 实现文本到图像、图像到文本检索，解释对比预训练，并分析组合性匹配的局限。不从零训练基础模型；微调只在基线验收、训练数据和计算预算明确后考虑。

## 已确定的设计

- 冻结 OpenAI CLIP ViT-B/32，使用配套预处理与 tokenizer。
- 有条件优先使用 Flickr30k 的公开检索划分；获取受阻时使用 COCO val2017 的显式自定义开发/评估划分，不能冒称 Karpathy test。
- L2 归一化、精确余弦排序、缓存特征；双向分别报告 Recall@1/5/10。
- 比较原始描述与固定 `a photo of ` 前缀；依据是 CLIP 论文附录 E.1，不预设会提升。
- Winoground 仅作最终组合性诊断，报告 text/image/group 三项分数；须先取得合规访问。
- 一个受预算约束的适应实验作为可选扩展，不是基线完成条件。

“Zero-shot”只表示本项目不对任务数据更新模型权重，不意味着能够排除预训练数据重叠。图像到文本检索必须接受该图像的全部标注描述作为正例。

## 文档入口

- [英文 Proposal](PROJECT_PROPOSAL.md) / [正式排版 PDF](output/pdf/Project_Proposal.pdf)
- [独立审核](docs/REVIEW.md)：要求追踪、问题分级、依据与未知项
- [项目计划](docs/PROJECT_PLAN.md)：拟建目录、接口、阶段、执行与审查机制
- [评估协议](docs/EVALUATION.md)：数据获取、划分、指标和实验隔离
- [进度台账](docs/PROGRESS.md)：唯一任务状态源；[检查清单](CHECKLIST.zh-CN.md)只作导航
- [Agent 工作约定](AGENTS.md)

计划中的代码目录尚未创建。`requirements.txt` 是历史候选清单，未锁版本、未验证，不可视为可复现安装方案。当前不展示尚未实现的命令。

## 论文

- CLIP：[原文](https://arxiv.org/abs/2103.00020) / [本地 PDF](paper/Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision.pdf)
- Winoground：[原文](https://arxiv.org/abs/2204.03162) / [本地 PDF](paper/Winoground.pdf)

论文事实、项目设计与运行结果分别记录；完整外部依据见评估协议。
