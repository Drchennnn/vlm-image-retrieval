# CS5489 项目要求与评分标准

适用学期：2026/27 Semester A。合并自原始选题公告（留档见 [archive/1.md](archive/1.md)）与 2026-11-24 核对的 CityU Canvas 作业页、项目公告和课程大纲。

## 1. 原始选题要求

> ***[Multi-Modal]*** **Image Retrieval with Natural Language**
>
> Build a system that searches for images based on natural language descriptions. Your main task is to utilize a pre-trained [vision-language common representation (feature) space](https://arxiv.org/pdf/2103.00020) for cross-modal retrieval, i.e., using language to retrieve image or use image to retrieve language. Note, you are not asked to train a foundation model from scratch by yourself. Instead, you should analyze the concept of contrastive pre-training and explain how it works. Then, you can choose to finetune your model on some specific or challenging datasets such as [winoground](https://arxiv.org/pdf/2204.03162) to improve the performance.

核心约束：
- 利用预训练视觉语言共享表征空间，实现**文本检图（T2I）**与**图像检文（I2T）**双向检索。
- **不从零训练基础模型**；必须深入解释**对比预训练（contrastive pre-training）**原理。
- 微调为可选项（如面向 Winoground 等挑战基准的组合性改进）。

---

## 2. 截止时间与 Canvas 提交入口

三项作业正式截止时间均为 **2026 年 12 月 1 日（周二）23:59（香港时间）**（Canvas 另列 12 月 6 日 23:59 为入口关闭时间，以 12 月 1 日为准；未按指示提交可能延迟评分或扣分）。

| Canvas 作业 | 提交物 | 页面满分 |
|---|---|---:|
| [Course Project (2026/27A)](https://canvas.cityu.edu.hk/courses/70596/assignments/325686) | 一份英文最终报告 PDF（无指定模板） | 100 |
| [Course Project Presentations (Code + Video)](https://canvas.cityu.edu.hk/courses/70596/assignments/325688) | 代码 ZIP + Canvas 文本框嵌入 Panopto 视频 | 15 |
| [Project Group Contribution](https://canvas.cityu.edu.hk/courses/70596/assignments/325692) | 文本框填每位组员 `Name, EID, percentage`（总计 100%） | 0 |

Project 占课程总评 **30%**。报告 100 分 rubric 已含演示 15 分与代码 5 分；演示入口亦单独显示 15 分。组员贡献低于平均时会按比例扣分（Canvas 两人示例：贡献 `A% > B%` 时，贡献少者得分 `x × B / A`）。

### 代码 ZIP 与 Panopto 视频规范

- **代码 ZIP**：命名 `ProjectGroup_[YourGroupNumber]_Code.zip`；必须附 `README.md`（或 `README.txt`），写明数据下载准确 URL / 公共仓库名、预处理命令或脚本、数据放置目录结构；**不要把大型公开数据集打进 ZIP**。
- **Panopto 视频**：最长 **5 分钟**；保存至 `2026/27 Semester A - CS5489 Machine Learning:Algorit &Apns [assignments]`；标题格式 `Group X: Y`（视频口播亦须报组号）；在 Canvas 文本框点击 **Apps（插头图标）→ Panopto Recordings** 嵌入。
- **截止后互评**：每位同学须观看至少 **3 个**其他组视频并留言评论或提问。

---

## 3. Course Project Rubric（100 分）与本项目对应策略

教学团队明确指出：**更看重方法分析与实验比较，单纯刷高指标不是评分重点。**

| 评分项 | 满分 | 阈值 | 满档要求 | 本项目覆盖策略（详见 [PLAN.md](PLAN.md) / [EVALUATION.md](EVALUATION.md)） |
|---|---:|---:|---|---|
| Results — Methods | 15 | 10 | 充分探索多种方法 | 冻结基线 + 前缀对照 + 多对比损失/骨干 + CSLS / Dual-Softmax 重排（+ 可选 Hard-Neg Adapter） |
| Results — Experiment Setup | 10 | 7 | 训练集/验证集 + 交叉验证调参 | 严格按 image ID 隔离 dev / test；在 dev 上做 **image-grouped 5-fold CV** 调优超参后冻结 |
| Results — Features | 10 | 5 | 充分比较特征表示 | 对比 CLIP ViT-B/32（ViT + InfoNCE）、CLIP RN50（CNN + InfoNCE）、SigLIP（Sigmoid 对比损失） |
| Results — DimRed/Clustering | 5 | 3 | 使用降维或聚类 | PCA/UMAP 可视化 Modality Gap；PCA 维度压缩曲线（512→32）；K-Means 语义簇检索难度分析 |
| Extra — Data | 5 | 3 | 充分数据探索与可视化 | Caption 长度与 77-token 截断分布、每图描述数、相似度分布与 Hubness 长尾频次可视化 |
| Extra — Method | 10 | 5 | 使用**超过一种**新方法 | 方法 A：CSLS 跨模态局部缩放；方法 B：Dual-Softmax 双向重排；Phase 2 冲刺：NegCLIP-lite 组合性微调 |
| Extra — Features | 5 | 5 | 使用新特征表示 | 针对 Modality Gap 构造 **Mean-Centering + Whitening（去中心化与白化对齐）** 跨模态特征表示 |
| Extra — Justification | 5 | 3 | 尝试新方法的理由充分 | 从 Modality Gap、高维 Hubness 与词序组合性盲区三个机理出发，逐项给出假设与实证检验 |
| Report — Descriptions | 9 | 7 | 深入说明做了什么及原因 | 完整推导 InfoNCE vs Sigmoid 对比损失、CSLS/Dual-Softmax 公式、协议设计与每项决策依据 |
| Report — Discussion | 3 | 3 | 讨论有洞见 | Paired bootstrap 置信区间、分层失败案例、Winoground 对象/关系错误拆解、局限性与负结果分析 |
| Report — Figures | 3 | 2 | 图表清晰有效 | 架构图、Modality Gap 前后投影图、PCA 压缩折线图、Hubness 分布图、语义簇误差图、定性案例图 |
| Presentation | 15 | 10 | 5 分钟讲解清晰有洞见 | 紧扣“现象 → 机理 → 方法 → 实证”讲清楚核心发现，不念流水账 |
| Code | 5 | — | 代码结构清晰可复现 | 单包 `src/vlm_retrieval/` + 合成单元测试 + 锁定依赖 + 完整运行文档 |
| **合计** | **100** | | | 组员署名：Benyi Chen、Muyao Yan、Jianheng Qiao（最终提交时补齐组号与 EID） |
