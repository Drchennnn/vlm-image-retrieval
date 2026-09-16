[English](README.md)

# 基于预训练视觉语言模型的自然语言图像检索

## 项目状态

本项目目前处于规划阶段。

目标是基于预训练视觉语言模型，建立可复现的双向图文检索基线：

- 文本到图像检索；
- 图像到文本检索。

项目首先实现可信的基线，并完成验证；只有在基线结果明确后，才决定是否进行微调。

## 方法概述

计划中的基线将使用 CLIP 或兼容的预训练模型。图像编码器和文本编码器把两种模态映射到共享的嵌入空间。归一化后的嵌入可以使用余弦相似度或等价的点积进行比较。

CLIP 风格的对比预训练把匹配的图文对作为正样本，把同一批次中的其他组合视为负样本。对称式目标提高匹配图文对的相似度，并降低不匹配组合的相似度。

以上内容是对参考论文的简要概述，不代表本项目已经训练模型、下载数据或复现原始训练设置。

## 评估计划

项目计划评估：

- 文本到图像检索；
- 图像到文本检索；
- 作为组合性压力测试的 Winoground。

数据集、检查点、预处理、指标、索引方法和计算环境尚未确定。候选检索指标包括 Recall@K、中位排名、平均排名和平均倒数排名。

除非未来实验明确记录其他用途，Winoground 计划仅用于评估。

## 建议的项目结构

首次实现可以采用以下布局：

```text
.
├── README.md
├── README.zh-CN.md
├── CHECKLIST.md
├── requirements.txt
├── data/
├── src/
│   ├── data/
│   ├── models/
│   ├── retrieval/
│   └── evaluation/
├── scripts/
└── tests/
```

该布局仍是临时建议。只有在成为已批准任务的一部分后，才创建对应目录。

## 当前证据

- 项目根目录已有参考论文。
- 尚未验证项目实现。
- 尚未验证数据集、模型运行、实验结果或指标。
- `.agents`、`.specify`、`openspec` 和 `.omx` 是本地工具目录，不属于项目上传集。
- 候选依赖列在 `requirements.txt` 中，目前尚未安装。
- 计划创建名为 `vlm` 的虚拟环境，但当前本地尚未创建。

待完成步骤和可能的后续方向见 [CHECKLIST.zh-CN.md](CHECKLIST.zh-CN.md)。

## 参考文献

- [Learning Transferable Visual Models From Natural Language Supervision](./Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision.pdf) · [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- [Winoground](./Winoground.pdf) · [arXiv:2204.03162](https://arxiv.org/abs/2204.03162)
