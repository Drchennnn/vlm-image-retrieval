# 评估协议 v2

更新日期：2026-11-24（由 v1 升级，对齐 CS5489 100 分 Rubric 与 7 天冲刺架构；因尚未产生实验结果，无旧结果失效问题）。性质：待实现的实验契约，不是实验报告。任何影响已运行结果的协议改动必须升版本并标注旧产物失效。

## 1. 数据集与划分策略

| 用途 | 方案 | 访问、规模与隔离要求 |
|---|---|---|
| 主检索（优先） | Flickr30k + Karpathy/Fei-Fei retrieval split | 仅当 Day 1 内已获合法授权并验证官方 split JSON（target 1K val / 1K test）时采用；图像受非商业研究条款约束 |
| 主检索（7 天务实首选/备选） | COCO val2017 官方直下（5K 图 ~1GB + 2017 annotations ~241MB） | 明确标注自定义协议 `coco2017-custom-split-v2`，**绝不冒称 Karpathy 5K test**。按数值 image ID 升序，用固定 `np.random.Generator(np.random.PCG64(5489))` 打乱：仅跑 Phase 1 时切 **1,000 dev / 4,000 eval**（与 v1 一致）；若启用 Phase 2 轻量微调且未另下 train2017，则切 **3,000 train / 1,000 dev / 1,000 test** 并保存显式 ID 清单与 SHA256（两套划分互斥，不得混比） |
| 组合性诊断 | Winoground 官方 400 test groups | HF gated，需合法账户接受条款；**严格 evaluation-only**，不参与任何调参、特征拟合、重排选参或早停。若未获批，交付经合成验证的 scorer 并在报告中显式交代缺失 |

数据验收必须检查：唯一且稳定的 `image_id / caption_id`、文件存在与可解码、图像到全部描述的一对多映射、**跨划分按 image ID 严格互斥**、内容哈希重复检查、样本数与缺失记录。原图不打包进代码 ZIP 或公开发布。

官方来源：
- [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) · [Karpathy splits](https://cs.stanford.edu/people/karpathy/deepimagesent/)
- [COCO Downloads](https://cocodataset.org/#download) & [Terms of Use](https://cocodataset.org/#termsofuse)（annotations 为 CC BY 4.0，图片另受原权利与 Flickr 条款约束）
- [Winoground Dataset Card](https://huggingface.co/datasets/facebook/winoground)

---

## 2. 预训练模型、特征与缓存规范

### 2.1 多表征模型矩阵（Phase 1）
| ID | 模型 | 视觉骨干 / 对比损失 | 用途 |
|---|---|---|---|
| M0 | OpenAI CLIP `ViT-B/32` | ViT-B/32 · Softmax InfoNCE | 核心基线（参考 `openai/CLIP@d05afc4`，SHA256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`） |
| M1 | OpenAI CLIP `RN50` | ResNet-50 (CNN) · Softmax InfoNCE | 同对比损失下的跨视觉架构对比（CNN vs ViT） |
| M2 | SigLIP (`ViT-B-16-SigLIP` / `google/siglip-base-patch16-224`) | ViT-B/16 · Sigmoid Pairwise Loss | 跨对比预训练损失范式对比（Softmax InfoNCE vs Sigmoid） |

规范：
- 每个模型严格使用自身配套的图像预处理（resize/crop/normalization）与 tokenizer，不混用。
- 图像特征 $v$ 与文本特征 $t$ 分别做 L2 归一化，float32 精确内积 $s(I, C) = v \cdot t$；拒绝 NaN/Inf、零范数和维度不匹配。
- 固定官方 truncation 行为（CLIP 为 77 tokens，SigLIP 为 64 tokens），记录每个模型在 `raw` 与 `a photo of ` 前缀下的截断数量与比例。
- 缓存键至少绑定：`model_id + weight_sha256 + preprocess_id + text_policy + manifest_sha256 + dtype + dim`，显式保存 ID 顺序。

---

## 3. 核心检索指标与 Winoground 计分

### 3.1 双向检索（T2I & I2T）
设 split 含 $N$ 张唯一图像、$M$ 条描述，相似度矩阵 $S \in \mathbb{R}^{N \times M}$，$S_{i,j} = s(I_i, C_j)$：
- **Text-to-Image (T2I)**：每条描述为查询，在全部 $N$ 张图中排序；对应 `image_id` 为正例。
- **Image-to-Text (I2T)**：每张唯一图像为查询，在全部 $M$ 条描述中排序；该图像的**全部标注 `caption_id`** 均为正例。
- **Recall@K**：每个查询的 `rank` 为首个标注正例的 1-based 排名，$\text{R@K} = 100 \times \text{mean}(\text{rank} \le K)$，报告 **K = 1, 5, 10**（可附 MedR / MRR 作为辅助）。
- **确定性平分规则**：相似度降序，完全相等时按候选稳定 ID 字典/数值升序排列。
- 每次报告必须附方向、查询数、候选数、每图正例数分布；不把 `smoke` 子集结果与完整候选库混表。

### 3.2 Winoground 严格计分
令 $s_{i,j} = s(C_i, I_j)$（第一维为 caption），正确配对为 $(C_0, I_0)$ 与 $(C_1, I_1)$：
```text
text_success  = (s00 > s10) and (s11 > s01)
image_success = (s00 > s01) and (s11 > s10)
group_success = text_success and image_success
```
- 相等视为失败（严格大于，不使用 tie-break 或容差）。
- 随机连续评分参考基线：text 25%、image 25%、group $1/6 \approx 16.67\%$。
- 保存全部 4 个原始分数与 3 个布尔结果；按官方 `object / relation / both` 标签做描述性误差分析并标明重叠与样本数。

---

## 4. 新特征表示、降维聚类与新检索方法（v2 新增）

### 4.1 数据探索与表征诊断（`features.py`）
1. **数据探索**：统计并可视化 caption 词数/token 数、截断率、每图描述数、相似度分布与 **Hubness 分布**（各候选图像/文本落入 Top-10 的被检索频次长尾分布，附 $k$-occurrence 偏度/基尼系数）。
2. **Modality Gap 可视化**：将配对图像/文本嵌入联合做 **PCA / UMAP** 2D 投影，展示双模态分离锥体及去中心对齐前后的几何距离（$\|\bar{v} - \bar{t}\|_2$）变化。
3. **PCA 维度压缩实验**：仅在 dev/train 上拟合 PCA 投影矩阵，将测试集特征压缩至 $d \in \{512, 256, 128, 64, 32\}$ 维（再 L2 归一化），绘出 R@1/5/10 与存储开销折中曲线。
4. **K-Means 语义聚类分析**：对 dev/test 图像嵌入做 K-Means 聚类（如 $K \in \{8, 12\}$，用 Silhouette score 辅助选 $K$），按语义簇汇总平均 R@1 与代表样例，分析哪些视觉场景检索最困难。

### 4.2 新特征表示：模态去中心化与白化对齐（`E_feat`）
- **Mean-Centering（模态中心对齐）**：分别计算 dev 集图像均值 $\mu_v$ 与文本均值 $\mu_t$，变换 $v' = \text{norm}(v - \mu_v)$、$t' = \text{norm}(t - \mu_t)$，缩减 Modality Gap。
- **ZCA / Reg-Whitening（正则化白化）**：在 dev 集上估计模态内协方差 $\Sigma + \epsilon I$，对去中心向量做 $(\Sigma + \epsilon I)^{-1/2}$ 白化并重新 L2 归一化，缓解各向异性（anisotropy）。
- **防泄漏**：$\mu_v, \mu_t, \Sigma$ 严格仅由 dev/train 拟合，绝不使用 held-out test 统计量。

### 4.3 无训练重排新方法与 5-Fold 交叉验证（`rerank.py`）
针对高维跨模态嵌入的 **Hubness** 效应，在相似度矩阵 $S$ 上实现两种方法，并用 **image-grouped 5-fold CV**（在 dev 集按 `image_id` 分 5 折，每次在 4 折或折内验证，以双向 R@1 均值作为选参目标）调优超参：

1. **方法 A — CSLS（Cross-Domain Similarity Local Scaling）**：
   $$\text{CSLS}(x, y) = 2\cos(x, y) - r_T(x) - r_I(y)$$
   其中 $r_T(x)$、$r_I(y)$ 分别为 $x, y$ 与其跨模态 $k$ 近邻的平均余弦相似度。CV 搜索空间：$k \in \{5, 10, 20, 50\}$。
2. **方法 B — Dual-Softmax（QB-Norm / 双向温度归一化）**：
   $$S_{\text{DS}} = S \odot \text{softmax}_{\text{row}}(S / \tau) \quad \text{或双向} \quad \text{softmax}_{\text{row}}(S / \tau) \odot \text{softmax}_{\text{col}}(S / \tau)$$
   对整库相似度矩阵做跨查询上下文惩罚，抑制高频 hub 候选。CV 搜索空间：$\tau \in \{0.01, 0.02, 0.05, 0.1, 0.2\}$（记录精确公式变体并固定）。

> 注：CSLS / Dual-Softmax 依赖候选库或查询批量上下文（transductive / gallery-level calibration），报告中必须如实说明这一前提（单查询在线场景需借助历史/背景 query bank 近似），且**不应用于 Winoground 2×2 诊断**（Winoground 仅评估原始余弦及新特征/微调表示本身，避免将库级统计混入单组组合性测试）。

### 4.4 Phase 2 可选冲刺：组合性硬负例轻量适应（`E_adapt` / NegCLIP-lite）
- **前提**：Phase 1 通过且时间/算力预算落定；仅使用独立 `train` split，**严禁接触 Winoground 与 held-out test**。
- **方法**：对每条训练 caption 自动构造 1 条词序/主宾或属性置换的 Hard Negative caption，与 batch 内常规负例共同计算对称对比损失（避免同图多描述互作假负例：每 batch 每图仅采 1 条正描述）。仅解冻文本编码器顶层 LoRA 或轻量 Residual Adapter（最多 3 组超参配置 × 最多 3 epochs，总时长封顶 2 小时），按 dev 集双向 R@1 均值选定并冻结，随后统一在 held-out test 与 Winoground 上评测。

---

## 5. 实验矩阵、统计检验与复现要求

| 实验 ID | 内容 | 目的 |
|---|---|---|
| `E0` | 冻结 `CLIP ViT-B/32` + raw captions | 核心基线 |
| `E1` | 冻结 `CLIP ViT-B/32` + `a photo of ` 前缀 | 受控文本提示对照（CLIP Appendix E.1） |
| `E2` | 冻结 `CLIP RN50` 与 `SigLIP`（raw / prefix） | 跨视觉架构（CNN vs ViT）与跨对比损失（InfoNCE vs Sigmoid）对比 |
| `E3` | 新特征表示：Mean-Centering / Whitening + PCA 压缩 | 检验消除 Modality Gap 与降维的影响（dev 拟合） |
| `E4` | 新方法：CSLS & Dual-Softmax 重排（5-fold CV 调参） | 检验消除 Hubness 的收益与交叉验证选参稳定性 |
| `E5`（可选） | NegCLIP-lite 硬负例轻量适应（Phase 2） | 检验组合性微调对常规检索与 Winoground 的影响 |

- **不确定性估计**：固定 `seed=5489` 的 **1,000 次 paired bootstrap**（检索按 `image_id` 整组重采样，固定候选库；Winoground 按 group 重采样），报告关键对比（如 `E1−E0`、`E3−E0`、`E4−E0`）的 R@1 差值与 95% 置信区间。
- **失败案例抽样**：固定 `seed=5489`，按 `E0` 的 R@1 成功/失败分层，每方向各抽最多 10 例，对比展示各方法输出；Winoground 按 `E0` group 成功/失败各抽最多 10 组并附官方标签。补充手工案例另标 `illustrative`。
- **单元测试必覆盖项**：完美匹配、已知排名失败、多正例 I2T、轴交换、ID 打乱、跨 split 泄漏检测、零向量/NaN 拦截、平分确定性、分块与全矩阵一致、Winoground 24 排列随机基线校验（4/24）、CSLS/Dual-Softmax 数值稳定性与 5-fold image-grouped 无泄漏断言、Mean-Centering/PCA 无 test 泄漏断言。
- **Run Bundle 必备产物**：`config.json`、`provenance.json`（含 git commit、dirty diff hash、依赖版本、数据/split/权重 SHA256、设备与耗时/峰值内存）、`metrics.json`、`ranks.jsonl`、`run.log`（Winoground 另存逐组 4 分数）。所有报告表格与图表必须由产物脚本生成，禁止手填数字。
