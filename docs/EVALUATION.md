# 评估协议 v1

日期：2026-10-20。性质：待实现的实验契约，不是实验报告。修改会影响已运行结果时，必须升版本、写原因并标记结果是否失效。

## 1. 数据与来源

| 用途 | 方案 | 访问与核实要求 |
|---|---|---|
| 主检索 | Flickr30k + Karpathy/Fei-Fei 发布的 retrieval split | 原作者页申请；图像限非商业研究/教学且受 Flickr 条款约束；核查图像与标注分别适用的条款 |
| 主方案的开发/测试 | 发布划分中的 validation / test；目标为 published 1K test | split JSON 与图像分别获取；精确 train/val/test 数量以 manifest 审核为准，不由原始总数推算 |
| 访问备选 | COCO val2017，明确标注 `coco2017-custom-1k-dev-4k-eval-v1` | 官方 5K 图像约 1GB，2017 annotations 包约 241MB；访问阶段重核容量与条款 |
| 组合性诊断 | Winoground 官方 400 test groups | HF gated，需接受条款并分享账户信息；本轮未代为接受，完整许可待合法获取后阅读 |

Flickr30k 在第一周数据关卡结束仍不可获取，则记录原因并采用备选，不能无期限等待。COCO 备选构建规则：按数值 image ID 升序，使用固定 NumPy 版本的 `Generator(PCG64(5489))` 打乱，前 1,000 图为 dev，余下 4,000 图为 eval；每张图的全部描述随图分配。保存显式 ID 列表和 SHA256，后续以清单为准。两集合各自构成检索候选库。该方案不是 Karpathy 5K test，不与其指标直接比较；它不提供适应训练集，可选训练默认延期，除非另外获取独立训练数据并修订计划。

数据验收需检查：稳定且唯一的 image/caption ID、文件存在与可解码、图像到全部描述的映射、image ID 跨划分互斥、内容哈希重复、实际样本数与标注数、缺失文件和排除理由。标准划分发现异常时停止并调查，禁止静默删样本后仍宣称标准评测。图片不进入公开发布包，提交示例须符合原数据条款。

官方来源（2026-10-20 核查，尚未下载数据）：

- [Flickr30k 原作者页](https://shannon.cs.illinois.edu/DenotationGraph/)：31,783 images / 158,915 captions；申请入口与图像使用限制。
- [Karpathy & Fei-Fei 作者发布页](https://cs.stanford.edu/people/karpathy/deepimagesent/)及[论文 §4](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf)：split 背景；页面提供 `caption_datasets.zip`，本轮只核实入口，未检验归档。
- [COCO 下载页](https://cocodataset.org/#download)、[条款](https://cocodataset.org/#termsofuse)：annotation 为 CC BY 4.0，不能泛化为所有图片统一 CC BY；图片另受原始权利和 Flickr 条款约束。
- [Winoground 官方数据卡与 FAQ](https://huggingface.co/datasets/facebook/winoground)：访问限制、test groups 和评分示例。

## 2. 模型、预处理与缓存

主模型为 OpenAI CLIP `ViT-B/32`，冻结全部权重，eval/inference 模式。优先复用官方 loader 返回的预处理及 tokenizer，不混用不同模型的图像均值、方差或 tokenizer。图像按配套 resize / center crop / RGB 处理；文本不额外重写、去停用词或翻译。

官方源码参考固定为 [openai/CLIP@d05afc4](https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6)，此时 ViT-B/32 checkpoint 的预期 SHA256 为 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`。这是来源元数据，不代表本机已下载。环境阶段选择兼容依赖后，记录实际 commit、权重 hash 和所有版本。

对图像特征 v、文本特征 t 分别 L2 归一化，评分 `s(I,C) = v · t`。拒绝 NaN/Inf、零范数和维度不匹配；使用 float32 评分。使用余弦分数，不对每行/列单独 softmax；后者会改变 Winoground 的跨行/列比较。固定正 logit scale 不改变排序，但 softmax 输出不能当跨候选库可比概率。

tokenizer 上限为 77 tokens（含特殊 token）；固定采用官方 truncation 行为并记录截断样本 ID/数量。raw 与 prefix 各自记录截断率，讨论前缀占用长度的混杂因素。缓存键至少含模型/权重、预处理、文本策略、manifest/split hash、dtype 和特征维度；任一变化则失效。保存明确 ID 顺序，禁止按文件系统遍历顺序猜映射。

## 3. 双向检索计分

令图像数 N、描述数 M，`S[i,j]=s(image_i, caption_j)`。

- **Text-to-image (T2I)**：每条描述为查询，候选为该 split 的全部 N 张唯一图像；关联 image ID 为标注正例。
- **Image-to-text (I2T)**：每张唯一图像为查询，候选为该 split 的全部 M 条描述；该图像的全部标注 caption ID 为正例。
- 每个查询的 `rank` 是第一个标注正例的 1-based 排名。`R@K = 100 × mean(rank <= K)`，分别报告 K=1,5,10。这是检索文献的“至少一个正确结果”定义，不是找回正例总比例。
- 相似度降序，完全平分按候选稳定 ID 的规范升序决定；排序规则在运行前固定。Winoground 不采用该平分解法，见下节。
- 必报方向、query 数、candidate 数、实际正例数分布与百分比单位。补充 MedR/MRR 可在主指标稳定后添加，但不是主任务必需项。
- caption 字符串重复不能擅自合并 ID。未标注但语义合理的匹配可能被标准指标判错，误差分析需要说明这一局限。

开发集与测试集均使用各自完整候选库。快速子集只做 smoke check，标为 `smoke`，不与完整 benchmark 混表。分块评分必须与小矩阵的完整评分一致；不使用 FAISS 或近似索引作为首版默认。

## 4. Winoground

令 `s_ij = s(C_i, I_j)`（这里第一维为 caption，区别于上一节 S），每组两个正确配对是 `(C0,I0)`、`(C1,I1)`：

```text
text_success  = (s00 > s10) and (s11 > s01)
image_success = (s00 > s01) and (s11 > s10)
group_success = text_success and image_success
```

三项均逐组计 0/1，再对 400 组求平均百分比；相等视为失败，不用 tie-break 或容差放宽成功条件。保存四个原始分数和三个布尔值。缺失组不能被静默剔除；不足 400 组必须标为 partial，不冒称 full benchmark。

随机独立连续评分的参考率为 text 25%、image 25%、group 1/6≈16.67%。Group 不是 25%×25%，因为比较并不独立。[官方解释](https://huggingface.co/datasets/facebook/winoground#why-is-the-group-score-for-a-random-model-equal-to-1667)

在所有方案冻结之后，固定评估 raw 与 prefix 两种配置；不根据 Winoground 表现选择提示词、checkpoint、学习率或训练轮次。按官方 object/relation/both 标签作描述性误差分析并报告组数；不把重叠标签当互斥类。若访问最终未获批，交付明确缺失原因和评分合成测试，不能声称已完成此诊断。

## 5. 实验矩阵与防泄漏

| ID | 模型/训练 | 描述 | 用途 |
|---|---|---|---|
| E0 | 冻结 ViT-B/32 | raw，原文不变 | 必做基线 |
| E1 | 相同模型、无训练 | 精确字符串 `a photo of ` + 原始描述 | 唯一必做对比；保持其余配置、候选库一致 |
| E2 | 可选的单一适应方法 | 由 dev 阶段预先固定 | 基线后单独设实验预算；默认 deferred |

E0/E1 均报告，不只展示较优者。若演示需要选一种配置，以 dev 的双向 R@1 算术均值选择，平分选 E0；该均值仅作选择规则，报告仍保留两个方向。冻结后再运行最终 retrieval test 与 Winoground。不把测试表现用来决定要不要启动 E2；E2 的去留在最终测试前决定。

置信区间采用固定 seed=5489 的 1,000 次 paired bootstrap：在每方向内按 image ID 重采样查询单元，T2I 将同一图片的所有 caption 查询一起抽样；候选库保持固定，估计的是给定候选库下的查询不确定性。比较 E1−E0 的 R@1 差值并报告 95% percentile interval。Winoground 按 group 重采样；不要把四个配对当四个独立样本。区间不覆盖预训练污染、候选库变更或域转移的不确定性。

若启动 E2：只用独立 train split，dev 选超参数；batch 内同一图像的多描述需用 multi-positive loss 或每图仅采一个描述，禁止互作假负例。须先锁定方法、可训练参数、最多 3 个配置、每个最多 3 epochs、总 accelerator-hour 上限与早停条件。实际小时上限由 smoke timing 和可用资源决定，未填数值不得开训。最终选定配置可运行 3 个训练 seed（计入预算）；不足时如实报告单 seed 限制。没有 dev 增益或预算不足即停止，不扩展搜索。

## 6. 正确性与可复现证据

实现阶段的必要 fixtures：完美匹配、已知排名失败、多正例、轴交换、caption 打乱、重复 ID/跨划分 ID、零向量/NaN、平分、batch 与分块一致；Winoground 全成功、仅 text 成功、仅 image 成功、平分失败。可枚举四个不同分数的 24 种排列核验 group 随机率为 4/24。

每次正式运行保存 `run_id`、协议版本、UTC 时间、git commit 与 dirty diff hash、完整配置、环境锁与设备、来源和 split/manifest hashes、权重 hash、seed、预处理/截断统计、逐查询 rank/top-K ID、Winoground 四分数、metrics JSON、日志、耗时与峰值内存。对比配置使用同一查询顺序和候选库。固定缓存重算计分应完全一致；重新编码的数值容差在设备 smoke 阶段记录，并检查临界平分。

失败案例抽样固定 seed=5489：每个方向按 E0 的 R@1 成功/失败分层，各抽最多 10 个查询；同一组查询展示 E0/E1 输出。Winoground 按 E0 group 成功/失败各抽最多 10 组，附官方标签。类别不足时全取并写实际数量；人工挑选的补充案例另标 illustrative，不当总体证据。

成功标准是“正确、可追溯、可复现并能解释局限”，不是达到未证实的指标阈值。论文结果只在模型、预处理、split 和候选规模一致时作有限对照。
