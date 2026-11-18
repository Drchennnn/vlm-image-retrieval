# 项目结构与执行计划

日期：2026-10-20。本文件为规划，不代表下列模块存在。目标为一个可复现、能解释局限的课程研究原型；不建设生产服务、向量数据库、复杂 Web UI 或多 agent 产品。

## 1. 最小交付与研究问题

- RQ1：冻结 CLIP 在规定候选库中的双向检索表现如何？
- RQ2：固定 `a photo of ` 前缀相对 raw captions 是否改变检索表现？
- RQ3：检索表现与 Winoground 组合性诊断呈现什么差异？这是描述性问题，不预设相关性或因果解释。

必交：可运行的双向检索 CLI、可信计分、E0/E1 结果与失败分析、来源/命令/环境/产物记录、最终报告。Winoground 是计划中的独立诊断，访问不可用时显式交代缺失并保留可验证 scorer；不拿合成测试替代真实分数。微调为可选 E2，不以提升为验收条件。

## 2. 拟建结构

当前只创建文档及 Proposal 输出目录，以下代码与数据目录在对应实现任务启动时再建：

```text
.
├── 1.md                         # 原始课程题目，保持不变
├── AGENTS.md                    # Agent 工作契约
├── README.md / README.zh-CN.md  # 项目入口
├── CHECKLIST*.md                # 验收导航，不独立维护状态
├── PROJECT_PROPOSAL.md          # 正式英文 Proposal 源文
├── paper/                      # 本地参考论文
├── docs/
│   ├── REVIEW.md               # 本轮独立审核
│   ├── PROJECT_PLAN.md         # 结构、流程、审查
│   ├── EVALUATION.md           # 唯一实验协议
│   └── PROGRESS.md             # 唯一进度台账及决策记录
├── output/pdf/Project_Proposal.pdf
├── pyproject.toml              # 待建：包、入口、测试/静态检查配置
├── requirements.txt            # 现有候选表；P2 替换为验证后的依赖策略
├── configs/                    # 待建：固定 dataset/model/eval 配置
├── src/vlm_retrieval/           # 待建：单包，不拆微服务
│   ├── data.py                 # manifest 解析、ID 校验、split 检查
│   ├── model.py                # 唯一 CLIP wrapper、预处理、归一化
│   ├── cache.py                # 特征与来源 metadata、缓存有效性
│   ├── retrieval.py            # 双向精确排名、分块检索
│   ├── metrics.py              # 纯函数 R@K / Winoground scorer
│   ├── evaluate.py             # 编排评估、置信区间、产物写出
│   └── cli.py                  # encode / search / evaluate 入口
├── tests/                      # 待建：纯合成 fixture + 小规模集成
├── data/                       # 待建：raw / manifests，原图不入 Git
├── artifacts/                  # 待建：特征缓存、运行日志，不入 Git
└── reports/                    # 待建：精选结果、图表、最终报告
```

不预建空目录；需要新模块时再扩展。当前 `.gitignore` 为白名单策略；实现阶段须同时允许新增源代码、配置、测试及脱敏结果，继续忽略原始数据、权重、密钥、缓存和工作流目录。不要把 `.omx`、`.specify`、`openspec` 当模型实现。

### 模块契约

| 边界 | 输入 → 输出 | 不变量 |
|---|---|---|
| Data | split manifest → image rows + caption rows | image ID 唯一；caption 引用存在的图像；同图全部描述同 split |
| Model | 图像/描述 batch → `[B,D]` 特征 | 冻结权重；有限值；L2 范数约 1；记录截断 |
| Cache | embeddings + ID list + metadata → cache | 数据、模型、预处理版本一致才可复用 |
| Retrieval | query/candidate embeddings → ranked IDs/scores | 相同候选库与确定性平分；分块与完整矩阵一致 |
| Metrics | rankings + relevance / 四个组内分数 → metrics | 不读取模型或下载数据；可用合成 fixture 验证 |
| Evaluation | config + manifest + cache → run bundle | 产物含查询数、候选数、协议和来源；不能仅打印一个分数 |

manifest 建议 JSONL：image 行含 `image_id, relative_path, split, file_sha256`；caption 行含 `caption_id, image_id, text`。配置与 ID 清单可进入版本控制；包含受限文本的完整清单是否公开需按许可决定，至少记录生成方式和哈希。

依赖策略：首版围绕 PyTorch/torchvision、官方 CLIP、Pillow、NumPy；CLIP 自身所需依赖随锁定源码确认。测试和静态检查工具在 P2 确定。Transformers、datasets、FAISS、Jupyter、pandas、scikit-learn 不因历史清单存在而全部安装；非必要依赖延期。使用现有算法先完成精确检索，再凭实际瓶颈决定优化。

## 3. 六周相对计划与退出条件

没有课程截止日期；周次从实现启动计算，互不表示已完成。

| 周次 / 关卡 | 工作 | 通过证据 / 停止条件 |
|---|---|---|
| 第 1 周 G1 | 数据访问、split 审核、环境锁定、小批模型 smoke | 条款/来源/哈希与计数；模型加载和缓存往返；实测时长/内存。Flickr30k 不可得则记决策切换 COCO |
| 第 2 周 G2 | 数据层、双向排名、纯函数指标与合成测试 | 多正例、轴、平分、泄漏检查通过；独立审查无未解决严重项 |
| 第 3 周 G3 | dev 全候选库基线；CLI 查询；复现运行 | 双向 R@1/5/10、逐查询产物；缓存重算一致；演示能关联实际 image/caption ID |
| 第 4 周 G4 | E0/E1 dev 对比；训练可行性决策；冻结配置 | 两种配置都保留；E2 明确 deferred，或完成训练与 dev 选择后再冻结；冻结记录含 revision/hash |
| 第 5 周 G5 | 冻结后 test、Winoground、置信区间、失败分析 | 完整 run bundle；数据不可得须单列限制；不据测试继续选择模型 |
| 第 6 周 G6 | 报告、演示、复现与提交包审查 | 第二执行者用文档重算指标；图表可追溯；无未实现命令、泄漏文件或虚构结论 |

最小可行性 PoC：在开发划分的 100 张图及全部关联描述上编码与双向搜索（标为 smoke）；验证加载、计分和缓存后测量资源。最多两轮 batch/device 调整仍不能稳定运行，则记录异常并采用 CPU 小批/分块作为恢复；如仍不可行，回报真实阻塞，不承诺训练。PoC 通过不等于全量指标已验证。

完整 1K 图/5K 描述的 float32 相似度矩阵约 20 MB（只计分数，不计模型/特征/数据加载），说明首版无需近似索引；这是算术估计，不是内存实测。总运行时间由设备实测估计后写入台账。

## 4. 执行与审查机制

1. **认领**：从台账选择一个 task ID，填写实际执行者、审查者、依赖与成功证据，改为 `doing`。未认领成员不默认承担任务。
2. **实现**：限制一个可验收切片；算法优先纯函数。对评估逻辑先构造已知答案 fixture；不为文档改动启动模型实验。
3. **验证**：先聚焦测试，再运行适用的 lint、typecheck、完整测试与 CLI smoke。若某类检查尚未配置，明确写出缺口，不声称通过。
4. **审查**：转 `review`，由另一 agent/成员检查 diff、测试、数据隔离和证据。严重项退回 `doing`；轻微项记录后修复或明确延期。
5. **收口**：审查通过才改为 `done`，写命令、结果摘要、产物路径、修订号；同步 README 的里程碑摘要。

状态转换：`todo → doing → review → done`；可进入 `blocked` 并写原因/恢复条件，解除后回到 `doing`；可选任务可为 `deferred`。禁止把“预计完成”转成 `done`。每日执行结束写一条 delta（完成、证据、阻塞、下一步），每周核对关卡；无活跃工作时不制造日报。

每个变更只需一个执行者和一个审查者。独立实现任务可并行，但协议、台账和集成由主执行者维护；共享文件先协调。署名成员为 Benyi Chen、Muyao Yan、Jianheng Qiao；具体编码/数据/报告分工尚未指定，不从署名推断贡献。

## 5. 研究审查与停止规则

- 数据审查：训练/开发/测试角色、原始条款、映射、缺失、重复和公开范围。
- 指标审查：T2I/I2T 方向、多正例、候选库、tie policy、Winoground index convention。
- 实验审查：一次只改一个因素；两种主配置都报告；训练参数只看 dev；测试后新想法标探索性。
- 结论审查：区分论文结果、本项目结果、推测；“无提升”也是有效结果；小样本类别不能过度解释。
- 停止规则：核心结果可复现且研究问题已回答即停止扩展；访问受阻走事先备选；训练预算未落实则不启动；不追求任意高分。

风险优先级：数据访问/许可 > 指标错误和泄漏 > 环境兼容/算力 > scope 扩张。缓解措施依次为访问备选、合成验证和独立审查、冻结模型/小批缓存、单一消融与条件训练。

## 6. 运行与提交材料

每个 run bundle 建议含 `config.json, environment.txt, provenance.json, metrics.json, ranks.jsonl, run.log`，Winoground 另存逐组 scores；表格从机器可读结果生成，人工分析通过 run ID 引用。公开报告不附受限原图或凭证。最终报告需包含对称对比损失解释、实验设计、双向结果、组合性诊断或缺失说明、受控比较、失败案例、资源测量及局限。

正式提交前检查课程要求是否新增模板、截止日或文件格式。当前 Proposal 是研究计划，不能替代未来最终报告；当前 PDF 使用组员汉语拼音署名。
