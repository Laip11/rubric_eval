# MCCR-Eval 当前方法与实验协议

更新时间：2026-11-07

本文档是当前执行口径。完整数学定义见 `new_our_method.md`；运行命令见 `EXPERIMENT_SETUP.md`。此前的 PAOR、AlpacaEval/MT-Bench/BiGGen 主表及 Pointwise 计划均属于历史方案，不与本轮结果混算。

## 1. 研究目标与对照关系

本轮主方法是 **MCCR-Eval（Multi-view Consensus Clustering and Residual Orthogonal Filtering）**，旧 PAOR 只作为同协议对照。

原论文 *Generating and Refining Dynamic Evaluation Rubrics for LLM-as-a-Judge* 及其仓库仅用于提供 CheckEval、DnA-Eval、RubricHub 的算法组件。本轮不声称复现该论文的主结果，也不把其 dynamic/DPO 方法与 MCCR 混为一谈。

当前方法角色：

| 方法 | 作用 |
|---|---|
| `mccr` | 新主方法 |
| `paor` | 旧方法对照，验证 MCCR 是否真正带来改进 |
| `direct` | 无准则直接裁判，正反两个顺序 |
| `checkeval` | 原仓库 CheckEval prompt、parser、评分逻辑 |
| `dna` | 原仓库 DnA aspect、weight、逐 aspect 评分与加权逻辑 |
| `rubrichub` | 原仓库参考回答、带权准则、逐 criterion 判定与加权逻辑 |
| `direct_k` | 近似调用数对齐的采样裁判；不视为严格 token/FLOP 对齐 |
| `dynamic` | 可选历史参考，不进入默认主表 |

## 2. 新数据集

### 2.1 RewardBench 2（非 Ties）

来源：[RewardBench 2 论文](https://arxiv.org/html/2506.01937v2)与[官方数据](https://huggingface.co/datasets/allenai/reward-bench-2)。使用 Factuality、Precise IF、Math、Safety、Focus 五个子集。

- 原始 1,763 个 prompt，每个 1 个正确候选和 3 个错误候选。
- 每个正确–错误组合构造成一个 pairwise 样本；3 个重复 rejected 文本按 query 去重，最终 5,286 个 pair。
- Ties 子集有多个正确/错误候选及专门 margin 指标，不进入当前普通 pairwise 主表。
- A/B 方向由 `sha1(subset + id + choice_index)` 确定性打乱，防止位置标签泄漏。
- 固定分层划分：100 dev query；1,663 test query；4,986 test pair。
- 指标是**派生 correct-vs-incorrect pairwise accuracy**，不是官方 best-of-4 leaderboard 分数。

### 2.2 HelpSteer3-Preference validation

来源：[HelpSteer3-Preference 论文](https://arxiv.org/html/2505.11475v1)与[官方数据](https://huggingface.co/datasets/nvidia/HelpSteer3)。只使用官方 `preference/validation.jsonl.gz`，不读取 train。

- 原始 2,017 个 validation pair；总体偏好来自 3–5 名人工标注者，过滤高分歧样本。
- 排除 1 个两个回答文本完全相同但标签非 tie 的退化样本，保留 2,016 个 query。
- `overall_preference < 0` 对应 response1 更好，`> 0` 对应 response2 更好，`0` 为人类 tie。
- 保留原始 A/B 方向和 `overall_preference`；先固定划分，再在推理前过滤人类 tie，避免改变 dev/test 成员。
- 按 domain 分层，在 validation 内固定划分 100 dev query 和 1,916 个原始 test query；test 中过滤 93 个人类 tie 后，正式评估为 1,823 query / 1,823 pair。

### 2.3 不再作为默认主基准的数据

RewardBench、JudgeBench、AlpacaEval、MT-Bench、BiGGen 的 loader 保留用于历史或附加分析，但不进入本轮默认主表。Pointwise 仍暂停。

## 3. MCCR-Eval 实现

### 3.1 完整探测

每个 query 使用 `N=6` 条探测，3 条来自生成器 `G`，3 条来自弱模型，温度按 `{0.3,0.7,1.0}` 循环。探测而不是待测回答用于准则生成和筛选，避免 target 信息泄漏。

每条探测最多 4,096 tokens，并记录 `finish_reason`、`completion_tokens` 和尝试次数：

- `finish_reason=length`、达到输出上限或空文本均视为无效；
- 无效探测使用新 seed 重新生成，最多 7 次；第 2 次要求不超过 1,200 tokens，第 3–4 次降温到 0.2，要求不超过 3,000 tokens 直接给出最终答案，长序列用范围/省略号紧凑表示，多项请求保持显式数量但压缩每项解释；
- 第 4 次在强生成器与弱模型之间切换；第 5 次改用裁判模型，只允许 800 tokens 内的最终回答，不确定时明确说明而非无限推导；第 6–7 次将原会话标为数据，使用 guided JSON Schema 强制 `answer/explanation/confidence` 结构，答案/解释分别最多 200/500 字符，输出硬上限 1,024 tokens，不保留推导或重复分析；实际 `probe_roles` 和 `probe_role_recoveries` 全量记录；
- 只有完整探测才能进入后续阶段；
- 每次失败的原始文本、finish reason 和 token 数写入 `probe_failure_records`；
- 超过重试次数则显式终止，不把截断文本静默写入得分矩阵，断点恢复也不能绕过次数上限。

### 3.2 多视图原子准则

对每条 query：

1. 先验视图只读取 query，生成 1–5 条必要原子准则；单一独立要求只保留 1 条，不强行拆分。
2. 每个探测分别形成一个 response-grounded 视图，生成 1–6 条原子准则，同样允许单准则视图。
3. 合并 `M+1=7` 个视图，为每条准则保留全局唯一 ID 和来源 view。

准则 JSON 使用初始生成加最多 2 次确定性重试；最终失败会记录原始输出并显式终止。对 malformed JSON，只恢复完整闭合且具有 ID/criterion 的对象，忽略不完整尾部并记录 repair 警告，不补写准则。批处理先保存全部成功 query，再报告失败，断点续跑不会丢弃同批次已成功结果。

### 3.3 聚类、支持度与核心分流

使用 `Qwen3-Embedding-0.6B` 的归一化嵌入，按余弦距离和 average linkage 做凝聚层次聚类，默认 `theta_sim=0.80`。

每簇选择平均簇内相似度最高的 medoid。支持度为

$$
\rho_c=\beta\,\mathbb{I}[c\cap R^{(0)}\neq\varnothing]+(1-\beta)\frac1M\sum_{m=1}^M\mathbb{I}[c\cap R^{(m)}\neq\varnothing],
$$

默认 `beta=0.4`、`tau_core=0.5`。`rho>=tau_core` 的 medoid 进入核心集，其余进入补充候选池。若没有核心簇，取支持度最高的一个簇作为数值安全回退并记录 `core_fallback=true`。

聚类和 medoid 是经验去重手段，不把“严格最小覆盖”或“严格无蕴涵”当作已证明结论。

### 3.4 实证残差正交筛选

裁判对 6 条探测在所有核心及候选 medoid 上评分，得到 `F_core` 和 `F_ext`。列中心化后，对核心矩阵做 SVD，逐步选择正交补空间中残差方差最大的候选：

$$
p^*=\arg\max_p\max(\rho_p,\rho_{\min})\frac{\|e_p\|_2^2}{M}.
$$

默认 `L=4`、`tau_var=0.04`、`rho_min=0.2`。若最佳残差方差低于阈值则提前停止。实现同时记录核心秩、最终秩、每次入选的残差方差和标准差。

由于只有 6 条探测，中心化后的有效秩至多为 5；该限制会在诊断中显式保留，不把正交选择解释为真实独立性的证明。

### 3.5 核心门控聚合

核心和补充准则分别按支持度、入选残差标准差归一化加权：

$$
U(y)=S_{core}(y)+\eta[S_{core}(y)]^\gamma S_{ext}(y),
$$

默认 `gamma=2.0`、`eta=0.5`。无补充准则时 `S_ext=0`，退化为 `U=S_core`。

Pairwise 使用 `epsilon=1e-4` 判定带内平局；需要时使用同一裁判做正反顺序 fallback。

## 4. 统一模型与长度预算

| 角色 | 模型 | 上下文上限 | 用途 |
|---|---|---:|---|
| `G` | Qwen3.8-27B | 40,960 | 全部生成侧调用、MCCR/PAOR 强探测 |
| `J` | Qwen3.5-9B | 32,768 | 全部最终评分、pairwise 裁判和 fallback |
| 弱探测 | Qwen2.5-7B-Instruct | 16,384 | 仅弱探测 |
| 嵌入 | Qwen3-Embedding-0.6B | 模型默认 | 语义聚类与 PAOR 冗余度 |

长度设置依据真实新数据统计，而不是任意增大：

- query 上限 8,192：HelpSteer3 最大约 4,359 tokens，保留余量；
- 待测 response 上限 4,096：新数据最大约 2,406 tokens，无目标回答截断；
- probe 上限 4,096：用于完整重采样，不代表希望每条都生成到上限；
- MCCR 原子裁判输出上限 4,096：允许一次返回较多 medoid 的评分；
- MCCR 先验/响应视图输出分别为 1,024/768 tokens；
- baseline 使用原仓库预算：CheckEval checklist 1,024、DnA aspect/weight 512/256、逐 aspect 评分 64、RubricHub reference/rubric 1,024/2,048、逐 criterion 评分 256。

Qwen3.5/Qwen3.8 均设置 `enable_thinking=false`。GPU0–5 均为通用 worker，按批次动态加载 G/J/weak；每个角色批次拆分到全部可用 GPU，角色变化时关闭旧引擎并释放显存。不同方法协程并发，同一方法内部依赖阶段保持顺序。

## 5. 判定、同分和指标

- **固定评估集合**：人类 gold tie 在固定划分后、推理前统一过滤，不进入 Accuracy、显著性、bootstrap 或成本；所有方法必须使用完全相同的 pair ID 集合。
- **Pairwise accuracy**：分母固定为全部评估 pair；预测缺失、模型无法区分和解析失败均按该样本 correct=0，不使用 valid-only accuracy，也不剔除失败样本。
- **正反顺序**：普通 direct 每对运行原始/交换两次；结论不一致时计错并记录 `n_order_disagreements`，不冒充 JSON 解析失败。`direct_k` 是额外的多次采样预算基线，不受两次限制。
- **独立评分同分**：按 `sha256(seed + pair_id)` 做可复现随机分配，单独报告 `n_model_score_ties`，不再固定选 A。
- **Fallback**：MCCR/PAOR 带内平局率单独报告；fallback 的正反不一致同样计错。
- **置信区间**：query-level bootstrap 1,000 次，避免把同一 query 的多个 pair 当作独立样本。
- **显著性**：主方法与最强 baseline 做非人类 tie pair 的 McNemar 检验；同时保存 `mccr_vs_paor.json`。
- **成本**：报告 LLM 调用、输入/输出 tokens 和墙钟时间。`direct_k` 只近似匹配理论调用数，MCCR 的基础预算为每 query `19 + 待测 response 数`，重试和 fallback 另计。
- **Pointwise logprob 公平性**：Pointwise 当前暂停。恢复比较时，所有方法必须使用一致的 logprob/软期望协议；只要有一个方法不能提供可比 logprob，则所有方法统一使用离散分数，禁止 MCCR/PAOR 用软期望而 baseline 用离散值直接比较。

## 6. 缓存与诊断

- MCCR、PAOR 使用不同的 method/cache namespace。
- 每个 query 保存累计调用/token `cost_checkpoint`，每个方法保存累计墙钟 checkpoint；断点恢复只加新进程增量，正式成本覆盖从冷启动开始的全部尝试。
- 缓存签名绑定生成器、裁判、弱探测、嵌入模型、长度、N/L 和全部选择/聚合超参数；配置不一致时不复用。
- 保存多视图全集、簇成员、medoid、支持度、核心/候选/入选集合、F 矩阵、残差、目标原子分数和 raw judge 输出。
- baseline 保存逐 aspect/criterion 原始输出；direct/dynamic 保存逐顺序投票与原始输出。
- 诊断写入当前 output root 内，避免跨实验覆盖。

## 7. 当前实验边界

- 当前只运行 Pairwise，seed0；CI 不是跨 seed 方差。
- 默认 MCCR 超参数来自 `new_our_method.md`，在正式声称优于 PAOR 前必须完成同协议对照；不能仅凭公式推断改进。
- 尚未完成 MCCR 的系统开发集调参、跨 seed 实验和独立长文压力测试。
- LongJudgeBench 等长文数据需要单独的上下文预算，不与当前 8K/4K 协议混跑。
- 旧 `outputs/main*`、`outputs/full_pairwise_qwen` 只作历史归档。
