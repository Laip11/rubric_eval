# MCCR 审查与顶会实验设置重构报告

## 背景
用户要求以审稿人视角审查 `rubric_eval` 仓库中 MCCR（前身 PAOR）方法、数据集与 baseline，并基于 2025–2026 文献重新设计实验。本文件为分析报告，不包含代码修改。

审查依据：`new_our_method.md`、`method.md`、`EXPERIMENT_SETUP.md`、`src/mccr_core.py`、`src/mccr_pipeline.py`、`src/paor_core.py`、`src/paor_pipeline.py`、`src/baselines.py`、`src/paor_prompts.py`、`src/run_main.py`。

---

## 1. 方法理解与核心问题

### 1.1 MCCR 流程（按代码实际行为）
1. 生成 N=6 条探测回答（3 条来自 G=Qwen3.8-27B，3 条来自弱模型 Qwen2.5-7B；截断后最多重试 7 次，中途会换模型甚至换成裁判模型）。
2. 生成 7 个视图的原子准则：1 个只看 query 的先验视图，6 个"query + 探测"视图。
3. 用 Qwen3-Embedding 做凝聚聚类（θ=0.8），每簇取 medoid，计算支持度 ρ=0.4·[先验视图中出现] + 0.6·[响应视图中出现的比例]；ρ≥0.5 的进入 core。
4. J（Qwen3.5-9B）在全部 medoid 上给 6 条探测打分，得到 F；在 core 子空间的正交补里贪心选残差方差最大的补充准则，最多 L=4 条。
5. 目标回答按 core+ext 打分（代码里用 **logprob 软期望**），聚合 U=S_core+η·S_core^γ·S_ext，平局时退回 direct 正反裁判。

**PAOR→MCCR 演进**：PAOR 由"先验基 + 一次性看全部探测的后验扩展 + std−λ·嵌入冗余贪心 + 词典序缺陷带 η=c/2K"组成；MCCR 把先验基换成多视图共识，把嵌入冗余换成分数空间正交，把有保证的词典序带换成启发式门控。两者的主张其实相同：**核心要求优先，判别性补充准则用于拉开差距**。MCCR 复杂度更高，但理论保证反而更弱：PAOR 的缺陷带是一个可证明的性质（"一条补充准则不能抵消一条核心缺陷"），门控公式没有对应保证。

### 1.2 最核心的问题（审稿人视角，按严重程度排序）

**P1. 增益无法归因：公平性漏洞（必须先修，否则结果不可信）**
- *强模型探测相当于参考答案*：MCCR 能看到 6 条探测，其中 3 条来自 27B 的 G，响应视图写出的准则会把 G 的答案/解法写进去（例如"最终答案是 X"）。在 RewardBench 2 的 Math/Factuality 子集上，这等于让 9B 裁判拿到 27B 的参考答案，而 Direct/CheckEval/DnA 拿不到。缺少 **参考答案引导的 Direct**（G 的回答作为参考）这一对照，审稿人会把 MCCR 的增益归因于"知识蒸馏"而不是准则结构。
- *打分分辨率不对等*：`mccr_pipeline._target_scores` 用 `softmax_expected_from_entries` 的软期望做 U；CheckEval（Yes/No）、RubricHub（布尔）、DnA（1–10 整数）、Direct（二元胜负）都是离散值。软分数能自动打破平局，pairwise 下同样不公平；`method.md` 只在 pointwise 部分规定了这一点。
- *解析失败被静默中和*：准则缺失时按 0.5 记分（`_score_result`、`_target_scores`），只累加 `parse_fail`；baseline 的失败则计为 correct=0。这与 `method.md` 中"失败即错"的口径不一致。
- *判定协议不对等*：Direct 正反不一致时直接计错；MCCR 走独立打分，只有带内平局才 fallback。两者的错误来源完全不同，需要统一协议，或同时报告 strict 与 tie=0.5 两种口径。
- *算力不对等*：MCCR 每 query 有 13 次 27B 长输出调用；`direct_k` 只按调用次数对齐（9B、256 tokens），差了一个数量级以上的 FLOPs。

**P2. 残差正交筛选在 M=6 下统计上近乎空转**
- 中心化后秩 ≤5。core 一般有 3–6 条，`core_rank` 往往已经占满 3–5 维，正交补只剩 0–2 维，残差基本为 0，L′ 绝大多数情况下会是 0，MCCR 退化为只用 core。必须报告 L′、`core_rank` 的分布来证实或推翻这一点。
- 6 个样本、近似 {0, 0.5, 1} 的方差估计噪声极大，τ_var=0.04 实际上是在对噪声设阈值。
- **方差 ≠ 判别力**：探测按强/弱模型交替，方差最大的方向通常是"27B vs 7B"的风格、长度、格式差异，与目标对的质量差异无关。这是无监督选择，又没有用到已知的方向信息（强模型通常更好）。RRD（2602.05125）恰好用"偏好弱模型的准则视为错向"来过滤，还用 Σ^{-1/2} 白化加权，比 MCCR 的硬选择更合理，也是直接竞品。

**P3. "共识 = 重要"的假设有偏，而且 core 实际上 ≈ 先验视图**
- 按 β=0.4、τ=0.5 计算：先验簇只要再被 1/6 个响应视图命中，就有 ρ=0.4+0.6/6=0.5，进入 core；不含先验的簇需要 ≥5/6 个视图命中才行。所以 core ≈"被回声一次的先验准则 + 几乎普适的通用准则"，与 PAOR 的先验基高度同构，**"多视图共识"带来的新信息很少**。
- 高频簇往往是泛化准则（准确、清晰、切题）；真正决定 RB2 正误的往往是长尾簇（某个具体事实、某一步计算）。门控 S_core^γ 恰好压低了这部分判别信息。
- 余弦阈值聚类会把"≤100 词"和"≤200 词"、"包含 X"和"不包含 X"这类语义相反但字面相近的准则合并。"严格最小、无蕴涵"的表述不成立（文档虽有弱化，`new_our_method.md` 仍保留这些措辞）。

**P4. 聚合粒度粗，fallback 可能主导结果**：每条准则只有 0/1/2 三档、K 很小，独立打分容易平局。如果 fallback 率较高，MCCR 的准确率有相当一部分其实是 Direct 的准确率。必须单独报告 fallback 率和 fallback 子集上的准确率。

**P5. 超参数多、工程补丁多**：θ、β、τ_core、L、τ_var、ρ_min、γ、η、ε、N 共 10 个超参数，只用 100 条 dev 调。7 级探测恢复链会在 G、弱模型、裁判之间切换，并强制 JSON 输出，探测分布因 query 而异，方法不再"良定义"。审稿人会认为这是 hack。

**P6. 没有利用 pairwise 信号**：先独立打分再比较，丢掉了"对比更容易"的优势（OnlineRubrics, ICML 2026 报告 pairwise 方式引出准则优于 pointwise）。

### 1.3 改进建议（MCCR v2：保留卖点，修正机制）
1. **保留**多视图生成 + 聚类共识，把它定位为"准则层面的 self-consistency"（GenRubric 2608.29856 用了同样的直觉）。同时承认 core ≈ 先验，把卖点改为"共识过滤先验视图中的幻觉要求 + 召回响应视图中暴露的隐含要求"，并在 RubricBench 上用专家准则召回率直接验证。
2. **用弱监督对比取代无监督方差**：探测构造成方向已知的对。强 vs 弱是天然的一对；更好的做法是对 G 的回答注入可控缺陷（违反某条约束、篡改关键事实），得到反事实降级版本。对每条准则计算"已知更好 > 已知更差"的方向一致率（AUC），过滤错向准则。权重改用收缩估计（Ledoit-Wolf）后的白化 Σ^{-1/2}·1，并与 RRD_WU 正面比较。增加 M（8–16 条短探测）以缓解秩约束。
3. **重要性的定义**：频率支持度之外，再加一个由 LLM 判定的"硬约束 / 质量项"标签（参照 RaR 的 Essential/Important/Optional/Pitfall）。硬约束用 PAOR 式可证明的词典序缺陷带做门控，质量项线性加权。以"频率 vs LLM 标签 vs 二者结合"做消融。
4. **平局处理**：先做准则级 pairwise 比较（同一条准则下比较 A/B），再退回整体 direct。所有方法统一使用离散分数，或统一使用 logprob。
5. **简化探测**：固定 max tokens，截断探测一律丢弃，生成模型保持固定，并报告丢弃率。删去切换裁判、强制 JSON 等恢复补丁。
6. **收敛主张**：去掉"严格最小基/无冗余"的表述，改为可检验的三个假设：H1 覆盖（专家准则召回）、H2 门控抗表面偏差（LLMBar/RM-Bench）、H3 对比选择优于无监督选择（消融 + 算力曲线）。

---

## 2. 数据集推荐

| 数据集 | 年份/规模 | 评测 | 用于验证什么 | 结论 |
|---|---|---|---|---|
| RewardBench 2 [论文](https://arxiv.org/abs/2506.01937) / [代码](https://github.com/allenai/reward-bench) | 2025（ICLR 2026），1,763 prompt | **官方 best-of-4 准确率**；派生 pair 放附录 | 客观正确性与精确指令遵循；分子集看准则在 Math/Factuality 上是否有用 | 主表；补报官方口径，派生 pair 不能单独当主指标 |
| HelpSteer3-Preference [论文](https://arxiv.org/abs/2505.11475) | 2025，val 2,017 | pair acc | 多领域真实人类偏好（含代码、多语言） | 主表（保留） |
| JudgeBench [论文](https://arxiv.org/abs/2410.12784) / [代码](https://github.com/ScalerLab/JudgeBench) | ICLR 2025，GPT-4o 子集 350 | 官方脚本，正反两序 | 困难正确性；可与 RRD、Chasing the Tail 的已报告数字直接对照 | 主表（新增） |
| PPE human preference [论文](https://arxiv.org/abs/2410.14872) / [代码](https://github.com/lmarena/PPE) | ICLR 2025，10.2K | pair acc | 大规模真实 Arena 偏好；同样可与 RRD 对照 | 主表（新增；算力不足时用官方分层子集） |
| RubricBench [论文](https://arxiv.org/abs/2603.01562) / [代码](https://github.com/planepig/rubricbench) / [数据](https://huggingface.co/datasets/DonJoey/rubricbench) | 2026，1,147 pair + 专家原子准则 | pair acc + 准则对齐 | **直接验证准则质量**：专家准则召回率与冗余；把生成准则替换为专家准则做 oracle 上界 | 主表 + 核心分析（新增，最契合 MCCR 的 coverage 主张） |
| LLMBar [论文](https://arxiv.org/abs/2310.07641) / [代码](https://github.com/princeton-nlp/LLMBar) | ICLR 2024，Natural+Adversarial 419 | pair acc | 回答表面好但违反指令；验证硬约束门控（H2） | 诊断（新增，经典权威） |
| RM-Bench [论文](https://arxiv.org/abs/2410.16184) / [代码](https://github.com/THU-KEG/RM-Bench) | ICLR 2025 | Easy/Normal/Hard | 风格与长度偏差的鲁棒性 | 诊断（新增） |
| HealthBench [论文](https://arxiv.org/abs/2505.08775) / ProfBench [论文](https://arxiv.org/abs/2510.18941) | 2025，专家 rubric | 准则召回 | 专家领域的准则覆盖 | 可选（附录） |
| MT-Bench human / BiGGen（仓库已有） | — | pair acc | 与来源论文 2605.30568 的数字对齐 | 可选附录 |

不推荐：AlpacaEval（标注噪声大）、RewardBench v1（已饱和）、RubricEval（2603.25133；评测准则级裁判能力，不是准则生成，可作为引用）。

---

## 3. Baseline 审查与补充

**现有 baseline 的问题**
- Direct：缺少"pointwise 直接打分"版本，导致"pairwise vs pointwise 打分方式"的差异与准则本身的作用混在一起；缺少参考答案引导版本（见 P1）。
- direct_k：只按调用次数对齐，且 T=0.7、HS3 上 k≈21，需要改为按 token/FLOPs 对齐，并画出 accuracy-vs-compute 曲线。
- CheckEval：使用数据集级静态清单（与原论文一致），可以保留，但只作为较弱的参照。
- "RubricHub"：实现的是"参考答案 → 带权准则"，**不是** RubricHub（[2601.08430](https://arxiv.org/abs/2601.08430)）的 coarse-to-fine 多模型聚合。应改名为 RubricHub-lite，或补齐其 response-grounded 合成与多模型聚合；后者与 MCCR 的多视图思想非常接近，是必须正面对比的对象。
- DnA：可保留为可选项。

**必须补充（按优先级）**
1. **RRD**（Shen et al., 2026）[2602.05125](https://arxiv.org/abs/2602.05125)：递归分解/过滤 + 白化加权，在 JudgeBench、PPE 上是 SOTA。机制几乎与 MCCR 一一对应（样本回答、弱/强过滤、相关性感知权重），是最关键的竞品。未找到官方代码，需要按论文 Algorithm 1 复现，三个变体 uniform/LLM/WU 都要做。
2. **参考答案引导的 Direct**（G 的回答作参考）和 **G=J 设置**：P1 的归因对照。
3. **LLM Rubrics w/o resp. / w. resp.**（单次调用；RRD 表中的标准基线）：等价于"只用 MCCR 先验视图"和"一次调用看全部探测"，检验多视图聚类是否有必要。
4. **Chasing the Tail RTD** [2509.21500](https://arxiv.org/abs/2509.21500)（ICLR 2026）/ [代码](https://github.com/Jun-Kai-Zhang/rubrics)：基于探测对的差异做细化，属于响应驱动准则的强基线。
5. **OpenRubrics CRG** [2510.07743](https://arxiv.org/abs/2510.07743) / [模型与数据](https://huggingface.co/OpenRubrics)：硬规则/原则二分，对应 MCCR 的 core/ext。推理时用探测对代替带标签的 chosen/rejected 做 prompt 版复现（只比较 training-free 方法，不使用 Rubric-RM）。
6. **Self-consistency / SPCT 式原则 + 批评 + 投票@k** [2504.02495](https://arxiv.org/abs/2504.02495)：算力对齐下的推理时扩展对照。

**可选**：RLCF candidate-based checklist [2507.18624](https://arxiv.org/abs/2507.18624) / [代码](https://github.com/viswavi/RLCF)；MRRG [2607.01830](https://arxiv.org/abs/2607.01830)（多角色准则聚合，无代码，可按描述 prompt 复现）；TICK [2410.03608](https://arxiv.org/abs/2410.03608)；RocketEval [2503.05142](https://arxiv.org/abs/2503.05142)；来源论文 Dynamic Rubrics [2605.30568](https://arxiv.org/abs/2605.30568)（只比较其 training-free 部分）；PAOR（作为自身消融）。**比较范围限定为 training-free 方法**，Prometheus 2、RM-R1、Rubric-RM、DeepSeek-GRM 等训练型裁判不纳入比较。

**证明必要性的强启发式（作为消融，不进主表）**：①全部视图取并集、不聚类、均匀权重；②聚类后全部候选均匀加权，不做残差筛选；③随机 L 条 / Top-variance L 条；④线性聚合 vs PAOR 缺陷带 vs 门控；⑤RaR 式 LLM 标签代替频率支持度；⑥RRD 白化代替 Gram-Schmidt；⑦N∈{2,4,6,8,12}。

---

## 4. 推荐的最终实验设置

**模型与信息控制**
- 主设置 **G=J**：所有生成和评分调用都使用同一个裁判，例如 Qwen3.5-9B。次设置 G=强模型（27B），且**所有**准则类 baseline 使用同一个 G，并加入参考答案引导的 Direct。
- 裁判至少覆盖 2 个模型家族（Qwen3.5-9B + Llama-3.1-8B 或 gpt-oss-20b）和 1 个强裁判（27B 或 GPT-4o/4.1-mini，用于和 RRD 的 JudgeBench/PPE 数字对照）。
- 所有方法统一离散分数（或统一 logprob）；解析失败对所有方法统一计 correct=0；平局统一走同一套 fallback 并计入成本。

**评测口径**
- 每个数据集使用官方脚本/口径（RB2 best-of-4、JudgeBench 正反两序）；另报 strict 口径（不一致即错）和 tie=0.5 口径。
- 指标：Accuracy、位置一致率、平局/fallback 率、fallback 子集准确率；准则统计（数量、L′ 和 core_rank 分布、RubricBench 专家召回、冗余率）。

**算力公平**
- 报告调用数、输入/输出 tokens、GPU 秒，以及参数加权 token（≈FLOPs）。
- 主表之外画 accuracy-vs-compute 曲线：Direct@k、SPCT@k、RRD、MCCR(N 变化) 放在同一张图上。

**统计**
- 超参数只在单一 dev 源上调（RB2 dev 100 + HS3 dev 100，或 HS3 train 抽样），冻结后跨数据集统一使用，禁止按数据集调参。
- 3 个 seed 覆盖准则生成的随机性，报 mean±std，再加 query-level bootstrap CI。
- 主方法 vs 每个 baseline 做配对 bootstrap / McNemar，并做 Holm 校正。

**表与分析**
- 表 1：主表（RB2、HS3、JudgeBench、PPE、RubricBench × 方法），附带成本列。
- 表 2：RB2 分子集 + LLMBar + RM-Bench 诊断。
- 表 3：组件消融（第 3 节 ①–⑦）。
- 图：算力曲线、超参数敏感性（θ、τ_core、β）。
- 准则质量：RubricBench 专家召回，并以专家准则作为 oracle 上界。
- 定性错误分析，外加小规模人工评估（约 100 份准则集，评必要性、正确性、冗余）。
- 可选下游：Best-of-N 重排或 GRPO 奖励（OpenRubrics、RRD、Chasing the Tail 都有这一项，对 ICLR/NeurIPS 加分明显）。

**执行顺序**
1. 修复 P1 的公平性问题，用现有 dev 跑诊断（L′、core_rank、fallback 率、core 与先验视图的重合度），用数据确认 P2/P3。
2. 实现 MCCR v2 的对比选择与白化，以及 RRD、参考 Direct、LLM Rubrics 基线。
3. 加入 JudgeBench、PPE、RubricBench、LLMBar、RM-Bench 的 loader。
4. 跑全量 3 seed，再做消融和算力曲线。

---


---

## 实施状态（2026-11-08）

下表对照本报告的各项建议，列出已在代码中落实的内容。目前**尚无真实模型结果**，所有方法优劣结论都必须在按 `EXPERIMENT_SETUP.md` 实际运行后再下。

| 报告项 | 实施内容 | 位置 |
|---|---|---|
| P1 打分分辨率不对等 | `--score-mode discrete` 为默认值，所有准则方法统一使用离散分数；logprob 只能在所有方法都使用时开启 | `paor_pipeline.PAORPipeline._score_values` |
| P1 解析失败被静默中和 | `--missing-policy fail` 为默认值：目标回答有准则缺失时直接计 parse error，不再按 0.5 处理 | `_resolve_missing`, `decide_pair` |
| P1 强模型探测相当于参考答案 | 新增 `direct_ref`（以探测作参考答案）；主设置改为 G = J；共享探测 | `rubric_baselines.py`, `ProbeStore` |
| P1 判定协议与算力不对等 | 所有分数型方法共用同一个 fallback；记录按角色拆分的 token 和参数加权 token；新增 strict / lenient 两种口径；提供准确率–算力曲线的运行命令 | `metrics.py`, `EXPERIMENT_SETUP.md` §7 |
| P2 残差选择空转 | 诊断脚本输出 L′ 分布和秩饱和比例；v2 改为方向对齐选择，配合共线性检验 | `mccr_diagnostics.py`, `mccr_core.contrastive_select` |
| P2 方差 ≠ 判别力 | 用强 ≻ 弱、原始 ≻ 反事实两类有序对计算方向对齐度，并用它过滤错向准则（包括 core 中的准则） | `mccr_core.directional_alignment`, `MCCRPipeline._counterfactuals` |
| P3 core ≈ 先验视图 | 诊断项 `core_share_prior_echoed_once`；`new_our_method.md` 删除了"严格最小 / 无冗余"的说法 | `mccr_diagnostics.py`, `new_our_method.md` |
| P4 fallback 可能主导结果 | 新增 `fallback_accuracy` 和 `non_fallback_accuracy` 两项指标 | `metrics.py` |
| P5 工程补丁过多 | 默认改为 `--probe-recovery simple`（同一模型、一次简洁提示重试、丢弃用尽重试的槽位）；v2 去掉了 4 个超参数 | `paor_pipeline._generate_probes` |
| 1.3 白化加权与词典序聚合 | Ledoit–Wolf 白化；带可证明保证的词典序聚合 | `mccr_core.whitened_weights`, `hierarchical_aggregate` |
| §2 数据集 | 新增 RubricBench、LLMBar、RM-Bench、PPE 的 loader（JudgeBench 已有）；新增 `--split all`；RB2 best-of-4、RM-Bench 风格矩阵、分子集指标 | `paor_data.py`, `metrics.py` |
| §3 Baseline | 新增 RRD（WU / 均匀权重）、CRG、RTD、rubric_query / rubric_resp、SPCT@k、direct_point、direct_ref、expert_rubric；RubricHub 标注为 lite 版 | `rubric_baselines.py`, `baselines.py`, `docs/BASELINES.md` |
| §4 统计 | 主方法对全部其他方法做 McNemar 检验和配对 bootstrap，并做 Holm 校正；3 个 seed 的运行命令 | `run_main.significance_against_all` |
| H1 准则覆盖 | 计算 RubricBench 专家准则的召回率、精确率和冗余率 | `rubric_recall.py` |

**尚未实现：** RLCF、MRRG、TICK、RocketEval、Dynamic Rubrics training-free 部分的完整复现；准则级 pairwise 平局处理（建议 1.3 第 4 条中的第一步）；Best-of-N / GRPO 下游实验。
