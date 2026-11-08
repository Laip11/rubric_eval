# MCCR-Eval 当前方法与实验协议

更新时间：2026-11-08

本文档是当前的执行口径。数学定义见 `new_our_method.md`（第 4 节为 MCCR v2），审查结论和改动理由见 `docs/MCCR_REVIEW.md`，baseline 的实现细节和与原论文的偏差见 `docs/BASELINES.md`，运行命令见 `EXPERIMENT_SETUP.md`。

**与 2026-11-07 口径相比的变更：** ① 只使用离散分数（logprob 已彻底移除），解析失败直接计错；② 所有依赖探测的方法共享同一组探测；③ 新增 MCCR v2 以及 RRD 等 9 个对照方法；④ 新增 JudgeBench、PPE、RubricBench、LLMBar、RM-Bench；⑤ 新增官方口径指标、FLOPs 代理指标和 Holm 校正。旧缓存的签名已失效，不会被误用。

**2026-11-08 第二次更新（`mccr-v8-coverage-n8`）：** 先验视图和响应视图改为“覆盖优先、要求级粒度、不限条数”；默认探测数 N=8（4 强 + 4 弱）；反事实降级数默认 4；彻底移除 logprob。

## 1. 方法角色

**比较范围：只比较 training-free 方法。** 所有方法都只通过 prompt 调用同一组现成模型；不纳入任何训练或微调过的裁判、奖励模型或准则生成器（如 Rubric-RM、DeepSeek-GRM、Prometheus 2、RM-R1）。

| 方法 | 角色 | 是否使用探测 |
|---|---|---|
| `mccr_me` | **主方法**：把 rubric 构造为最小证据集合——query 视图（显式/隐含 + 原文片段校验）与 query+response 视图对比分类，见证测试检验必要性，去冗余，一轮反例检验充分性；见 `new_our_method.md` §5 | 是 |
| `mccr_v2` | 前作/消融：多视图共识 + 方向性弱监督筛选 + 白化权重 + 词典序聚合 | 是（另加反事实降级探测） |
| `mccr` | MCCR v1 原始规范（残差方差选择 + 门控），作为自身前身对照 | 是 |
| `paor` | 更早的前身（先验基 + 后验扩展 + 缺陷带） | 是 |
| `direct` | 正反两序直接 pairwise 裁判 | 否 |
| `direct_point` | 逐回答 1–10 直接打分再比较：把“pointwise 打分方式”与“准则本身”的作用分开 | 否 |
| `direct_k` / `direct_k<k>` | 采样多数投票；显式 `--direct-k` 用于画算力曲线 | 否 |
| `direct_ref` | **参考答案引导**的 direct（参考答案 = 第一个强生成器探测）：检验“探测其实充当了参考答案”这一解释 | 是 |
| `sc_principles_k<k>` | SPCT 式“原则 + 批评 + 打分”，正反交替投票 @k | 否 |
| `rubric_query` | 单次调用、只看 query 生成准则（RRD 表中的 “LLM Rubrics w/o resp.”） | 否 |
| `rubric_resp` | 单次调用、看 query + 全部探测生成准则（“LLM Rubrics w. resp.”） | 是 |
| `rrd` / `rrd_uniform` | RRD（arXiv:2602.05125）：递归分解、错向/冗余过滤、WU 白化或均匀权重 | 是 |
| `crg` | OpenRubrics 的对比式准则生成（prompt 版，用强/弱探测构成伪偏好对） | 是 |
| `rtd` | Chasing the Tail 的 Refinement-through-Differentiation | 是 |
| `expert_rubric` | **Oracle 上界**：RubricBench 专家准则 + 同一裁判 | 否 |
| `checkeval` / `dna` / `rubrichub` | 原仓库算法组件（`rubrichub` 实为 RubricHub-lite，见 `docs/BASELINES.md`） | 否 |

## 2. 公平性协议（所有方法一致）

- **共享探测。** `ProbeStore`（`src/paor_pipeline.py`）对每个 (benchmark, query, seed) 只生成一次探测。所有使用探测的方法读取同一组探测，**每个方法都计入全部探测成本**。可用 `--no-share-probes` 关闭，但不建议。
- **分数分辨率。** 所有准则类方法只使用离散分数 {0, 0.5, 1}，与 baseline 的离散输出可比。logprob 软期望已从代码中移除：它依赖分词和采样细节，部分 baseline 拿不到，也会让评估结果不稳定。裁判请求不再申请 logprob。
- **解析失败。** `--missing-policy fail`（默认）：目标回答只要有任何一条准则未解析，该回答的分数就记为缺失，相关 pair 计为 parse error（correct=0），与 baseline 一致。旧行为（缺失按 0.5 处理）需显式指定 `neutral`。探测侧的缺失分数只影响准则筛选，仍按 0.5 处理并计入 `parse_fail`。
- **平局。** 所有分数型准则方法（MCCR、PAOR、RRD、CRG、RTD、rubric_*、expert）对 |ΔU| ≤ 1e-4 的 pair 走同一个正反两序 direct fallback；fallback 内正反结论不一致时计错。Direct 类方法的正反不一致计错；`direct_point` 和 `sc_principles` 的同分按 `sha256(seed+pair_id)` 随机分配，并记录 `model_score_tie`。
- **探测恢复。** `--probe-recovery simple`（默认）：每个槽位固定使用同一模型，被截断后加一句简洁提示重试，用尽 `--probe-max-attempts` 次后丢弃该槽位；完整探测少于 `--min-probes` 时显式失败。旧的 7 级恢复链（中途换模型、换裁判、强制 JSON）保留为 `full`，只用于复现旧结果。
- **算力。** 记录每个方法每个 query 的调用数、输入/输出 tokens、墙钟时间，以及按角色拆分的 `tokens:<role>`。`--role-params strong=9 generator=27 weak=7` 给出参数加权 token（`param_tokens_per_sample`，近似 FLOPs）。主表同时报告准确率和算力，并画准确率–算力曲线（第 6 节）。
- **信息设置。** 主设置为 **G = J**：生成与评分使用同一个模型，`--generator-model-path` 指向裁判模型。次设置为 G = 强模型，此时**所有**使用生成器的方法共用同一个 G，并且必须同时报告 `direct_ref`。

## 3. 数据集

| 数据集 | loader | 规模 | 主指标 | 验证目标 |
|---|---|---|---|---|
| RewardBench 2（非 Ties） | `rewardbench2` | 1,763 query / 5,286 pair | **官方 best-of-4**（`query_all_correct`）+ 派生 pair acc + 分子集 | 客观正确性、精确 IF；准则是否有助于 Math/Factuality |
| HelpSteer3-Preference val | `helpsteer3` | 2,016 → test 1,823 非 tie | pair acc（分 domain） | 真实多领域人类偏好 |
| JudgeBench（GPT-4o 子集） | `judgebench_gpt` | 350 pair | pair acc（正反两序） | 困难正确性；可与 RRD 已报告数字对照 |
| PPE Human Preference v1 | `ppe` | 10,211 非 tie pair | pair acc（分类别） | 大规模 Arena 偏好；可与 RRD 对照 |
| RubricBench | `rubricbench` | 1,147 pair + 专家原子准则 | 官方分组 acc（IF/STEM/CODE/SAFE/CHAT）+ **专家准则召回率** | H1：准则覆盖；`expert_rubric` 给出上界 |
| LLMBar | `llmbar` | 419 pair（Natural + 4 个 Adversarial 子集） | 分子集 acc | H2：表面好但违反指令 |
| RM-Bench | `rmbench` | 1,327 prompt × 3×3 风格矩阵 = 11,941 pair | 官方 Hard/Normal/Easy（按领域平均） | H2：风格/长度偏差鲁棒性 |

数据来源和检查见 `generating_dynamic_rubric/data/README_DATA.md`。PPE 不在 git 中，运行前需执行 `python scripts/download_ppe.py`。注意 RubricBench 中有 510 条来自 HelpSteer3、132 条来自 RewardBench 2，与上述两个数据集可能重叠，因此它**不能**用于调参；它和 LLMBar、RM-Bench、JudgeBench、PPE 一样，用 `--split all` 冻结超参数后整体评估。

**划分规则：** 只有 RewardBench 2 和 HelpSteer3 有固定 dev（各 100 query，`cache/splits/*.json`），**全部超参数只在这两个 dev 上确定**；其余数据集不调参。

## 4. MCCR-ME 流程（`mccr_me`，主方法）

1. query 视图：覆盖式、要求级粒度、不限条数；每条准则标注 `explicit/implicit`，`explicit` 需给出 query 原文片段，由代码校验其确实是 query 的子串。
2. 共享探测 N=8，每条探测生成一个响应视图（覆盖式、不限条数）。
3. 聚类对齐后分为四类：`explicit`、`both`（Q∩R）直接入选；`query_implicit`（Q-only）、`response_only`（R-only）做必要性检验，每个 query 最多检验 12 条。
4. 必要性（默认 `witness`）：以强探测为底稿，生成只违反该准则的见证回答；J 在正反两序中都判底稿更好，才算必要。可切换为 `judgment`、`hitting_set`、`none` 做消融。
5. 去冗余：一次蕴含判断，加上见证阶段的 IMPOSSIBLE 信号，按优先级做线性删除极小化。
6. 充分性（默认开启）：生成满足全部准则但更差的对抗回答；若 J 确认，就提取遗漏的要求，经必要性检验后补入。
7. 用均匀权重给目标回答打分（离散分数），平局走共享 fallback。

**调用预算：** 约 30 + 3T + R（T 为被检候选数，≤ 12），再加 fallback。参数：`--me-necessity --me-max-tests --me-sufficiency`。

## 4b. MCCR v2 流程（`mccr_v2`，前作/消融）

1. 先验视图：只看 query，**尽可能覆盖**一个优秀回答必须满足的全部要求（显式约束、隐含前提、核心目标、格式/长度/语言、安全边界）。每条停留在“要求”层级：同一要求的子要点合并，不写实现细节或具体答案。**不设条数**，只保留 50 条的防失控上限，不作为超参数。
2. 共享探测 N=8（偶数槽为强生成器、奇数槽为弱模型，各 4 条；温度 0.3/0.7/1.0 循环）。
3. 每个有效探测生成一个响应视图：覆盖该样本揭示出的全部要求和失败模式，粒度与先验视图相同，**不设条数**，并禁止写针对该样本本身的准则。
4. 反事实：对全部（最多 4 个）强生成器探测各生成一份只含**一个**实质缺陷的降级版本。由此得到 4×4 个强≻弱对和 4 个原始≻反事实对，共 20 对，评分行共 12 行。
5. Qwen3-Embedding 余弦距离 + average linkage 聚类（θ=0.8），取 medoid，计算支持度 ρ，ρ ≥ 0.5 的进入 core。
6. 裁判对全部有效探测和反事实在所有 medoid 上打分；用强≻弱、原始≻反事实两类有序对计算方向对齐度 $a_r$。
7. 从 core 中移除 $a_r<-0.05$ 的错向准则；补充准则按 $a_p$ 贪心选择，带共线性检验，最多 L=4 条。
8. core 权重用 Ledoit–Wolf 收缩白化；采用词典序聚合 $U=S_{core}+c\cdot\min_k w_k/2\cdot S_{ext}$（c=0.8）。离散分数下补充项可证明不会推翻 core 上的差距。
9. 平局走共享 fallback。

v1（`mccr`）保留原规范：支持度权重、残差方差选择（只用探测行）、门控聚合，但同样运行在第 2 节的公平协议下。各组件可通过 `--mccr-selection/--mccr-core-weighting/--mccr-aggregation/--mccr-contrast/--mccr-filter-core` 单独替换，配合 `--method-suffix` 做消融。

**每 query 调用预算（N=8、R 个目标回答）：** 1（先验）+ 8（探测）+ 8（响应视图）+ 4（反事实）+ 12（探测 + 反事实评分）+ R（目标评分）= **33+R**，再加 fallback。v1 为 25+R。旧的计数式 prompt 保留为 `--view-prompts legacy`，仅用于复现旧结果。

**上下文：** `rubric_resp`、`rrd`、`paor` 会把全部探测放进同一个生成器 prompt。N=8 时如果超出生成器上下文，每条探测会被截断到相等的份额，并计入 `probe_prompt_truncated`。MCCR 的每个视图只看一条探测，不受影响。

## 5. 指标与统计

- **固定分母。** 人类 tie 在固定划分之后、推理之前过滤；所有方法使用完全相同的 pair ID 集合。缺失、解析失败和正反不一致都保留在分母中并计错（strict）。
- **同时报告：** `accuracy`（strict）、`accuracy_lenient`（未解决的平局记 0.5）、`query_all_correct`（RB2 官方 best-of-4）、`accuracy_by_subset` 及其宏平均、RM-Bench Hard/Normal/Easy、`position_consistency`、`fallback_rate` 及 fallback / 非 fallback 子集准确率、调用数 / token / 参数加权 token。
- **CI 与显著性：** query-level bootstrap（默认 1,000 次）。主方法对**每一个**其他方法做 McNemar 检验和配对 query-bootstrap 差值检验，并做 Holm 校正（`<primary>_vs_all.json`）；同时单独输出 `mccr_v2_vs_mccr`、`mccr_vs_paor`、`mccr_v2_vs_rrd`。
- **Seed：** 正式结果至少跑 3 个 seed（`--seeds 0 1 2`），报告均值 ± 标准差。确定性方法（direct、checkeval、dna、dynamic）在非零 seed 上复用 seed0 的结果。

## 6. 机制诊断与准则质量

- `src/mccr_diagnostics.py`：L′ 分布、L′=0 的比例、core 秩是否饱和、core 中含先验成员 / 只被回声一次 / 只来自响应视图的比例、fallback 比例与 fallback 子集准确率、探测丢弃数、错向 core 准则数。用于实证检验审查报告中的 P2–P4。
- `src/rubric_recall.py`：在 RubricBench 上计算各方法生成准则对专家准则的召回率、精确率和冗余率（嵌入匹配，或 LLM 匹配器）。

## 7. 当前边界

- 本仓库的测试只覆盖逻辑（用脚本化的假模型池），**尚无任何真实模型结果**。任何关于方法优劣的结论都需要按 `EXPERIMENT_SETUP.md` 实际运行后再下。
- RRD、OpenRubrics CRG、Chasing the Tail RTD、SPCT 均为按论文描述的 prompt 版复现，与原文的偏差列在 `docs/BASELINES.md`。
- 长文评测（LongJudgeBench 等）需要单独的上下文预算，不与当前 8K/4K 协议混跑。Pointwise 仍暂停。
