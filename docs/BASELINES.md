# Baseline 实现说明

所有准则类 baseline 与 MCCR 在以下方面保持一致：
- **共享探测：** 使用 `ProbeStore` 生成的同一组探测，并完整计入探测成本；
- **共享裁判：** 使用同一个原子裁判 prompt（`PROMPT_ATOMIC_JUDGE`），每个目标回答调用一次，分数取 0/1/2；
- **共享协议：** 采用同一套离散分数和"缺失即失败"规则，平局走同一个正反两序 direct fallback。

因此，方法之间的差异只来自**准则如何构造和加权**。下表中的"偏差"是指与原论文设置的不同，这些不同在写论文时必须如实报告。

| 方法 | 论文 | 官方代码 | 本仓库实现 | 与原文的偏差 |
|---|---|---|---|---|
| `rrd`, `rrd_uniform` | Shen et al., *Rethinking Rubric Generation for Improving LLM Judge and Reward Modeling for Open-ended Tasks*, [arXiv:2602.05125](https://arxiv.org/abs/2602.05125) | 未找到 | `src/rubric_baselines.py::_build_rrd`。流程：<br>① 用 query 和全部探测生成初始准则；<br>② 对满足它的探测数 > 2 的准则逐条分解为 2–4 条更细的准则；<br>③ 若某条准则在弱模型探测上的平均分高于强模型探测，判为错向并丢弃；<br>④ 与已保留准则的嵌入余弦 ≥ 0.9 时判为冗余（不与自身的父准则比较）；<br>⑤ 至少有一个子准则被接受时，删除父准则；<br>⑥ 最多 2 轮，或拒绝数超过 15 时停止；<br>⑦ WU 权重 Σ^{-1/2}1（Ledoit–Wolf 收缩），或均匀权重 | 原文使用 m=8 个样本（GPT-4o / Gemini 生成），错向检测用 GPT-4o vs Llama3-8B，冗余判定由 LLM 完成，没有明确的轮数上限。<br>本仓库使用 N=6 个共享探测（强生成器 vs 弱模型），用嵌入判定冗余，最多 2 轮 |
| `crg` | Liu et al., *OpenRubrics*, [arXiv:2510.07743](https://arxiv.org/abs/2510.07743) | [HF: OpenRubrics](https://huggingface.co/OpenRubrics) | 以强探测为 preferred、弱探测为 rejected 构成伪偏好对，生成 hard rules（1–5 条）和 principles（2–6 条）。hard rules 作为 core，principles 作为补充，用词典序带聚合 | 原文用真实偏好标签构造对比对，并做偏好-标签一致性过滤；测试时没有标签，因此改用伪标签，也不做一致性过滤。另外，可把官方的 Rubric-RM-8B 作为训练型参考行单独报告（不计入同算力比较） |
| `rtd` | Zhang et al., *Chasing the Tail*, [arXiv:2509.21500](https://arxiv.org/abs/2509.21500)（ICLR 2026） | [github.com/Jun-Kai-Zhang/rubrics](https://github.com/Jun-Kai-Zhang/rubrics) | 从只看 query 的准则出发，每轮用当前准则给探测打分，取得分最高的两条探测让 proposer 细化准则，共 2 轮，均匀权重 | 原文的候选池是强模型输出或改写版本，用于训练奖励；本仓库只用共享探测，并且只用于评测 |
| `rubric_query` / `rubric_resp` | RRD 论文中的 "LLM Rubrics w/o / w. resp." 基线 | — | 单次生成调用，均匀权重 | — |
| `sc_principles_k<k>` | Liu et al., *Inference-Time Scaling for Generalist Reward Modeling*（DeepSeek-GRM / SPCT），[arXiv:2504.02495](https://arxiv.org/abs/2504.02495) | 模型已在 HF 公开 | prompt 格式为"原则 → 批评 → `\boxed{s1, s2}`"；采样 k 次，正反顺序交替，分数求和投票 | 没有经过 SPCT 训练，也没有 meta RM。这里只是同算力下的推理时扩展对照 |
| `direct_ref` | 参考答案引导的 LLM-as-a-Judge（MT-Bench / JudgeBench 惯例） | — | 参考答案 = 第一个强生成器探测；正反两序比较，不一致即判错 | — |
| `direct_point` | — | — | 逐回答 1–10 打分后再比较 | — |
| `expert_rubric` | RubricBench, [arXiv:2603.01562](https://arxiv.org/abs/2603.01562) | [github.com/planepig/rubricbench](https://github.com/planepig/rubricbench) | 用专家原子准则替代生成准则，裁判和协议不变，均匀权重 | Oracle 上界，不属于可比方法 |
| `checkeval` | Lee et al., *CheckEval*, [arXiv:2403.18771](https://arxiv.org/abs/2403.18771) | 原仓库组件 | 每个数据集一份 20 题静态清单 | 清单生成只调用一次，没有摊入每个 query 的成本 |
| `dna` | Li et al., *Decompose and Aggregate (DnA-Eval)* | 原仓库组件 | aspect → 权重 → 逐 aspect 打分 | — |
| `rubrichub` | *RubricHub*, [arXiv:2601.08430](https://arxiv.org/abs/2601.08430) | 原仓库组件 | 由参考答案生成带权准则，再逐条判定 | **只是 RubricHub-lite**：没有实现原文的 coarse-to-fine 多模型聚合和难度演化。论文中应写作 "RubricHub-lite (reference-grounded weighted rubric)" |

## 可选，尚未实现

- **RLCF** candidate-based checklist，[arXiv:2507.18624](https://arxiv.org/abs/2507.18624)，[代码](https://github.com/viswavi/RLCF)
- **MRRG**（多角色准则生成），[arXiv:2607.01830](https://arxiv.org/abs/2607.01830)，未找到代码
- **TICK**，[arXiv:2410.03608](https://arxiv.org/abs/2410.03608)
- **RocketEval**，[arXiv:2503.05142](https://arxiv.org/abs/2503.05142)
- **Prometheus 2**：已有原仓库的 `eval_prometheus2.py`，尚未接入 `run_main`
- 来源论文 **Dynamic Rubrics**，[arXiv:2605.30568](https://arxiv.org/abs/2605.30568)：提供 DPO 生成器 checkpoint，可作为训练型参考行

## 预算口径

所有方法记录的成本都包含以下部分：自己的全部调用、共享探测、重试和 fallback。参数加权 token 用 `--role-params` 计算。`direct_k` 在默认情况下按 MCCR 的理论调用数近似对齐，但这**不是** FLOPs 对齐；严格的比较请使用第 7 节的准确率–算力曲线（见 `EXPERIMENT_SETUP.md`）。
