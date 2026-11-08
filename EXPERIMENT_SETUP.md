# MCCR-Eval 实验运行手册

更新时间：2026-11-08

方法协议见 `method.md`；MCCR v1/v2 的数学定义见 `new_our_method.md`；审查报告见 `docs/MCCR_REVIEW.md`；baseline 的实现细节见 `docs/BASELINES.md`。

> 所有结论都必须来自实际运行。本仓库的单元测试只覆盖逻辑（使用脚本化的假模型池）。

## 0. 公共变量

```bash
cd /data/project/sustech/rubric_eval
M=/data/project/sustech/model
J=$M/Qwen3.5-9B            # 裁判
G=$M/Qwen3.8-27B           # 强生成器（次设置）
W=$M/Qwen2.5-7B-Instruct   # 弱探测
```

公共参数：

```bash
COMMON="--backend offline --model-path $J --strong-model qwen35-9b \
  --weak-model-path $W --weak-model qwen25-7b \
  --judge-gpus 0 1 2 3 4 5 --generator-gpus 0 1 2 3 4 5 --weak-gpus 0 1 2 3 4 5 \
  --missing-policy fail --probe-recovery simple \
  --bootstrap 1000 --skip-existing"
```

方法组：

```bash
MAIN="mccr_v2 mccr paor direct direct_point direct_k sc_principles direct_ref \
  rubric_query rubric_resp rrd rrd_uniform crg rtd checkeval rubrichub"
```

## 1. 冒烟测试（2 条 query，只检查流程和输出结构）

PPE 原始数据不在 git 中，首次运行前先执行：

```bash
python scripts/download_ppe.py
```

```bash
python src/run_main.py $COMMON \
  --generator-model-path $J --generator-model qwen35-9b --role-params strong=9 generator=9 weak=7 \
  --benchmarks rewardbench2 helpsteer3 rubricbench --methods $MAIN \
  --seeds 0 --split test --max-queries 2 --bootstrap 100 \
  --output-root outputs/v2_smoke --cache-root cache/v2_smoke
```

`expert_rubric` 只能在 RubricBench 上运行（其他数据集没有专家准则，会显式报错），因此要单独运行（第 4 节）。G = J 时生成器上下文同样需要 ≥ query + N×probe + 2048 tokens（`rubric_resp/rrd/paor` 会把全部探测放进一个 prompt）；如果裁判模型的上下文不够，请调小 `--probe-max-tokens` 或 `--N`。

## 2. Dev：调参与机制诊断（只用 RB2 / HS3 的 dev）

```bash
python src/run_main.py $COMMON \
  --generator-model-path $J --generator-model qwen35-9b --role-params strong=9 generator=9 weak=7 \
  --benchmarks rewardbench2 helpsteer3 --methods mccr_v2 mccr paor rrd direct \
  --seeds 0 --split dev --output-root outputs/v2_dev --cache-root cache/v2_dev
python src/mccr_diagnostics.py --output-root outputs/v2_dev --methods mccr mccr_v2
```

在 `MCCR_DIAGNOSTICS.md` 中重点检查以下几项，用于实证确认或推翻审查报告的 P2–P4：
- `share_L_prime_zero` 和 `share_rank_saturated`（P2：v1 的残差选择是否基本失效）；
- `core_share_prior_echoed_once`（P3：core 是否约等于先验视图）；
- `fallback_rate` 与 `fallback_accuracy`（P4：结果是否由 fallback 主导）。

只在这一步调整 `--theta-sim --tau-core --tau-align --max-collinearity --n-counterfactual --L`，确定后冻结。

## 3. 主表：G = J（主设置）

```bash
python src/run_main.py $COMMON \
  --generator-model-path $J --generator-model qwen35-9b --role-params strong=9 generator=9 weak=7 \
  --benchmarks rewardbench2 helpsteer3 --methods $MAIN \
  --seeds 0 1 2 --split test --output-root outputs/v2_main_gj --cache-root cache/v2_main_gj

python src/run_main.py $COMMON \
  --generator-model-path $J --generator-model qwen35-9b --role-params strong=9 generator=9 weak=7 \
  --benchmarks judgebench_gpt rubricbench llmbar rmbench ppe --methods $MAIN \
  --seeds 0 1 2 --split all --output-root outputs/v2_main_gj --cache-root cache/v2_main_gj
```

PPE（10,211 对）和 RM-Bench（11,941 对）开销最大。算力不足时，先用 `--max-queries 2000` 跑固定的前 2,000 条 query（qid 按哈希排序，近似随机），并在论文中写明。

## 4. 准则质量与 Oracle 上界（RubricBench）

```bash
python src/run_main.py $COMMON \
  --generator-model-path $J --generator-model qwen35-9b \
  --benchmarks rubricbench --methods expert_rubric --seeds 0 --split all \
  --output-root outputs/v2_main_gj --cache-root cache/v2_main_gj

python src/rubric_recall.py --output-root outputs/v2_main_gj \
  --embedding-model $M/Qwen3-Embedding-0.6B \
  --methods mccr_v2 mccr paor rrd rrd_uniform rubric_query rubric_resp crg rtd checkeval rubrichub
```

如需 LLM 匹配器（更可靠），先启动一个 OpenAI 兼容服务，然后加 `--matcher llm --endpoint http://127.0.0.1:8000/v1 --model <served-name>`。

## 5. 次设置：G = 强生成器（信息控制）

```bash
python src/run_main.py $COMMON \
  --generator-model-path $G --generator-model qwen38-27b --role-params strong=9 generator=27 weak=7 \
  --generator-max-model-len 40960 \
  --benchmarks rewardbench2 helpsteer3 judgebench_gpt --methods $MAIN \
  --seeds 0 --split test --output-root outputs/v2_main_g27 --cache-root cache/v2_main_g27
```

`judgebench_gpt` 没有 test 划分之外的 dev，这里使用 `--split test`，只排除 10% 的 dev；如需全集，请单独用 `--split all` 运行。审读结果时必须对比 `mccr_v2` 与 `direct_ref`：如果 `direct_ref` 已经拿到大部分增益，MCCR 的贡献就主要来自强模型提供的信息，而不是准则结构本身。

## 6. 组件消融（dev 上选设计，test 上报告）

每个变体用 `--method-suffix` 生成独立的方法名和缓存。探测在同一 cache root 内共享，所以每个变体只新增准则评分和目标评分的调用。

```bash
ABL="$COMMON --generator-model-path $J --generator-model qwen35-9b --role-params strong=9 generator=9 weak=7 \
  --benchmarks rewardbench2 helpsteer3 llmbar --methods mccr_v2 --seeds 0 --split test \
  --output-root outputs/v2_ablation --cache-root cache/v2_ablation"
python src/run_main.py $ABL                                                    # full v2
python src/run_main.py $ABL --method-suffix _residual   --mccr-selection residual
python src/run_main.py $ABL --method-suffix _nosel      --mccr-selection none
python src/run_main.py $ABL --method-suffix _roles      --mccr-contrast roles
python src/run_main.py $ABL --method-suffix _cf         --mccr-contrast counterfactual
python src/run_main.py $ABL --method-suffix _nofilter   --mccr-filter-core off
python src/run_main.py $ABL --method-suffix _support    --mccr-core-weighting support
python src/run_main.py $ABL --method-suffix _uniform    --mccr-core-weighting uniform
python src/run_main.py $ABL --method-suffix _gated      --mccr-aggregation gated
python src/run_main.py $ABL --method-suffix _linear     --mccr-aggregation linear
for n in 4 6 12; do python src/run_main.py $ABL --method-suffix _N$n --N $n --n-counterfactual $((n / 2)); done   # 默认 N=8
python src/run_main.py $ABL --method-suffix _legacyviews --view-prompts legacy   # 旧的计数式视图 prompt
```

`--N` 不同时探测签名也不同，会重新生成探测，这一点在设计上是有意的。v1 组件的消融可用 `--methods mccr --method-suffix ...` 实现，同样的参数会作用到 v1 上。旧的 `src/mccr_coverage_ablation.py`（只调 τ_core / τ_var / L）仍可用于 v1。

## 7. 准确率–算力曲线

```bash
CURVE="$COMMON --generator-model-path $J --generator-model qwen35-9b --role-params strong=9 generator=9 weak=7 \
  --benchmarks rewardbench2 helpsteer3 --seeds 0 --split test \
  --output-root outputs/v2_compute --cache-root cache/v2_compute"
for k in 2 4 8 16 32; do python src/run_main.py $CURVE --methods direct_k --direct-k $k; done
for k in 1 4 8 16; do python src/run_main.py $CURVE --methods sc_principles --sc-k $k; done
for n in 4 6 8 12; do python src/run_main.py $CURVE --methods mccr_v2 rrd --N $n --n-counterfactual $((n / 2)) --method-suffix _N$n; done
```

以每个方法的 `param_tokens_per_sample`（或 `avg_*_tokens_per_sample`）为横轴、`accuracy` 为纵轴作图。注意：`rrd` 的输出名不带 N 后缀，若要把不同 N 的 `rrd` 都放进同一张图，需为每个 N 使用不同的 `--output-root`。

## 8. 汇总

```bash
python src/summarize_results.py --output-root outputs/v2_main_gj
python src/mccr_diagnostics.py  --output-root outputs/v2_main_gj --methods mccr mccr_v2
```

`SUMMARY.md` 包含以下内容：strict / lenient 准确率、RB2 best-of-4、位置一致率、fallback 及其子集准确率、分子集准确率、RM-Bench Hard/Normal/Easy、调用数、token、参数加权 token，以及 `<primary>_vs_all.json` 中经 Holm 校正的配对检验。

## 9. 断点续跑与缓存

- 保持命令不变，并加 `--skip-existing`。完整的 `seed*.json` 会被直接复用；准则类方法按 query 缓存各阶段的中间结果。
- 共享探测存放在 `cache/<root>/probes/<benchmark>/seed*/`，签名包括生成器、弱模型、N、长度、恢复模式和尝试次数。
- 方法缓存的签名包括全部协议设置（score/missing/recovery）和方法超参数。签名一变，就不会复用旧缓存。
- 旧的 `outputs/mccr_pairwise_v1` 等目录使用旧协议（软分数、缺失按 0.5 处理、7 级恢复），**不能**与新结果混算。

## 10. 验证

```bash
python -m unittest discover -s tests -v
python -m compileall -q src
```

测试覆盖以下内容：
- 协议：只用离散分数（CLI 中已无 logprob 选项）、缺失即失败、简单探测恢复、共享探测成本、多探测 prompt 的上下文截断；
- MCCR v2：方向对齐、Ledoit–Wolf 白化、对比式选择、词典序保证，以及 v1/v2 的端到端运行；
- 新 baseline：RRD 错向过滤、CRG、RTD、rubric_*、direct_ref、expert、SPCT 投票、direct_point；
- 新指标：lenient、best-of-N、分子集、RM-Bench 风格矩阵、FLOPs 代理指标、Holm 校正、配对 bootstrap；
- 新数据：RubricBench、LLMBar、RM-Bench、PPE 的规模和结构；
- 以及原有测试。
