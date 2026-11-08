# MCCR-Eval Pairwise 实验运行手册

更新时间：2026-11-07

本文档给出当前唯一有效的主实验运行方式。方法协议见 `method.md`，MCCR 数学定义见 `new_our_method.md`。

## 1. 主实验范围

| 数据集 | 原始范围 | Dev | Test query | Test 非 tie pair | 人类 tie |
|---|---|---:|---:|---:|---:|
| RewardBench 2（非 Ties） | 1,763 query / 5,286 pair | 100 | 1,663 | 4,986 | 0 |
| HelpSteer3-Preference validation | 2,016 query / 2,016 pair | 100 | 1,823 | 1,823 | 93（推理前过滤） |

- RewardBench 2 使用正确–错误派生 pairwise，不报告官方 best-of-4 分数。
- HelpSteer3 只使用官方 validation，在其内部按 domain 分层划分；不读取 train。
- 当前只运行 Pairwise、seed0，不设置 `--max-queries` 时即为全部 test query。
- 默认主方法是 `mccr`；`paor` 是旧方法对照，不是原论文方法。
- CheckEval、DnA、RubricHub 使用 `generating_dynamic_rubric/src` 中的原始算法组件和离线批处理适配器。

## 2. 模型、GPU 与长度

| 角色 | 模型 | GPU | 上下文 |
|---|---|---|---:|
| `G` | Qwen3.8-27B | GPU0–5，按生成批次动态加载 | 40,960 |
| `J` | Qwen3.5-9B | GPU0–5，按裁判批次动态加载 | 32,768 |
| 弱探测 | Qwen2.5-7B-Instruct | GPU0–5，按弱探测批次动态加载 | 16,384 |
| 嵌入 | Qwen3-Embedding-0.6B | CPU | 默认 |

统一长度预算：

```text
query_max_tokens=8192
response_max_tokens=4096
probe_max_tokens=4096
probe_max_attempts=7
```

达到输出上限或为空的探测会重新生成；第 2 次要求不超过 1,200 tokens，第 3–4 次降温到 0.2，并要求不超过 3,000 tokens 直接给出最终答案，长序列用范围/省略号，多项请求保持显式数量但压缩每项解释。第 4 次在强生成器与弱模型之间切换，第 5 次改用裁判模型并限制 800 tokens 内给出最终回答；第 6–7 次把原会话标为数据，使用 guided JSON Schema 强制 `answer/explanation/confidence` 结构，答案/解释最多 200/500 字符，硬上限 1,024 tokens。实际 `probe_roles` 和切换 slot 会写入诊断。连续 7 次无效则运行失败，不会把截断探测用于聚类或评分。失败原文与 token/finish reason 保存在 `probe_failure_records`，断点恢复同样受最大尝试次数约束。准则 JSON 允许初始生成加 2 次确定性重试；MCCR prior 输出上限为 1,024 tokens。malformed JSON 仅恢复完整闭合对象并记录警告，不保留半截准则；批处理先保存全部成功结果，再报告失败并重跑缺失项。

MCCR 默认参数：

```text
N=6
L=4
theta_sim=0.80
tau_core=0.50
beta=0.40
tau_var=0.04
rho_min=0.20
gamma=2.0
eta=0.5
```

旧 PAOR 对照仍使用 `c=0.8`、`tau_d=0.15`、`tau_rho=0.85`、`lambda=0.5`。

## 3. 数据文件

真实数据已下载到：

```text
generating_dynamic_rubric/data/
  rewardbench2/test.parquet
  helpsteer3/preference_validation.jsonl.gz
```

加载器会检查 Git LFS 指针、RewardBench 2 的 subset+ID、重复 rejected 以及 HelpSteer3 的退化文本 pair。不要在未重新校验的情况下手工替换这些文件。

## 4. 小规模真实模型检查

```bash
cd /data/project/sustech/rubric_eval

python src/run_main.py \
  --backend offline \
  --benchmarks rewardbench2 helpsteer3 \
  --methods mccr paor direct checkeval dna rubrichub direct_k \
  --seeds 0 --split test --max-queries 2 \
  --model-path /data/project/sustech/model/Qwen3.5-9B \
  --strong-model qwen35-9b --judge-gpus 1 2 3 4 \
  --generator-model-path /data/project/sustech/model/Qwen3.8-27B \
  --generator-model qwen38-27b --generator-gpus 0 \
  --weak-model-path /data/project/sustech/model/Qwen2.5-7B-Instruct \
  --weak-model qwen25-7b --weak-gpus 5 \
  --output-root outputs/mccr_smoke \
  --cache-root cache/mccr_smoke \
  --bootstrap 100 --skip-existing \
  > outputs/mccr_smoke.log 2>&1
```

该结果只验证推理和输出结构，不能用于方法优劣结论。

## 5. 全量主实验

```bash
python src/run_main.py \
  --backend offline \
  --benchmarks rewardbench2 helpsteer3 \
  --methods mccr paor direct checkeval dna rubrichub direct_k \
  --seeds 0 \
  --split test \
  --model-path /data/project/sustech/model/Qwen3.5-9B \
  --strong-model qwen35-9b \
  --judge-gpus 1 2 3 4 \
  --generator-model-path /data/project/sustech/model/Qwen3.8-27B \
  --generator-model qwen38-27b \
  --generator-gpus 0 \
  --weak-model-path /data/project/sustech/model/Qwen2.5-7B-Instruct \
  --weak-model qwen25-7b \
  --weak-gpus 5 \
  --output-root outputs/mccr_pairwise_v1 \
  --cache-root cache/mccr_pairwise_v1 \
  --bootstrap 1000 \
  --skip-existing \
  > outputs/mccr_pairwise_v1.log 2>&1
```

不需要启动 HTTP server。同一数据集内的 7 个方法使用协程并发。GPU0–5 都是通用 worker：每个角色批次切分到全部可用 GPU，按当前角色加载 G/J/weak；角色变化时显式关闭旧引擎、释放显存再加载新模型。`--judge-gpus`、`--generator-gpus`、`--weak-gpus` 的并集决定可用 worker，不再代表静态模型分配。同一方法内部存在依赖的 prior/probe/response-view/judge 阶段仍保持顺序。

只跑主方法与原始 rubric baselines时，可省略旧 PAOR：

```bash
python src/run_main.py \
  --backend offline \
  --benchmarks rewardbench2 helpsteer3 \
  --methods mccr direct checkeval dna rubrichub direct_k \
  --seeds 0 --split test \
  --model-path /data/project/sustech/model/Qwen3.5-9B \
  --strong-model qwen35-9b --judge-gpus 1 2 3 4 \
  --generator-model-path /data/project/sustech/model/Qwen3.8-27B \
  --generator-model qwen38-27b --generator-gpus 0 \
  --weak-model-path /data/project/sustech/model/Qwen2.5-7B-Instruct \
  --weak-model qwen25-7b --weak-gpus 5 \
  --output-root outputs/mccr_pairwise_v1 \
  --cache-root cache/mccr_pairwise_v1 \
  --bootstrap 1000 --skip-existing
```

## 6. 断点续跑与缓存规则

保留完全相同的命令和 `--skip-existing`：

- 完整 method/seed JSON 会直接复用；
- MCCR/PAOR 按 query 缓存中间阶段；
- `cost_checkpoint` 累计跨进程调用/token，`wall-*.json` 累计方法墙钟；断点恢复只添加新增成本；
- 缓存签名包括模型、N/L、长度和全部方法超参数；
- 更换数据、prompt、模型、长度或超参数后，应使用新的 output/cache root；
- 新探测长度与旧 512-token 缓存不兼容，不能混用；
- 正式结果目录与 smoke 目录必须隔离。

## 7. 汇总

```bash
python src/summarize_results.py \
  --output-root /data/project/sustech/rubric_eval/outputs/mccr_pairwise_v1
```

输出结构：

```text
outputs/mccr_pairwise_v1/
  manifest.json
  SUMMARY.md
  summary.json
  summary.csv
  rewardbench2/<method>/seed0.json
  rewardbench2/<method>/seed0.metrics.json
  rewardbench2/mccr_vs_strongest.json
  rewardbench2/mccr_vs_paor.json
  helpsteer3/<method>/seed0.json
  helpsteer3/<method>/seed0.metrics.json
  helpsteer3/mccr_vs_strongest.json
  helpsteer3/mccr_vs_paor.json
  diagnostics/<benchmark>.<method>.seed0.jsonl
```

主表同时报告 Accuracy、query bootstrap CI、fallback、正反顺序不一致、模型同分、解析失败、调用和 token 成本。人类 gold tie 在固定划分后、推理前统一过滤；所有方法使用完全相同的 pair ID 集合。Accuracy 的分母固定为全部评估 pair，模型无法区分、顺序冲突、缺预测和解析失败均按 correct=0，不使用 valid-only accuracy。普通 direct 每对运行正反两次；`direct_k` 是额外的多次采样预算基线。MCCR 是否优于 PAOR 必须依据相同数据上的准确率、显著性和成本共同判断。

Pointwise 继续暂停。恢复时必须对所有方法统一 logprob/软期望协议；若任何方法无法提供可比 logprob，则全部使用离散分数，禁止只对 MCCR/PAOR 使用软期望后与离散 baseline 比较。

## 8. 验证

```bash
python -m unittest discover -s tests -v
python -m compileall -q src
```

当前测试覆盖 MCCR 聚类支持度、残差选择、门控、截断探测重采样、新数据数量/去重、顺序不一致、固定分母以及消融缓存重选。

### Dev 覆盖率消融

默认 dev 对照完成后，运行：

```bash
python src/mccr_coverage_ablation.py \
  --output-root outputs/mccr_dev_ablation_v1 \
  --cache-root cache/mccr_dev_ablation_v1 \
  --benchmarks rewardbench2 helpsteer3 --bootstrap 1000
```

比较 `(tau_core=0.50,tau_var=0.04,L=4)`、`(0.30,0.04,4)`、`(0.50,0.02,8)` 和 `(0.30,0.02,8)`。变体复用探测、原始候选、聚类及全部 medoid 探测评分，只新增目标评分和 fallback；报告的新增成本不能解释为冷启动总成本。结果和完整 query/rubric 示例写入 `DEV_COVERAGE_ABLATION.md`，不根据 test 结果选择配置。

## 9. 结果解释限制

- 当前仅 seed0，不能报告跨 seed 稳定性。
- 新默认超参数仍需开发集验证，不能在测试集上调参。
- RewardBench 2 的 Ties 子集未进入普通 pairwise 主表。
- `direct_k` 是近似调用数对照，不是严格 token/FLOP 对照。
- LongJudgeBench 的 4K–36K 长文需要另一套上下文预算，不能套用当前 8K/4K 配置。
- 旧 `outputs/full_pairwise_qwen` 存在数据与协议差异，不与本轮直接比较或混算。
