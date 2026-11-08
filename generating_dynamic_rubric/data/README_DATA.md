# 数据来源

所有文件均为官方发布的原始数据，未做任何修改，直接以普通 git blob 存放（不使用 LFS）。唯一的例外是 PPE：它的原始 Arena prompt 里有一段字符串会被 GitHub secret scanning 识别为 API key，因此不能提交，需要用脚本下载。
加载逻辑见 `src/paor_data.py`。各数据集的许可证以原始发布方为准，使用前请分别核对。

| 目录 | 来源 | 获取方式 | loader | 说明 |
|---|---|---|---|---|
| `rewardbench2/test.parquet` | [allenai/reward-bench-2](https://huggingface.co/datasets/allenai/reward-bench-2) | 原有 | `rewardbench2` | 不含 Ties 子集 |
| `helpsteer3/preference_validation.jsonl.gz` | [nvidia/HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) | 原有 | `helpsteer3` | 只用官方 validation |
| `judgebench/{gpt,claude}.jsonl` | [ScalerLab/JudgeBench](https://huggingface.co/datasets/ScalerLab/JudgeBench) | 原有 | `judgebench_gpt` / `judgebench_claude` | GPT-4o 子集共 350 对 |
| `rubricbench/train.parquet` | [DonJoey/rubricbench](https://huggingface.co/datasets/DonJoey/rubricbench)（arXiv:2603.01562） | 2026-11-08 下载 | `rubricbench` | 1,147 对；`label` 为 0 表示 response_a 更好（依据官方 `eval_submission.py`）；含专家原子准则；子集按官方分组（IF/STEM/CODE/SAFE/CHAT） |
| `llmbar/<subset>/dataset.json` | [princeton-nlp/LLMBar](https://github.com/princeton-nlp/LLMBar)：`Dataset/LLMBar/{Natural, Adversarial/*}` | 2026-11-08 下载 | `llmbar` | 419 对；`label` 为 1 表示 output_1 更好 |
| `rmbench/total_dataset.json` | [THU-KEG/RM-Bench](https://huggingface.co/datasets/THU-KEG/RM-Bench) | 2026-11-08 下载 | `rmbench` | 1,327 个 prompt，3 个 chosen × 3 个 rejected 风格变体；有 2 个 chosen 与 rejected 文本完全相同的组合被跳过，剩 11,941 对 |
| `ppe/human_preference_v1_test.parquet` | [lmarena-ai/PPE-Human-Preference-V1](https://huggingface.co/datasets/lmarena-ai/PPE-Human-Preference-V1) | **不在 git 中**：运行 `python scripts/download_ppe.py`（校验 SHA-256） | `ppe` | 16,038 行；过滤 tie、tie (bothbad) 和两条回答完全相同的样本后剩 10,211 对；`response_1` 对应 `model_a` |

**数据重叠：** RubricBench 中有 510 对来自 HelpSteer3，132 对来自 RewardBench 2。因此 RubricBench 不能用于调参，只能在冻结超参数后整体评估（`--split all`）。
