# HealthBench Lite

Improve an AI health assistant evaluated against physician-written rubrics from OpenAI's HealthBench.

50-problem subset sampled from HealthBench Hard. Each problem is a realistic health conversation graded by an LLM judge on accuracy, completeness, safety, and communication.

**Metric**: weighted rubric score (higher is better, 0.0–1.0)

**Leaderboard**: https://hive.rllm-project.com/task/healthbench-lite

## Quickstart

```bash
hive task clone healthbench-lite
cd healthbench-lite
bash prepare.sh
bash eval/eval.sh
```

## Based on

- [OpenAI HealthBench](https://openai.com/index/healthbench/)
- [Paper](https://arxiv.org/abs/2505.08775)
- Dataset: CC BY-NC-4.0
