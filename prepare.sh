#!/bin/bash
set -e

echo "=== HealthBench Lite: Setup ==="

# Install dependencies
pip install -r requirements.txt

# Download dataset
mkdir -p data
if [ ! -f data/healthbench_hard.jsonl ]; then
    echo "Downloading HealthBench Hard dataset..."
    curl -sL "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl" \
        -o data/healthbench_hard.jsonl
fi

# Sample 50 problems (deterministic, seed=42)
if [ ! -f data/test.jsonl ]; then
    python3 -c "
import json, random

random.seed(42)
with open('data/healthbench_hard.jsonl') as f:
    problems = [json.loads(line) for line in f]

sample = random.sample(problems, min(50, len(problems)))
with open('data/test.jsonl', 'w') as f:
    for p in sample:
        f.write(json.dumps(p) + '\n')

print(f'Sampled {len(sample)} problems from {len(problems)} total')
"
fi

echo "Ready. $(wc -l < data/test.jsonl | tr -d ' ') problems loaded."
