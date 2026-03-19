#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# Check data exists
if [ ! -f data/test.jsonl ]; then
    echo "ERROR: data/test.jsonl not found. Run 'bash prepare.sh' first." >&2
    exit 1
fi

# Check OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set." >&2
    exit 1
fi

python3 eval/run_all.py
