"""HealthBench Lite — evaluation runner.

Reads problems from data/test.jsonl, runs agent.py on each, grades with rubrics.
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from grader import score_problem


DATA_FILE = os.environ.get("DATA_FILE", "data/test.jsonl")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "8"))


def format_conversation(messages: list[dict]) -> str:
    """Format conversation messages for the grader."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        parts.append(f"[{role}]: {content}")
    return "\n\n".join(parts)


def run_one(problem: dict, idx: int) -> dict:
    """Run agent.py on one problem and grade the response."""
    prompt_id = problem.get("prompt_id", f"problem-{idx}")
    messages = problem["prompt"]
    rubrics = problem.get("rubrics", [])

    try:
        # Run agent.py with problem as stdin
        proc = subprocess.run(
            [sys.executable, "agent.py"],
            input=json.dumps(problem),
            capture_output=True,
            text=True,
            timeout=120,
        )

        if proc.returncode != 0:
            print(f"  [{idx+1}] {prompt_id}: AGENT ERROR: {proc.stderr[:200]}", file=sys.stderr)
            return {"id": prompt_id, "score": 0.0, "error": proc.stderr[:200]}

        response = proc.stdout.strip()
        if not response:
            print(f"  [{idx+1}] {prompt_id}: EMPTY RESPONSE", file=sys.stderr)
            return {"id": prompt_id, "score": 0.0, "error": "empty response"}

        # Grade against rubrics
        conversation_str = format_conversation(messages)
        score = score_problem(conversation_str, response, rubrics)

        print(f"  [{idx+1}] {prompt_id}: {score:.4f} ({len(rubrics)} rubrics)", file=sys.stderr)
        return {"id": prompt_id, "score": score, "n_rubrics": len(rubrics)}

    except subprocess.TimeoutExpired:
        print(f"  [{idx+1}] {prompt_id}: TIMEOUT", file=sys.stderr)
        return {"id": prompt_id, "score": 0.0, "error": "timeout"}
    except Exception as e:
        print(f"  [{idx+1}] {prompt_id}: ERROR: {e}", file=sys.stderr)
        return {"id": prompt_id, "score": 0.0, "error": str(e)}


def main():
    with open(DATA_FILE) as f:
        problems = [json.loads(line) for line in f]

    print(f"Evaluating {len(problems)} problems with {MAX_WORKERS} workers...", file=sys.stderr)

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_one, p, i): i for i, p in enumerate(problems)}
        for future in as_completed(futures):
            results.append(future.result())

    # Sort by original order
    total = len(results)
    scores = [r["score"] for r in results]
    avg_score = sum(scores) / total if total > 0 else 0.0
    avg_rubrics = sum(r.get("n_rubrics", 0) for r in results) / total if total > 0 else 0.0
    errors = sum(1 for r in results if "error" in r)

    # Save detailed results
    os.makedirs("eval_results", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(f"eval_results/results_{ts}.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Print summary
    print(f"\n{'='*40}", file=sys.stderr)
    print(f"Score: {avg_score:.4f} ({total} problems, {errors} errors)", file=sys.stderr)
    print(f"{'='*40}", file=sys.stderr)

    # Machine-readable output
    print("---")
    print(f"score:            {avg_score:.4f}")
    print(f"problems:         {total}")
    print(f"avg_rubrics:      {avg_rubrics:.1f}")
    print(f"errors:           {errors}")


if __name__ == "__main__":
    main()
