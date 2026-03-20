"""HealthBench grader — evaluates a response against physician-written rubrics."""

import json
import os

from openai import OpenAI

GRADER_MODEL = os.environ.get("GRADER_MODEL", "gpt-4.1-mini")

GRADER_TEMPLATE = """You are a medical expert evaluating an AI health assistant's response.

## Conversation
{conversation}

## AI Assistant's Response
{response}

## Criterion to Evaluate
{criterion}

## Task
Determine if the AI assistant's response meets the above criterion.
Respond with a JSON object:
{{"criteria_met": true/false, "explanation": "brief explanation"}}

Only output the JSON object, nothing else."""


def grade_rubric(conversation_str: str, response: str, criterion: str) -> bool:
    """Grade a single rubric criterion. Returns True if met."""
    client = OpenAI()

    prompt = GRADER_TEMPLATE.format(
        conversation=conversation_str,
        response=response,
        criterion=criterion,
    )

    for attempt in range(3):
        try:
            result = client.chat.completions.create(
                model=GRADER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            text = result.choices[0].message.content.strip()
            # Parse JSON from response
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text)
            return bool(parsed.get("criteria_met", False))
        except (json.JSONDecodeError, Exception):
            if attempt == 2:
                return False
    return False


def score_problem(conversation_str: str, response: str, rubrics: list[dict]) -> float:
    """Score a response against all rubrics for a problem.

    Returns a score in [0, 1] = sum(met positive points) / sum(all positive points),
    with negative rubric points subtracted from numerator if met.
    """
    positive_total = 0.0
    earned = 0.0

    for rubric in rubrics:
        points = rubric.get("points", 1.0)
        criterion = rubric.get("criterion", "")
        met = grade_rubric(conversation_str, response, criterion)

        if points > 0:
            positive_total += points
            if met:
                earned += points
        elif points < 0:
            # Negative rubrics: subtract from earned if met (penalize undesirable behavior)
            if met:
                earned += points  # points is negative, so this subtracts

    if positive_total == 0:
        return 0.0

    return max(0.0, min(1.0, earned / positive_total))
