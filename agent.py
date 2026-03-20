"""HealthBench Lite — Health Assistant Agent

This agent receives a multi-turn health conversation and generates a response.
The response is graded against physician-written rubrics.

Modify this file to improve the score. You can change the prompt, model,
reasoning strategy, or anything else.
"""

import json
import os
import sys

from openai import OpenAI

MODEL = os.environ.get("SOLVER_MODEL", "o3")

SYSTEM_PROMPT = """You are a helpful, accurate, and empathetic health assistant.

When responding to health questions:
- Provide medically accurate information
- Be thorough but clear
- Use appropriate medical disclaimers
- Show empathy and understanding
- Address the specific question asked
- Suggest seeking professional medical advice when appropriate"""


def generate_response(messages: list[dict]) -> str:
    """Generate a response to a health conversation."""
    client = OpenAI()

    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    response = client.chat.completions.create(
        model=MODEL,
        messages=full_messages,
        reasoning_effort="high",
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    # Read problem from stdin
    problem = json.loads(sys.stdin.read())
    messages = problem["prompt"]

    response = generate_response(messages)
    print(response)
