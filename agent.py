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

MODEL = os.environ.get("SOLVER_MODEL", "gpt-4.1")

SYSTEM_PROMPT = """You are a helpful, accurate, and empathetic health assistant.

When responding to health questions:
- Provide medically accurate information
- Be thorough but clear
- Use appropriate medical disclaimers
- Show empathy and understanding
- Address the specific question asked
- Suggest seeking professional medical advice when appropriate"""

CRITIQUE_PROMPT = """You are a physician reviewing an AI health assistant's response to a patient conversation. Identify specific problems:

1. ACCURACY: Any medically incorrect statements? Outdated information? Unsupported claims?
2. MISSING INFO: Did it fail to ask important clarifying questions? Miss key differential diagnoses or safety concerns?
3. OVERSTEPPING: Did it fabricate clinical details not provided? Give overly definitive diagnoses? Include irrelevant tangential information?
4. SAFETY: Any potentially harmful advice? Did it fail to flag red flags or urgent situations?
5. COMMUNICATION: Is the tone appropriate? Is it too long or too short?

List the specific issues that need fixing. Be concrete."""

REFINE_PROMPT = """You are a helpful, accurate, and empathetic health assistant.

You previously gave a response that had some issues. Here is the critique:
<critique>
{critique}
</critique>

Now provide an improved response that addresses these issues. Respond directly to the patient — do not reference the critique or your previous response."""


def generate_response(messages: list[dict]) -> str:
    """Generate a response using self-refine: generate, critique, then improve."""
    client = OpenAI()

    # Step 1: Generate initial response
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    initial = client.chat.completions.create(
        model=MODEL,
        messages=full_messages,
        temperature=0.3,
        max_tokens=2048,
    )
    initial_response = initial.choices[0].message.content

    # Step 2: Critique the response
    conversation_text = "\n".join(f"[{m['role']}]: {m['content']}" for m in messages)
    critique_messages = [
        {"role": "system", "content": CRITIQUE_PROMPT},
        {"role": "user", "content": f"## Conversation\n{conversation_text}\n\n## AI Response\n{initial_response}\n\nProvide your critique."}
    ]
    critique = client.chat.completions.create(
        model=MODEL,
        messages=critique_messages,
        temperature=0.2,
        max_tokens=1024,
    )
    critique_text = critique.choices[0].message.content

    # Step 3: Generate refined response
    refine_system = REFINE_PROMPT.format(critique=critique_text)
    refined_messages = [{"role": "system", "content": refine_system}] + messages
    refined = client.chat.completions.create(
        model=MODEL,
        messages=refined_messages,
        temperature=0.3,
        max_tokens=2048,
    )

    return refined.choices[0].message.content


if __name__ == "__main__":
    # Read problem from stdin
    problem = json.loads(sys.stdin.read())
    messages = problem["prompt"]

    response = generate_response(messages)
    print(response)
