import json

JUDGE_SYSTEM_PROMPT = (
    "You are an evaluator for an AI agent platform. You'll be shown a user message sent to "
    "one agent, that agent's actual response, and a rubric describing what a passing response "
    "looks like. Judge strictly against the rubric only -- not general response quality, "
    "style, or length. A response can be a pass even if you'd have written it differently, as "
    "long as it satisfies the rubric.\n\n"
    'Respond with ONLY a JSON object: {"passed": <true or false>, "reasoning": "<one sentence '
    'citing what in the response drove the verdict>"}. No markdown, no extra text -- valid '
    "JSON only."
)


def build_judge_messages(case: dict, response: str) -> list[dict]:
    lines = [
        f"User message sent to '{case['agent']}':",
        case["message"],
        "",
        "Agent's actual response:",
        response,
        "",
        "Rubric:",
        case["rubric"],
    ]
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def parse_judge_verdict(raw_reply: str) -> dict:
    """Strict parse of a judge reply. Raises ValueError on anything malformed -- callers
    decide how to record that (e.g. a case that failed to even get judged should never be
    silently counted as a pass).
    """
    parsed = json.loads(raw_reply)
    passed = parsed["passed"]
    reasoning = parsed.get("reasoning", "")
    if not isinstance(passed, bool):
        raise ValueError(f"'passed' must be a boolean, got {passed!r}")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    return {"passed": passed, "reasoning": reasoning}
