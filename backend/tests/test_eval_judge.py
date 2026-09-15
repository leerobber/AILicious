import json

import pytest

from evals.cases import EVAL_CASES
from evals.judge import build_judge_messages, parse_judge_verdict


def test_eval_cases_are_well_formed():
    assert len(EVAL_CASES) > 0
    ids = [case["id"] for case in EVAL_CASES]
    assert len(ids) == len(set(ids)), "eval case ids must be unique"
    for case in EVAL_CASES:
        assert case["agent"] and isinstance(case["agent"], str)
        assert case["message"] and isinstance(case["message"], str)
        assert case["rubric"] and isinstance(case["rubric"], str)


def test_build_judge_messages_includes_message_response_and_rubric():
    case = {"agent": "nexus", "message": "hello there", "rubric": "must say hi back"}

    messages = build_judge_messages(case, "hi!")

    user_content = messages[1]["content"]
    assert "hello there" in user_content
    assert "hi!" in user_content
    assert "must say hi back" in user_content
    assert "nexus" in user_content


def test_parse_judge_verdict_valid_pass():
    verdict = parse_judge_verdict(json.dumps({"passed": True, "reasoning": "met the rubric"}))

    assert verdict == {"passed": True, "reasoning": "met the rubric"}


def test_parse_judge_verdict_valid_fail():
    verdict = parse_judge_verdict(json.dumps({"passed": False, "reasoning": "missed the rubric"}))

    assert verdict == {"passed": False, "reasoning": "missed the rubric"}


def test_parse_judge_verdict_missing_reasoning_defaults_to_empty_string():
    verdict = parse_judge_verdict(json.dumps({"passed": True}))

    assert verdict == {"passed": True, "reasoning": ""}


def test_parse_judge_verdict_non_bool_passed_raises():
    with pytest.raises(ValueError):
        parse_judge_verdict(json.dumps({"passed": "yes", "reasoning": "r"}))


def test_parse_judge_verdict_malformed_json_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_judge_verdict("not json at all")


def test_parse_judge_verdict_missing_passed_key_raises():
    with pytest.raises(KeyError):
        parse_judge_verdict(json.dumps({"reasoning": "r"}))


def test_parse_judge_verdict_non_string_reasoning_is_stringified():
    verdict = parse_judge_verdict(json.dumps({"passed": True, "reasoning": 123}))

    assert verdict == {"passed": True, "reasoning": "123"}
