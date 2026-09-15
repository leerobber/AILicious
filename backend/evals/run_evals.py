"""Runs backend/evals/cases.py against a real, running AILicious backend and judges each
response with a real Mistral call. This is an integration/E2E tool -- it needs a real
MISTRAL_API_KEY and a live backend (local or deployed), so it is NOT part of the pytest
suite and is NOT wired into the required CI gate. See the README's "Automated evals"
section for why, and for how to opt into running it on a schedule or in CI yourself.

Usage:
    cd backend
    EVAL_BASE_URL=https://ailicious-backend.onrender.com \
    APP_API_KEY=... \
    MISTRAL_API_KEY=... \
    python -m evals.run_evals

Each run appends one compact line to evals/history.jsonl (git-tracked, so trend over time
is visible in the repo's own history) and writes a full per-case report to
evals/results/<timestamp>.json (gitignored -- local inspection only).
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

from evals.cases import EVAL_CASES
from evals.judge import build_judge_messages, parse_judge_verdict

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
HISTORY_PATH = Path(__file__).resolve().parent / "history.jsonl"


async def _call_mistral_judge(messages: list[dict], api_key: str, model: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            MISTRAL_API_URL,
            json={"model": model, "messages": messages},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def run_case(case: dict, base_url: str, app_api_key: str, mistral_api_key: str, mistral_model: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{base_url}/agents/{case['agent']}",
                json={"message": case["message"]},
                headers={"X-API-Key": app_api_key},
            )
            resp.raise_for_status()
            response_text = resp.json()["response"]
    except (httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
        return {**case, "response": None, "passed": False, "reasoning": f"backend call failed: {exc}"}

    judge_messages = build_judge_messages(case, response_text)
    try:
        raw_reply = await _call_mistral_judge(judge_messages, mistral_api_key, mistral_model)
        verdict = parse_judge_verdict(raw_reply)
    except Exception as exc:  # noqa: BLE001 - any judge failure must still produce a recorded, failed case
        verdict = {"passed": False, "reasoning": f"judge call/parse failed: {exc}"}

    return {**case, "response": response_text, **verdict}


async def run_all(base_url: str, app_api_key: str, mistral_api_key: str, mistral_model: str) -> dict:
    results = await asyncio.gather(
        *(run_case(case, base_url, app_api_key, mistral_api_key, mistral_model) for case in EVAL_CASES)
    )
    passed = sum(1 for r in results if r["passed"])
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "total": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 3) if results else 0.0,
        "results": results,
    }


def _write_reports(report: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{report['timestamp'].replace(':', '-')}.json"
    out_path.write_text(json.dumps(report, indent=2))

    summary_line = {
        "timestamp": report["timestamp"],
        "base_url": report["base_url"],
        "total": report["total"],
        "passed": report["passed"],
        "pass_rate": report["pass_rate"],
    }
    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps(summary_line) + "\n")

    return out_path


def _print_summary(report: dict) -> None:
    for r in report["results"]:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {r['id']}: {r['reasoning']}")
    print(f"\n{report['passed']}/{report['total']} passed ({report['pass_rate']:.0%})")


def main() -> None:
    base_url = os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:8000")
    app_api_key = os.environ.get("APP_API_KEY", "")
    mistral_api_key = os.environ.get("MISTRAL_API_KEY", "")
    mistral_model = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")

    if not mistral_api_key:
        print("MISTRAL_API_KEY is required to run the judge -- see this file's module docstring.", file=sys.stderr)
        sys.exit(2)

    report = asyncio.run(run_all(base_url, app_api_key, mistral_api_key, mistral_model))
    out_path = _write_reports(report)
    _print_summary(report)
    print(f"\nFull report: {out_path}")

    sys.exit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()
