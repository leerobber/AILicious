import pytest

from core import cost as cost_module


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(cost_module, "DB_PATH", tmp_path / "test_cost.db")
    cost_module.init_db()
    yield


def test_record_usage_and_summary_roundtrip():
    cost_module.record_usage("chat", "mistral-small-latest", 100, 50, 150, agent="nexus")

    summary = cost_module.get_usage_summary()

    assert summary["by_category"] == [
        {
            "category": "chat",
            "model": "mistral-small-latest",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "calls": 1,
        }
    ]
    assert summary["totals"] == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "calls": 1}


def test_get_usage_summary_empty_when_nothing_recorded():
    assert cost_module.get_usage_summary() == {
        "by_category": [],
        "totals": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0},
    }


def test_get_usage_summary_aggregates_same_category_and_model():
    cost_module.record_usage("digest", "mistral-small-latest", 100, 20, 120, agent="nexus")
    cost_module.record_usage("digest", "mistral-small-latest", 200, 30, 230, agent="forge")

    summary = cost_module.get_usage_summary()

    assert len(summary["by_category"]) == 1
    row = summary["by_category"][0]
    assert row["prompt_tokens"] == 300
    assert row["completion_tokens"] == 50
    assert row["total_tokens"] == 350
    assert row["calls"] == 2


def test_get_usage_summary_separates_different_categories():
    cost_module.record_usage("chat", "mistral-small-latest", 100, 50, 150)
    cost_module.record_usage("embedding", "mistral-embed", 10, 0, 10)

    summary = cost_module.get_usage_summary()

    assert {row["category"] for row in summary["by_category"]} == {"chat", "embedding"}
    assert summary["totals"]["total_tokens"] == 160


def test_get_usage_summary_filters_by_category():
    cost_module.record_usage("chat", "mistral-small-latest", 100, 50, 150)
    cost_module.record_usage("embedding", "mistral-embed", 10, 0, 10)

    summary = cost_module.get_usage_summary(category="embedding")

    assert [row["category"] for row in summary["by_category"]] == ["embedding"]
    assert summary["totals"]["total_tokens"] == 10


def test_record_usage_swallows_failure_instead_of_raising(monkeypatch):
    """Same posture as core.events.record_event: a failure to log usage must never break
    the Mistral call that generated it."""
    monkeypatch.setattr(cost_module, "DB_PATH", cost_module.DB_PATH.parent / "never_initialized.db")

    cost_module.record_usage("chat", "mistral-small-latest", 1, 1, 2)  # must not raise
