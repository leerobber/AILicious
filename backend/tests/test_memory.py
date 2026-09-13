import pytest

from core import memory as memory_module


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_module, "DB_PATH", tmp_path / "test_memory.db")
    memory_module.init_db()
    yield


def test_add_message_returns_new_row_id():
    id1 = memory_module.add_message("s1", "nexus", "user", "a")
    id2 = memory_module.add_message("s1", "nexus", "user", "b")

    assert isinstance(id1, int)
    assert id2 == id1 + 1


def test_set_feedback_roundtrip():
    import libsql

    message_id = memory_module.add_message("s1", "nexus", "assistant", "reply")

    assert memory_module.set_feedback(message_id, 1) is True

    conn = libsql.connect(str(memory_module.DB_PATH))
    row = conn.execute("SELECT feedback FROM messages WHERE id = ?", (message_id,)).fetchone()
    conn.close()
    assert row[0] == 1


def test_set_feedback_unknown_id_returns_false():
    assert memory_module.set_feedback(999999, 1) is False


def test_add_and_get_history_roundtrip():
    memory_module.add_message("s1", "nexus", "user", "hello")
    memory_module.add_message("s1", "nexus", "assistant", "hi there")

    history = memory_module.get_history("s1", "nexus")

    assert history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_history_scoped_per_agent():
    memory_module.add_message("s1", "forge", "user", "forge secret")

    assert memory_module.get_history("s1", "oracle") == []
    assert len(memory_module.get_history("s1", "forge")) == 1


def test_history_scoped_per_session():
    memory_module.add_message("s1", "nexus", "user", "in s1")
    memory_module.add_message("s2", "nexus", "user", "in s2")

    assert len(memory_module.get_history("s1", "nexus")) == 1
    assert len(memory_module.get_history("s2", "nexus")) == 1


def test_history_respects_limit_and_chronological_order():
    for i in range(5):
        memory_module.add_message("s1", "nexus", "user", f"msg{i}")

    history = memory_module.get_history("s1", "nexus", limit=3)

    assert [m["content"] for m in history] == ["msg2", "msg3", "msg4"]


def test_high_utility_message_survives_over_recent_filler():
    # 25 short (near-zero heuristic score) messages; a plain recency LIMIT of 20 would
    # keep only the newest 20, dropping the oldest 5 (msg0..msg4).
    ids = [memory_module.add_message("s1", "nexus", "user", f"msg{i}") for i in range(25)]

    # Upvote the message that would otherwise be the very first one dropped, boosting
    # its utility_score enough to outrank plain recency.
    memory_module.set_feedback(ids[0], 1)

    history = memory_module.get_history("s1", "nexus", limit=20)
    contents = [m["content"] for m in history]

    assert "msg0" in contents  # survived despite being older than the recency cutoff
    assert len(contents) == 20
    assert contents == sorted(contents, key=lambda c: int(c.removeprefix("msg")))  # still chronological


def test_zero_utility_history_matches_plain_recency():
    # No feedback anywhere: KAIROS's ranking must collapse to exactly what a bare
    # `ORDER BY id DESC LIMIT` would return — no behavior change for the common case.
    for i in range(25):
        memory_module.add_message("s1", "nexus", "user", f"msg{i}")

    history = memory_module.get_history("s1", "nexus", limit=20)
    contents = [m["content"] for m in history]

    assert contents == [f"msg{i}" for i in range(5, 25)]


def test_get_feedback_examples_returns_empty_list_when_no_feedback():
    memory_module.add_message("s1", "nexus", "user", "hello")
    memory_module.add_message("s1", "nexus", "assistant", "hi there")

    assert memory_module.get_feedback_examples("nexus") == []


def test_get_feedback_examples_pairs_user_and_assistant_messages():
    memory_module.add_message("s1", "nexus", "user", "what's the weather?")
    assistant_id = memory_module.add_message("s1", "nexus", "assistant", "sunny and warm")
    memory_module.set_feedback(assistant_id, 1)

    examples = memory_module.get_feedback_examples("nexus")

    assert examples == [{"user_message": "what's the weather?", "assistant_reply": "sunny and warm", "feedback": 1}]


def test_get_feedback_examples_only_includes_feedback_messages_most_recent_first():
    memory_module.add_message("s1", "nexus", "user", "q1")
    id1 = memory_module.add_message("s1", "nexus", "assistant", "a1")
    memory_module.add_message("s1", "nexus", "user", "q2")
    id2 = memory_module.add_message("s1", "nexus", "assistant", "a2")
    memory_module.add_message("s1", "nexus", "user", "q3")
    memory_module.add_message("s1", "nexus", "assistant", "a3")  # no feedback, excluded

    memory_module.set_feedback(id1, -1)
    memory_module.set_feedback(id2, 1)

    examples = memory_module.get_feedback_examples("nexus")

    assert [e["assistant_reply"] for e in examples] == ["a2", "a1"]
    assert [e["feedback"] for e in examples] == [1, -1]


def test_get_feedback_examples_scoped_per_agent():
    memory_module.add_message("s1", "forge", "user", "q")
    forge_reply_id = memory_module.add_message("s1", "forge", "assistant", "forge reply")
    memory_module.set_feedback(forge_reply_id, 1)

    assert memory_module.get_feedback_examples("oracle") == []
    assert len(memory_module.get_feedback_examples("forge")) == 1


def test_get_stats():
    memory_module.add_message("s1", "nexus", "user", "a")
    memory_module.add_message("s2", "forge", "user", "b")

    assert memory_module.get_stats() == {"total_messages": 2, "total_sessions": 2}


def test_init_db_is_idempotent_against_pre_agent_column_schema(tmp_path, monkeypatch):
    import libsql

    db_path = tmp_path / "legacy.db"
    conn = libsql.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        ("legacy-session", "user", "legacy row", "2020-01-01"),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(memory_module, "DB_PATH", db_path)
    memory_module.init_db()  # must not raise on the pre-existing schema

    history = memory_module.get_history("legacy-session", "nexus")
    assert history == [{"role": "user", "content": "legacy row"}]
