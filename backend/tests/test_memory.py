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


def test_get_messages_since_none_returns_everything_for_the_agent():
    memory_module.add_message("s1", "nexus", "user", "a")
    memory_module.add_message("s2", "nexus", "assistant", "b")
    memory_module.add_message("s1", "forge", "user", "not nexus")

    messages = memory_module.get_messages_since("nexus", None)

    assert [m["content"] for m in messages] == ["a", "b"]


def test_get_messages_since_timestamp_excludes_earlier_messages():
    import libsql

    memory_module.add_message("s1", "nexus", "user", "old")

    conn = libsql.connect(str(memory_module.DB_PATH))
    boundary = conn.execute("SELECT created_at FROM messages ORDER BY id DESC LIMIT 1").fetchone()[0]
    conn.close()

    memory_module.add_message("s1", "nexus", "user", "new")

    messages = memory_module.get_messages_since("nexus", boundary)

    assert [m["content"] for m in messages] == ["new"]


def test_get_messages_since_spans_sessions_and_includes_feedback():
    id1 = memory_module.add_message("s1", "nexus", "user", "q")
    memory_module.add_message("s2", "nexus", "assistant", "a")
    memory_module.set_feedback(id1, 1)

    messages = memory_module.get_messages_since("nexus", None)

    assert messages[0]["feedback"] == 1
    assert messages[1]["feedback"] is None


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
