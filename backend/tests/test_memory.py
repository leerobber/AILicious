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


def test_get_recent_message_ids_scoped_per_session_and_agent():
    id1 = memory_module.add_message("s1", "nexus", "user", "in s1")
    memory_module.add_message("s2", "nexus", "user", "in s2")
    memory_module.add_message("s1", "forge", "user", "different agent")

    assert memory_module.get_recent_message_ids("s1", "nexus", limit=10) == {id1}


def test_get_recent_message_ids_respects_limit():
    ids = [memory_module.add_message("s1", "nexus", "user", f"msg{i}") for i in range(5)]

    recent = memory_module.get_recent_message_ids("s1", "nexus", limit=2)

    assert recent == set(ids[-2:])


def test_set_embedding_and_get_embedded_messages_roundtrip():
    message_id = memory_module.add_message("s1", "nexus", "user", "my favorite color is teal")
    memory_module.set_embedding(message_id, [1.0, 0.0, 0.0])

    embedded = memory_module.get_embedded_messages("nexus", exclude_ids=set())

    assert embedded == [
        {"id": message_id, "role": "user", "content": "my favorite color is teal", "embedding": [1.0, 0.0, 0.0]}
    ]


def test_get_embedded_messages_excludes_messages_without_an_embedding():
    memory_module.add_message("s1", "nexus", "user", "never embedded")

    assert memory_module.get_embedded_messages("nexus", exclude_ids=set()) == []


def test_get_embedded_messages_excludes_given_ids():
    id1 = memory_module.add_message("s1", "nexus", "user", "a")
    id2 = memory_module.add_message("s1", "nexus", "user", "b")
    memory_module.set_embedding(id1, [1.0, 0.0])
    memory_module.set_embedding(id2, [0.0, 1.0])

    embedded = memory_module.get_embedded_messages("nexus", exclude_ids={id1})

    assert [m["id"] for m in embedded] == [id2]


def test_get_embedded_messages_spans_sessions():
    id1 = memory_module.add_message("s1", "nexus", "user", "session one")
    id2 = memory_module.add_message("s2", "nexus", "user", "session two")
    memory_module.set_embedding(id1, [1.0])
    memory_module.set_embedding(id2, [1.0])

    embedded = memory_module.get_embedded_messages("nexus", exclude_ids=set())

    assert {m["id"] for m in embedded} == {id1, id2}


def test_get_embedded_messages_scoped_per_agent():
    id1 = memory_module.add_message("s1", "nexus", "user", "nexus message")
    id2 = memory_module.add_message("s1", "forge", "user", "forge message")
    memory_module.set_embedding(id1, [1.0])
    memory_module.set_embedding(id2, [1.0])

    embedded = memory_module.get_embedded_messages("nexus", exclude_ids=set())

    assert [m["id"] for m in embedded] == [id1]


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


def _flaky_once(real_get_connection):
    """Returns a `_get_connection` replacement that fails with Turso's exact observed
    "retry the transaction" error on its first call, then delegates to the real one --
    reproducing the production failure (an idle interactive transaction rolled back
    under load) without needing a real Turso connection.
    """
    calls = {"n": 0}

    class _FlakyConnection:
        def execute(self, *args, **kwargs):
            raise ValueError(
                'Hrana: `stream error: `Error { message: "SQLite error: interactive transaction was '
                'rolled back because the stream was idle for too long; retry the transaction", '
                'code: "SQLITE_BUSY" }`'
            )

        def close(self):
            pass

    def _get_connection():
        calls["n"] += 1
        if calls["n"] == 1:
            return _FlakyConnection()
        return real_get_connection()

    return _get_connection, calls


def test_add_message_retries_transient_turso_error_and_succeeds(monkeypatch):
    fake_get_connection, calls = _flaky_once(memory_module._get_connection)
    monkeypatch.setattr(memory_module, "_get_connection", fake_get_connection)

    message_id = memory_module.add_message("s1", "nexus", "user", "hi")  # must not raise

    assert isinstance(message_id, int)
    assert calls["n"] == 2  # one failed attempt, one real retry


def test_set_feedback_retries_transient_turso_error_and_succeeds(monkeypatch):
    message_id = memory_module.add_message("s1", "nexus", "assistant", "reply")

    fake_get_connection, calls = _flaky_once(memory_module._get_connection)
    monkeypatch.setattr(memory_module, "_get_connection", fake_get_connection)

    assert memory_module.set_feedback(message_id, 1) is True
    assert calls["n"] == 2


def test_set_embedding_retries_transient_turso_error_and_succeeds(monkeypatch):
    message_id = memory_module.add_message("s1", "nexus", "user", "hi")

    fake_get_connection, calls = _flaky_once(memory_module._get_connection)
    monkeypatch.setattr(memory_module, "_get_connection", fake_get_connection)

    memory_module.set_embedding(message_id, [1.0, 2.0, 3.0])  # must not raise
    assert calls["n"] == 2

    stored = memory_module.get_embedded_messages("nexus", exclude_ids=set())
    assert any(m["id"] == message_id and m["embedding"] == [1.0, 2.0, 3.0] for m in stored)
