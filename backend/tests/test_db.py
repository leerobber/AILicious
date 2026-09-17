import pytest

from core.db import execute_with_retry


class _FakeConnection:
    def __init__(self, fail_times: int = 0, message: str = "Hrana: stream error: ... retry the transaction ..."):
        self.fail_times = fail_times
        self.message = message
        self.calls = 0
        self.closed = False

    def execute(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ValueError(self.message)
        return "ok"

    def close(self):
        self.closed = True


def test_execute_with_retry_succeeds_first_try_without_retrying():
    conn = _FakeConnection(fail_times=0)

    result = execute_with_retry(lambda: conn, lambda c: c.execute())

    assert result == "ok"
    assert conn.calls == 1
    assert conn.closed is True


def test_execute_with_retry_retries_transient_failure_with_a_fresh_connection():
    conns = [_FakeConnection(fail_times=1), _FakeConnection(fail_times=0)]
    connection_iter = iter(conns)

    result = execute_with_retry(lambda: next(connection_iter), lambda c: c.execute())

    assert result == "ok"
    # The first (failed) connection must be closed and never reused -- a fresh connection
    # is what the retry actually needs, since whatever tripped "idle too long" on the first
    # one isn't fixed by calling it again.
    assert conns[0].calls == 1
    assert conns[0].closed is True
    assert conns[1].calls == 1
    assert conns[1].closed is True


def test_execute_with_retry_gives_up_after_max_retries_and_raises_the_last_error():
    conns = [_FakeConnection(fail_times=99) for _ in range(3)]
    connection_iter = iter(conns)

    with pytest.raises(ValueError, match="retry the transaction"):
        execute_with_retry(lambda: next(connection_iter), lambda c: c.execute(), max_retries=2)

    assert all(c.closed for c in conns)


def test_execute_with_retry_does_not_retry_an_unrelated_value_error():
    conn = _FakeConnection(fail_times=1, message="something unrelated broke")

    with pytest.raises(ValueError, match="something unrelated broke"):
        execute_with_retry(lambda: conn, lambda c: c.execute())

    assert conn.calls == 1  # never retried
    assert conn.closed is True


def test_execute_with_retry_closes_connection_even_when_operation_raises_immediately():
    conn = _FakeConnection(fail_times=1, message="not a retryable message")

    with pytest.raises(ValueError):
        execute_with_retry(lambda: conn, lambda c: c.execute())

    assert conn.closed is True
