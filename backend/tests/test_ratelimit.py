import pytest

from core.ratelimit import check_rate_limit, reset


@pytest.fixture(autouse=True)
def _reset_state():
    reset()
    yield
    reset()


def test_allows_up_to_limit_then_blocks():
    key = "test-key-1"
    for _ in range(3):
        allowed, _ = check_rate_limit(key, 3)
        assert allowed is True

    allowed, retry_after = check_rate_limit(key, 3)
    assert allowed is False
    assert retry_after > 0


def test_different_keys_are_independent():
    assert check_rate_limit("key-a", 1) == (True, 0.0)
    assert check_rate_limit("key-b", 1) == (True, 0.0)
    assert check_rate_limit("key-a", 1)[0] is False
