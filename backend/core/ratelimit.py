import threading
import time
from collections import defaultdict, deque

WINDOW_SECONDS = 60
_lock = threading.Lock()
_requests: dict[str, deque[float]] = defaultdict(deque)


def check_rate_limit(key: str, max_requests: int) -> tuple[bool, float]:
    """Sliding-window rate limit. Returns (allowed, retry_after_seconds)."""
    now = time.monotonic()
    with _lock:
        window = _requests[key]
        cutoff = now - WINDOW_SECONDS
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= max_requests:
            retry_after = WINDOW_SECONDS - (now - window[0])
            return False, max(retry_after, 0.0)

        window.append(now)
        return True, 0.0


def reset() -> None:
    """Clear all rate-limit state. For test isolation only."""
    with _lock:
        _requests.clear()
