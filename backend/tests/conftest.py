import os
import sys
from pathlib import Path

# Set required env vars before any test module imports main.py / core modules,
# so module-level constants (e.g. MISTRAL_API_KEY) pick these up correctly.
os.environ.setdefault("MISTRAL_API_KEY", "test-mistral-key")
os.environ.setdefault("APP_API_KEY", "test-app-key")
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "1000")
os.environ.setdefault("MAX_MESSAGE_LENGTH", "4000")
os.environ.pop("TURSO_DATABASE_URL", None)
os.environ.pop("TURSO_AUTH_TOKEN", None)

# Make `core` and `main` importable the same way they are at runtime
# (uvicorn is run with backend/ as the working directory).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
