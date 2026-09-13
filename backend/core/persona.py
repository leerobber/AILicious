from pathlib import Path

import yaml

PERSONA_DIR = Path(__file__).resolve().parent.parent / "config" / "personas"

_personas: dict[str, dict] = {}


def load_personas() -> None:
    _personas.clear()
    for path in PERSONA_DIR.glob("*.yaml"):
        with open(path) as f:
            _personas[path.stem] = yaml.safe_load(f)


def get_system_prompt(name: str = "nexus") -> str:
    persona = _personas.get(name)
    if not persona:
        raise KeyError(f"Persona '{name}' is not loaded")
    return persona["system_prompt"]
