"""Small shared helpers."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(p: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
