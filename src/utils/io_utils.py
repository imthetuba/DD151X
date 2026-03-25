from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_prefix(feature_set: str, model_name: str, split_type: str, date_str: str | None = None) -> str:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    return f"run_{date_str}_{feature_set}_{model_name}_{split_type}"


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
