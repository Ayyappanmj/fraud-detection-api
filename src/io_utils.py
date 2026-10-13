"""Small helpers shared by the training pipeline and the API."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def json_default(obj: Any) -> Any:
    """Convert numpy / pathlib objects so ``json.dumps`` can serialise them."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=json_default), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
