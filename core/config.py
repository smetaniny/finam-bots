from __future__ import annotations

import os
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    for key, value in os.environ.items():
        if key.startswith("FINAM_"):
            data[key] = value
    return data
