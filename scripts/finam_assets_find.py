from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import FinamRestClient
from core import get_logger


def _as_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        items = payload.get("assets")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def _matches(asset: Dict[str, Any], query: str) -> bool:
    haystack: Iterable[str] = (
        str(asset.get("symbol", "")),
        str(asset.get("code", "")),
        str(asset.get("ticker", "")),
        str(asset.get("short_name", "")),
        str(asset.get("name", "")),
    )
    query_lower = query.lower()
    return any(query_lower in value.lower() for value in haystack if value)


def _format_asset(asset: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("symbol", "code", "ticker", "name", "market", "board")
    return {key: asset.get(key) for key in keys if key in asset}


def main() -> None:
    load_dotenv()
    log = get_logger("finam_assets_find")

    query = os.getenv("FINAM_ASSET_QUERY") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not query:
        raise RuntimeError("Provide FINAM_ASSET_QUERY or pass a query as argv[1].")

    market_filter = os.getenv("FINAM_ASSET_MARKET")

    client = FinamRestClient()
    try:
        payload = client.get_assets()
    finally:
        client.close()

    assets = _as_list(payload)
    matches = []
    for asset in assets:
        if market_filter and str(asset.get("market", "")).upper() != market_filter.upper():
            continue
        if _matches(asset, query):
            matches.append(_format_asset(asset))

    log.info({"query": query, "market": market_filter, "matches": matches})


if __name__ == "__main__":
    main()
