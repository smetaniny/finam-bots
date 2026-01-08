from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List

import httpx

from core.interfaces import Bar, BrokerAdapter, DataAdapter, OrderRequest

from .finam_rest import FinamRestClient


class FinamRestDataAdapter(DataAdapter):
    def __init__(self, client: FinamRestClient) -> None:
        self.client = client
        self.cache_dir = os.path.join("data", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_bars(self, symbol: str, timeframe: str, start_ts: int, end_ts: int) -> Iterable[Bar]:
        try:
            payload = self.client.get_bars(symbol, timeframe, start_ts, end_ts)
            self._write_cache(symbol, timeframe, payload)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 503:
                raise
            payload = self._read_cache(symbol, timeframe)
            if payload is None:
                raise
        for item in payload.get("bars", []):
            yield Bar(
                symbol=symbol,
                timestamp=item["timestamp"],
                open=float(item["open"]["value"]),
                high=float(item["high"]["value"]),
                low=float(item["low"]["value"]),
                close=float(item["close"]["value"]),
                volume=float(item["volume"]["value"]),
            )

    def _cache_path(self, symbol: str, timeframe: str) -> str:
        safe_symbol = symbol.replace("@", "_").replace("/", "_")
        return os.path.join(self.cache_dir, f"bars_{safe_symbol}_{timeframe}.json")

    def _write_cache(self, symbol: str, timeframe: str, payload: Dict[str, Any]) -> None:
        path = self._cache_path(symbol, timeframe)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def _read_cache(self, symbol: str, timeframe: str) -> Dict[str, Any] | None:
        path = self._cache_path(symbol, timeframe)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)


class FinamRestBrokerAdapter(BrokerAdapter):
    def __init__(self, client: FinamRestClient) -> None:
        self.client = client

    def place_order(self, order: OrderRequest) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "account_id": order.account_id,
            "symbol": order.symbol,
            "quantity": {"value": str(order.quantity)},
            "side": order.side,
            "type": order.order_type,
        }
        if order.limit_price is not None:
            payload["limit_price"] = {"value": str(order.limit_price)}
        if order.stop_price is not None:
            payload["stop_price"] = {"value": str(order.stop_price)}
        if order.time_in_force is not None:
            payload["time_in_force"] = order.time_in_force
        if order.client_order_id is not None:
            payload["client_order_id"] = order.client_order_id
        if order.comment is not None:
            payload["comment"] = order.comment
        return self.client.place_order(payload)

    def cancel_order(self, account_id: str, order_id: str) -> Dict[str, Any]:
        return self.client.cancel_order(account_id, order_id)

    def get_order(self, account_id: str, order_id: str) -> Dict[str, Any]:
        return self.client.get_order(account_id, order_id)

    def get_account(self, account_id: str) -> Dict[str, Any]:
        return self.client.get_account(account_id)
