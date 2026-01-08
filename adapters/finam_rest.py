from __future__ import annotations

import os
import time
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, Optional

import httpx

from core.logger import get_logger

class FinamRestClient:
    def __init__(
        self,
        secret: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 15.0,
        rate_limit_per_minute: int = 200,
        max_retries: int = 3,
    ) -> None:
        self.secret = secret or os.getenv("FINAM_API_SECRET")
        self.base_url = (base_url or os.getenv("FINAM_API_BASE") or "https://api.finam.ru").rstrip("/")
        self.auth_header = os.getenv("FINAM_AUTH_HEADER", "Authorization")
        self.auth_scheme = os.getenv("FINAM_AUTH_SCHEME", "Bearer")
        self.timeout = timeout
        self.rate_limit_per_minute = rate_limit_per_minute
        self.max_retries = max_retries
        self._token: Optional[str] = None
        self._token_expires_at: Optional[float] = None
        self._client = httpx.Client(timeout=self.timeout, http2=True)
        self._request_times: Deque[float] = deque()
        self._log = get_logger("finam_rest")

    def close(self) -> None:
        self._client.close()

    def _auth_headers(self) -> Dict[str, str]:
        token = self._ensure_token()
        token = token.strip()
        if self.auth_scheme.lower() in ("", "none"):
            return {self.auth_header: token}
        return {self.auth_header: f"{self.auth_scheme} {token}"}

    def _ensure_token(self) -> str:
        if self._token and self._token_expires_at and time.time() < self._token_expires_at - 30:
            return self._token
        if not self.secret:
            raise RuntimeError("FINAM_API_SECRET is required for authentication")
        token = self.auth(self.secret)
        details = self.token_details(token)
        expires_at = details.get("expires_at")
        if expires_at:
            self._token_expires_at = _parse_ts(expires_at)
        self._token = token
        return token

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None, auth: bool = True) -> Dict[str, Any]:
        headers: Dict[str, str] = {}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if auth:
            headers.update(self._auth_headers())
            self._log.info("Auth header: %s %s", self.auth_header, self.auth_scheme)
        self._apply_rate_limit()
        url = f"{self.base_url}{path}"
        attempt = 0
        while True:
            try:
                response = self._client.request(method, url, json=payload, params=params, headers=headers)
                self._log.info("API %s %s -> %s", method, path, response.status_code)
                if response.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    self._backoff(attempt)
                    attempt += 1
                    continue
                response.raise_for_status()
                return response.json()
            except httpx.RequestError:
                if attempt >= self.max_retries:
                    raise
                self._backoff(attempt)
                attempt += 1

    def _post(self, path: str, payload: Dict[str, Any], auth: bool = True) -> Dict[str, Any]:
        return self._request("POST", path, payload=payload, auth=auth)

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None, auth: bool = True) -> Dict[str, Any]:
        return self._request("GET", path, params=params, auth=auth)

    def _apply_rate_limit(self) -> None:
        now = time.time()
        window_start = now - 60
        while self._request_times and self._request_times[0] < window_start:
            self._request_times.popleft()
        if len(self._request_times) >= self.rate_limit_per_minute:
            sleep_for = 60 - (now - self._request_times[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._request_times.append(time.time())

    def _backoff(self, attempt: int) -> None:
        time.sleep(min(2 ** attempt, 10))

    def auth(self, secret: str) -> str:
        data = self._post("/v1/sessions", {"secret": secret}, auth=False)
        return data["token"]

    def token_details(self, token: str) -> Dict[str, Any]:
        return self._post("/v1/sessions/details", {"token": token}, auth=False)

    def get_account(self, account_id: str) -> Dict[str, Any]:
        return self._get(f"/v1/accounts/{account_id}")

    def get_trades(self, account_id: str, limit: int = 100, start_ts: Optional[int] = None, end_ts: Optional[int] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if start_ts is not None and end_ts is not None:
            params["interval.from"] = start_ts
            params["interval.to"] = end_ts
        return self._get(f"/v1/accounts/{account_id}/trades", params=params)

    def get_transactions(self, account_id: str, limit: int = 100, start_ts: Optional[int] = None, end_ts: Optional[int] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if start_ts is not None and end_ts is not None:
            params["interval.from"] = start_ts
            params["interval.to"] = end_ts
        return self._get(f"/v1/accounts/{account_id}/transactions", params=params)

    def get_assets(self) -> Dict[str, Any]:
        return self._get("/v1/assets")

    def get_asset(self, symbol: str, account_id: Optional[str] = None) -> Dict[str, Any]:
        params = {"symbol": symbol}
        if account_id:
            params["account_id"] = account_id
        return self._get("/v1/assets/get", params=params)

    def get_asset_params(self, symbol: str, account_id: str) -> Dict[str, Any]:
        params = {"symbol": symbol, "account_id": account_id}
        return self._get("/v1/assets/params", params=params)

    def get_bars(self, symbol: str, timeframe: str, start_ts: int, end_ts: int) -> Dict[str, Any]:
        params = {"symbol": symbol, "timeframe": timeframe, "interval.from": start_ts, "interval.to": end_ts}
        return self._get("/v1/marketdata/bars", params=params)

    def get_last_quote(self, symbol: str) -> Dict[str, Any]:
        params = {"symbol": symbol}
        return self._get("/v1/marketdata/lastquote", params=params)

    def get_orderbook(self, symbol: str) -> Dict[str, Any]:
        params = {"symbol": symbol}
        return self._get("/v1/marketdata/orderbook", params=params)

    def place_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._post("/v1/orders", payload)

    def cancel_order(self, account_id: str, order_id: str) -> Dict[str, Any]:
        payload = {"account_id": account_id, "order_id": order_id}
        return self._post("/v1/orders/cancel", payload)

    def get_order(self, account_id: str, order_id: str) -> Dict[str, Any]:
        params = {"account_id": account_id, "order_id": order_id}
        return self._get("/v1/orders/get", params=params)

    def get_orders(self, account_id: str) -> Dict[str, Any]:
        params = {"account_id": account_id}
        return self._get("/v1/orders", params=params)


def _parse_ts(value: str) -> float:
    # ISO-8601 "2025-07-24T08:06:30Z" -> epoch seconds (UTC)
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).timestamp()
