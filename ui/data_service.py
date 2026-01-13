from __future__ import annotations

import time
from typing import Any, Dict, List

from adapters import FinamRestClient, FinamRestDataAdapter
from core.interfaces import Bar


class FinamDataService:
    def __init__(self, client: FinamRestClient) -> None:
        self.client = client
        self.data_adapter = FinamRestDataAdapter(client)

    def get_bars(self, symbol: str, timeframe: str, days_back: int) -> List[Bar]:
        end_ts = int(time.time())
        max_days = self._max_days_for_timeframe(timeframe)
        if days_back <= max_days:
            start_ts = end_ts - days_back * 86400
            return list(self.data_adapter.get_bars(symbol, timeframe, start_ts, end_ts))

        bars: List[Bar] = []
        remaining_days = days_back
        chunk_end = end_ts
        while remaining_days > 0:
            chunk_days = min(max_days, remaining_days)
            chunk_start = chunk_end - chunk_days * 86400
            bars.extend(self.data_adapter.get_bars(symbol, timeframe, chunk_start, chunk_end))
            remaining_days -= chunk_days
            chunk_end = chunk_start
        return bars

    def _max_days_for_timeframe(self, timeframe: str) -> int:
        # Finam API rejects too wide intervals for intraday timeframes.
        if timeframe in {"TIME_FRAME_H1", "TIME_FRAME_H4"}:
            return 30
        return 365

    def get_account(self, account_id: str) -> Dict[str, Any]:
        return self.client.get_account(account_id)

    def get_trades(self, account_id: str, limit: int = 50) -> Dict[str, Any]:
        return self.client.get_trades(account_id, limit=limit)

    def get_orders(self, account_id: str) -> Dict[str, Any]:
        return self.client.get_orders(account_id)
