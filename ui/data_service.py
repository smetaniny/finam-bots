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
        start_ts = end_ts - days_back * 86400
        return list(self.data_adapter.get_bars(symbol, timeframe, start_ts, end_ts))

    def get_account(self, account_id: str) -> Dict[str, Any]:
        return self.client.get_account(account_id)

    def get_trades(self, account_id: str, limit: int = 50) -> Dict[str, Any]:
        return self.client.get_trades(account_id, limit=limit)

    def get_orders(self, account_id: str) -> Dict[str, Any]:
        return self.client.get_orders(account_id)
