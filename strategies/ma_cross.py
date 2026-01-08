from __future__ import annotations

from collections import deque
from typing import Deque, Optional

from core.interfaces import Bar, OrderRequest
from strategies.base import BaseStrategy


class MACrossStrategy(BaseStrategy):
    def __init__(
        self,
        name: str,
        account_id: str,
        symbol: str,
        sma_short: int,
        sma_long: int,
        quantity: float,
    ) -> None:
        super().__init__(name)
        if sma_short >= sma_long:
            raise ValueError("sma_short must be меньше sma_long")
        self.account_id = account_id
        self.symbol = symbol
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.quantity = quantity
        self._closes: Deque[float] = deque(maxlen=sma_long + 1)
        self._last_signal: Optional[str] = None

    def on_bar(self, bar: Bar) -> Optional[OrderRequest]:
        if bar.symbol != self.symbol:
            return None
        self._closes.append(bar.close)
        if len(self._closes) < self.sma_long + 1:
            return None

        sma_short_now = sum(list(self._closes)[-self.sma_short :]) / self.sma_short
        sma_long_now = sum(list(self._closes)[-self.sma_long :]) / self.sma_long
        prev = list(self._closes)[:-1]
        sma_short_prev = sum(prev[-self.sma_short :]) / self.sma_short
        sma_long_prev = sum(prev[-self.sma_long :]) / self.sma_long

        if sma_short_prev <= sma_long_prev and sma_short_now > sma_long_now:
            if self._last_signal != "BUY":
                self._last_signal = "BUY"
                return OrderRequest(
                    account_id=self.account_id,
                    symbol=self.symbol,
                    quantity=self.quantity,
                    side="SIDE_BUY",
                    order_type="ORDER_TYPE_MARKET",
                )
        if sma_short_prev >= sma_long_prev and sma_short_now < sma_long_now:
            if self._last_signal != "SELL":
                self._last_signal = "SELL"
                return OrderRequest(
                    account_id=self.account_id,
                    symbol=self.symbol,
                    quantity=self.quantity,
                    side="SIDE_SELL",
                    order_type="ORDER_TYPE_MARKET",
                )
        return None
