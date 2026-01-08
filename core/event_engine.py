from __future__ import annotations

from typing import Iterable, List

from .interfaces import BrokerAdapter, DataAdapter, OrderRequest, RiskManager, Strategy
from .logger import get_logger


class EventEngine:
    def __init__(
        self,
        data_adapter: DataAdapter,
        broker_adapter: BrokerAdapter,
        risk_manager: RiskManager,
        strategies: Iterable[Strategy],
    ) -> None:
        self.data_adapter = data_adapter
        self.broker_adapter = broker_adapter
        self.risk_manager = risk_manager
        self.strategies: List[Strategy] = list(strategies)
        self.log = get_logger(__name__)

    def run_bars(self, symbol: str, timeframe: str, start_ts: int, end_ts: int) -> None:
        for bar in self.data_adapter.get_bars(symbol, timeframe, start_ts, end_ts):
            for strategy in self.strategies:
                order = strategy.on_bar(bar)
                if order is None:
                    continue
                if not self.risk_manager.allow_order(order):
                    self.log.warning("Order blocked by risk manager: %s", order)
                    continue
                self.log.info("Placing order: %s", order)
                self.broker_adapter.place_order(order)
