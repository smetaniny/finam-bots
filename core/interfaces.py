from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class Bar:
    symbol: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class OrderRequest:
    account_id: str
    symbol: str
    quantity: float
    side: str
    order_type: str
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: Optional[str] = None
    client_order_id: Optional[str] = None
    comment: Optional[str] = None


class DataAdapter(ABC):
    @abstractmethod
    def get_bars(self, symbol: str, timeframe: str, start_ts: int, end_ts: int) -> Iterable[Bar]:
        raise NotImplementedError


class BrokerAdapter(ABC):
    @abstractmethod
    def place_order(self, order: OrderRequest) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, account_id: str, order_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_order(self, account_id: str, order_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_account(self, account_id: str) -> Dict[str, Any]:
        raise NotImplementedError


class RiskManager(ABC):
    @abstractmethod
    def allow_order(self, order: OrderRequest) -> bool:
        raise NotImplementedError


class Strategy(ABC):
    @abstractmethod
    def on_bar(self, bar: Bar) -> Optional[OrderRequest]:
        raise NotImplementedError
