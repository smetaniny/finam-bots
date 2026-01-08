from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict

from .interfaces import OrderRequest, RiskManager


class AllowAllRiskManager(RiskManager):
    def allow_order(self, order: OrderRequest) -> bool:
        return True


@dataclass
class RiskLimits:
    max_position_per_symbol: float = 0.0
    max_trades_per_day: int = 0
    daily_loss_limit: float = 0.0


@dataclass
class RiskState:
    positions: Dict[str, float] = field(default_factory=dict)
    trades_today: int = 0
    day_pnl: float = 0.0
    day: date = field(default_factory=date.today)
    hard_stop: bool = False


class SimpleRiskManager(RiskManager):
    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits
        self.state = RiskState()

    def allow_order(self, order: OrderRequest) -> bool:
        self._roll_day()
        if self.state.hard_stop:
            return False
        if self.limits.max_trades_per_day and self.state.trades_today >= self.limits.max_trades_per_day:
            return False
        if self.limits.daily_loss_limit and self.state.day_pnl <= -abs(self.limits.daily_loss_limit):
            return False
        if self.limits.max_position_per_symbol:
            current = self.state.positions.get(order.symbol, 0.0)
            projected = current + order.quantity if order.side == "SIDE_BUY" else current - order.quantity
            if abs(projected) > self.limits.max_position_per_symbol:
                return False
        return True

    def register_fill(self, symbol: str, qty: float, pnl: float) -> None:
        self._roll_day()
        self.state.trades_today += 1
        self.state.day_pnl += pnl
        self.state.positions[symbol] = self.state.positions.get(symbol, 0.0) + qty

    def set_hard_stop(self, enabled: bool) -> None:
        self.state.hard_stop = enabled

    def _roll_day(self) -> None:
        today = date.today()
        if self.state.day != today:
            self.state.day = today
            self.state.trades_today = 0
            self.state.day_pnl = 0.0
