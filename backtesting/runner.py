from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional

import yaml
from dotenv import load_dotenv

from adapters import FinamRestClient, FinamRestDataAdapter
from core import get_logger
from core.interfaces import Bar, OrderRequest
from strategies.ma_cross import MACrossStrategy


@dataclass
class Trade:
    side: str
    price: float
    timestamp: str
    qty: float
    pnl: float


def _handle_order(order: OrderRequest, bar: Bar, position: float, entry_price: Optional[float]) -> tuple[float, Optional[float], float, Optional[Trade]]:
    pnl = 0.0
    trade: Optional[Trade] = None
    price = bar.close
    qty = order.quantity

    if order.side == "SIDE_BUY":
        if position < 0 and entry_price is not None:
            pnl = (entry_price - price) * abs(position)
            trade = Trade("COVER", price, bar.timestamp, abs(position), pnl)
            position = 0
            entry_price = None
        if position == 0:
            position = qty
            entry_price = price
            trade = Trade("BUY", price, bar.timestamp, qty, pnl)
    elif order.side == "SIDE_SELL":
        if position > 0 and entry_price is not None:
            pnl = (price - entry_price) * abs(position)
            trade = Trade("SELL", price, bar.timestamp, abs(position), pnl)
            position = 0
            entry_price = None
        if position == 0:
            position = -qty
            entry_price = price
            trade = Trade("SHORT", price, bar.timestamp, qty, pnl)

    return position, entry_price, pnl, trade


def run_backtest() -> None:
    load_dotenv()
    log = get_logger("backtest")

    account_id = os.getenv("FINAM_ACCOUNT_ID")
    if not account_id:
        raise RuntimeError("FINAM_ACCOUNT_ID is required")

    with open("configs/main.yaml", "r", encoding="utf-8") as handle:
        main_cfg = yaml.safe_load(handle) or {}
    with open("configs/strategy_ma_cross.yaml", "r", encoding="utf-8") as handle:
        strat_cfg = yaml.safe_load(handle) or {}

    strat_cfg["account_id"] = account_id
    strategy = MACrossStrategy(
        name=strat_cfg["name"],
        account_id=strat_cfg["account_id"],
        symbol=strat_cfg["symbol"],
        sma_short=int(strat_cfg["sma_short"]),
        sma_long=int(strat_cfg["sma_long"]),
        quantity=float(strat_cfg["quantity"]),
    )

    client = FinamRestClient()
    try:
        data_adapter = FinamRestDataAdapter(client)
        timeframe = main_cfg.get("timeframe", "TIME_FRAME_M1")
        days_back = int(main_cfg.get("days_back", 1))
        end_ts = int(time.time())
        start_ts = end_ts - days_back * 86400
        bars = list(data_adapter.get_bars(strat_cfg["symbol"], timeframe, start_ts, end_ts))

        position = 0.0
        entry_price: Optional[float] = None
        trades: List[Trade] = []
        total_pnl = 0.0

        for bar in bars:
            order = strategy.on_bar(bar)
            if not order:
                continue
            position, entry_price, pnl, trade = _handle_order(order, bar, position, entry_price)
            total_pnl += pnl
            if trade:
                trades.append(trade)

        log.info("Backtest trades: %s", len(trades))
        log.info("Backtest total PnL: %.2f", total_pnl)
        if position != 0 and entry_price is not None:
            unrealized = (bars[-1].close - entry_price) * position
            log.info("Unrealized PnL: %.2f", unrealized)
    finally:
        client.close()


if __name__ == "__main__":
    run_backtest()
