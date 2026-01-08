from __future__ import annotations

import os
from typing import List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from dotenv import load_dotenv

from adapters import FinamRestClient
from core import get_logger
from core.interfaces import Bar
from ui.data_service import FinamDataService


def _bars_to_df(bars: List[Bar]) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]
    )
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _plot_candles(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
            )
        ]
    )
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Price", height=520)
    return fig


def _add_trade_markers(fig: go.Figure, trades: list, symbol: str) -> None:
    points = []
    for trade in trades:
        if trade.get("symbol") != symbol:
            continue
        ts = trade.get("timestamp")
        price_val = trade.get("price", {}).get("value")
        side = trade.get("side", "")
        if ts and price_val is not None:
            points.append((ts, float(price_val), side))
    if not points:
        return
    df = pd.DataFrame(points, columns=["timestamp", "price", "side"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    colors = df["side"].map({"SIDE_BUY": "green", "SIDE_SELL": "red"}).fillna("blue")
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["price"],
            mode="markers",
            marker=dict(color=colors, size=8, symbol="circle"),
            name="Trades",
        )
    )


def _tail_log(path: str, max_lines: int = 200) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    return "".join(lines[-max_lines:])


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Finam Bot Dashboard", layout="wide")
    log = get_logger("dashboard")

    account_id = os.getenv("FINAM_ACCOUNT_ID")
    if not account_id:
        st.error("FINAM_ACCOUNT_ID не задан в .env")
        return

    st.title("Finam Bot Dashboard")

    client = FinamRestClient()
    service = FinamDataService(client)

    with st.sidebar:
        st.header("Параметры")
        symbol = st.selectbox(
            "Инструмент (ticker@mic)",
            [
                "SBER@MISX",
                "GAZP@MISX",
                "LKOH@MISX",
                "VTBR@MISX",
                "GMKN@MISX",
                "ROSN@MISX",
                "SNGS@MISX",
                "ALRS@MISX",
                "CHMF@MISX",
                "NVTK@MISX",
            ],
            index=0,
        )
        timeframe = st.selectbox(
            "Таймфрейм",
            [
                "TIME_FRAME_M1",
                "TIME_FRAME_M5",
                "TIME_FRAME_M15",
                "TIME_FRAME_M30",
                "TIME_FRAME_H1",
                "TIME_FRAME_H2",
                "TIME_FRAME_H4",
                "TIME_FRAME_H8",
                "TIME_FRAME_D",
                "TIME_FRAME_W",
                "TIME_FRAME_MN",
                "TIME_FRAME_QR",
            ],
            index=0,
        )
        days_back = st.slider("Глубина (дней)", 1, 30, 7)
        trades_limit = st.slider("Сделки (лимит)", 10, 200, 50)
        st.subheader("Стоп/тейк (визуально)")
        stop_loss_pct = st.number_input("Stop Loss %", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
        take_profit_pct = st.number_input("Take Profit %", min_value=0.0, max_value=200.0, value=0.0, step=0.1)

    try:
        with open("configs/main.yaml", "r", encoding="utf-8") as handle:
            main_cfg = yaml.safe_load(handle) or {}

        bars = service.get_bars(symbol, timeframe, days_back)
        df = _bars_to_df(bars)
        if df.empty:
            st.warning("Нет данных по свечам")
        else:
            fig = _plot_candles(df, f"{symbol} {timeframe}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Позиции")
            account = service.get_account(account_id)
            positions = account.get("positions", [])
            daily_pnl = 0.0
            unrealized_pnl = 0.0
            for pos in positions:
                daily_pnl += float(pos.get("daily_pnl", {}).get("value", 0) or 0)
                unrealized_pnl += float(pos.get("unrealized_pnl", {}).get("value", 0) or 0)
            st.metric("PnL (дневной)", f"{daily_pnl:.2f}")
            st.metric("PnL (нереализ.)", f"{unrealized_pnl:.2f}")
            if positions:
                st.dataframe(pd.DataFrame(positions))
            else:
                st.info("Позиции отсутствуют")

        with col2:
            st.subheader("Сделки")
            trades = service.get_trades(account_id, limit=trades_limit).get("trades", [])
            if trades:
                st.dataframe(pd.DataFrame(trades))
            else:
                st.info("Сделок нет")

        if not df.empty:
            pos_match = next((p for p in positions if p.get("symbol") == symbol), None)
            if pos_match and (stop_loss_pct > 0 or take_profit_pct > 0):
                qty = float(pos_match.get("quantity", {}).get("value", 0) or 0)
                base_price = float(pos_match.get("average_price", {}).get("value", 0) or df["close"].iloc[-1])
                if qty != 0:
                    if qty > 0:
                        if stop_loss_pct > 0:
                            fig.add_hline(y=base_price * (1 - stop_loss_pct / 100), line_dash="dot", line_color="red")
                        if take_profit_pct > 0:
                            fig.add_hline(y=base_price * (1 + take_profit_pct / 100), line_dash="dot", line_color="green")
                    else:
                        if stop_loss_pct > 0:
                            fig.add_hline(y=base_price * (1 + stop_loss_pct / 100), line_dash="dot", line_color="red")
                        if take_profit_pct > 0:
                            fig.add_hline(y=base_price * (1 - take_profit_pct / 100), line_dash="dot", line_color="green")

            _add_trade_markers(fig, trades, symbol)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Ордера")
        orders = service.get_orders(account_id).get("orders", [])
        if orders:
            st.dataframe(pd.DataFrame(orders))
        else:
            st.info("Активных ордеров нет")

        st.subheader("Логи")
        log_path = os.getenv("FINAM_LOG_FILE", "logs/finam-bots.log")
        log_text = _tail_log(log_path)
        if log_text:
            st.text_area("Последние события", log_text, height=240)
        else:
            st.info("Логи пока пустые")

        st.subheader("Статус бота")
        mode = main_cfg.get("mode", "paper")
        limits = main_cfg.get("risk_limits", {})
        st.write(
            {
                "mode": mode,
                "max_position_per_symbol": limits.get("max_position_per_symbol"),
                "max_trades_per_day": limits.get("max_trades_per_day"),
                "daily_loss_limit": limits.get("daily_loss_limit"),
                "hard_stop": False,
            }
        )
    except Exception as exc:
        log.error("Dashboard error: %s", exc)
        st.error(f"Ошибка: {exc}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
