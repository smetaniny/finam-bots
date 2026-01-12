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

MOSCOW_TZ = "Europe/Moscow"


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
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
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
    _add_pattern_labels(fig, df)
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Price", height=520)
    return fig


def _pattern_flags(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    body = (result["close"] - result["open"]).abs()
    rng = (result["high"] - result["low"]).replace(0, 1e-9)
    upper = result["high"] - result[["open", "close"]].max(axis=1)
    lower = result[["open", "close"]].min(axis=1) - result["low"]

    pin_bar = ((upper >= 2 * body) & (lower <= 0.3 * body)) | (
        (lower >= 2 * body) & (upper <= 0.3 * body)
    )

    prev_open = result["open"].shift(1)
    prev_close = result["close"].shift(1)
    prev_body = (prev_close - prev_open).abs()
    engulfing = (body > prev_body) & (
        (result["open"] <= prev_close) & (result["close"] >= prev_open)
    ) | (
        (result["open"] >= prev_close) & (result["close"] <= prev_open)
    )

    prev_high = result["high"].shift(1)
    prev_low = result["low"].shift(1)
    prev_dir = (prev_close - prev_open).fillna(0)
    curr_dir = (result["close"] - result["open"]).fillna(0)
    key_reversal = (
        ((result["high"] > prev_high) & (curr_dir < 0) & (prev_dir > 0))
        | ((result["low"] < prev_low) & (curr_dir > 0) & (prev_dir < 0))
    )

    result["pattern_pb"] = pin_bar.fillna(False)
    result["pattern_e"] = engulfing.fillna(False)
    result["pattern_kr"] = key_reversal.fillna(False)
    return result


def _add_pattern_labels(fig: go.Figure, df: pd.DataFrame) -> None:
    flagged = _pattern_flags(df)
    if flagged.empty:
        return
    labels = []
    for _, row in flagged.iterrows():
        parts = []
        if row["pattern_pb"]:
            parts.append("PB")
        if row["pattern_e"]:
            parts.append("E")
        if row["pattern_kr"]:
            parts.append("KR")
        if parts:
            labels.append((row["timestamp"], row["high"], " ".join(parts)))

    if not labels:
        return
    label_df = pd.DataFrame(labels, columns=["timestamp", "high", "label"])
    offset = (df["high"] - df["low"])
    offset = offset.mask(offset == 0, df["high"] * 0.001)
    offset_val = float(offset.median()) if not offset.empty else 0.0
    label_df["y"] = label_df["high"] + offset_val * 0.2
    fig.add_trace(
        go.Scatter(
            x=label_df["timestamp"],
            y=label_df["y"],
            text=label_df["label"],
            mode="text",
            textposition="top center",
            name="Patterns",
            showlegend=False,
        )
    )


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
                "RMH6@RTSX",
            ],
            index=0,
        )
        timeframe = st.selectbox(
            "Таймфрейм",
            [
                "TIME_FRAME_H1",
                "TIME_FRAME_H4",
                "TIME_FRAME_D",
            ],
            index=0,
        )
        ranges = {
            "TIME_FRAME_D": 60,
            "TIME_FRAME_H4": 30,
            "TIME_FRAME_H1": 30,
        }
        days_back = ranges[timeframe]
        st.caption("Глубина: D=60 дней, H4/H1=30 дней")
        trades_limit = st.slider("Сделки (лимит)", 10, 200, 50)
        st.subheader("Стоп/тейк (визуально)")
        stop_loss_pct = st.number_input("Stop Loss %", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
        take_profit_pct = st.number_input("Take Profit %", min_value=0.0, max_value=200.0, value=0.0, step=0.1)
        if st.button("Проверить MarketData"):
            for check_symbol in ("SBER@MISX", "RMH6@RTSX"):
                try:
                    quote = client.get_last_quote(check_symbol)
                    log.info("Last quote check (%s): %s", check_symbol, quote)
                    st.success(f"{check_symbol}: OK")
                except Exception as exc:
                    log.warning("Last quote check (%s) failed: %s", check_symbol, exc)
                    st.error(f"{check_symbol}: ошибка lastquote")

    try:
        with open("configs/main.yaml", "r", encoding="utf-8") as handle:
            main_cfg = yaml.safe_load(handle) or {}

        try:
            log.info("Clock (server): %s", client.get_clock())
        except Exception:
            log.info("Clock (server): n/a")
        try:
            log.info("Last quote: %s", client.get_last_quote(symbol))
        except Exception:
            log.info("Last quote: n/a")
        bars = service.get_bars(symbol, timeframe, days_back)
        df = _bars_to_df(bars)
        if df.empty:
            st.warning("Нет данных по свечам")
        else:
            fig = _plot_candles(df, f"{symbol} {timeframe}")

        st.subheader("Свечи (O H L C V)")
        last_quote_value = None
        try:
            last_quote = client.get_last_quote(symbol)
            last_quote_value = float(last_quote.get("quote", {}).get("last", {}).get("value"))
        except Exception:
            last_quote_value = None
        for tf in ("TIME_FRAME_D", "TIME_FRAME_H4", "TIME_FRAME_H1"):
            tf_bars = service.get_bars(symbol, tf, ranges[tf])
            tf_df = _bars_to_df(tf_bars)
            st.markdown(f"**{tf} (последние {ranges[tf]} дней)**")
            if tf_df.empty:
                st.info("Нет данных")
                continue
            tf_df = tf_df.copy()
            tf_df["timestamp"] = tf_df["timestamp"].dt.tz_convert(MOSCOW_TZ)
            shift = {
                "TIME_FRAME_H1": pd.Timedelta(hours=1),
                "TIME_FRAME_H4": pd.Timedelta(hours=4),
                "TIME_FRAME_D": pd.Timedelta(days=1),
            }[tf]
            tf_df["timestamp"] = tf_df["timestamp"] + shift
            if tf == "TIME_FRAME_H1" and last_quote_value is not None:
                current_hour = pd.Timestamp.now(tz=MOSCOW_TZ).floor("H")
                last_ts = tf_df["timestamp"].iloc[-1]
                if last_ts < current_hour:
                    tf_df = pd.concat(
                        [
                            tf_df,
                            pd.DataFrame(
                                [
                                    {
                                        "timestamp": current_hour + shift,
                                        "open": last_quote_value,
                                        "high": last_quote_value,
                                        "low": last_quote_value,
                                        "close": last_quote_value,
                                        "volume": 0.0,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )
            ohlcv = tf_df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
            ohlcv = ohlcv.rename(
                columns={"open": "O", "high": "H", "low": "L", "close": "C", "volume": "V"}
            )
            st.dataframe(ohlcv, use_container_width=True)

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
            try:
                trades = service.get_trades(account_id, limit=trades_limit).get("trades", [])
                if trades:
                    st.dataframe(pd.DataFrame(trades))
                else:
                    st.info("Сделок нет")
            except Exception as exc:
                trades = []
                st.error(f"Ошибка загрузки сделок: {exc}")

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
