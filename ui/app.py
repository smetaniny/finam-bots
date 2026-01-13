from __future__ import annotations

import os
from typing import List

import numpy as np
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
FIGURE_POINTS = 100
TIMEFRAME_FIGURE_MULTIPLIER = {
    "TIME_FRAME_H1": 1.0,
    "TIME_FRAME_H4": 1.5,
    "TIME_FRAME_D": 2.0,
}


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
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _apply_time_shift(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df.empty:
        return df
    shift = {
        "TIME_FRAME_H1": pd.Timedelta(hours=1),
        "TIME_FRAME_H4": pd.Timedelta(hours=1),
    }.get(timeframe)
    result = df.copy()
    result["timestamp"] = result["timestamp"].dt.tz_convert(MOSCOW_TZ)
    if shift is not None:
        result["timestamp"] = result["timestamp"] + shift
    return result


def _append_synthetic_bar(df: pd.DataFrame, timeframe: str, last_quote_value: float | None) -> pd.DataFrame:
    if df.empty or last_quote_value is None:
        return df
    now_msk = pd.Timestamp.now(tz=MOSCOW_TZ)
    if timeframe == "TIME_FRAME_H1":
        current_mark = now_msk.floor("H") + pd.Timedelta(hours=1)
    elif timeframe == "TIME_FRAME_H4":
        current_mark = now_msk.floor("4H") + pd.Timedelta(hours=1)
    elif timeframe == "TIME_FRAME_D":
        current_mark = now_msk.normalize()
    else:
        return df
    synthetic = pd.DataFrame(
        [
            {
                "timestamp": current_mark,
                "open": last_quote_value,
                "high": last_quote_value,
                "low": last_quote_value,
                "close": last_quote_value,
                "volume": 0.0,
            }
        ]
    )
    combined = pd.concat(
        [
            df,
            synthetic,
        ],
        ignore_index=True,
    )
    return combined.drop_duplicates(subset=["timestamp"], keep="last")


def _price_step_from_asset(asset: dict) -> float | None:
    min_step = asset.get("minStep") or asset.get("min_step")
    decimals = asset.get("decimals")
    if min_step is None or decimals is None:
        return None
    try:
        return float(min_step) / (10 ** int(decimals))
    except (TypeError, ValueError):
        return None


def _point_value_from_asset(asset: dict) -> float | None:
    decimals = asset.get("decimals")
    if decimals is None:
        return None
    try:
        return 10 ** (-int(decimals))
    except (TypeError, ValueError):
        return None


def _auto_distance_for_timeframe(point_value: float, timeframe: str) -> float:
    multiplier = TIMEFRAME_FIGURE_MULTIPLIER.get(timeframe, 1.5)
    return point_value * FIGURE_POINTS * multiplier


def _important_extremes(df: pd.DataFrame, window: int, distance: float) -> tuple[pd.Series, pd.Series]:
    """Находит важные минимумы и максимумы по методике Валеева."""
    if df.empty or window < 2:
        return pd.Series(dtype=bool, index=df.index), pd.Series(dtype=bool, index=df.index)
    
    low = df['low'].values
    high = df['high'].values
    close = df['close'].values
    
    n = len(df)
    is_low = np.zeros(n, dtype=bool)
    is_high = np.zeros(n, dtype=bool)
    
    # Минимальное движение для подтверждения
    min_move = max(5.0, distance * 0.05)  # хотя бы 5 пунктов или 5% от distance
    
    for i in range(n):
        # ========== ПРОВЕРКА ВАЖНОГО МИНИМУМА ==========
        # Проверяем, что это локальный минимум
        if i > 0 and i < n - 1 and low[i] < low[i-1] and low[i] < low[i+1]:
            # 1. Проверка падения слева (хотя бы 1 свеча из window выше)
            left_ok = False
            look_left = min(window, i)
            for j in range(1, look_left + 1):
                if low[i] < low[i-j] - min_move:
                    left_ok = True
                    break
            
            # 2. Проверка роста справа (хотя бы 1 свеча из window/2 выше на distance)
            right_ok = False
            look_right = min(window // 2, n - i - 1)
            for j in range(1, look_right + 1):
                if low[i+j] > low[i] + distance:
                    right_ok = True
                    break
            
            # 3. Проверка закрытия (не на самом минимуме)
            candle_range = high[i] - low[i]
            close_ok = candle_range > 0 and (close[i] - low[i]) > (candle_range * 0.15)
            
            # 4. Проверка на ложный пробой (цена не падала сильно ниже)
            false_break = False
            for j in range(1, min(window, n - i - 1) + 1):
                if low[i+j] < low[i] - (distance * 0.4):
                    false_break = True
                    break
            
            is_low[i] = left_ok and right_ok and close_ok and not false_break
        
        # ========== ПРОВЕРКА ВАЖНОГО МАКСИМУМА ==========
        # Проверяем, что это локальный максимум
        if i > 0 and i < n - 1 and high[i] > high[i-1] and high[i] > high[i+1]:
            # 1. Проверка роста слева (хотя бы 1 свеча из window ниже)
            left_ok = False
            look_left = min(window, i)
            for j in range(1, look_left + 1):
                if high[i] > high[i-j] + min_move:
                    left_ok = True
                    break
            
            # 2. Проверка падения справа (хотя бы 1 свеча из window/2 ниже на distance)
            right_ok = False
            look_right = min(window // 2, n - i - 1)
            for j in range(1, look_right + 1):
                if high[i+j] < high[i] - distance:
                    right_ok = True
                    break
            
            # 3. Проверка закрытия (не на самом максимуме)
            candle_range = high[i] - low[i]
            close_ok = candle_range > 0 and (high[i] - close[i]) > (candle_range * 0.15)
            
            # 4. Проверка на ложный пробой (цена не поднималась сильно выше)
            false_break = False
            for j in range(1, min(window, n - i - 1) + 1):
                if high[i+j] > high[i] + (distance * 0.4):
                    false_break = True
                    break
            
            is_high[i] = left_ok and right_ok and close_ok and not false_break
    
    return pd.Series(is_low, index=df.index), pd.Series(is_high, index=df.index)


def _add_zones_from_extremes(fig: go.Figure, df: pd.DataFrame, is_low: pd.Series, is_high: pd.Series) -> None:
    """Добавляет зоны поддержки/сопротивления на график."""
    if df.empty:
        return
    
    # ЗОНЫ ПОДДЕРЖКИ (от важных минимумов)
    support_points = df[is_low]
    if not support_points.empty:
        for _, row in support_points.iterrows():
            if row['close'] > row['low']:  # Закрытие должно быть выше минимума
                fig.add_hrect(
                    y0=row['low'],
                    y1=row['close'],
                    fillcolor="rgba(34, 197, 94, 0.15)",
                    line_width=1,
                    line_color="rgba(34, 197, 94, 0.5)",
                    annotation_text=f"S: {row['low']:.1f}-{row['close']:.1f}",
                    annotation_position="top left",
                    annotation_font_size=10,
                    annotation_font_color="#22c55e"
                )
    
    # ЗОНЫ СОПРОТИВЛЕНИЯ (от важных максимумов)
    resistance_points = df[is_high]
    if not resistance_points.empty:
        for _, row in resistance_points.iterrows():
            if row['high'] > row['close']:  # Максимум должен быть выше закрытия
                fig.add_hrect(
                    y0=row['close'],
                    y1=row['high'],
                    fillcolor="rgba(239, 68, 68, 0.15)",
                    line_width=1,
                    line_color="rgba(239, 68, 68, 0.5)",
                    annotation_text=f"R: {row['close']:.1f}-{row['high']:.1f}",
                    annotation_position="bottom left",
                    annotation_font_size=10,
                    annotation_font_color="#ef4444"
                )


def _add_important_extremes_markers(fig: go.Figure, df: pd.DataFrame, is_low: pd.Series, is_high: pd.Series) -> None:
    """Добавляет маркеры важных экстремумов на график."""
    if df.empty:
        return
    
    low_points = df[is_low]
    high_points = df[is_high]
    
    # Вычисляем отступ для маркеров
    price_range = (df['high'].max() - df['low'].min())
    offset = price_range * 0.02 if price_range > 0 else 10.0
    
    if not low_points.empty:
        fig.add_trace(
            go.Scatter(
                x=low_points["timestamp"],
                y=low_points["low"] - offset,
                mode="markers+text",
                marker=dict(symbol="triangle-up", size=12, color="#22c55e", line=dict(width=2, color="white")),
                text=[f"{l:.1f}" for l in low_points["low"]],
                textposition="top center",
                name="Важные минимумы",
                showlegend=False,
                opacity=0.9,
            )
        )
    
    if not high_points.empty:
        fig.add_trace(
            go.Scatter(
                x=high_points["timestamp"],
                y=high_points["high"] + offset,
                mode="markers+text",
                marker=dict(symbol="triangle-down", size=12, color="#ef4444", line=dict(width=2, color="white")),
                text=[f"{h:.1f}" for h in high_points["high"]],
                textposition="bottom center",
                name="Важные максимумы",
                showlegend=False,
                opacity=0.9,
            )
        )


def _plot_candles(
    df: pd.DataFrame,
    title: str,
    markers_df: pd.DataFrame | None = None,
    important_window: int = 10,
    important_distance: float = 150.0,
) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                increasing_line_color="#22c55e",
                increasing_fillcolor="#22c55e",
                decreasing_line_color="#ef4444",
                decreasing_fillcolor="#ef4444",
                name="Цена",
            )
        ]
    )
    
    _add_pattern_labels(fig, df)
    
    if markers_df is None:
        markers_df = df
    
    # Находим важные экстремумы
    is_low, is_high = _important_extremes(markers_df, important_window, important_distance)
    
    # Добавляем зоны поддержки/сопротивления
    _add_zones_from_extremes(fig, markers_df, is_low, is_high)
    
    # Добавляем маркеры экстремумов
    _add_important_extremes_markers(fig, markers_df, is_low, is_high)
    
    fig.update_layout(
        title=title,
        height=560,
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor="#0b0f14",
        paper_bgcolor="#0b0f14",
        font=dict(color="#e5e7eb", size=12),
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor="#1f2937",
            type="category",
            categoryorder="array",
            categoryarray=df["timestamp"].tolist(),
            rangeslider=dict(visible=False),
            showspikes=True,
            spikemode="across",
            spikecolor="#374151",
            spikethickness=1,
            spikesnap="cursor",
        ),
        yaxis=dict(
            title="",
            showgrid=True,
            gridcolor="#1f2937",
            showspikes=True,
            spikemode="across",
            spikecolor="#374151",
            spikethickness=1,
            spikesnap="cursor",
            fixedrange=False,
        ),
        dragmode="pan",
        hovermode="x unified",
    )
    
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
    descriptions = {
        "PB": "Пин‑бар — свеча с длинным хвостом (тенью) и маленьким телом у одного края, сигнал возможного разворота.",
        "E": "Поглощение (Engulfing) — свеча, тело которой полностью перекрывает тело предыдущей в противоположную сторону.",
        "KR": "Key Reversal — разворотный бар: делает новый экстремум против тренда и закрывается в противоположной стороне (часто с большим объемом).",
    }
    labels = []
    for _, row in flagged.iterrows():
        parts = []
        hover_parts = []
        if row["pattern_pb"]:
            parts.append("PB")
            hover_parts.append(descriptions["PB"])
        if row["pattern_e"]:
            parts.append("E")
            hover_parts.append(descriptions["E"])
        if row["pattern_kr"]:
            parts.append("KR")
            hover_parts.append(descriptions["KR"])
        if parts:
            labels.append((row["timestamp"], row["high"], "<br>".join(parts), "<br>".join(hover_parts)))

    if not labels:
        return
    label_df = pd.DataFrame(labels, columns=["timestamp", "high", "label", "hover"])
    offset = (df["high"] - df["low"])
    offset = offset.mask(offset == 0, df["high"] * 0.001)
    offset_val = float(offset.median()) if not offset.empty else 0.0
    label_df["y"] = label_df["high"] + offset_val * 0.2
    fig.add_trace(
        go.Scatter(
            x=label_df["timestamp"],
            y=label_df["y"],
            text=label_df["label"],
            hovertext=label_df["hover"],
            mode="text",
            textposition="top center",
            name="Patterns",
            hovertemplate="%{hovertext}<extra></extra>",
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
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(MOSCOW_TZ)
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
        st.subheader("Важные экстремумы")
        auto_distance = None
        auto_step = None
        auto_point = None
        try:
            asset_info = client.get_asset(symbol, account_id=account_id)
            auto_step = _price_step_from_asset(asset_info)
            auto_point = _point_value_from_asset(asset_info)
            if auto_point is not None:
                auto_distance = _auto_distance_for_timeframe(auto_point, timeframe)
        except Exception:
            auto_distance = None
            auto_step = None
            auto_point = None
        
        use_auto_distance = st.checkbox(
            "D из параметров инструмента",
            value=auto_distance is not None,
        )
        
        if auto_distance is not None:
            multiplier = TIMEFRAME_FIGURE_MULTIPLIER.get(timeframe, 1.5)
            st.caption(
                f"Авто D = {auto_distance:g} (1 фигура = {FIGURE_POINTS} пунктов, "
                f"пункт = {auto_point or 0:g}, шаг цены = {auto_step or 0:g}, "
                f"множитель {multiplier:g})"
            )
        
        important_distance_manual = st.number_input(
            "Порог расстояния D (вручную)",
            min_value=0.0,
            value=float(auto_distance or 150.0),
            step=1.0,
        )
        
        important_window = st.number_input(
            "Окно анализа (свечей)",
            min_value=2,
            max_value=50,
            value=10,
            step=1,
            help="Сколько свечей анализировать слева и справа от экстремума"
        )
        
        important_distance = auto_distance if use_auto_distance and auto_distance is not None else important_distance_manual
        
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
        
        last_quote_value = None
        try:
            last_quote = client.get_last_quote(symbol)
            last_quote_value = float(last_quote.get("quote", {}).get("last", {}).get("value"))
        except Exception:
            last_quote_value = None

        tabs = st.tabs(["График", "Свечи (O H L C)"])
        
        with tabs[0]:
            bars = service.get_bars(symbol, timeframe, days_back)
            df = _bars_to_df(bars)
            
            if df.empty:
                st.warning("Нет данных по свечам")
            else:
                df_plot_base = _apply_time_shift(df, timeframe)
                
                # Находим важные экстремумы
                is_low, is_high = _important_extremes(df_plot_base, important_window, important_distance)
                
                st.caption(f"Важные экстремумы: минимумов={int(is_low.sum())}, максимумов={int(is_high.sum())}")
                
                if int(is_low.sum()) + int(is_high.sum()) == 0:
                    st.info("По текущим параметрам не найдено важных экстремумов. Попробуйте уменьшить расстояние D или увеличить окно.")
                else:
                    with st.expander("Список важных экстремумов"):
                        points = []
                        for idx, row in df_plot_base[is_low].iterrows():
                            points.append({
                                "type": "MIN",
                                "timestamp": row["timestamp"],
                                "low": row["low"],
                                "high": row["high"],
                                "close": row["close"],
                                "зона_поддержки": f"{row['low']:.1f}-{row['close']:.1f}"
                            })
                        for idx, row in df_plot_base[is_high].iterrows():
                            points.append({
                                "type": "MAX",
                                "timestamp": row["timestamp"],
                                "low": row["low"],
                                "high": row["high"],
                                "close": row["close"],
                                "зона_сопротивления": f"{row['close']:.1f}-{row['high']:.1f}"
                            })
                        
                        if points:
                            st.dataframe(pd.DataFrame(points), use_container_width=True)
                
                df_plot = _append_synthetic_bar(df_plot_base, timeframe, last_quote_value)
                
                fig = _plot_candles(
                    df_plot,
                    f"{symbol} {timeframe}",
                    markers_df=df_plot_base,
                    important_window=important_window,
                    important_distance=important_distance,
                )
                
                st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            st.subheader("Свечи (O H L C)")
            for tf in ("TIME_FRAME_D", "TIME_FRAME_H4", "TIME_FRAME_H1"):
                tf_bars = service.get_bars(symbol, tf, ranges[tf])
                tf_df = _bars_to_df(tf_bars)
                st.markdown(f"**{tf} (последние {ranges[tf]} дней)**")
                if tf_df.empty:
                    st.info("Нет данных")
                    continue
                tf_df = _apply_time_shift(tf_df, tf)
                tf_df = _append_synthetic_bar(tf_df, tf, last_quote_value)
                ohlcv = tf_df[["timestamp", "open", "high", "low", "close"]].copy()
                ohlcv = ohlcv.rename(columns={"open": "O", "high": "H", "low": "L", "close": "C"})
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

        if "df" in locals() and not df.empty:
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
        st.write({
            "mode": mode,
            "max_position_per_symbol": limits.get("max_position_per_symbol"),
            "max_trades_per_day": limits.get("max_trades_per_day"),
            "daily_loss_limit": limits.get("daily_loss_limit"),
            "hard_stop": False,
        })
        
    except Exception as exc:
        log.error("Dashboard error: %s", exc)
        st.error(f"Ошибка: {exc}")
    finally:
        client.close()


if __name__ == "__main__":
    main()