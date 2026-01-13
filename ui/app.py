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

    low = df["low"].values
    high = df["high"].values
    close = df["close"].values
    n = len(df)
    is_low = np.zeros(n, dtype=bool)
    is_high = np.zeros(n, dtype=bool)

    left_window = max(3, min(5, window))
    right_window = max(5, min(7, window))

    if n < left_window + right_window + 1:
        return pd.Series(is_low, index=df.index), pd.Series(is_high, index=df.index)

    for i in range(left_window, n - right_window):
        left_lows = low[i - left_window : i]
        right_lows = low[i + 1 : i + right_window + 1]
        window_lows = low[i - left_window : i + right_window + 1]

        is_lowest = low[i] == np.min(window_lows)
        left_falls = low[i] < np.min(left_lows)
        right_rises = np.max(right_lows) >= low[i] + distance
        right_no_retest = np.min(right_lows) > low[i]
        candle_ok = close[i] > low[i]
        no_late_retest = np.min(low[i + 1 :]) > low[i]

        if is_lowest and left_falls and right_rises and right_no_retest and candle_ok and no_late_retest:
            is_low[i] = True

        left_highs = high[i - left_window : i]
        right_highs = high[i + 1 : i + right_window + 1]
        window_highs = high[i - left_window : i + right_window + 1]

        is_highest = high[i] == np.max(window_highs)
        left_rises = high[i] > np.max(left_highs)
        right_drops = np.min(right_highs) <= high[i] - distance
        right_no_retest_high = np.max(right_highs) < high[i]
        candle_ok = close[i] < high[i]
        no_late_retest_high = np.max(high[i + 1 :]) < high[i]

        if is_highest and left_rises and right_drops and right_no_retest_high and candle_ok and no_late_retest_high:
            is_high[i] = True

    return pd.Series(is_low, index=df.index), pd.Series(is_high, index=df.index)


def _filter_close_extremes(
    is_low: pd.Series,
    is_high: pd.Series,
    df: pd.DataFrame,
    min_gap: float = 30.0,
) -> tuple[pd.Series, pd.Series]:
    """Фильтрует слишком близкие экстремумы."""
    filtered_low = is_low.copy()
    filtered_high = is_high.copy()

    # Фильтрация минимумов (по времени)
    low_indices = df[filtered_low].index.tolist()
    if len(low_indices) > 1:
        low_indices.sort()

        i = 0
        while i < len(low_indices):
            current_idx = low_indices[i]
            current_val = df.at[current_idx, "low"]

            j = i + 1
            while j < len(low_indices):
                next_idx = low_indices[j]
                next_val = df.at[next_idx, "low"]

                if abs(next_val - current_val) < min_gap:
                    current_body = abs(df.at[current_idx, "close"] - df.at[current_idx, "open"])
                    next_body = abs(df.at[next_idx, "close"] - df.at[next_idx, "open"])

                    if current_body >= next_body:
                        filtered_low.at[next_idx] = False
                        low_indices.pop(j)
                    else:
                        filtered_low.at[current_idx] = False
                        low_indices.pop(i)
                        break
                else:
                    j += 1
            else:
                i += 1

    # Фильтрация максимумов (по времени)
    high_indices = df[filtered_high].index.tolist()
    if len(high_indices) > 1:
        high_indices.sort()

        i = 0
        while i < len(high_indices):
            current_idx = high_indices[i]
            current_val = df.at[current_idx, "high"]

            j = i + 1
            while j < len(high_indices):
                next_idx = high_indices[j]
                next_val = df.at[next_idx, "high"]

                if abs(next_val - current_val) < min_gap:
                    current_body = abs(df.at[current_idx, "close"] - df.at[current_idx, "open"])
                    next_body = abs(df.at[next_idx, "close"] - df.at[next_idx, "open"])

                    if current_body >= next_body:
                        filtered_high.at[next_idx] = False
                        high_indices.pop(j)
                    else:
                        filtered_high.at[current_idx] = False
                        high_indices.pop(i)
                        break
                else:
                    j += 1
            else:
                i += 1

    return filtered_low, filtered_high


def _add_zones_from_extremes(fig: go.Figure, df: pd.DataFrame, is_low: pd.Series, is_high: pd.Series) -> None:
    """Добавляет зоны поддержки/сопротивления на график."""
    if df.empty:
        return
    
    # ЗОНЫ ПОДДЕРЖКИ (от важных минимумов)
    support_points = df[is_low]
    if not support_points.empty:
        for _, row in support_points.iterrows():
            zone_low = row['low']
            zone_high = row['close']
            # Если закрытие на минимуме или ниже, немного расширяем зону
            if zone_high <= zone_low:
                zone_high = zone_low + (df['high'].max() - df['low'].min()) * 0.001
            
            if zone_high > zone_low:
                fig.add_hrect(
                    y0=zone_low,
                    y1=zone_high,
                    fillcolor="rgba(34, 197, 94, 0.15)",
                    line_width=1,
                    line_color="rgba(34, 197, 94, 0.5)",
                    annotation_text=f"S: {zone_low:.1f}-{zone_high:.1f}",
                    annotation_position="top left",
                    annotation_font_size=10,
                    annotation_font_color="#22c55e"
                )
    
    # ЗОНЫ СОПРОТИВЛЕНИЯ (от важных максимумов)
    resistance_points = df[is_high]
    if not resistance_points.empty:
        for _, row in resistance_points.iterrows():
            zone_low = row['close']
            zone_high = row['high']
            # Если закрытие на максимуме или выше, немного расширяем зону
            if zone_low >= zone_high:
                zone_low = zone_high - (df['high'].max() - df['low'].min()) * 0.001
            
            if zone_high > zone_low:
                fig.add_hrect(
                    y0=zone_low,
                    y1=zone_high,
                    fillcolor="rgba(239, 68, 68, 0.15)",
                    line_width=1,
                    line_color="rgba(239, 68, 68, 0.5)",
                    annotation_text=f"R: {zone_low:.1f}-{zone_high:.1f}",
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
    important_window: int = 15,  # Увеличил для H4
    important_distance: float = 100.0,  # Уменьшил для H4
    show_extremes: bool = True,
    show_extremes_markers: bool = True,
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
    
    if show_extremes:
        # Находим важные экстремумы
        is_low, is_high = _important_extremes(markers_df, important_window, important_distance)

        # Добавляем зоны поддержки/сопротивления
        _add_zones_from_extremes(fig, markers_df, is_low, is_high)

        # Добавляем маркеры экстремумов
        if show_extremes_markers:
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
        
        # РЕКОМЕНДОВАННЫЕ НАСТРОЙКИ ДЛЯ RTS H4:
        if timeframe == "TIME_FRAME_H4":
            default_distance = 65.0
            default_window = 12
            st.info("Для H4 (RTS) рекомендую: D=65, окно=12")
        elif timeframe == "TIME_FRAME_H1":
            default_distance = 50.0
            default_window = 20
            st.info("Для H1 рекомендую: D=50, окно=20")
        else:  # D1
            default_distance = 200.0
            default_window = 20
            st.info("Для D1 рекомендую: D=200, окно=20")
        
        important_distance_manual = st.number_input(
            "Порог расстояния D",
            min_value=0.0,
            value=float(auto_distance or default_distance),
            step=5.0,
            help="Минимальное движение после экстремума. RTS H4: 80-120 пунктов"
        )
        
        important_window = st.number_input(
            "Окно анализа (свечей)",
            min_value=2,
            max_value=50,
            value=default_window,
            step=1,
            help="Сколько свечей анализировать слева и справа от экстремума"
        )
        
        important_distance = auto_distance if use_auto_distance and auto_distance is not None else important_distance_manual
        
        ranges = {
            "TIME_FRAME_D": 60,
            "TIME_FRAME_H4": 60,
            "TIME_FRAME_H1": 30,
        }
        days_back = ranges[timeframe]
        st.caption("Глубина: D=60 дней, H4=60 дней, H1=30 дней")
        
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
                show_extremes = True
                show_extremes_markers = timeframe == "TIME_FRAME_H4"
                extremes_df = df_plot_base

                if timeframe != "TIME_FRAME_H4":
                    h4_bars = service.get_bars(symbol, "TIME_FRAME_H4", ranges["TIME_FRAME_H4"])
                    h4_df = _bars_to_df(h4_bars)
                    h4_df = _apply_time_shift(h4_df, "TIME_FRAME_H4")
                    if not h4_df.empty:
                        extremes_df = h4_df

                if show_extremes and timeframe == "TIME_FRAME_H4":
                    # Находим важные экстремумы
                    is_low, is_high = _important_extremes(extremes_df, important_window, important_distance)
                    
                    found_lows = int(is_low.sum())
                    found_highs = int(is_high.sum())
                    st.caption(f"Важные экстремумы: минимумов={found_lows}, максимумов={found_highs}")
                    
                    if found_lows + found_highs == 0:
                        st.warning("Не найдено важных экстремумов. Возможные причины:")
                        st.write("1. Параметр D слишком большой - уменьшите до 50-100 для H4")
                        st.write("2. Окно слишком маленькое - увеличьте до 10-20")
                        st.write("3. На графике действительно нет четких разворотов")
                    else:
                        with st.expander("Список важных экстремумов"):
                            points = []
                            for idx, row in extremes_df[is_low].iterrows():
                                points.append({
                                    "type": "MIN",
                                    "timestamp": row["timestamp"],
                                    "low": row["low"],
                                    "high": row["high"],
                                    "close": row["close"],
                                    "зона_поддержки": f"{row['low']:.1f}-{row['close']:.1f}"
                                })
                            for idx, row in extremes_df[is_high].iterrows():
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
                elif timeframe != "TIME_FRAME_H4":
                    st.caption("Зоны поддержки/сопротивления рассчитаны по H4 и показаны поверх текущего таймфрейма.")
                
                df_plot = _append_synthetic_bar(df_plot_base, timeframe, last_quote_value)
                
                fig = _plot_candles(
                    df_plot,
                    f"{symbol} {timeframe}",
                    markers_df=extremes_df,
                    important_window=important_window,
                    important_distance=important_distance,
                    show_extremes=show_extremes,
                    show_extremes_markers=show_extremes_markers,
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
