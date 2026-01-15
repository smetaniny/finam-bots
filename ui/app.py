from __future__ import annotations

import os
from typing import List, Tuple, Dict, Any

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

# Константы для РТС
RTS_MIN_DISTANCE_H1 = 50.0   # Минимум 50 пунктов для H1
RTS_MIN_DISTANCE_H4 = 100.0  # Минимум 100 пунктов для H4 (ваша рекомендация "1 фигуры")
RTS_MIN_DISTANCE_D = 200.0   # Минимум 200 пунктов для D

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


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("@", "_").replace("/", "_")


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


def _trend_down(values: np.ndarray) -> bool:
    """Проверяет, есть ли падающий тренд в массиве значений."""
    if values.size < 2:
        return True
    # Простая проверка: общее движение вниз
    return values[-1] < values[0] or np.mean(np.diff(values) < 0) > 0.5


def _trend_up(values: np.ndarray) -> bool:
    """Проверяет, есть ли растущий тренд в массиве значений."""
    if values.size < 2:
        return True
    # Простая проверка: общее движение вверх
    return values[-1] > values[0] or np.mean(np.diff(values) > 0) > 0.5


def _is_local_extremum(values: np.ndarray, index: int, window: int = 15) -> bool:
    """Проверяет, является ли значение локальным экстремумом в окне."""
    if len(values) < window:
        return False
    
    half_window = window // 2
    start = max(0, index - half_window)
    end = min(len(values), index + half_window + 1)
    
    window_values = values[start:end]
    current_value = values[index]
    
    # Для минимума
    if current_value == np.min(window_values):
        return True
    # Для максимума
    if current_value == np.max(window_values):
        return True
    
    return False


def _find_important_minimum(df: pd.DataFrame, i: int, distance: float) -> Tuple[bool, Dict[str, Any]]:
    """
    Проверяет, является ли свеча i важным минимумом по методике Валеева.
    ПРАКТИЧЕСКИЙ ПОДХОД: Гибкость вместо догмы.
    """
    if i < 3 or i >= len(df) - 7:
        return False, {}
    
    low_i = df.at[i, "low"]
    close_i = df.at[i, "close"]
    
    # ШАГ 1: Найти кандидата (локальный минимум в окне 10-15 свечей)
    look_back = min(7, i)
    look_forward = min(7, len(df) - i - 1)
    
    window_lows = df["low"].iloc[i-look_back:i+look_forward+1].values
    is_local_min = low_i == np.min(window_lows)
    
    if not is_local_min:
        return False, {}
    
    # ШАГ 2: Проверить контекст ДО (3-5 свечей)
    prev_3_5 = df["close"].iloc[max(0, i-5):i].values
    # ГИБКОСТЬ: не обязательно строго каждая свеча, но общая тенденция вниз
    price_falling = len(prev_3_5) == 0 or prev_3_5[0] > low_i
    
    # ШАГ 3: Проверить свечу (ВАЖНОЕ УСЛОВИЕ!)
    candle_ok = close_i > low_i  # Close ДОЛЖЕН БЫТЬ ВЫШЕ Low
    
    if not candle_ok:
        return False, {}
    
    # ШАГ 4: Проверить реакцию ПОСЛЕ (5-7 свечей)
    next_5_7_indices = list(range(i+1, min(len(df), i+8)))
    
    # ГИБКОСТЬ: цена должна уйти от минимума на указанное расстояние
    price_moved_away = False
    max_price_after = 0
    
    if next_5_7_indices:
        highs_after = df["high"].iloc[next_5_7_indices].values
        max_price_after = np.max(highs_after) if len(highs_after) > 0 else 0
        price_moved_away = max_price_after >= low_i + distance
        
        # Если нет движения на полное расстояние, проверяем явный разворот
        if not price_moved_away and len(next_5_7_indices) >= 3:
            # Проверяем, что есть хотя бы 2 зеленые свечи подряд
            closes_after = df["close"].iloc[next_5_7_indices].values
            opens_after = df["open"].iloc[next_5_7_indices].values
            green_candles = sum(1 for c, o in zip(closes_after, opens_after) if c > o)
            if green_candles >= 2 and closes_after[-1] > closes_after[0]:
                price_moved_away = True
    
    # Проверка, что цена не возвращалась к минимуму в ближайшие 5-7 свечей
    no_immediate_retest = True
    if next_5_7_indices:
        lows_after = df["low"].iloc[next_5_7_indices].values
        # ГИБКОСТЬ: допускаем тест зоны, но не обновление минимума
        no_immediate_retest = len(lows_after) == 0 or np.min(lows_after) >= low_i - distance * 0.1
    
    # ШАГ 5: Проверить историю (более мягко!)
    # Ваше правило: "Если позже цена ВЕРНУЛАСЬ и ОБНОВИЛА этот минимум — он НЕ важный"
    # "Если цена лишь тестировала зону, но не обновила минимум — уровень остается важным"
    
    future_lows = df["low"].iloc[i+1:].values if i+1 < len(df) else []
    
    if len(future_lows) > 0:
        # Проверяем, был ли минимум обновлен ПОЗЖЕ
        min_future_low = np.min(future_lows)
        # Считаем обновлением только если новый минимум явно ниже
        level_broken_later = min_future_low < low_i - distance * 0.05
        
        if level_broken_later:
            # Проверяем, был ли это разовый спуск или реальное обновление
            # Смотрим на закрытие после предполагаемого обновления
            broken_idx = i + 1 + np.argmin(future_lows)
            if broken_idx < len(df):
                close_after_break = df.at[broken_idx, "close"]
                # Если закрытие выше минимума - возможно, это был ложный пробой
                if close_after_break > min_future_low + distance * 0.3:
                    level_broken_later = False
    
    # ГИБКОЕ РЕШЕНИЕ: если все основные условия выполнены, считаем важным
    is_important = (is_local_min and price_falling and candle_ok and 
                    price_moved_away and no_immediate_retest)
    
    # Дополнительное условие: минимум должен быть значительным
    # (не мелкая коррекция внутри тренда)
    if is_important:
        # Проверяем амплитуду движения до минимума
        if i > 10:
            highs_before = df["high"].iloc[max(0, i-10):i].values
            max_high_before = np.max(highs_before) if len(highs_before) > 0 else low_i
            drawdown = (max_high_before - low_i) / max_high_before * 100
            # Если просадка менее 1% - возможно, это незначительный минимум
            if drawdown < 1.0:
                is_important = False
    
    zone_info = {
        "type": "support",
        "zone_low": low_i,
        "zone_high": close_i,
        "timestamp": df.at[i, "timestamp"],
        "original_low": low_i,
        "original_close": close_i,
        "max_price_after": max_price_after,
        "move_after": max_price_after - low_i if max_price_after > 0 else 0
    }
    
    return is_important, zone_info


def _find_important_maximum(df: pd.DataFrame, i: int, distance: float) -> Tuple[bool, Dict[str, Any]]:
    """
    Проверяет, является ли свеча i важным максимумом по методике Валеева.
    ПРАКТИЧЕСКИЙ ПОДХОД: Гибкость вместо догмы.
    """
    if i < 3 or i >= len(df) - 7:
        return False, {}
    
    high_i = df.at[i, "high"]
    close_i = df.at[i, "close"]
    
    # ШАГ 1: Найти кандидата
    look_back = min(7, i)
    look_forward = min(7, len(df) - i - 1)
    
    window_highs = df["high"].iloc[i-look_back:i+look_forward+1].values
    is_local_max = high_i == np.max(window_highs)
    
    if not is_local_max:
        return False, {}
    
    # ШАГ 2: Проверить контекст ДО (3-5 свечей)
    prev_3_5 = df["close"].iloc[max(0, i-5):i].values
    # ГИБКОСТЬ: общая тенденция вверх
    price_rising = len(prev_3_5) == 0 or prev_3_5[0] < high_i
    
    # ШАГ 3: Проверить свечу (ВАЖНОЕ УСЛОВИЕ!)
    candle_ok = close_i < high_i  # Close ДОЛЖЕН БЫТЬ НИЖЕ High
    
    if not candle_ok:
        return False, {}
    
    # ШАГ 4: Проверить реакцию ПОСЛЕ (5-7 свечей)
    next_5_7_indices = list(range(i+1, min(len(df), i+8)))
    
    price_moved_away = False
    min_price_after = float("inf")
    
    if next_5_7_indices:
        lows_after = df["low"].iloc[next_5_7_indices].values
        min_price_after = np.min(lows_after) if len(lows_after) > 0 else high_i
        price_moved_away = min_price_after <= high_i - distance
        
        # Если нет движения на полное расстояние, проверяем явный разворот
        if not price_moved_away and len(next_5_7_indices) >= 3:
            closes_after = df["close"].iloc[next_5_7_indices].values
            opens_after = df["open"].iloc[next_5_7_indices].values
            red_candles = sum(1 for c, o in zip(closes_after, opens_after) if c < o)
            if red_candles >= 2 and closes_after[-1] < closes_after[0]:
                price_moved_away = True
    
    # Проверка, что цена не возвращалась к максимуму
    no_immediate_retest = True
    if next_5_7_indices:
        highs_after = df["high"].iloc[next_5_7_indices].values
        no_immediate_retest = len(highs_after) == 0 or np.max(highs_after) <= high_i + distance * 0.1
    
    # ШАГ 5: Проверить историю (более мягко!)
    future_highs = df["high"].iloc[i+1:].values if i+1 < len(df) else []
    
    if len(future_highs) > 0:
        max_future_high = np.max(future_highs)
        level_broken_later = max_future_high > high_i + distance * 0.05
        
        if level_broken_later:
            broken_idx = i + 1 + np.argmax(future_highs)
            if broken_idx < len(df):
                close_after_break = df.at[broken_idx, "close"]
                if close_after_break < max_future_high - distance * 0.3:
                    level_broken_later = False
    
    is_important = (is_local_max and price_rising and candle_ok and 
                    price_moved_away and no_immediate_retest)
    
    # Дополнительная проверка на значимость
    if is_important:
        if i > 10:
            lows_before = df["low"].iloc[max(0, i-10):i].values
            min_low_before = np.min(lows_before) if len(lows_before) > 0 else high_i
            rally = (high_i - min_low_before) / min_low_before * 100
            if rally < 1.0:
                is_important = False
    
    zone_info = {
        "type": "resistance",
        "zone_low": close_i,
        "zone_high": high_i,
        "timestamp": df.at[i, "timestamp"],
        "original_high": high_i,
        "original_close": close_i,
        "min_price_after": min_price_after,
        "move_after": high_i - min_price_after if min_price_after < float("inf") else 0
    }
    
    return is_important, zone_info


def _merge_close_zones(zones: List[Dict[str, Any]], merge_distance: float = 30.0) -> List[Dict[str, Any]]:
    """Объединяет близкие зоны в одну широкую зону."""
    if not zones:
        return []
    
    # Разделяем зоны поддержки и сопротивления
    support_zones = [z for z in zones if z["type"] == "support"]
    resistance_zones = [z for z in zones if z["type"] == "resistance"]
    
    merged_zones = []
    
    # Объединяем зоны поддержки
    if support_zones:
        support_zones.sort(key=lambda x: x["zone_low"])
        current_zone = support_zones[0].copy()
        
        for zone in support_zones[1:]:
            if zone["zone_low"] <= current_zone["zone_high"] + merge_distance:
                # Объединяем зоны
                current_zone["zone_low"] = min(current_zone["zone_low"], zone["zone_low"])
                current_zone["zone_high"] = max(current_zone["zone_high"], zone["zone_high"])
                # Сохраняем все исходные точки для отладки
                if "merged_points" not in current_zone:
                    current_zone["merged_points"] = [current_zone.copy()]
                current_zone["merged_points"].append(zone.copy())
            else:
                merged_zones.append(current_zone)
                current_zone = zone.copy()
        
        merged_zones.append(current_zone)
    
    # Объединяем зоны сопротивления
    if resistance_zones:
        resistance_zones.sort(key=lambda x: x["zone_low"])
        current_zone = resistance_zones[0].copy()
        
        for zone in resistance_zones[1:]:
            if zone["zone_low"] <= current_zone["zone_high"] + merge_distance:
                # Объединяем зоны
                current_zone["zone_low"] = min(current_zone["zone_low"], zone["zone_low"])
                current_zone["zone_high"] = max(current_zone["zone_high"], zone["zone_high"])
                if "merged_points" not in current_zone:
                    current_zone["merged_points"] = [current_zone.copy()]
                current_zone["merged_points"].append(zone.copy())
            else:
                merged_zones.append(current_zone)
                current_zone = zone.copy()
        
        merged_zones.append(current_zone)
    
    return merged_zones


def _important_extremes_with_zones(df: pd.DataFrame, distance: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Находит важные экстремумы с ПРАКТИЧЕСКИМ подходом.
    """
    if df.empty or len(df) < 20:
        return [], []
    
    zones = []
    extreme_points = []
    
    # Проходим по всем свечам
    for i in range(10, len(df) - 10):  # Начинаем с 10-й свечи, чтобы было больше истории
        # Проверяем на важный минимум
        is_important_min, zone_info_min = _find_important_minimum(df, i, distance)
        if is_important_min:
            # Проверяем, не является ли этот минимум частью уже существующей зоны
            is_new_zone = True
            for existing_zone in zones:
                if existing_zone["type"] == "support":
                    # Если новый минимум близко к существующей зоне поддержки
                    if abs(zone_info_min["zone_low"] - existing_zone["zone_low"]) < distance * 0.5:
                        is_new_zone = False
                        # Обновляем существующую зону
                        existing_zone["zone_low"] = min(existing_zone["zone_low"], zone_info_min["zone_low"])
                        existing_zone["zone_high"] = max(existing_zone["zone_high"], zone_info_min["zone_high"])
                        break
            
            if is_new_zone:
                zones.append(zone_info_min)
            
            extreme_points.append({
                "type": "MIN",
                "timestamp": df.at[i, "timestamp"],
                "price": df.at[i, "low"],
                "close": df.at[i, "close"],
                "zone": f"{zone_info_min['zone_low']:.1f}-{zone_info_min['zone_high']:.1f}"
            })
        
        # Проверяем на важный максимум
        is_important_max, zone_info_max = _find_important_maximum(df, i, distance)
        if is_important_max:
            # Проверяем, не является ли этот максимум частью уже существующей зоны
            is_new_zone = True
            for existing_zone in zones:
                if existing_zone["type"] == "resistance":
                    if abs(zone_info_max["zone_high"] - existing_zone["zone_high"]) < distance * 0.5:
                        is_new_zone = False
                        existing_zone["zone_low"] = min(existing_zone["zone_low"], zone_info_max["zone_low"])
                        existing_zone["zone_high"] = max(existing_zone["zone_high"], zone_info_max["zone_high"])
                        break
            
            if is_new_zone:
                zones.append(zone_info_max)
            
            extreme_points.append({
                "type": "MAX",
                "timestamp": df.at[i, "timestamp"],
                "price": df.at[i, "high"],
                "close": df.at[i, "close"],
                "zone": f"{zone_info_max['zone_low']:.1f}-{zone_info_max['zone_high']:.1f}"
            })
    
    # Объединяем близкие зоны
    merged_zones = _merge_close_zones(zones, merge_distance=distance * 0.7)
    
    return merged_zones, extreme_points


def _add_zones_to_chart(fig: go.Figure, zones: List[Dict[str, Any]]) -> None:
    """Добавляет зоны поддержки/сопротивления на график."""
    if not zones:
        return
    
    for zone in zones:
        if zone["type"] == "support":
            color = "rgba(34, 197, 94, 0.15)"  # Зеленый с прозрачностью
            line_color = "rgba(34, 197, 94, 0.5)"
            annotation_text = f"S: {zone['zone_low']:.1f}-{zone['zone_high']:.1f}"
            annotation_position = "top left"
        else:  # resistance
            color = "rgba(239, 68, 68, 0.15)"  # Красный с прозрачностью
            line_color = "rgba(239, 68, 68, 0.5)"
            annotation_text = f"R: {zone['zone_low']:.1f}-{zone['zone_high']:.1f}"
            annotation_position = "bottom left"
        
        fig.add_hrect(
            y0=zone["zone_low"],
            y1=zone["zone_high"],
            fillcolor=color,
            line_width=1,
            line_color=line_color,
            annotation_text=annotation_text,
            annotation_position=annotation_position,
            annotation_font_size=10,
            annotation_font_color=line_color.replace("0.5", "1.0")
        )


def _add_extreme_markers(fig: go.Figure, df: pd.DataFrame, extreme_points: List[Dict[str, Any]]) -> None:
    """Добавляет маркеры важных экстремумов на график."""
    if not extreme_points:
        return
    
    # Разделяем минимумы и максимумы
    min_points = [p for p in extreme_points if p["type"] == "MIN"]
    max_points = [p for p in extreme_points if p["type"] == "MAX"]
    
    # Вычисляем отступ для маркеров
    price_range = df["high"].max() - df["low"].min()
    offset = price_range * 0.02 if price_range > 0 else 10.0
    
    # Добавляем маркеры минимумов
    if min_points:
        min_timestamps = [p["timestamp"] for p in min_points]
        min_prices = [p["price"] for p in min_points]
        min_texts = [f"{p['price']:.1f}" for p in min_points]
        
        fig.add_trace(
            go.Scatter(
                x=min_timestamps,
                y=[p - offset for p in min_prices],
                mode="markers+text",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color="#22c55e",
                    line=dict(width=2, color="white")
                ),
                text=min_texts,
                textposition="top center",
                name="Важные минимумы",
                showlegend=False,
                opacity=0.9,
            )
        )
    
    # Добавляем маркеры максимумов
    if max_points:
        max_timestamps = [p["timestamp"] for p in max_points]
        max_prices = [p["price"] for p in max_points]
        max_texts = [f"{p['price']:.1f}" for p in max_points]
        
        fig.add_trace(
            go.Scatter(
                x=max_timestamps,
                y=[p + offset for p in max_prices],
                mode="markers+text",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color="#ef4444",
                    line=dict(width=2, color="white")
                ),
                text=max_texts,
                textposition="bottom center",
                name="Важные максимумы",
                showlegend=False,
                opacity=0.9,
            )
        )


def _plot_candles(
    df: pd.DataFrame,
    title: str,
    zones: List[Dict[str, Any]] = None,
    extreme_points: List[Dict[str, Any]] = None,
    show_zones: bool = True,
    show_extremes_markers: bool = True,
    overlay_text: str | None = None,
) -> go.Figure:
    """Создает свечной график с зонами поддержки/сопротивления."""
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
    
    # Добавляем зоны поддержки/сопротивления
    if show_zones and zones:
        _add_zones_to_chart(fig, zones)
    
    # Добавляем маркеры экстремумов
    if show_extremes_markers and extreme_points:
        _add_extreme_markers(fig, df, extreme_points)

    if overlay_text:
        fig.add_annotation(
            text=overlay_text,
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.99,
            showarrow=False,
            align="left",
            bgcolor="rgba(15, 23, 42, 0.7)",
            bordercolor="rgba(148, 163, 184, 0.4)",
            borderwidth=1,
            font=dict(size=11, color="#e5e7eb"),
        )
    
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


def _trend_summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "label": "НЕТ ДАННЫХ",
            "start": 0.0,
            "end": 0.0,
            "move": 0.0,
            "pct": 0.0,
            "low": 0.0,
            "high": 0.0,
        }
    start = float(df["close"].iloc[0])
    end = float(df["close"].iloc[-1])
    move = end - start
    pct = (move / start * 100.0) if start else 0.0
    low = float(df["low"].min())
    high = float(df["high"].max())
    rng = high - low
    ratio = abs(move) / rng if rng else 0.0
    if ratio < 0.35:
        label = "БОКОВИК"
    else:
        label = "ВОСХОДЯЩИЙ" if move > 0 else "НИСХОДЯЩИЙ"
    return {
        "label": label,
        "start": start,
        "end": end,
        "move": move,
        "pct": pct,
        "low": low,
        "high": high,
    }


def _zone_summary(zones: List[Dict[str, Any]], price: float) -> Dict[str, Any]:
    in_zone = None
    support = None
    resistance = None
    for zone in zones:
        if zone["zone_low"] <= price <= zone["zone_high"]:
            in_zone = zone
            break
    support_zones = [
        z for z in zones if z["type"] == "support" and z["zone_high"] <= price
    ]
    resistance_zones = [
        z for z in zones if z["type"] == "resistance" and z["zone_low"] >= price
    ]
    if support_zones:
        support = max(support_zones, key=lambda z: z["zone_high"])
    if resistance_zones:
        resistance = min(resistance_zones, key=lambda z: z["zone_low"])
    return {"in_zone": in_zone, "support": support, "resistance": resistance}


def _zone_label(zone: Dict[str, Any] | None) -> str:
    if not zone:
        return "н/д"
    return f"{zone['zone_low']:.1f}-{zone['zone_high']:.1f}"


def _pattern_summary(df: pd.DataFrame) -> str:
    flagged = _pattern_flags(df)
    if flagged.empty:
        return "паттерн: н/д"
    last = flagged.iloc[-1]
    patterns = []
    if last["pattern_pb"]:
        patterns.append("пин-бар")
    if last["pattern_e"]:
        patterns.append("поглощение")
    if last["pattern_kr"]:
        patterns.append("key reversal")
    if last["pattern_doji"]:
        patterns.append("доджи")
    if not patterns:
        return "паттерн: нет"
    return "паттерн: " + ", ".join(patterns)


def _build_market_analysis(
    symbol: str,
    d1_df: pd.DataFrame,
    h4_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    h4_zones: List[Dict[str, Any]],
    last_quote_value: float | None,
) -> Tuple[str, str]:
    trend = _trend_summary(d1_df)
    current_price = (
        float(last_quote_value)
        if last_quote_value is not None
        else (float(h1_df["close"].iloc[-1]) if not h1_df.empty else 0.0)
    )
    zones = _zone_summary(h4_zones, current_price)
    pattern_text = _pattern_summary(h1_df)
    now_msk = pd.Timestamp.now(tz=MOSCOW_TZ).strftime("%Y-%m-%d %H:%M")

    in_zone = zones["in_zone"]
    if in_zone:
        zone_type = "сопротивления" if in_zone["type"] == "resistance" else "поддержки"
        zone_note = f"Цена в зоне {zone_type}: {_zone_label(in_zone)}"
    else:
        zone_note = "Цена между зонами"

    scenario = "Ожидание подтверждения на H1"
    if in_zone and in_zone["type"] == "resistance":
        scenario = "Приоритет продавцов: ждать медвежий паттерн или пробой вверх"
    elif in_zone and in_zone["type"] == "support":
        scenario = "Приоритет покупателей: ждать бычий паттерн или пробой вниз"

    analysis_md = "\n".join(
        [
            f"# АНАЛИЗ РЫНКА {symbol} (авто)",
            "",
            f"Обновлено: **{now_msk} (MSK)**",
            "",
            "## 1. D1 — тренд",
            f"- Старт: **{trend['start']:.1f}**",
            f"- Текущая: **{trend['end']:.1f}**",
            f"- Диапазон: **{trend['low']:.1f} – {trend['high']:.1f}**",
            f"- Движение: **{trend['move']:.1f}** ({trend['pct']:.1f}%)",
            f"- Режим: **{trend['label']}**",
            "",
            "## 2. H4 — уровни",
            f"- Поддержка: **{_zone_label(zones['support'])}**",
            f"- Сопротивление: **{_zone_label(zones['resistance'])}**",
            f"- {zone_note}",
            "",
            "## 3. H1 — текущая ситуация",
            f"- Цена: **{current_price:.1f}**",
            f"- {pattern_text}",
            "",
            "## 4. Сценарий",
            f"- {scenario}",
        ]
    )

    overlay_text = "<br>".join(
        [
            f"D1: {trend['label']} ({trend['low']:.1f}-{trend['high']:.1f})",
            f"H4: S {_zone_label(zones['support'])} | R {_zone_label(zones['resistance'])}",
            f"H1: {current_price:.1f} | {pattern_text}",
            zone_note,
        ]
    )

    return analysis_md, overlay_text


def _pattern_flags(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    body = (result["close"] - result["open"]).abs()
    rng = (result["high"] - result["low"]).replace(0, 1e-9)
    upper = result["high"] - result[["open", "close"]].max(axis=1)
    lower = result[["open", "close"]].min(axis=1) - result["low"]
    body_top = result[["open", "close"]].max(axis=1)
    body_bottom = result[["open", "close"]].min(axis=1)

    long_shadow = 2.0
    small_body = 0.3
    doji_body = 0.1
    opp_shadow_max = 0.3
    body_upper_pos = result["low"] + rng * 0.6
    body_lower_pos = result["low"] + rng * 0.4

    lower_long = lower >= long_shadow * body
    upper_long = upper >= long_shadow * body
    both_long = lower_long & upper_long
    doji = body <= doji_body * rng
    small_body_flag = body <= small_body * rng

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
    result["pattern_ls"] = (lower_long | upper_long).fillna(False)
    result["pattern_ls_lower"] = lower_long.fillna(False)
    result["pattern_ls_upper"] = upper_long.fillna(False)
    result["pattern_ls_both"] = both_long.fillna(False)
    result["pattern_doji"] = doji.fillna(False)
    result["pattern_doji_long"] = (doji & both_long).fillna(False)
    result["pattern_gravestone"] = (doji & upper_long & (lower <= opp_shadow_max * body)).fillna(False)
    result["pattern_rickshaw"] = (doji & both_long & (body_top >= body_lower_pos) & (body_bottom <= body_upper_pos)).fillna(False)
    result["pattern_hammer_shape"] = (
        lower_long & small_body_flag & (upper <= opp_shadow_max * body) & (body_bottom >= body_upper_pos)
    ).fillna(False)
    result["pattern_inverted_shape"] = (
        upper_long & small_body_flag & (lower <= opp_shadow_max * body) & (body_top <= body_lower_pos)
    ).fillna(False)
    result["pattern_doji_hammer"] = (doji & lower_long).fillna(False)
    result["pattern_doji_star"] = (doji & upper_long).fillna(False)
    result["pattern_spinning_lower"] = (lower_long & ~small_body_flag).fillna(False)
    result["pattern_spinning_upper"] = (upper_long & ~small_body_flag).fillna(False)
    result["pattern_inside_ls"] = ((result["high"] <= prev_high) & (result["low"] >= prev_low) & (lower_long | upper_long)).fillna(False)
    result["pattern_outside_ls"] = ((result["high"] >= prev_high) & (result["low"] <= prev_low) & (lower_long | upper_long)).fillna(False)
    return result


def _add_pattern_labels(fig: go.Figure, df: pd.DataFrame) -> None:
    flagged = _pattern_flags(df)
    if flagged.empty:
        return
    descriptions = {
        "PB": "Пин‑бар — свеча с длинным хвостом (тенью) и маленьким телом у одного края, сигнал возможного разворота.",
        "E": "Поглощение (Engulfing) — свеча, тело которой полностью перекрывает тело предыдущей в противоположную сторону.",
        "KR": "Key Reversal — разворотный бар: делает новый экстремум против тренда и закрывается в противоположной стороне (часто с большим объемом).",
        "LS": "Длинная тень (правило 1‑2‑3) — тень в 2–3 раза больше тела.",
        "HAM": "Молот / висящий человек — маленькое тело у верхней части, длинная нижняя тень (контекст зависит от тренда).",
        "IH": "Перевернутый молот / падающая звезда — маленькое тело у нижней части, длинная верхняя тень (контекст зависит от тренда).",
        "DH": "Доджи‑молот — почти нет тела, длинная нижняя тень.",
        "DS": "Доджи‑звезда — почти нет тела, длинная верхняя тень.",
        "SP-L": "Волчок с длинной нижней тенью — тело среднее, нижняя тень заметно длиннее.",
        "SP-U": "Волчок с длинной верхней тенью — тело среднее, верхняя тень заметно длиннее.",
        "DJ": "Доджи — тело очень маленькое, тени примерно равные.",
        "DJL": "Длинноногий доджи — тело маленькое, обе тени очень длинные.",
        "GR": "Надгробие доджи — тело внизу, нижней тени почти нет, верхняя очень длинная.",
        "RK": "Рикша — тело в середине, обе тени очень длинные.",
        "IN": "Внутренний бар с длинной тенью — диапазон внутри предыдущей свечи.",
        "OUT": "Внешний бар с акцентом на тень — поглощает предыдущую свечу.",
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
        if row["pattern_ls"]:
            parts.append("LS")
            hover_parts.append(descriptions["LS"])
        if row["pattern_hammer_shape"]:
            parts.append("HAM")
            hover_parts.append(descriptions["HAM"])
        if row["pattern_inverted_shape"]:
            parts.append("IH")
            hover_parts.append(descriptions["IH"])
        if row["pattern_doji_hammer"]:
            parts.append("DH")
            hover_parts.append(descriptions["DH"])
        if row["pattern_doji_star"]:
            parts.append("DS")
            hover_parts.append(descriptions["DS"])
        if row["pattern_spinning_lower"]:
            parts.append("SP-L")
            hover_parts.append(descriptions["SP-L"])
        if row["pattern_spinning_upper"]:
            parts.append("SP-U")
            hover_parts.append(descriptions["SP-U"])
        if row["pattern_doji"]:
            parts.append("DJ")
            hover_parts.append(descriptions["DJ"])
        if row["pattern_doji_long"]:
            parts.append("DJL")
            hover_parts.append(descriptions["DJL"])
        if row["pattern_gravestone"]:
            parts.append("GR")
            hover_parts.append(descriptions["GR"])
        if row["pattern_rickshaw"]:
            parts.append("RK")
            hover_parts.append(descriptions["RK"])
        if row["pattern_inside_ls"]:
            parts.append("IN")
            hover_parts.append(descriptions["IN"])
        if row["pattern_outside_ls"]:
            parts.append("OUT")
            hover_parts.append(descriptions["OUT"])
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
        ranges = {
            "TIME_FRAME_D": 60,
            "TIME_FRAME_H4": 60,
            "TIME_FRAME_H1": 60,
        }
        days_back = ranges[timeframe]
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

        analysis_d1_df = _apply_time_shift(
            _bars_to_df(service.get_bars(symbol, "TIME_FRAME_D", ranges["TIME_FRAME_D"])),
            "TIME_FRAME_D",
        )
        analysis_h4_df = _apply_time_shift(
            _bars_to_df(service.get_bars(symbol, "TIME_FRAME_H4", ranges["TIME_FRAME_H4"])),
            "TIME_FRAME_H4",
        )
        analysis_h1_df = _apply_time_shift(
            _bars_to_df(service.get_bars(symbol, "TIME_FRAME_H1", ranges["TIME_FRAME_H1"])),
            "TIME_FRAME_H1",
        )
        h4_zones, _ = _important_extremes_with_zones(analysis_h4_df, RTS_MIN_DISTANCE_H4)
        analysis_md, overlay_text = _build_market_analysis(
            symbol=symbol,
            d1_df=analysis_d1_df,
            h4_df=analysis_h4_df,
            h1_df=analysis_h1_df,
            h4_zones=h4_zones,
            last_quote_value=last_quote_value,
        )
        overlay_text = ""
        analysis_path = os.path.join(
            "Фьючерсы",
            "RTSM",
            "RTSM-3.26 (15.01.2026)",
            "Анализ рынка.md",
        )
        try:
            os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
            with open(analysis_path, "w", encoding="utf-8") as handle:
                handle.write(analysis_md)
        except Exception:
            pass

        tabs = st.tabs(["График", "Свечи (O H L C)", "Анализ"])
        
        with tabs[0]:
            bars = service.get_bars(symbol, timeframe, days_back)
            df = _bars_to_df(bars)
            
            if df.empty:
                st.warning("Нет данных по свечам")
            else:
                df_plot_base = _apply_time_shift(df, timeframe)
                
                # Находим важные экстремумы и зоны (авто по 1.5 * AvgRange(20))
                avg_range = float((df_plot_base["high"] - df_plot_base["low"]).tail(20).mean())
                important_distance = max(0.0, 1.5 * avg_range)
                zones, extreme_points = _important_extremes_with_zones(df_plot_base, important_distance)

                if not zones:
                    st.warning("Не найдено важных экстремумов. Возможные причины:")
                    st.write("1. Параметр D слишком большой - уменьшите")
                    st.write("2. На графике нет четких разворотов")
                    st.write("3. Недостаточно данных для анализа")
                
                df_plot = _append_synthetic_bar(df_plot_base, timeframe, last_quote_value)
                
                fig = _plot_candles(
                    df_plot,
                    f"{symbol} {timeframe} - Важные экстремумы",
                    zones=zones,
                    extreme_points=extreme_points,
                    show_zones=True,
                    show_extremes_markers=True,
                    overlay_text=overlay_text,
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
                if tf != "TIME_FRAME_D":
                    tf_df = _append_synthetic_bar(tf_df, tf, last_quote_value)
                ohlcv = tf_df[["timestamp", "open", "high", "low", "close"]].copy()
                ohlcv = ohlcv.rename(columns={"open": "O", "high": "H", "low": "L", "close": "C"})
                st.dataframe(ohlcv, use_container_width=True)
                export_ts = pd.Timestamp.now(tz=MOSCOW_TZ).strftime("%Y-%m-%dT%H-%M")
                export_name = f"{_safe_symbol(symbol)}_{tf}_{export_ts}_export.csv"
                st.download_button(
                    label="Скачать CSV",
                    data=ohlcv.to_csv(index=False),
                    file_name=export_name,
                    mime="text/csv",
                    key=f"download_{tf}",
                )

        with tabs[2]:
            st.subheader("Анализ рынка")
            if os.path.exists(analysis_path):
                with open(analysis_path, "r", encoding="utf-8") as handle:
                    st.markdown(handle.read())
            else:
                st.info("Файл анализа рынка не найден, показываю авто‑анализ.")
                st.markdown(analysis_md)

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
                trades = service.get_trades(account_id, limit=50).get("trades", [])
                if trades:
                    st.dataframe(pd.DataFrame(trades))
                else:
                    st.info("Сделок нет")
            except Exception as exc:
                trades = []
                st.error(f"Ошибка загрузки сделок: {exc}")

        if "df" in locals() and not df.empty:
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
