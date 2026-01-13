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
        
        # Автоматические настройки для РТС
        if timeframe == "TIME_FRAME_H4":
            default_distance = RTS_MIN_DISTANCE_H4
            st.info(f"Для H4 (RTS) рекомендую: D={default_distance} (минимум 100 пунктов)")
        elif timeframe == "TIME_FRAME_H1":
            default_distance = RTS_MIN_DISTANCE_H1
            st.info(f"Для H1 рекомендую: D={default_distance} (минимум 50 пунктов)")
        else:  # D1
            default_distance = RTS_MIN_DISTANCE_D
            st.info(f"Для D1 рекомендую: D={default_distance} (минимум 200 пунктов)")
        
        important_distance = st.number_input(
            "Порог расстояния D (пунктов)",
            min_value=0.0,
            value=float(default_distance),
            step=10.0,
            help="Минимальное движение после экстремума ('1 фигура')"
        )
        
        st.caption("ПРАКТИЧЕСКАЯ ЛОГИКА ПОИСКА ВАЖНЫХ ЭКСТРЕМУМОВ:")
        st.caption("1. Ключевой принцип: гибкость вместо догмы")
        st.caption("2. Минимум: Close > Low, цена уходит от минимума")
        st.caption("3. Максимум: Close < High, цена падает от максимума")
        st.caption("4. '1 фигура' для РТС H4: 100+ пунктов")
        
        ranges = {
            "TIME_FRAME_D": 60,
            "TIME_FRAME_H4": 60,
            "TIME_FRAME_H1": 30,
        }
        days_back = ranges[timeframe]
        st.caption(f"Глубина: {days_back} дней")
        
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
                
                # Находим важные экстремумы и зоны
                zones, extreme_points = _important_extremes_with_zones(df_plot_base, important_distance)
                
                # Показываем статистику
                support_zones = len([z for z in zones if z["type"] == "support"])
                resistance_zones = len([z for z in zones if z["type"] == "resistance"])
                min_points = len([p for p in extreme_points if p["type"] == "MIN"])
                max_points = len([p for p in extreme_points if p["type"] == "MAX"])
                
                st.caption(f"Найдено: {support_zones} зон поддержки, {resistance_zones} зон сопротивления")
                st.caption(f"Точек экстремумов: {min_points} минимумов, {max_points} максимумов")
                
                if not zones:
                    st.warning("Не найдено важных экстремумов. Возможные причины:")
                    st.write("1. Параметр D слишком большой - уменьшите")
                    st.write("2. На графике нет четких разворотов")
                    st.write("3. Недостаточно данных для анализа")
                else:
                    with st.expander("Детали зон поддержки/сопротивления"):
                        zones_df = pd.DataFrame([
                            {
                                "Тип": "Поддержка" if z["type"] == "support" else "Сопротивление",
                                "Зона": f"{z['zone_low']:.1f}-{z['zone_high']:.1f}",
                                "Ширина": f"{z['zone_high'] - z['zone_low']:.1f}",
                                "Точка": f"{z.get('original_low', z['zone_low']):.1f}" if z["type"] == "support" else f"{z.get('original_high', z['zone_high']):.1f}"
                            }
                            for z in zones
                        ])
                        st.dataframe(zones_df, use_container_width=True)
                    
                    with st.expander("Список важных экстремумов"):
                        if extreme_points:
                            extremes_df = pd.DataFrame(extreme_points)
                            extremes_df["timestamp"] = pd.to_datetime(extremes_df["timestamp"])
                            extremes_df = extremes_df[["type", "timestamp", "price", "close", "zone"]]
                            extremes_df.columns = ["Тип", "Время", "Экстремум", "Закрытие", "Зона"]
                            st.dataframe(extremes_df, use_container_width=True)
                    
                    with st.expander("Детальная отладка экстремумов"):
                        if extreme_points:
                            st.write("Найденные точки экстремумов:")
                            for i, point in enumerate(extreme_points):
                                st.write(
                                    f"{i}. {point['type']} в {point['timestamp']}: "
                                    f"цена={point['price']}, зона={point['zone']}"
                                )
                        
                        st.write("\nПроверка конкретных свечей из вашего примера:")
                        test_points = [
                            ("2025-12-19 14:00:00", "MIN", 1063.0),
                            ("2025-12-22 12:00:00", "MIN", 1076.0),
                            ("2025-12-29 13:00:00", "MAX", 1151.5),
                        ]
                        
                        for ts_str, expected_type, expected_price in test_points:
                            try:
                                target_ts = pd.Timestamp(ts_str).tz_localize(MOSCOW_TZ)
                                idx = (df_plot_base["timestamp"] - target_ts).abs().argmin()
                                
                                if expected_type == "MIN":
                                    is_important, zone_info = _find_important_minimum(
                                        df_plot_base, idx, important_distance
                                    )
                                    st.write(
                                        f"{ts_str} (idx={idx}): MIN={df_plot_base.at[idx, 'low']:.1f}, "
                                        f"Важный={is_important}, Close>Low="
                                        f"{df_plot_base.at[idx, 'close'] > df_plot_base.at[idx, 'low']}"
                                    )
                                else:
                                    is_important, zone_info = _find_important_maximum(
                                        df_plot_base, idx, important_distance
                                    )
                                    st.write(
                                        f"{ts_str} (idx={idx}): MAX={df_plot_base.at[idx, 'high']:.1f}, "
                                        f"Важный={is_important}, Close<High="
                                        f"{df_plot_base.at[idx, 'close'] < df_plot_base.at[idx, 'high']}"
                                    )
                            except Exception as exc:
                                st.write(f"Ошибка при проверке {ts_str}: {exc}")
                
                df_plot = _append_synthetic_bar(df_plot_base, timeframe, last_quote_value)
                
                fig = _plot_candles(
                    df_plot,
                    f"{symbol} {timeframe} - Важные экстремумы",
                    zones=zones,
                    extreme_points=extreme_points,
                    show_zones=True,
                    show_extremes_markers=(timeframe == "TIME_FRAME_H4"),
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
