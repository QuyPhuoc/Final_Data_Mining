"""
feature_engineering.py
======================
Pipeline tạo features cho bài toán dự đoán giá cổ phiếu / coin.

THAY ĐỔI SAU PHÂN TÍCH TƯƠNG QUAN:
  ✗ LOẠI : rsi_14 (tương quan >0.86 với price_pos_20, trend_direction)
  ✗ LOẠI : bb_pct_b_20 (tương quan 0.932 với price_pos_20)
  ✗ LOẠI : obv_ma_ratio, macd_hist, regime_num, vol_ratio_20, bb_width_pct_20 (|corr target| < 0.05)
  ✓ GIỮ  : log_return_1/5/20d, rvol_5/21d, atr_pct_14, macd_signal, price_pos_20,
            close_to_low, range_pct, trend_direction
  ✓ THÊM : momentum_ratio = log_return_20d / log_return_5d

Nguyên tắc thiết kế:
  1. Không đưa giá tuyệt đối vào model
  2. Không dùng .shift(-n) trước khi split → target được tạo tại thời điểm dự đoán
  3. Rolling z-score với expanding min_periods (không lookahead)
  4. Mọi feature chỉ dùng data ≤ t
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional, Tuple

TRADING_DAYS = 252


# ═══════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột về lowercase."""
    return df.rename(columns={c: c.lower().strip() for c in df.columns})


def rolling_zscore(series: pd.Series, window: int = 63) -> pd.Series:
    """Rolling z-score không lookahead bias."""
    mu = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    return (series - mu) / std.replace(0, np.nan)


# ═══════════════════════════════════════════════════════════════
# 1. LOG RETURN FEATURES
# ═══════════════════════════════════════════════════════════════

def add_log_return_features(
    df: pd.DataFrame,
    close_col: str = "close",
    periods: List[int] = [1, 5, 20],
) -> pd.DataFrame:
    """Log return theo các horizon."""
    df = df.copy()
    c = df[close_col]
    for p in periods:
        df[f"log_return_{p}d"] = np.log(c / c.shift(p))
    return df


# ═══════════════════════════════════════════════════════════════
# 2. VOLATILITY FEATURES
# ═══════════════════════════════════════════════════════════════

def add_volatility_features(
    df: pd.DataFrame,
    close_col: str = "close",
    windows: List[int] = [5, 21],
    annualize: bool = True,
) -> pd.DataFrame:
    """Realized volatility (std của log return)."""
    df = df.copy()
    log_ret = np.log(df[close_col] / df[close_col].shift(1))
    factor = np.sqrt(TRADING_DAYS) if annualize else 1.0
    for w in windows:
        df[f"rvol_{w}d"] = log_ret.rolling(w, min_periods=w // 2).std() * factor
    return df


def add_atr_pct(
    df: pd.DataFrame,
    high_col: str = "high", low_col: str = "low", close_col: str = "close",
    window: int = 14,
) -> pd.DataFrame:
    """ATR chuẩn hóa theo giá (ATR%)."""
    df = df.copy()
    h, l, c = df[high_col], df[low_col], df[close_col]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window, min_periods=window // 2).mean()
    df[f"atr_pct_{window}"] = atr / c.replace(0, np.nan)
    return df


# ═══════════════════════════════════════════════════════════════
# 3. MOMENTUM OSCILLATORS (CHỈ GIỮ MACD_SIGNAL)
# ═══════════════════════════════════════════════════════════════

def add_macd(
    df: pd.DataFrame,
    close_col: str = "close",
    fast: int = 12, slow: int = 26, signal: int = 9,
) -> pd.DataFrame:
    """
    MACD - chỉ giữ signal line (tương quan target cao nhất).
    Bỏ histogram và line gốc.
    """
    df = df.copy()
    ema_fast = df[close_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[close_col].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_sig = macd_line.ewm(span=signal, adjust=False).mean()
    df["macd_signal"] = macd_sig
    return df


# ═══════════════════════════════════════════════════════════════
# 4. VOLUME FEATURES (CHỈ GIỮ VOL_RATIO - TUY NHIÊN CORR THẤP, CÓ THỂ LOẠI)
# Ở ĐÂY TẠM GIỮ VOL_RATIO_20 ĐỂ THỬ, NHƯNG THEO PHÂN TÍCH NÊN LOẠI.
# ĐỂ BẢO TOÀN, TA SẼ KHÔNG THÊM VOLUME FEATURE NÀO.
# ═══════════════════════════════════════════════════════════════

# Không thêm volume ratio hay obv ratio vì tương quan quá thấp.
def add_volume_features_removed():
    pass  # chủ động bỏ qua


# ═══════════════════════════════════════════════════════════════
# 5. BOLLINGER BANDS (CHỈ GIỮ WIDTH NẾU CẦN, NHƯNG CORR THẤP → BỎ)
# ═══════════════════════════════════════════════════════════════

# Bỏ hẳn Bollinger bands vì bb_pct_b đã loại (cộng tuyến), bb_width_pct quá yếu.


# ═══════════════════════════════════════════════════════════════
# 6. CANDLESTICK STRUCTURE (GIỮ range_pct, close_to_low)
# ═══════════════════════════════════════════════════════════════

def add_candlestick_features(
    df: pd.DataFrame,
    high_col: str = "high", low_col: str = "low", close_col: str = "close",
) -> pd.DataFrame:
    """
    range_pct    : (high-low)/close
    close_to_low : (close-low)/(high-low)
    """
    df = df.copy()
    h, l, c = df[high_col], df[low_col], df[close_col]
    rng = (h - l).replace(0, np.nan)
    df["range_pct"] = rng / c.replace(0, np.nan)
    df["close_to_low"] = (c - l) / rng
    return df


# ═══════════════════════════════════════════════════════════════
# 7. REGIME & TREND (CHỈ GIỮ price_pos_20, trend_direction)
# ═══════════════════════════════════════════════════════════════

def add_regime_features(
    df: pd.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    window: int = 20,
) -> pd.DataFrame:
    """
    price_pos_20 : vị trí close trong kênh 20 ngày [0,1]
    trend_direction : DI-based direction [-1, +1]
    Bỏ regime_num (tương quan target quá thấp).
    """
    df = df.copy()
    c = df[close_col]

    # price_pos_20
    lo20 = df[low_col].rolling(window, min_periods=window // 2).min()
    hi20 = df[high_col].rolling(window, min_periods=window // 2).max()
    rng20 = (hi20 - lo20).replace(0, np.nan)
    df["price_pos_20"] = (c - lo20) / rng20

    # trend_direction (DI-based)
    h, l = df[high_col], df[low_col]
    prev_h = h.shift(1)
    prev_l = l.shift(1)
    dm_plus = (h - prev_h).clip(lower=0)
    dm_minus = (prev_l - l).clip(lower=0)
    dm_plus = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
    adm_plus = dm_plus.rolling(14, min_periods=7).mean()
    adm_minus = dm_minus.rolling(14, min_periods=7).mean()
    total = (adm_plus + adm_minus).replace(0, np.nan)
    df["trend_direction"] = (adm_plus - adm_minus) / total
    return df


# ═══════════════════════════════════════════════════════════════
# 8. MOMENTUM RATIO (MỚI)
# ═══════════════════════════════════════════════════════════════

def add_momentum_ratio(
    df: pd.DataFrame,
    short_period: int = 5,
    long_period: int = 20,
) -> pd.DataFrame:
    """
    momentum_ratio = log_return_{long}d / (log_return_{short}d + epsilon)
    Thể hiện sự khuếch đại/ suy giảm xu hướng.
    """
    df = df.copy()
    short_ret = df[f"log_return_{short_period}d"]
    long_ret = df[f"log_return_{long_period}d"]
    df["momentum_ratio"] = long_ret / (short_ret + 1e-6)
    return df


# ═══════════════════════════════════════════════════════════════
# 9. ROLLING Z-SCORE (CHỐNG LOOKAHEAD)
# ═══════════════════════════════════════════════════════════════

def apply_rolling_zscore(
    df: pd.DataFrame,
    cols: List[str],
    window: int = 63,
    inplace: bool = True,
) -> pd.DataFrame:
    """Chuẩn hóa rolling z-score cho danh sách cột."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        z = rolling_zscore(df[col], window=window)
        if inplace:
            df[col] = z
        else:
            df[f"{col}_z"] = z
    return df


# ═══════════════════════════════════════════════════════════════
# 10. TARGET ENGINEERING
# ═══════════════════════════════════════════════════════════════

def add_target(
    df: pd.DataFrame,
    close_col: str = "close",
    horizon: int = 5,
    target_type: str = "log_return",
    thresholds: Tuple[float, float] = (-0.01, 0.01),
) -> pd.DataFrame:
    """Tạo target (log return / direction / tertile)."""
    df = df.copy()
    log_ret = np.log(df[close_col].shift(-horizon) / df[close_col])

    if target_type == "log_return":
        df["target"] = log_ret
    elif target_type == "direction":
        df["target"] = np.where(log_ret > 0, 1, -1)
    elif target_type == "tertile":
        lo, hi = thresholds
        df["target"] = np.where(log_ret > hi, 1, np.where(log_ret < lo, -1, 0))
    else:
        raise ValueError("target_type must be 'log_return', 'direction', or 'tertile'")
    return df


# ═══════════════════════════════════════════════════════════════
# 11. PIPELINE TỔNG HỢP
# ═══════════════════════════════════════════════════════════════

# Các cột nên được rolling z-score (chỉ còn macd_signal và momentum_ratio)
_COLS_TO_NORMALIZE = ["macd_signal", "momentum_ratio"]

# Feature set mặc định SAU KHI TỐI ƯU
FEATURE_COLS = [
    "log_return_1d",
    "log_return_5d",
    "log_return_20d",
    "rvol_5d",
    "rvol_21d",
    "atr_pct_14",
    "macd_signal",
    "price_pos_20",
    "close_to_low",
    "range_pct",
    "trend_direction",
    "momentum_ratio",
]

# Các cột không phải feature (để loại trừ trong get_feature_cols)
_NON_FEATURE_COLS = {
    "date", "datetime", "time",
    "open", "high", "low", "close", "volume",
    "target",
    "log_return",  # raw
    "ret",
}

# Prefix patterns để nhận diện feature động
_FEATURE_PREFIXES = (
    "log_return_",
    "rvol_",
    "atr_pct_",
)

# Exact-match feature names
_FEATURE_EXACT = {
    "macd_signal",
    "price_pos_20",
    "close_to_low",
    "range_pct",
    "trend_direction",
    "momentum_ratio",
}


def get_feature_cols(df: pd.DataFrame) -> list:
    """Trả về danh sách feature columns có trong df."""
    result = []
    for col in df.columns:
        if col in _NON_FEATURE_COLS:
            continue
        if col in _FEATURE_EXACT:
            result.append(col)
            continue
        if any(col.startswith(pref) for pref in _FEATURE_PREFIXES):
            result.append(col)
    return result


def build_features(
    df: pd.DataFrame,
    # Log return
    return_periods: List[int] = [1, 5, 20],
    # Volatility
    vol_windows: List[int] = [5, 21],
    atr_window: int = 14,
    # MACD
    macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
    # Regime
    price_window: int = 20,
    # Normalization
    normalize_cols: Optional[List[str]] = None,
    zscore_window: int = 63,
    zscore_inplace: bool = True,
    # Target
    horizon: int = 5,
    target_type: str = "log_return",
    target_thresholds: Tuple[float, float] = (-0.01, 0.01),
    # General
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Pipeline hoàn chỉnh - tạo features đã tối ưu.
    """
    df = _normalize_cols(df.copy())

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột: {missing}")

    # 1. Log returns
    df = add_log_return_features(df, periods=return_periods)

    # 2. Volatility
    df = add_volatility_features(df, windows=vol_windows)
    df = add_atr_pct(df, window=atr_window)

    # 3. MACD signal
    df = add_macd(df, fast=macd_fast, slow=macd_slow, signal=macd_signal)

    # 4. Candlestick
    df = add_candlestick_features(df)

    # 5. Regime & trend
    df = add_regime_features(df, window=price_window)

    # 6. Momentum ratio
    # Chỉ thêm nếu có đủ các cột log_return cần thiết
    if all(f"log_return_{p}d" in df.columns for p in [5, 20]):
        df = add_momentum_ratio(df, short_period=5, long_period=20)

    # 7. Rolling z-score
    cols_z = normalize_cols if normalize_cols is not None else _COLS_TO_NORMALIZE
    df = apply_rolling_zscore(df, cols=cols_z, window=zscore_window, inplace=zscore_inplace)

    # 8. Target
    df = add_target(df, horizon=horizon, target_type=target_type, thresholds=target_thresholds)

    if drop_na:
        actual_feature_cols = get_feature_cols(df)
        subset = actual_feature_cols + (["target"] if "target" in df.columns else [])
        df = df.dropna(subset=subset).reset_index(drop=True)

    return df


# ═══════════════════════════════════════════════════════════════
# 12. CORRELATION ANALYSIS & REPORT
# ═══════════════════════════════════════════════════════════════

def correlation_analysis(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: Optional[str] = "target",
    threshold: float = 0.85,
) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    corr = df[feature_cols].corr().abs()
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            v = corr.iloc[i, j]
            if v > threshold:
                pairs.append((corr.columns[i], corr.columns[j], round(float(v), 4)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return corr, pairs


def display_correlation_in_streamlit(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: Optional[str] = "target",
    threshold: float = 0.85,
) -> None:
    import streamlit as st
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    corr_matrix = df[feature_cols].corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Ma trận tương quan giữa các features",
    )
    fig.update_layout(height=600, font=dict(size=10))
    st.plotly_chart(fig, use_container_width=True)

    _, high_pairs = correlation_analysis(df, feature_cols, threshold=threshold)
    if high_pairs:
        st.warning(f"⚠️ {len(high_pairs)} cặp có |corr| > {threshold}")
        for f1, f2, v in high_pairs[:10]:
            st.write(f"- {f1} ↔ {f2} : {v:.3f}")
    else:
        st.success(f"✅ Không có cặp feature nào vượt ngưỡng {threshold}.")

    if target_col and target_col in df.columns:
        st.markdown("**🎯 Tương quan với target:**")
        target_corr = df[feature_cols + [target_col]].corr()[target_col].drop(target_col).sort_values(key=abs, ascending=False)
        fig2 = go.Figure(go.Bar(
            x=target_corr.values,
            y=target_corr.index,
            orientation="h",
            marker_color=["#3fb950" if v > 0 else "#f85149" for v in target_corr.values],
            text=[f"{v:+.3f}" for v in target_corr.values],
            textposition="outside",
        ))
        fig2.update_layout(height=max(300, len(target_corr) * 26))
        st.plotly_chart(fig2, use_container_width=True)


def get_correlation_report(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target",
    threshold: float = 0.85
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[str, str, float]]]:
    corr_matrix = df[feature_cols].corr()
    target_corr = df[feature_cols + [target_col]].corr()[target_col].drop(target_col).sort_values(key=abs, ascending=False)
    high_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            v = corr_matrix.iloc[i, j]
            if abs(v) > threshold:
                high_pairs.append((feature_cols[i], feature_cols[j], v))
    high_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return corr_matrix, target_corr, high_pairs


# ═══════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    n = 600
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 20000 * np.cumprod(1 + np.random.normal(0, 0.015, n))
    high = close * np.random.uniform(1.005, 1.02, n)
    low = close * np.random.uniform(0.98, 0.995, n)
    open_ = close * np.random.uniform(0.98, 1.02, n)
    volume = np.random.randint(500_000, 5_000_000, n).astype(float)

    df_raw = pd.DataFrame({
        "date": dates, "open": open_, "high": high,
        "low": low, "close": close, "volume": volume,
    })

    print("Building features...")
    df_feat = build_features(df_raw, horizon=5, target_type="log_return")
    print(f"Shape: {df_feat.shape}")
    feature_cols = get_feature_cols(df_feat)
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"\nSample:\n{df_feat[feature_cols + ['target']].tail(5).to_string()}")

    corr, pairs = correlation_analysis(df_feat, threshold=0.85)
    print(f"\nCặp tương quan > 0.85: {len(pairs)}")
    for f1, f2, v in pairs[:5]:
        print(f"  {f1} ↔ {f2} : {v:.4f}")