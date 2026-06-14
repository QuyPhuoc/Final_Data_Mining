import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# =========================================================
# UTILS (CLEAN ARCHITECTURE)
# =========================================================

REQUIRED_COLS = ['date', 'open', 'high', 'low', 'close', 'volume']


def _safe_lower(df: pd.DataFrame) -> pd.DataFrame:
    """Không mutate input"""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Safe datetime conversion"""
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values('date')
    return df.reset_index(drop=True)


def _has_ohlcv(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in REQUIRED_COLS)


# =========================================================
# UI HELPERS
# =========================================================

def color_class(val, good_above=None, bad_below=None):
    if good_above is not None and val > good_above:
        return "metric-ok"
    if bad_below is not None and val < bad_below:
        return "metric-bad"
    return "metric-warn"


def metric_card(label, value, sub="", color_cls=""):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_cls}">{value}</div>
        {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """


def section(title):
    return f'<div class="section-header">{title}</div>'


# =========================================================
# LOAD DATA
# =========================================================

def load_df(uploaded):
    name = uploaded.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    elif name.endswith(".parquet"):
        df = pd.read_parquet(uploaded)
    else:
        return None

    df = _safe_lower(df)

    date_col = next((c for c in df.columns if "date" in c or "time" in c), None)

    if date_col:
        df = df.rename(columns={date_col: "date"})
        df = _ensure_datetime(df)

    return df


# =========================================================
# EDA QUALITY
# =========================================================

def analyse_quality(df: pd.DataFrame) -> dict:
    df = _safe_lower(df)

    q = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
    }

    # date range safe
    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce")
        q["date_min"] = d.min()
        q["date_max"] = d.max()
    else:
        q["date_min"] = None
        q["date_max"] = None

    # missing
    miss = df.isnull().sum()
    q["missing_total"] = int(miss.sum())
    q["missing_pct"] = round(miss.sum() / max(df.size, 1) * 100, 2)
    q["missing_by_col"] = miss[miss > 0].to_dict()

    # duplicates
    q["dup_rows"] = int(df.duplicated().sum())
    q["dup_dates"] = int(df.duplicated(subset="date").sum()) if "date" in df.columns else 0

    # =====================================================
    # OHLC VALIDATION (FIX DOUBLE COUNTING LOGIC)
    # =====================================================

    if _has_ohlcv(df):
        o = pd.to_numeric(df['open'], errors='coerce')
        h = pd.to_numeric(df['high'], errors='coerce')
        l = pd.to_numeric(df['low'], errors='coerce')
        c = pd.to_numeric(df['close'], errors='coerce')

        error_mask = (
            (h < l) |
            (c > h) |
            (c < l) |
            (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)
        )

        q["ohlc_ok"] = True
        q["ohlc_error_rows"] = int(error_mask.sum())

        # FIX: không double count quá mức
        q["neg_price"] = int(((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)).sum())

        # RETURNS SAFE
        ret = c.pct_change().dropna()

        if len(ret) > 1:
            q["ret_mean"] = float(ret.mean())
            q["ret_std"] = float(ret.std())
            q["ret_skew"] = float(ret.skew()) if len(ret) > 2 else 0
            q["ret_kurt"] = float(ret.kurtosis()) if len(ret) > 2 else 0
            q["max_daily_gain"] = float(ret.max())
            q["max_daily_loss"] = float(ret.min())

            # OUTLIER SAFE
            if ret.std() > 0:
                z = (ret - ret.mean()) / ret.std()
                q["outlier_3s"] = int((z.abs() > 3).sum())
                q["outlier_5s"] = int((z.abs() > 5).sum())
            else:
                q["outlier_3s"] = 0
                q["outlier_5s"] = 0
        else:
            q["ret_mean"] = q["ret_std"] = 0
            q["outlier_3s"] = q["outlier_5s"] = 0

        # GAP SAFE
        if "date" in df.columns and len(df) > 1:
            d = pd.to_datetime(df["date"], errors="coerce")
            gaps = d.diff().dt.days.dropna()

            q["max_gap_days"] = int(gaps.max()) if not gaps.empty else 0
            q["gaps_gt5"] = int((gaps > 5).sum())
            q["gaps_gt20"] = int((gaps > 20).sum())
        else:
            q["max_gap_days"] = q["gaps_gt5"] = q["gaps_gt20"] = 0

    else:
        q["ohlc_ok"] = False
        q["ohlc_error_rows"] = -1

    return q


# =========================================================
# CHARTS (FULL SAFE MODE)
# =========================================================

def chart_candlestick(df):
    df = _ensure_datetime(_safe_lower(df))

    if not _has_ohlcv(df):
        return None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing_line_color="#3fb950",
        decreasing_line_color="#f85149"
    ), row=1, col=1)

    if 'volume' in df.columns:
        colors = np.where(df['close'] >= df['open'], "#3fb950", "#f85149")

        fig.add_trace(go.Bar(
            x=df['date'],
            y=df['volume'],
            marker_color=colors,
            opacity=0.6
        ), row=2, col=1)

    fig.update_layout(template="plotly_dark", height=500, showlegend=False)
    return fig


def chart_missing(df):
    df = _safe_lower(df)

    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values()

    if miss.empty:
        return None

    fig = go.Figure(go.Bar(
        x=miss.values,
        y=miss.index,
        orientation="h",
        marker_color="#d29922"
    ))

    fig.update_layout(template="plotly_dark", height=300)
    return fig


def chart_returns_dist(df):
    df = _safe_lower(df)

    if 'close' not in df.columns:
        return None

    ret = df['close'].pct_change().dropna() * 100

    if len(ret) < 5 or ret.std() == 0:
        return None

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=ret,
        nbinsx=60,
        marker_color="#388bfd"
    ))

    x = np.linspace(ret.min(), ret.max(), 200)
    pdf = norm.pdf(x, ret.mean(), ret.std())
    scale = len(ret) * (ret.max() - ret.min()) / 60

    fig.add_trace(go.Scatter(
        x=x,
        y=pdf * scale,
        line=dict(color="#f0883e"),
        name="Normal fit"
    ))

    fig.update_layout(template="plotly_dark", height=350)
    return fig


def chart_rolling_vol(df):
    df = _safe_lower(df)

    if 'close' not in df.columns or len(df) < 30:
        return None

    ret = df['close'].pct_change()

    vol5 = ret.rolling(5).std() * np.sqrt(252)
    vol21 = ret.rolling(21).std() * np.sqrt(252)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['date'], y=vol5, name="Vol 5d"))
    fig.add_trace(go.Scatter(x=df['date'], y=vol21, name="Vol 21d"))

    fig.update_layout(template="plotly_dark", height=320)
    return fig


def chart_gaps(df):
    df = _safe_lower(df)

    if 'date' not in df.columns or len(df) < 2:
        return None

    gaps = pd.to_datetime(df['date'], errors='coerce').diff().dt.days.dropna()

    if gaps.empty:
        return None

    fig = go.Figure(go.Histogram(
        x=gaps,
        nbinsx=30,
        marker_color="#8957e5"
    ))

    fig.update_layout(template="plotly_dark", height=260)
    return fig