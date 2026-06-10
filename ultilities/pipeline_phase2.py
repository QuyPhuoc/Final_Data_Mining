import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf

def perform_adf_test(series: pd.Series) -> dict:
    """
    Kiểm định tính dừng Augmented Dickey-Fuller (ADF)
    """
    result = adfuller(series.dropna(), autolag='AIC')
    return {
        "adf_stat": result[0],
        "p_value": result[1],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05
    }

def compute_acf_pacf(series: pd.Series, lags: int = 20) -> tuple:
    """
    Tính toán hệ số tự tương quan (ACF) và tự tương quan từng phần (PACF)
    """
    cleaned_series = series.dropna()
    acf_vals = acf(cleaned_series, nlags=lags, fft=True)
    pacf_vals = pacf(cleaned_series, nlags=lags, method='ols')
    return acf_vals, pacf_vals

def decompose_time_series(df: pd.DataFrame, period: int = 20) -> tuple:
    """
    Phân rã cấu trúc chuỗi thời gian (Trend, Seasonal, Residuals) trên tập huấn luyện
    """
    df_decomp = df.set_index('Date').sort_index()
    df_decomp = df_decomp['Close'].resample('D').mean().ffill()
    
    decomposition = seasonal_decompose(df_decomp, model='additive', period=period)
    return decomposition.trend, decomposition.seasonal, decomposition.resid