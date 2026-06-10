import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler

def apply_stationarity(df: pd.DataFrame, target_col: str = 'Close') -> pd.DataFrame:
    """
    Thực hiện biến đổi sai phân bậc 1 để đưa chuỗi giá về trạng thái dừng ổn định.
    """
    df_stat = df.copy()
    df_stat['Stationary_Target'] = df_stat[target_col].diff()
    # Phiên đầu tiên sẽ bị NaN do sai phân, điền bằng 0 để giữ nguyên kích thước ma trận
    df_stat['Stationary_Target'] = df_stat['Stationary_Target'].fillna(0)
    return df_stat

def scale_datasets(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Chuẩn hóa dữ liệu chống rò rỉ (Data Leakage).
    Chỉ FIT trên tập Train và TRANSFORM trên cả 3 tập.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Ép kiểu và trích xuất ma trận số
    train_scaled = df_train[feature_cols].copy()
    val_scaled = df_val[feature_cols].copy()
    test_scaled = df_test[feature_cols].copy()
    
    # Học tham số phân phối từ riêng tập Train
    scaler.fit(train_scaled)
    
    # Áp đặt hệ số chuẩn hóa lên các tập
    train_scaled[feature_cols] = scaler.transform(train_scaled[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val_scaled[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_scaled[feature_cols])
    
    return train_scaled, val_scaled, test_scaled, scaler

def create_sliding_windows(df_scaled: pd.DataFrame, target_series: pd.Series, lookback_window: int = 10) -> tuple:
    """
    Chuyển đổi DataFrame thành cấu trúc Tensor 3D Cửa sổ trượt phục vụ mạng LSTM/Transformer.
    Đầu vào: Matrix [Mẫu x Đặc trưng]
    Đầu ra: X_tensor [Mẫu x Bước thời gian x Đặc trưng], y_tensor [Mẫu]
    """
    X, y = [], []
    data_matrix = df_scaled.values
    target_matrix = target_series.values
    
    for i in range(len(data_matrix) - lookback_window):
        X.append(data_matrix[i:(i + lookback_window), :])
        y.append(target_matrix[i + lookback_window])
        
    return np.array(X), np.array(y)