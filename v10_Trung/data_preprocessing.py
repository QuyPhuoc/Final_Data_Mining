# data_preprocessing.py (FIXED)
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Literal, Optional, Tuple, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import warnings

# ============================
# 1. NORMALIZER (giữ nguyên)
# ============================
class Normalizer:
    def __init__(self, method: Literal['minmax', 'standard', 'robust'] = 'standard'):
        self.method = method
        if method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("method must be 'minmax', 'standard', or 'robust'")
        self.fitted = False
        self.columns = None

    def fit(self, X: pd.DataFrame, columns: Optional[list] = None):
        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.columns = columns
        self.scaler.fit(X[columns])
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Phải gọi fit() trước khi transform")
        X = X.copy()
        X[self.columns] = self.scaler.transform(X[self.columns])
        return X

    def fit_transform(self, X: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
        self.fit(X, columns)
        return self.transform(X)

def sort_by_date(df: pd.DataFrame, sort_col: str = 'date') -> pd.DataFrame:
    """
    Sắp xếp DataFrame theo cột thời gian tăng dần.
    Thường được gọi trước khi chia tập.
    """
    if sort_col not in df.columns:
        raise ValueError(f"Cột '{sort_col}' không tồn tại.")
    return df.sort_values(sort_col).reset_index(drop=True)

def fix_ohlc_errors(df: pd.DataFrame, 
                    ohlc_cols: Tuple[str, str, str, str] = ('open', 'high', 'low', 'close')) -> pd.DataFrame:
    """
    Sửa lỗi OHLC: Đảm bảo high = max(open, high, low, close), low = min(open, high, low, close).
    Chỉ áp dụng nếu các cột tồn tại. Trả về DataFrame đã sửa (các cột khác không đổi).
    """
    df = df.copy()
    open_col, high_col, low_col, close_col = ohlc_cols
    
    # Kiểm tra sự tồn tại của các cột
    if all(c in df.columns for c in ohlc_cols):
        # Tính đúng high và low từ 4 giá trị
        df[high_col] = df[[open_col, high_col, low_col, close_col]].max(axis=1)
        df[low_col]  = df[[open_col, high_col, low_col, close_col]].min(axis=1)
    else:
        raise ValueError(f"Thiếu một trong các cột: {ohlc_cols}")
    return df

# ============================
# 2. CHIA TẬP (KHÔNG SHUFFLE, GIỮ TẤT CẢ COLUMNS)
# ============================
def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    sort_by: str = "date"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chia dữ liệu theo thời gian - GIỮ NGUYÊN TẤT CẢ CỘT
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Tổng tỷ lệ phải bằng 1"
    
    df = df.copy(deep=True)
    
    # Sắp xếp theo thời gian
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()
    
    # Reset index
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    return train, val, test

# ============================
# 3. XỬ LÝ MISSING (SAU KHI CHIA, AN TOÀN)
# ============================
def remove_rows_by_missing_threshold(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Xóa dòng có tỷ lệ missing > threshold (0-1)"""
    missing_ratio = df.isnull().mean(axis=1)
    return df[missing_ratio <= threshold].copy()

def fill_forward(df: pd.DataFrame) -> pd.DataFrame:
    return df.ffill()

def fill_missing_with_mean(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mean_val = train[column].mean()
    train = train.copy(); val = val.copy(); test = test.copy()
    train[column] = train[column].fillna(mean_val)
    val[column] = val[column].fillna(mean_val)
    test[column] = test[column].fillna(mean_val)
    return train, val, test

def fill_missing_with_median(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    median_val = train[column].median()
    train = train.copy(); val = val.copy(); test = test.copy()
    train[column] = train[column].fillna(median_val)
    val[column] = val[column].fillna(median_val)
    test[column] = test[column].fillna(median_val)
    return train, val, test

def fill_missing_with_mode(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mode_val = train[column].mode()
    mode_val = mode_val.iloc[0] if not mode_val.empty else 0
    train = train.copy(); val = val.copy(); test = test.copy()
    train[column] = train[column].fillna(mode_val)
    val[column] = val[column].fillna(mode_val)
    test[column] = test[column].fillna(mode_val)
    return train, val, test

def fill_missing_with_ml(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    feature_cols: list
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = train.copy(); val = val.copy(); test = test.copy()
    
    # 1. Xử lý missing trong feature_cols (dùng mean của train)
    for col in feature_cols:
        mean_val = train[col].mean()
        train[col] = train[col].fillna(mean_val)
        val[col] = val[col].fillna(mean_val)
        test[col] = test[col].fillna(mean_val)
    
    # 2. Huấn luyện trên các dòng có target không missing
    known = train[train[target_col].notna()]
    if known.empty or len(feature_cols) == 0:
        return train, val, test
    
    X_train = known[feature_cols].values
    y_train = known[target_col].values
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 3. Dự đoán cho các dòng missing
    for df in (train, val, test):
        mask = df[target_col].isna()
        if mask.any():
            X_pred = df.loc[mask, feature_cols].values
            df.loc[mask, target_col] = model.predict(X_pred)
    return train, val, test

# ============================
# 4. XỬ LÝ TRÙNG LẶP (TRƯỚC KHI CHIA)
# ============================
def remove_duplicate_rows(df: pd.DataFrame, subset: list = None, keep: str = 'first') -> pd.DataFrame:
    return df.drop_duplicates(subset=subset, keep=keep)

# ============================
# 5. CHUẨN HÓA TEXT (GIỮ NGUYÊN DẤU TIẾNG VIỆT)
# ============================
def standardize_text_columns(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Chuẩn hóa cột text: strip, lower, nhưng giữ nguyên dấu tiếng Việt.
    Không dùng regex xóa ký tự đặc biệt.
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    return df

# ============================
# 6. HÀM HELPER: LẤY DANH SÁCH CỘT SỐ CÓ THỂ NORMALIZE
# ============================
def get_numeric_feature_cols(
    df: pd.DataFrame,
    exclude_cols: List[str] = None
) -> List[str]:
    """
    Lấy danh sách cột số có thể chuẩn hóa.
    Loại trừ date, target, và các cột trong exclude_cols.
    """
    if exclude_cols is None:
        exclude_cols = ['date', 'datetime', 'target']
    else:
        exclude_cols = list(exclude_cols) + ['date', 'datetime', 'target']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude_cols]

# ============================
# 7. CÁC HÀM CŨ KHÔNG AN TOÀN (DEPRECATED, CÓ CẢNH BÁO)
# ============================
def fill_missing_mean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    warnings.warn("fill_missing_mean trên toàn bộ dữ liệu có thể gây leakage. Hãy dùng fill_missing_with_mean sau split.", UserWarning)
    df = df.copy()
    df[column] = df[column].fillna(df[column].mean())
    return df

def fill_missing_median(df: pd.DataFrame, column: str) -> pd.DataFrame:
    warnings.warn("fill_missing_median trên toàn bộ dữ liệu có thể gây leakage.", UserWarning)
    df = df.copy()
    df[column] = df[column].fillna(df[column].median())
    return df

def fill_missing_mode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    warnings.warn("fill_missing_mode trên toàn bộ dữ liệu có thể gây leakage.", UserWarning)
    df = df.copy()
    mode_val = df[column].mode()
    if not mode_val.empty:
        df[column] = df[column].fillna(mode_val.iloc[0])
    return df

def fill_missing_ml(df: pd.DataFrame, target_col: str, feature_cols: list) -> pd.DataFrame:
    warnings.warn("fill_missing_ml trên toàn bộ dữ liệu gây leakage nghiêm trọng. Hãy dùng fill_missing_with_ml sau split.", UserWarning)
    df = df.copy()
    known = df[df[target_col].notna()]
    unknown = df[df[target_col].isna()]
    if unknown.empty or known.empty:
        return df
    X_train = known[feature_cols].values
    y_train = known[target_col].values
    model = LinearRegression()
    model.fit(X_train, y_train)
    X_pred = unknown[feature_cols].values
    preds = model.predict(X_pred)
    df.loc[df[target_col].isna(), target_col] = preds
    return df

def remove_missing_rows(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """Xóa dòng có missing (chỉ nên dùng trước split)"""
    if subset is None:
        return df.dropna()
    return df.dropna(subset=subset)