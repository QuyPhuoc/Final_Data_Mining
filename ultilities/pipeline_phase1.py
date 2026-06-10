import os
import numpy as np
import pandas as pd
import yfinance as yf

def fetch_raw_data(ticker_symbol: str, source: str) -> pd.DataFrame:
    """
    Hàm gọi API kết nối sàn giao dịch lấy dữ liệu thô gốc và đồng bộ hóa Schema.
    """
    ticker_clean = ticker_symbol.strip().upper()
    
    if source == "Yahoo Finance (Crypto, Quốc tế)":
        df_raw = yf.download(tickers=ticker_clean, period="max", interval="1d", auto_adjust=False)
        if df_raw.empty:
            raise ValueError(f"Không tìm thấy dữ liệu cho mã '{ticker_clean}' trên Yahoo Finance.")
        
        df_raw = df_raw.reset_index()
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = [col[0] if isinstance(col, tuple) else col for col in df_raw.columns]
            
        df_raw['Ticker'] = ticker_clean.split('-')[0]
        df_raw['DTYYYYMMDD'] = df_raw['Date'].dt.strftime('%Y%m%d')
        
        rename_map = {col: col.capitalize() for col in df_raw.columns if str(col).lower() in ['open', 'high', 'low', 'close', 'volume']}
        df_raw.rename(columns=rename_map, inplace=True)
        
    elif source == "Vnstock (Chứng khoán Việt Nam)":
        try:
            from vnstock.api.quote import Quote
            q = Quote(symbol=ticker_clean, source='VCI')
            df_raw = q.history(start='2010-01-01', end='2026-12-31')
        except Exception:
            from vnstock import stock_historical_data
            df_raw = stock_historical_data(symbol=ticker_clean, start_date='2010-01-01', end_date='2026-12-31', resolution='1D', type='stock')
            
        if df_raw is None or df_raw.empty:
            raise ValueError(f"API Vnstock không trả về dữ liệu cho mã '{ticker_clean}'.")
            
        df_raw = df_raw.reset_index()
        df_raw.columns = [str(col) for col in df_raw.columns]
        
        rename_map = {}
        for col in df_raw.columns:
            if col.lower() in ['open', 'high', 'low', 'close', 'volume']:
                rename_map[col] = col.capitalize()
            elif col.lower() in ['time', 'date', 'tradingdate']:
                rename_map[col] = 'Date'
                
        df_raw.rename(columns=rename_map, inplace=True)
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])
        df_raw['DTYYYYMMDD'] = df_raw['Date'].dt.strftime('%Y%m%d')
        df_raw['Ticker'] = ticker_clean
    else:
        raise ValueError("Nguồn cấp dữ liệu không hợp lệ.")

    required_cols = ['Ticker', 'DTYYYYMMDD', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df_final = df_raw[required_cols].sort_values('Date').reset_index(drop=True)
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
    df_final.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
    
    return df_final

def compute_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tự động trích xuất các đặc trưng chỉ báo kỹ thuật cơ bản bổ trợ cho mô hình ML.
    """
    df = df.copy()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility_20d'] = df['Log_Return'].rolling(window=20).std()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    df.dropna(inplace=True)
    return df.reset_index(drop=True)

def process_and_split_pipeline(df: pd.DataFrame, start_year: int, end_year: int, train_ratio: float, val_ratio: float, ticker: str, output_dir: str = "../data") -> tuple:
    """
    Hàm cắt lọc thời gian, phân chia Train/Validation/Test theo dòng thời gian tuyến tính và ghi tệp CSV.
    """
    df['Year'] = df['Date'].dt.year
    df_filtered = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)].copy()
    df_filtered.drop(columns=['Year'], inplace=True)
    
    if df_filtered.empty:
        raise ValueError(f"Không có dữ liệu của mã {ticker} trong giai đoạn {start_year} - {end_year}.")
    
    total_len = len(df_filtered)
    train_end_idx = int(total_len * train_ratio)
    val_end_idx = int(total_len * (train_ratio + val_ratio))
    
    df_train = df_filtered.iloc[:train_end_idx].copy()
    df_val = df_filtered.iloc[train_end_idx:val_end_idx].copy()
    df_test = df_filtered.iloc[val_end_idx:].copy()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    ticker_clean = ticker.split('-')[0].upper()
    full_path = os.path.abspath(os.path.join(output_dir, f"{ticker_clean}_full.csv"))
    train_path = os.path.abspath(os.path.join(output_dir, f"{ticker_clean}_train.csv"))
    val_path = os.path.abspath(os.path.join(output_dir, f"{ticker_clean}_val.csv"))
    test_path = os.path.abspath(os.path.join(output_dir, f"{ticker_clean}_test.csv"))
    
    df_filtered.to_csv(full_path, index=False)
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    paths_dict = {"full": full_path, "train": train_path, "val": val_path, "test": test_path}
    return df_filtered, df_train, df_val, df_test, paths_dict