import streamlit as st
import pandas as pd
import yfinance as yf
from vnstock import Vnstock  # Sử dụng cú pháp hợp nhất chuẩn 4.x
import os

# Cấu hình giao diện hiển thị góc rộng cho Data Pipeline
st.set_page_config(page_title="Data Mining Pipeline - Phase 1", layout="wide")

# ═════════════════════════════════════════════════════════════════
# CORE FUNCTIONS: INGESTION, SPLITTING & EXPORT
# ═════════════════════════════════════════════════════════════════

def fetch_raw_data(ticker_symbol: str, source: str) -> pd.DataFrame:
    """
    Hàm gọi API kết nối sàn giao dịch lấy dữ liệu thô gốc (Schema Alignment)
    Đã sửa lỗi MultiIndex Tuple bằng cách làm phẳng tên cột (Flatten Columns).
    """
    ticker_clean = ticker_symbol.strip().upper()
    
    if source == "Yahoo Finance (Crypto, Quốc tế)":
        df_raw = yf.download(tickers=ticker_clean, period="max", interval="1d", auto_adjust=False)
        if df_raw.empty:
            raise ValueError(f"Không tìm thấy dữ liệu cho mã '{ticker_clean}' trên Yahoo Finance.")
        
        df_raw = df_raw.reset_index()
        
        # CHỐNG LỖI TUPLE: Nếu cột là MultiIndex, chỉ lấy phần tử tên cột đầu tiên (ví dụ: 'Open')
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = [col[0] if isinstance(col, tuple) else col for col in df_raw.columns]
        else:
            df_raw.columns = [col for col in df_raw.columns]
            
        df_raw['Ticker'] = ticker_clean.split('-')[0]
        df_raw['DTYYYYMMDD'] = df_raw['Date'].dt.strftime('%Y%m%d')
        
        # Đồng bộ hóa tên các cột OHLCV viết hoa chữ cái đầu
        rename_map = {col: col.capitalize() for col in df_raw.columns if str(col).lower() in ['open', 'high', 'low', 'close', 'volume']}
        df_raw.rename(columns=rename_map, inplace=True)
        
    elif source == "Vnstock (Chứng khoán Việt Nam)":
        try:
            stock = Vnstock().stock(symbol=ticker_clean, source='VCI')
            df_raw = stock.quote.history(start='2010-01-01', end='2026-12-31')
        except Exception:
            from vnstock import stock_historical_data
            df_raw = stock_historical_data(symbol=ticker_clean, start_date='2010-01-01', end_date='2026-12-31', resolution='1D', type='stock')
            
        if df_raw is None or df_raw.empty:
            raise ValueError(f"API Vnstock không trả về dữ liệu cho mã '{ticker_clean}'.")
            
        df_raw = df_raw.reset_index()
        
        # Đảm bảo ép kiểu chuỗi cho tên cột của Vnstock trước khi xử lý lower
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

    # Khóa các trường dữ liệu bắt buộc cấu thành Feature Matrix
    required_cols = ['Ticker', 'DTYYYYMMDD', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df_final = df_raw[required_cols].sort_values('Date').reset_index(drop=True)
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
    df_final.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
    
    return df_final


def process_and_split_pipeline(df: pd.DataFrame, start_year: int, end_year: int, train_ratio: float, ticker: str, output_dir: str = "../data") -> tuple:
    """
    Hàm cắt lọc dữ liệu theo khoảng thời gian, phân chia tập Train/Test theo chuỗi thời gian tuyến tính,
    và thực thi ghi trực tiếp các tệp CSV cục bộ.
    """
    df['Year'] = df['Date'].dt.year
    df_filtered = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)].copy()
    df_filtered.drop(columns=['Year'], inplace=True)
    
    if df_filtered.empty:
        raise ValueError(f"Không có dữ liệu của mã {ticker} trong giai đoạn {start_year} - {end_year}.")
    
    # Phân chia tập dữ liệu theo Time-Series Split (Nghiêm cấm Shuffle ngẫu nhiên)
    total_len = len(df_filtered)
    train_size = int(total_len * train_ratio)
    
    df_train = df_filtered.iloc[:train_size].copy()
    df_test = df_filtered.iloc[train_size:].copy()
    
    # Khởi tạo đường dẫn lưu trữ Data Lake
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    ticker_clean = ticker.split('-')[0].upper()
    full_path = os.path.abspath(os.path.join(output_dir, f"{ticker_clean}_full.csv"))
    train_path = os.path.abspath(os.path.join(output_dir, f"{ticker_clean}_train.csv"))
    test_path = os.path.abspath(os.path.join(output_dir, f"{ticker_clean}_test.csv"))
    
    # Ghi dữ liệu xuống ổ đĩa
    df_filtered.to_csv(full_path, index=False)
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    paths_dict = {"full": full_path, "train": train_path, "test": test_path}
    return df_filtered, df_train, df_test, paths_dict


# ═════════════════════════════════════════════════════════════════
# STREAMLIT USER INTERFACE
# ═════════════════════════════════════════════════════════════════

st.title("🛡️ Machine Learning Pipeline — Phase 1: Data Preparation")
st.markdown("Hệ thống nạp dữ liệu từ các sàn giao dịch, cấu hình tham số Dataset và phân tách tập dữ liệu chuẩn bị cho huấn luyện mô hình.")
st.markdown("---")

# Cấu hình thanh Sidebar điều khiển tham số
st.sidebar.header("⚙️ Cấu Hình Tham Số Dataset")

selected_source = st.sidebar.radio(
    "1. Chọn Cổng Kết Nối API:",
    ["Yahoo Finance (Crypto, Quốc tế)", "Vnstock (Chứng khoán Việt Nam)"]
)

if selected_source == "Yahoo Finance (Crypto, Quốc tế)":
    ticker_input = st.sidebar.text_input("Nhập mã Asset Quốc tế:", "BTC-USD")
else:
    ticker_input = st.sidebar.text_input("Nhập mã Cổ phiếu VN:", "BID")

st.sidebar.markdown("---")

# Bộ lọc khoảng thời gian (Năm)
start_yr, end_yr = st.sidebar.slider(
    "2. Chọn Khoảng Thời Gian (Năm):",
    min_value=2010, max_value=2026, value=(2018, 2026)
)

# Tỷ lệ phân chia tập huấn luyện
train_pct = st.sidebar.slider(
    "3. Tỷ Lệ Chia Tập Huấn Luyện (Train Size):",
    min_value=0.50, max_value=0.95, value=0.80, step=0.05
)

trigger_pipeline = st.sidebar.button("🚀 KÍCH HOẠT PHASE 1 PIPELINE", use_container_width=True)

# ═════════════════════════════════════════════════════════════════
# EXECUTION & OUTPUT IN SCREEN
# ═════════════════════════════════════════════════════════════════

if trigger_pipeline:
    with st.spinner("⏳ Đang thiết lập luồng tải dữ liệu và phân chia tập dữ liệu..."):
        try:
            # 1. Gọi API lấy dữ liệu thô
            df_raw = fetch_raw_data(ticker_input, selected_source)
            
            # 2. Xử lý cắt lọc thời gian, chia Train/Test và ghi file CSV
            df_filtered, df_train, df_test, paths = process_and_split_pipeline(
                df=df_raw, start_year=start_yr, end_year=end_yr, train_ratio=train_pct, ticker=ticker_input
            )
            
            # 3. In toàn bộ kết quả trực quan ra màn hình
            st.balloons()
            st.markdown("## 📊 KẾT QUẢ PHÂN TÁCH VÀ XUẤT TỆP CSV THÀNH CÔNG")
            
            st.info(f"📁 **Vị trí tệp CSV cục bộ vừa xuất:**\n"
                    f"* Toàn bộ tập dữ liệu (Lọc năm): `{paths['full']}`\n"
                    f"* Tập Huấn luyện (Train Set): `{paths['train']}`\n"
                    f"* Tập Kiểm thử (Test Set): `{paths['test']}`")
            
            # Hiển thị Metrics tổng quan số lượng dòng dữ liệu sau chia
            c1, c2, c3 = st.columns(3)
            c1.metric("Tổng Số Phiên Sau Lọc", f"{len(df_filtered):,} dòng", "100%")
            c2.metric("Tập Huấn Luyện (Train Set)", f"{len(df_train):,} dòng", f"{train_pct*100:.0f}%")
            c3.metric("Tập Kiểm Thử (Test Set)", f"{len(df_test):,} dòng", f"{(1-train_pct)*100:.0f}%")
            
            # In toàn bộ dữ liệu ra màn hình thông qua các Tab dữ liệu lớn
            data_tabs = st.tabs(["📋 Toàn Bộ Dataset Sau Lọc", "🧠 Tập Huấn Luyện (Train Set)", "🎯 Tập Kiểm Thử (Test Set)"])
            
            with data_tabs[0]:
                st.caption(f"Dữ liệu từ ngày {df_filtered['Date'].min().date()} đến {df_filtered['Date'].max().date()}")
                st.dataframe(df_filtered, use_container_width=True)
                
            with data_tabs[1]:
                st.caption(f"Dữ liệu dùng để học máy (Từ ngày {df_train['Date'].min().date()} đến {df_train['Date'].max().date()})")
                st.dataframe(df_train, use_container_width=True)
                
            with data_tabs[2]:
                st.caption(f"Dữ liệu dùng để backtest đánh giá (Từ ngày {df_test['Date'].min().date()} đến {df_test['Date'].max().date()})")
                st.dataframe(df_test, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Pipeline gặp lỗi hệ thống: {str(e)}")
else:
    st.info("💡 Hệ thống đang sẵn sàng. Hãy điều chỉnh cấu hình ở Sidebar bên trái và bấm nút kích hoạt.")