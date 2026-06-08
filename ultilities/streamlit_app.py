import streamlit as st
import pandas as pd
import yfinance as yf
from vnstock import Vnstock
import os
import numpy as np

# Import mô-đun báo cáo thống kê tự động nâng cao
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report

# Cấu hình giao diện hiển thị góc rộng cho Data Pipeline
st.set_page_config(page_title="Data Mining Pipeline - Phase 1 Advanced", layout="wide")

# ═════════════════════════════════════════════════════════════════
# CORE FUNCTIONS: INGESTION, FEATURE ENGINEERING & SPLITTING
# ═════════════════════════════════════════════════════════════════

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
            stock = Vnstock().stock(symbol=ticker_clean, source='VCI')
            df_raw = stock.quote.history(start='2010-01-01', end='2026-12-31')
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
    Tự động trích xuất các đặc trưng chỉ báo kỹ thuật cơ bản
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
    Hàm cắt lọc thời gian, phân chia Train/Validation/Test theo chuỗi thời gian tuyến tính và ghi tệp CSV cục bộ.
    """
    df['Year'] = df['Date'].dt.year
    df_filtered = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)].copy()
    df_filtered.drop(columns=['Year'], inplace=True)
    
    if df_filtered.empty:
        raise ValueError(f"Không có dữ liệu của mã {ticker} trong giai đoạn {start_year} - {end_year}.")
    
    # Tính toán điểm cắt dòng thời gian dựa trên tỷ lệ được phân bổ
    total_len = len(df_filtered)
    train_end_idx = int(total_len * train_ratio)
    val_end_idx = int(total_len * (train_ratio + val_ratio))
    
    # Phân chia dữ liệu theo đúng trình tự thời gian
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
    
    # Ghi dữ liệu trực tiếp xuống ổ đĩa cục bộ
    df_filtered.to_csv(full_path, index=False)
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    paths_dict = {"full": full_path, "train": train_path, "val": val_path, "test": test_path}
    return df_filtered, df_train, df_val, df_test, paths_dict


# ═════════════════════════════════════════════════════════════════
# STREAMLIT USER INTERFACE
# ═════════════════════════════════════════════════════════════════

st.title("🛡️ Machine Learning Pipeline — Phase 1: Advanced Data Engine")
st.markdown("Hệ thống nạp dữ liệu thông minh, tích hợp tự động trích xuất chỉ báo đặc trưng kỹ thuật, lọc thời gian và phân chia Dataset.")
st.markdown("---")

# Cấu hình thanh Sidebar
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

start_yr, end_yr = st.sidebar.slider(
    "2. Chọn Khoảng Thời Gian (Năm):",
    min_value=2010, max_value=2026, value=(2018, 2026)
)

st.sidebar.markdown("**3. Phân chia tỷ lệ Dataset (Time-Series Split):**")
# Sử dụng 2 thanh trượt độc lập, tự động điều chỉnh logic tổng không vượt quá 100%
train_pct = st.sidebar.slider("Tỷ Lệ Huấn Luyện (Train Size):", min_value=0.40, max_value=0.90, value=0.70, step=0.05)
val_pct = st.sidebar.slider("Tỷ Lệ Kiểm Định (Validation Size):", min_value=0.05, max_value=0.30, value=0.15, step=0.05)
test_pct = round(1.0 - train_pct - val_pct, 2)

# Hiển thị phân bổ trực quan cho lập trình viên theo dõi
if test_pct < 0:
    st.sidebar.error(f"❌ Tổng tỷ lệ vượt quá 100% ({int((train_pct + val_pct)*100)}%). Hãy giảm bớt tỷ lệ Train hoặc Validation.")
else:
    st.sidebar.markdown(f"📊 *Phân bổ thực tế: Train {int(train_pct*100)}% | Val {int(val_pct*100)}% | Test {int(test_pct*100)}%*")

st.sidebar.markdown("---")
enable_features = st.sidebar.checkbox("Trích xuất chỉ báo đặc trưng (SMA, RSI, Volatility)", value=True)
generate_report = st.sidebar.checkbox("Tự động sinh báo cáo ydata-profiling chuyên sâu", value=False)

trigger_pipeline = st.sidebar.button("🚀 KÍCH HOẠT PHASE 1 PIPELINE", use_container_width=True, disabled=(test_pct < 0))

# Khởi tạo trạng thái bộ nhớ đệm Streamlit (session_state) để lưu dữ liệu khi render tab nâng cao
if 'data_cache' not in st.session_state:
    st.session_state['data_cache'] = None

# ═════════════════════════════════════════════════════════════════
# EXECUTION & OUTPUT DISPLAY
# ═════════════════════════════════════════════════════════════════

if trigger_pipeline:
    with st.spinner("⏳ Đang thiết lập luồng tải dữ liệu và tính toán phân tách hệ thống..."):
        try:
            df_processed = fetch_raw_data(ticker_input, selected_source)
            
            if enable_features:
                df_processed = compute_basic_features(df_processed)
                st.success("⚙️ Đã hoàn thành trích xuất ma trận đặc trưng kỹ thuật bổ trợ (Feature Matrix)!")
            else:
                st.warning("⚠️ Bỏ qua bước trích xuất chỉ báo, chỉ giữ lại các cột giá thô.")
                
            df_filtered, df_train, df_val, df_test, paths = process_and_split_pipeline(
                df=df_processed, start_year=start_yr, end_year=end_yr, train_ratio=train_pct, val_ratio=val_pct, ticker=ticker_input
            )
            
            # Lưu trữ vào session_state
            st.session_state['data_cache'] = {
                "df_filtered": df_filtered, "df_train": df_train, "df_val": df_val, "df_test": df_test,
                "paths": paths, "ticker": ticker_input.strip().upper(), "test_pct": test_pct
            }
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Pipeline gặp lỗi hệ thống: {str(e)}")

# Đọc dữ liệu từ bộ nhớ trạng thái để hiển thị lên UI
if st.session_state['data_cache'] is not None:
    cache = st.session_state['data_cache']
    
    st.markdown("## 📊 KẾT QUẢ ĐỒNG BỘ VÀ PHÂN TÁCH PIPELINE THÀNH CÔNG")
    st.info(f"📁 **Vị trí file CSV được ghi tự động vào Data Lake cục bộ:**\n"
            f"* Toàn bộ dữ liệu sau lọc: `{cache['paths']['full']}`\n"
            f"* Tập Huấn luyện (Train Set): `{cache['paths']['train']}`\n"
            f"* Tập Kiểm định (Validation Set): `{cache['paths']['val']}`\n"
            f"* Tập Kiểm thử (Test Set): `{cache['paths']['test']}`")
    
    st.write("📥 **Tải trực tiếp các file đã xử lý về máy tính cá nhân:**")
    dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)
    with dl_col1:
        st.download_button(label="📥 Tải file Full CSV", data=cache['df_filtered'].to_csv(index=False), file_name=f"{cache['ticker']}_full.csv", mime="text/csv")
    with dl_col2:
        st.download_button(label="🧠 Tải file Train CSV", data=cache['df_train'].to_csv(index=False), file_name=f"{cache['ticker']}_train.csv", mime="text/csv")
    with dl_col3:
        st.download_button(label="🧪 Tải file Validation CSV", data=cache['df_val'].to_csv(index=False), file_name=f"{cache['ticker']}_val.csv", mime="text/csv")
    with dl_col4:
        st.download_button(label="🎯 Tải file Test CSV", data=cache['df_test'].to_csv(index=False), file_name=f"{cache['ticker']}_test.csv", mime="text/csv")
        
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng Số Phiên Thống Kê", f"{len(cache['df_filtered']):,} dòng", "100%")
    c2.metric("Tập Huấn Luyện (Train)", f"{len(cache['df_train']):,} dòng", f"{train_pct*100:.0f}%")
    c3.metric("Tập Kiểm Định (Validation)", f"{len(cache['df_val']):,} dòng", f"{val_pct*100:.0f}%")
    c4.metric("Tập Kiểm Thử (Test)", f"{len(cache['df_test']):,} dòng", f"{cache['test_pct']*100:.0f}%")
    
    # Thiết lập cấu trúc các Tab đầu ra trực quan
    data_tabs = st.tabs([
        "📋 Toàn Bộ Dataset Sau Lọc", 
        "🧠 Tập Huấn Luyện (Train)", 
        "🧪 Tập Kiểm Định (Validation)",
        "🎯 Tập Kiểm Thử (Test)", 
        "🧬 Báo cáo Tự động ydata-profiling"
    ])
    
    with data_tabs[0]:
        st.caption(f"Khung thời gian biểu diễn: {cache['df_filtered']['Date'].min().date()} đến {cache['df_filtered']['Date'].max().date()}")
        st.dataframe(cache['df_filtered'], use_container_width=True)
        
    with data_tabs[1]:
        st.caption(f"Dữ liệu dùng để huấn luyện mô hình (Từ ngày {cache['df_train']['Date'].min().date()} đến {cache['df_train']['Date'].max().date()})")
        st.dataframe(cache['df_train'], use_container_width=True)
        
    with data_tabs[2]:
        st.caption(f"Dữ liệu dùng để tối ưu siêu tham số & chống quá khớp (Từ ngày {cache['df_val']['Date'].min().date()} đến {cache['df_val']['Date'].max().date()})")
        st.dataframe(cache['df_val'], use_container_width=True)
        
    with data_tabs[3]:
        st.caption(f"Dữ liệu out-of-sample dùng để chấm điểm kiểm thử cuối cùng (Từ ngày {cache['df_test']['Date'].min().date()} đến {cache['df_test']['Date'].max().date()})")
        st.dataframe(cache['df_test'], use_container_width=True)
        
    with data_tabs[4]:
        st.subheader("Báo Cáo Khai Phá Dữ Liệu Tự Động Toàn Diện (Automated EDA)")
        if generate_report:
            with st.spinner("📊 Đang phân tích ma trận dữ liệu và xây dựng báo cáo ydata-profiling..."):
                profile = ProfileReport(cache['df_filtered'], title=f"EDA Report: {cache['ticker']}", explorative=True, minimal=False)
                st_profile_report(profile)
        else:
            st.info("💡 Bạn chưa tích chọn tính năng này. Hãy tích vào ô 'Tự động sinh báo cáo ydata-profiling chuyên sâu' ở thanh Sidebar bên trái và ấn nút khởi chạy để quét dữ liệu nâng cao.")
else:
    st.info("💡 Hệ thống đang sẵn sàng. Hãy điều chỉnh cấu hình ở Sidebar bên trái và bấm nút kích hoạt.")