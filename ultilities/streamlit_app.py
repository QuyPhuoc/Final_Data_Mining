import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Import các phân hệ Core Functions đã bóc tách từ các file phase tương ứng
import pipeline_phase1 as p1
import pipeline_phase2 as p2

# Import mô-đun báo cáo thống kê tự động nâng cao
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report

# Cấu hình giao diện hiển thị góc rộng cho Data Pipeline
st.set_page_config(page_title="Data Mining Pipeline - Multi-Phase Architecture", layout="wide")

# ═════════════════════════════════════════════════════════════════
# STREAMLIT USER INTERFACE
# ═════════════════════════════════════════════════════════════════

st.title("🛡️ Machine Learning Pipeline — Enterprise Multi-Phase Data Engine")
st.markdown("Hệ thống xử lý phân tầng cấu trúc: **Phase 1 (Nạp & Tách)** và **Phase 2 (EDA Thống Kê & Phân Rã Toán Học)** độc lập.")
st.markdown("---")

# Cấu hình thanh Sidebar tham số
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
train_pct = st.sidebar.slider("Tỷ Lệ Huấn Luyện (Train Size):", min_value=0.40, max_value=0.90, value=0.70, step=0.05)
val_pct = st.sidebar.slider("Tỷ Lệ Kiểm Định (Validation Size):", min_value=0.05, max_value=0.30, value=0.15, step=0.05)
test_pct = round(1.0 - train_pct - val_pct, 2)

if test_pct < 0:
    st.sidebar.error(f"❌ Tổng tỷ lệ vượt quá 100% ({int((train_pct + val_pct)*100)}%). Hãy giảm bớt tỷ lệ Train hoặc Validation.")
else:
    st.sidebar.markdown(f"📊 *Phân bổ thực tế: Train {int(train_pct*100)}% | Val {int(val_pct*100)}% | Test {int(test_pct*100)}%*")

st.sidebar.markdown("---")
enable_features = st.sidebar.checkbox("Trích xuất chỉ báo đặc trưng (SMA, RSI, Volatility)", value=True)
generate_report = st.sidebar.checkbox("Tự động sinh báo cáo ydata-profiling chuyên sâu", value=False)

trigger_pipeline = st.sidebar.button("🚀 KÍCH HOẠT HỆ THỐNG PIPELINE", use_container_width=True, disabled=(test_pct < 0))

# Khởi tạo trạng thái bộ nhớ đệm (Session State) để lưu trữ dữ liệu
if 'data_cache' not in st.session_state:
    st.session_state['data_cache'] = None

# ═════════════════════════════════════════════════════════════════
# EXECUTION PIPELINE TRIGGER
# ═════════════════════════════════════════════════════════════════

if trigger_pipeline:
    with st.spinner("⏳ Hệ thống đang kích hoạt luồng xử lý phân tầng qua các Phase..."):
        try:
            # Thực thi Phase 1
            df_processed = p1.fetch_raw_data(ticker_input, selected_source)
            
            if enable_features:
                df_processed = p1.compute_basic_features(df_processed)
                st.success("⚙️ [Phase 1] Hoàn thành cấu trúc ma trận đặc trưng (Feature Matrix)!")
            else:
                st.warning("⚠️ [Phase 1] Chỉ sử dụng thuộc tính giá gốc thô.")
                
            df_filtered, df_train, df_val, df_test, paths = p1.process_and_split_pipeline(
                df=df_processed, start_year=start_yr, end_year=end_yr, train_ratio=train_pct, val_ratio=val_pct, ticker=ticker_input
            )
            
            # Đóng gói kết quả lưu vào bộ nhớ đệm
            st.session_state['data_cache'] = {
                "df_filtered": df_filtered, "df_train": df_train, "df_val": df_val, "df_test": df_test,
                "paths": paths, "ticker": ticker_input.strip().upper(), "test_pct": test_pct,
                "train_pct": train_pct, "val_pct": val_pct
            }
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Toàn hệ thống gặp sự cố: {str(e)}")

# ═════════════════════════════════════════════════════════════════
# RENDERING DASHBOARD INTERFACE
# ═════════════════════════════════════════════════════════════════

if st.session_state['data_cache'] is not None:
    cache = st.session_state['data_cache']
    
    st.markdown("## 📊 KẾT QUẢ ĐỒNG BỘ VÀ PHÂN TÁCH PIPELINE THÀNH CÔNG")
    st.info(f"📁 **Vị trí file CSV được ghi tự động vào Data Lake cục bộ:**\n"
            f"* Toàn bộ dữ liệu sau lọc: `{cache['paths']['full']}`\n"
            f"* Tập Huấn luyện (Train Set): `{cache['paths']['train']}`\n"
            f"* Tập Kiểm định (Validation Set): `{cache['paths']['val']}`\n"
            f"* Tập Kiểm thử (Test Set): `{cache['paths']['test']}`")
    
    # Khu vực download nhanh dữ liệu
    dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)
    with dl_col1: st.download_button(label="📥 Tải file Full CSV", data=cache['df_filtered'].to_csv(index=False), file_name=f"{cache['ticker']}_full.csv", mime="text/csv")
    with dl_col2: st.download_button(label="🧠 Tải file Train CSV", data=cache['df_train'].to_csv(index=False), file_name=f"{cache['ticker']}_train.csv", mime="text/csv")
    with dl_col3: st.download_button(label="🧪 Tải file Validation CSV", data=cache['df_val'].to_csv(index=False), file_name=f"{cache['ticker']}_val.csv", mime="text/csv")
    with dl_col4: st.download_button(label="🎯 Tải file Test CSV", data=cache['df_test'].to_csv(index=False), file_name=f"{cache['ticker']}_test.csv", mime="text/csv")
        
    st.markdown("---")
    metric_c1, metric_c2, metric_c3, metric_c4 = st.columns(4)
    metric_c1.metric("Tổng Số Phiên Thống Kê", f"{len(cache['df_filtered']):,} dòng", "100%")
    metric_c2.metric("Tập Huấn Luyện (Train)", f"{len(cache['df_train']):,} dòng", f"{cache['train_pct']*100:.0f}%")
    metric_c3.metric("Tập Kiểm Định (Validation)", f"{len(cache['df_val']):,} dòng", f"{cache['val_pct']*100:.0f}%")
    metric_c4.metric("Tập Kiểm Thử (Test)", f"{len(cache['df_test']):,} dòng", f"{cache['test_pct']*100:.0f}%")
    
    # 🌟 KHỞI TẠO HỆ THỐNG PHÂN TẦNG TABS CHO CẢ 2 PHASE
    data_tabs = st.tabs([
        "📋 [P1] Data Lake Sets", 
        "📊 [P2] Phân Phối Năm & Mốc Thăm Dò",
        "📈 [P2] Phân Rã Chuỗi & Tính Dừng (ADF)",
        "🎯 [P2] Tương Quan Chuỗi ACF/PACF",
        "🧬 [P2] Ma Trận Tương Quan Đa Biến",
        "📊 [P1] Báo cáo Ydata-Profiling"
    ])
    
    # TAB PHASE 1: HIỂN THỊ DỮ LIỆU THÔ VÀ DỮ LIỆU ĐÃ CHIA TẬP
    with data_tabs[0]:
        st.markdown("### 🗂️ Quản Trị Các Tập Dữ Liệu After Split")
        sub_c1, sub_c2, sub_c3 = st.tabs(["Toàn bộ dữ liệu", "Tập Huấn luyện (Train)", "Tập Kiểm định & Kiểm thử (Val/Test)"])
        with sub_c1: st.dataframe(cache['df_filtered'], use_container_width=True)
        with sub_c2: st.dataframe(cache['df_train'], use_container_width=True)
        with sub_c3: 
            st.write("🧪 **Tập Validation:**")
            st.dataframe(cache['df_val'], use_container_width=True)
            st.write("🎯 **Tập Test:**")
            st.dataframe(cache['df_test'], use_container_width=True)
        
    # TAB PHASE 2: PHÂN PHỐI HẰNG NĂM
    with data_tabs[1]:
        st.subheader(f"🔍 Khám Phá Mật Độ Phân Phối Hằng Năm — Asset: {cache['ticker']}")
        df_eda = cache['df_filtered'].copy()
        df_eda['Year_Label'] = df_eda['Date'].dt.year
        
        annual_stats = df_eda.groupby('Year_Label').agg(
            Volume_Samples=('Close', 'count'), Avg_Close=('Close', 'mean'),
            Max_Close=('Close', 'max'), Min_Close=('Close', 'min')
        ).reset_index()
        
        eda_col1, eda_col2 = st.columns([3, 2])
        with eda_col1:
            fig_annual = go.Figure()
            fig_annual.add_trace(go.Bar(
                x=annual_stats['Year_Label'].astype(str), y=annual_stats['Volume_Samples'],
                text=annual_stats['Volume_Samples'], textposition='auto',
                marker_color='#1E88E5', name='Số phiên'
            ))
            fig_annual.update_layout(xaxis_title="Năm", yaxis_title="Tổng số phiên", template="plotly_white", height=340)
            st.plotly_chart(fig_annual, use_container_width=True)
            
        with eda_col2:
            annual_display = annual_stats.copy()
            annual_display['Avg_Close'] = annual_display['Avg_Close'].round(2)
            annual_display.columns = ['Năm tài chính', 'Số lượng phiên', 'Giá Đóng Cửa TB', 'Giá Đỉnh Cao Nhất', 'Giá Đáy Thấp Nhất']
            st.dataframe(annual_display.set_index('Năm tài chính'), use_container_width=True)
            
        st.markdown("---")
        st.markdown("### 🗓️ Đề Xuất Các Mốc Thời Gian Thăm Dò Dữ Liệu")
        years_available = sorted(df_eda['Year_Label'].unique())
        if len(years_available) >= 1:
            m1, m2, m3 = st.columns(3)
            with m1: st.info(f"**📍 Mốc 1: Khởi Nguyên Baseline**\n* **Thời gian:** Năm {years_available[0]}\n* Phân tích đặc trưng tĩnh ban đầu.")
            with m2: 
                mid_idx = len(years_available) // 2
                st.success(f"**📍 Mốc 2: Biến Động Trung Hạn**\n* **Thời gian:** Năm {years_available[mid_idx]}\n* Đánh giá độ lệch và Outliers cấu trúc.")
            with m3: st.warning(f"**📍 Mốc 3: Cận Vệ Hiện Tại (Data Drift)**\n* **Thời gian:** Năm {years_available[-1]}\n* Đo lường độ dịch chuyển phân phối thực tế.")
                
            st.markdown("---")
            selected_year = st.selectbox("🎯 Chọn riêng phân phối của năm cụ thể:", years_available, index=len(years_available)-1)
            df_single_year = df_eda[df_eda['Year_Label'] == selected_year]
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=df_single_year['Close'], nbinsx=30, marker_color='#5E35B1', opacity=0.8))
            fig_hist.update_layout(xaxis_title="Vùng Giá Close", yaxis_title="Tần suất phiên", template="plotly_white", height=300)
            st.plotly_chart(fig_hist, use_container_width=True)

    # TAB PHASE 2: PHÂN RÃ CHUỖI & TÍNH DỪNG
    with data_tabs[2]:
        st.subheader("📈 Phân Phối Cấu Trúc Chuỗi Thời Gian & Thống Kê Toán Học (Vận hành trên Train Set)")
        
        adf_results = p2.perform_adf_test(cache['df_train']['Close'])
        
        c_adf1, c_adf2 = st.columns([1, 2])
        with c_adf1:
            st.markdown("#### 🔬 Kết quả Kiểm định ADF (Stationarity Test)")
            st.metric("ADF Statistic", f"{adf_results['adf_stat']:.4f}")
            st.metric("p-value", f"{adf_results['p_value']:.4e}")
            
            if adf_results['is_stationary']:
                st.success("🎉 Kết luận: Chuỗi ĐÃ ĐẠT TÍNH DỪNG. Sẵn sàng nạp vào mô hình toán học.")
            else:
                # FIXED: Thay thế st.danger bằng st.error chuẩn của Streamlit
                st.error("⚠️ Kết luận: Chuỗi KHÔNG DỪNG. Cần biến đổi sai phân hoặc Log Return tại Phase 3!")
                
            with st.expander("Xem Vùng Ngưỡng Tới Hạn (Critical Values)"):
                st.json(adf_results['critical_values'])
                
        with c_adf2:
            st.markdown("#### 🪵 Phân Rã Thành Phần Chuỗi (Time-Series Decomposition)")
            trend, seasonal, resid = p2.decompose_time_series(cache['df_train'], period=20)
            
            fig_decomp = go.Figure()
            fig_decomp.add_trace(go.Scatter(x=trend.index, y=trend, name='Xu hướng (Trend)', line=dict(color='#FF5722')))
            fig_decomp.add_trace(go.Scatter(x=seasonal.index, y=seasonal, name='Chu kỳ (Seasonal)', line=dict(color='#4CAF50')))
            fig_decomp.add_trace(go.Scatter(x=resid.index, y=resid, name='Nhiễu (Residuals)', mode='markers', marker=dict(size=4, color='#9E9E9E')))
            fig_decomp.update_layout(template="plotly_white", height=350)
            st.plotly_chart(fig_decomp, use_container_width=True)

    # TAB PHASE 2: ACF & PACF
    with data_tabs[3]:
        st.subheader("🎯 Phân Tích Hệ Số Tự Tương Quan (ACF) & Tự Tương Quan Từng Phần (PACF)")
        st.markdown("Xác định mốc trễ (Lookback Window) tối ưu cho các cấu trúc mạng LSTM / Transformer.")
        
        max_lags = st.slider("Chọn số lượng khoảng trễ (Lags) phân tích:", min_value=10, max_value=50, value=25)
        acf_v, pacf_v = p2.compute_acf_pacf(cache['df_train']['Close'], lags=max_lags)
        lag_axis = list(range(max_lags + 1))
        
        c_ap1, c_ap2 = st.columns(2)
        conf = 1.96 / np.sqrt(len(cache['df_train']))
        with c_ap1:
            fig_acf = go.Figure()
            fig_acf.add_trace(go.Bar(x=lag_axis, y=acf_v, marker_color='#009688', name='ACF'))
            fig_acf.add_hline(y=conf, line_dash="dash", line_color="red")
            fig_acf.add_hline(y=-conf, line_dash="dash", line_color="red")
            fig_acf.update_layout(title="Hệ số tự tương quan (ACF)", template="plotly_white", height=300)
            st.plotly_chart(fig_acf, use_container_width=True)
            
        with c_ap2:
            fig_pacf = go.Figure()
            fig_pacf.add_trace(go.Bar(x=lag_axis, y=pacf_v, marker_color='#E91E63', name='PACF'))
            fig_pacf.add_hline(y=conf, line_dash="dash", line_color="red")
            fig_pacf.add_hline(y=-conf, line_dash="dash", line_color="red")
            fig_pacf.update_layout(title="Hệ số tự tương quan từng phần (PACF)", template="plotly_white", height=300)
            st.plotly_chart(fig_pacf, use_container_width=True)

    # TAB PHASE 2: MA TRẬN ĐA BIẾN
    with data_tabs[4]:
        st.subheader("🧬 Ma Trận Tương Quan Tuyến Tính Pearson")
        st.markdown("Rà soát rủi ro đa cộng tuyến giữa các chỉ báo bổ trợ kỹ thuật.")
        
        numeric_cols = cache['df_train'].select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ['DTYYYYMMDD']]
        
        if len(numeric_cols) > 1:
            corr_matrix = cache['df_train'][numeric_cols].corr(method='pearson')
            fig_heatmap = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            fig_heatmap.update_layout(height=450, template="plotly_white")
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("⚠️ Vui lòng bật chức năng 'Trích xuất chỉ báo đặc trưng' ở Sidebar để kích hoạt ma trận đa biến.")

    # TAB PHASE 1: YDATA-PROFILING AUTOMATED REPORT
    with data_tabs[5]:
        st.subheader("Báo Cáo Khai Phá Dữ Liệu Tự Động Toàn Diện (Automated EDA)")
        if generate_report:
            with st.spinner("📊 Đang xây dựng cấu trúc báo cáo ydata-profiling..."):
                # FIXED: Thêm interactions=None để chặn render hexbin/scatter của matplotlib gây crash ứng dụng
                profile = ProfileReport(
                    cache['df_filtered'], 
                    title=f"EDA Report: {cache['ticker']}", 
                    explorative=True, 
                    minimal=False,
                    interactions=None
                )
                st_profile_report(profile)
        else:
            st.info("💡 Tính năng sinh báo cáo tự động đang tắt. Hãy bật ở thanh Sidebar trái nếu cần quét dữ liệu phân vị rộng.")
else:
    st.info("💡 Hệ thống đang sẵn sàng. Hãy cấu hình tham số đầu vào ở thanh Sidebar bên trái và ấn nút khởi chạy.")