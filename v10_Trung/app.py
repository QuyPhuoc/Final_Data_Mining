import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import modules
from data_collector import fetch_data, detect_source
from analyse_data import (
    load_df, analyse_quality, 
    chart_candlestick, chart_missing, chart_returns_dist,
    chart_rolling_vol, chart_gaps,
    metric_card, section
)

st.set_page_config(
    page_title="Stock Analysis Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Font & base */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #0f1117; border-right: 1px solid #1e2130; }
    [data-testid="stSidebar"] * { color: #c9d1d9 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextInput label { color: #8b949e !important; font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }

    /* Main background */
    .main { background: #0d1117; }
    .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

    /* Metric cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .metric-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 4px; }
    .metric-value { font-size: 26px; font-weight: 600; color: #e6edf3; font-family: 'JetBrains Mono', monospace; }
    .metric-sub   { font-size: 12px; color: #8b949e; margin-top: 3px; }
    .metric-ok    { color: #3fb950; }
    .metric-warn  { color: #d29922; }
    .metric-bad   { color: #f85149; }

    /* Section headers */
    .section-header {
        font-size: 13px; font-weight: 600; color: #8b949e;
        text-transform: uppercase; letter-spacing: .1em;
        border-bottom: 1px solid #21262d;
        padding-bottom: 8px; margin: 24px 0 16px;
    }

    /* Code blocks */
    .code-block {
        background: #161b22;
        border: 1px solid #21262d;
        border-left: 3px solid #388bfd;
        border-radius: 6px;
        padding: 14px 18px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12.5px;
        color: #e6edf3;
        white-space: pre;
        overflow-x: auto;
        margin: 8px 0 16px;
    }

    /* Explainer cards */
    .explain-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 18px 22px;
        margin: 10px 0;
    }
    .explain-title { font-size: 14px; font-weight: 600; color: #79c0ff; margin-bottom: 8px; }
    .explain-body  { font-size: 13.5px; color: #c9d1d9; line-height: 1.7; }
    .explain-body b { color: #e6edf3; }
    .tag {
        display: inline-block;
        font-size: 11px; font-weight: 500;
        padding: 2px 8px; border-radius: 12px;
        margin-right: 4px; margin-bottom: 4px;
    }
    .tag-blue   { background: #1f3a5f; color: #79c0ff; }
    .tag-green  { background: #1a3a28; color: #3fb950; }
    .tag-orange { background: #3a2a10; color: #d29922; }
    .tag-red    { background: #3a1a1a; color: #f85149; }

    /* Alert boxes */
    .alert-warn {
        background: #272115; border: 1px solid #4a3720;
        border-radius: 8px; padding: 12px 16px;
        font-size: 13px; color: #d29922; margin: 8px 0;
    }
    .alert-ok {
        background: #162217; border: 1px solid #255534;
        border-radius: 8px; padding: 12px 16px;
        font-size: 13px; color: #3fb950; margin: 8px 0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; padding: 4px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #8b949e; border-radius: 6px; padding: 8px 20px; font-size: 13px; font-weight: 500; }
    .stTabs [aria-selected="true"] { background: #21262d; color: #e6edf3; }

    /* Hide streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📈 Stock Analysis Tool")
    st.markdown("---")
    
    # Cho phép chọn phương thức lấy dữ liệu
    data_mode = st.radio("Chọn phương thức nạp dữ liệu:", ["Crawl Tự động (API)", "Upload File Thủ công"])
    
    current_active_ticker = "Uploaded_File"
    
    if data_mode == "Crawl Tự động (API)":
        ticker_input = st.text_input("Nhập mã (Ticker):", value="FPT", help="Mã VN (FPT, HPG), US (AAPL), Crypto (BTC)").strip().upper()
        current_active_ticker = ticker_input
        
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input("Từ ngày", value=datetime.today() - timedelta(days=3*365))
        with col_end:
            end_date = st.date_input("Đến ngày", value=datetime.today())
            
        use_cache_opt = st.toggle("Sử dụng Cache Parquet", value=True)
        
        if ticker_input:
            src_detected = detect_source(ticker_input)
            st.info(f"🔍 Tự nhận diện nguồn: `{src_detected}`")
            
        btn_crawl = st.button("🚀 Bắt đầu Thu thập", use_container_width=True)
        
        if btn_crawl:
            if start_date > end_date:
                st.error("Ngày bắt đầu phải nhỏ hơn ngày kết thúc.")
            else:
                with st.spinner(f"Đang crawl dữ liệu {ticker_input}..."):
                    try:
                        df_crawled = fetch_data(
                            ticker=ticker_input,
                            start=start_date.strftime("%Y-%m-%d"),
                            end=end_date.strftime("%Y-%m-%d"),
                            use_cache=use_cache_opt
                        )
                        # Chuẩn hóa chữ cái đầu của cột để khớp với module phân tích cũ
                        st.session_state["processed_df"] = df_crawled
                        st.session_state["active_ticker"] = ticker_input
                        st.success("Crawl thành công! Chuyển sang phân tích.")
                    except Exception as e:
                        st.error(f"Lỗi crawl: {e}")
    else:
        uploaded = st.file_uploader(
            "Upload dữ liệu (CSV / Excel / Parquet)",
            type=["csv", "xlsx", "xls", "parquet"],
            help="File cần có cột: Date, Open, High, Low, Close, Volume"
        )
        if uploaded is not None:
            df_uploaded = load_df(uploaded)
            if df_uploaded is not None:
                st.session_state["processed_df"] = df_uploaded
                st.session_state["active_ticker"] = uploaded.name
                st.success("Đã load file thành công!")

    st.markdown("---")
    st.markdown(
        '<div style="font-size:11px;color:#484f58;line-height:1.6">'
        'Pipeline CRISP-DM<br>'
        '① <b style="color:#79c0ff">Data collection ✔</b><br>'
        '② <b style="color:#8b949e">Data analysis ← bạn đang ở đây</b><br>'
        '③ Feature engineering<br>'
        '④ Modeling<br>'
        '⑤ Backtesting<br>'
        '⑥ Deployment</div>',
        unsafe_allow_html=True
    )
    
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔬  Thu thập & Phân tích dữ liệu", "🧹 Tiền xử lý (Preprocessing)", "🤖 Modeling & Training", "🤖 Evaluation", "📊 Backtest"])

# ══════════════════════════════════════════
# TAB 1 — DATA COLLECTION & ANALYSIS
# ══════════════════════════════════════════
with tab1:

    if "processed_df" not in st.session_state:
        st.markdown("""
        <div style="text-align:center;padding:80px 20px">
            <div style="font-size:48px;margin-bottom:16px">📥</div>
            <div style="font-size:18px;font-weight:600;color:#e6edf3;margin-bottom:8px">
                Chưa có dữ liệu vận hành
            </div>
            <div style="font-size:14px;color:#8b949e">
                Vui lòng chọn <b>Crawl Tự động</b> hoặc <b>Upload file</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        df = st.session_state["processed_df"]
        active_name = st.session_state.get("active_ticker", "Data")

        q = analyse_quality(df)

        # =====================================================
        # SAFE GET HELPER (tránh KeyError)
        # =====================================================
        def safe(key, default=0):
            return q.get(key, default)

        # =====================================================
        # HEADER
        # =====================================================
        section(f"Tổng quan chất lượng dữ liệu: {active_name}")

        c1, c2, c3, c4 = st.columns(4)

        # ── ROW COUNT ──
        with c1:
            date_min = q.get("date_min")
            date_max = q.get("date_max")

            if hasattr(date_min, "strftime"):
                date_min = date_min.strftime("%Y-%m-%d")
            if hasattr(date_max, "strftime"):
                date_max = date_max.strftime("%Y-%m-%d")

            st.markdown(
                metric_card(
                    "Số dòng dữ liệu",
                    f"{q['n_rows']:,}",
                    f"{date_min or '?'} → {date_max or '?'}"
                ),
                unsafe_allow_html=True
            )

        # ── MISSING ──
        with c2:
            miss_cls = "metric-ok" if safe("missing_pct") == 0 else \
                       ("metric-warn" if safe("missing_pct") < 2 else "metric-bad")

            st.markdown(
                metric_card(
                    "Missing values",
                    f"{safe('missing_pct')}%",
                    f"{safe('missing_total')} ô trống",
                    miss_cls
                ),
                unsafe_allow_html=True
            )

        # ── DUPLICATES ──
        with c3:
            dup_cls = "metric-ok" if safe("dup_rows") == 0 else "metric-bad"

            st.markdown(
                metric_card(
                    "Duplicate rows",
                    str(safe("dup_rows")),
                    f"Dup dates: {safe('dup_dates')}",
                    dup_cls
                ),
                unsafe_allow_html=True
            )

        # ── OHLC ──
        with c4:
            if q.get("ohlc_ok", False):

                ohlc_errors = safe("ohlc_errors")  # FIX KEY ERROR

                err_cls = "metric-ok" if ohlc_errors == 0 else "metric-bad"

                st.markdown(
                    metric_card(
                        "OHLC errors",
                        str(ohlc_errors),
                        f"High<Low: {safe('high_lt_low')} | Neg: {safe('neg_price')}",
                        err_cls
                    ),
                    unsafe_allow_html=True
                )
            else:
                st.markdown(metric_card("OHLC", "N/A", "Không đủ dữ liệu"), unsafe_allow_html=True)

        # =====================================================
        # CANDLESTICK
        # =====================================================
        section("Biểu đồ giá & Volume")

        fig_c = chart_candlestick(df)
        if fig_c:
            st.plotly_chart(fig_c, use_container_width=True)

        # =====================================================
        # MISSING
        # =====================================================
        if safe("missing_total") > 0:
            section("Chi tiết Missing Values")

            col_a, col_b = st.columns([1.2, 1])

            with col_a:
                fig_m = chart_missing(df)
                if fig_m:
                    st.plotly_chart(fig_m, use_container_width=True)

            with col_b:
                miss_by_col = q.get("missing_by_col", {})

                miss_df = pd.DataFrame({
                    "Cột": list(miss_by_col.keys()),
                    "Missing": list(miss_by_col.values()),
                    "Tỷ lệ %": [
                        round(v / max(q["n_rows"], 1) * 100, 2)
                        for v in miss_by_col.values()
                    ]
                })

                st.dataframe(miss_df, use_container_width=True, hide_index=True)

        else:
            st.markdown(
                '<div class="alert-ok">✅ Dữ liệu không có missing values</div>',
                unsafe_allow_html=True
            )

        # =====================================================
        # OHLC ERRORS DETAIL
        # =====================================================
        if q.get("ohlc_ok") and safe("ohlc_errors") > 0:

            section("OHLC Consistency Errors")

            cols = st.columns(4)

            labels = [
                ("High < Low", safe("high_lt_low")),
                ("Close > High", safe("close_gt_high")),
                ("Close < Low", safe("close_lt_low")),
                ("Neg price", safe("neg_price")),
            ]

            for col, (label, val) in zip(cols, labels):
                with col:
                    cls = "metric-ok" if val == 0 else "metric-bad"
                    st.markdown(metric_card(label, str(val), "", cls), unsafe_allow_html=True)

        # =====================================================
        # RETURNS
        # =====================================================
        if q.get("ohlc_ok") and len(df) > 2:

            section("Phân phối Daily Return")

            col_l, col_r = st.columns([1.5, 1])

            with col_l:
                fig_r = chart_returns_dist(df)
                if fig_r:
                    st.plotly_chart(fig_r, use_container_width=True)

            with col_r:
                st.markdown(metric_card("Kurtosis", f"{safe('ret_kurt',0):.2f}"), unsafe_allow_html=True)
                st.markdown(metric_card("Skewness", f"{safe('ret_skew',0):.3f}"), unsafe_allow_html=True)

                st.markdown(metric_card(
                    "Outliers > 3σ",
                    str(safe("outlier_3s")),
                    f"Max gain: {safe('max_daily_gain',0):+.1%}"
                ), unsafe_allow_html=True)

        # =====================================================
        # VOLATILITY
        # =====================================================
        section("Rolling Volatility")

        fig_v = chart_rolling_vol(df)
        if fig_v:
            st.plotly_chart(fig_v, use_container_width=True)

        # =====================================================
        # RAW DATA
        # =====================================================
        section("Dữ liệu thô")

        st.dataframe(df.head(50), use_container_width=True, height=280)

        # =====================================================
        # EXPORT
        # =====================================================
        section("Báo cáo tóm tắt")

        summary_text = "\n".join([
            f"Ticker: {active_name}",
            f"Rows: {q['n_rows']}",
            f"Missing: {safe('missing_pct')}%",
            f"Dup rows: {safe('dup_rows')}",
            f"OHLC errors: {safe('ohlc_errors')}",
        ])

        st.download_button(
            "⬇ Download report",
            data=summary_text,
            file_name=f"report_{active_name}.txt",
            mime="text/plain",
        )

with tab2:
    if "processed_df" not in st.session_state:
        st.markdown("""
        <div style="text-align:center;padding:80px 20px">
            <div style="font-size:48px;margin-bottom:16px">⚠️</div>
            <div style="font-size:18px;font-weight:600;color:#e6edf3;margin-bottom:8px">Chưa có dữ liệu</div>
            <div style="font-size:14px;color:#8b949e">Vui lòng thu thập hoặc upload dữ liệu ở tab <b>Thu thập & Phân tích</b> trước.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Khởi tạo raw_df
        if "raw_df" not in st.session_state:
            st.session_state["raw_df"] = st.session_state["processed_df"].copy()
        df_raw = st.session_state["raw_df"].copy()

        st.subheader("🔧 Tiền xử lý & Feature Engineering")
        st.markdown("""
        **Quy trình (tránh leakage, tránh mất dữ liệu):**  
        0️⃣ Sửa lỗi OHLC & Sắp xếp  1️⃣ Xóa trùng  2️⃣ Forward Fill  
        3️⃣ **Feature Engineering** → ghi vào `fe_df`  
        4️⃣ Chia tập (đọc từ `fe_df`)  5️⃣ Chuẩn hóa
        """)

        # Nút reset
        col_reset = st.columns([3, 1])
        with col_reset[1]:
            if st.button("🔄 Reset toàn bộ"):
                for key in ["raw_df","fe_df","feature_list","feature_cols","fe_done",
                            "train_raw","val_raw","test_raw","train","val","test","scaler"]:
                    st.session_state.pop(key, None)
                st.session_state["raw_df"] = st.session_state["processed_df"].copy()
                st.rerun()

        # Status bar
        s = {
            "raw": "raw_df" in st.session_state,
            "fe":  st.session_state.get("fe_done", False),
            "split": "train_raw" in st.session_state,
            "scale": "train" in st.session_state,
        }
        c0,c1,c2,c3 = st.columns(4)
        for col,(lbl,done) in zip([c0,c1,c2,c3],[
            ("0–2 Raw",s["raw"]),("3 FE",s["fe"]),
            ("4 Split",s["split"]),("5 Scale",s["scale"])]):
            color = "#3fb950" if done else "#8b949e"
            col.markdown(
                f'<div style="text-align:center;padding:6px;background:#161b22;'
                f'border-radius:8px;border:1px solid #21262d;font-size:13px;color:{color}">'
                f'{"✅" if done else "○"} {lbl}</div>', unsafe_allow_html=True)
        st.markdown("")

        # ── BƯỚC 0: SỬA OHLC ────────────────────────────────────────
        with st.expander("0️⃣ Sửa lỗi OHLC & Sắp xếp theo ngày", expanded=False):
            if s["fe"]:
                st.warning("⚠️ FE đã chạy. Sửa bước này sẽ cần chạy lại FE.")
            if st.button("🔧 Sửa lỗi và sắp xếp", key="fix_ohlc_and_sort"):
                from data_preprocessing import fix_ohlc_errors, sort_by_date
                df_fixed = fix_ohlc_errors(df_raw)
                df_sorted = sort_by_date(df_fixed, sort_col="date")
                st.session_state["raw_df"] = df_sorted
                # Xoá các bước sau
                for k in ["fe_df","fe_done","feature_list","feature_cols",
                          "train_raw","val_raw","test_raw","train","val","test","scaler"]:
                    st.session_state.pop(k, None)
                st.success(f"✅ Đã sửa OHLC — {len(df_sorted):,} dòng.")
                st.rerun()

        # ── BƯỚC 1: XÓA TRÙNG ───────────────────────────────────────
        with st.expander("1️⃣ Xóa dữ liệu trùng lặp", expanded=False):
            if s["fe"]:
                st.warning("⚠️ FE đã chạy. Sửa bước này sẽ cần chạy lại FE.")
            n_dup = df_raw.duplicated().sum()
            st.info(f"Phát hiện **{n_dup}** dòng trùng.")
            if st.button("🗑️ Xóa trùng", key="dup_all"):
                from data_preprocessing import remove_duplicate_rows
                df_dedup = remove_duplicate_rows(df_raw)
                st.session_state["raw_df"] = df_dedup
                for k in ["fe_df","fe_done","feature_list","feature_cols",
                          "train_raw","val_raw","test_raw","train","val","test","scaler"]:
                    st.session_state.pop(k, None)
                st.success(f"✅ Đã xóa {n_dup} dòng trùng.")
                st.rerun()

        # ── BƯỚC 2: FORWARD FILL ────────────────────────────────────
        with st.expander("2️⃣ Xử lý missing (Forward Fill)", expanded=False):
            if s["fe"]:
                st.warning("⚠️ FE đã chạy. Sửa bước này sẽ cần chạy lại FE.")
            n_miss = df_raw.isnull().sum().sum()
            st.info(f"Tổng **{n_miss}** ô missing trong raw data.")
            if st.button("▶️ Forward Fill toàn bộ", key="ffill_all"):
                from data_preprocessing import fill_forward
                df_filled = fill_forward(df_raw)
                st.session_state["raw_df"] = df_filled
                for k in ["fe_df","fe_done","feature_list","feature_cols",
                          "train_raw","val_raw","test_raw","train","val","test","scaler"]:
                    st.session_state.pop(k, None)
                st.success("✅ Đã forward fill.")
                st.rerun()

        # ── BƯỚC 3: FEATURE ENGINEERING ─────────────────────────────
        with st.expander("3️⃣ Feature Engineering", expanded=False):
            st.info("⚠️ FE luôn đọc từ raw_df (6 cột). Kết quả lưu vào fe_df.", icon="ℹ️")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                horizon = st.number_input("Horizon (ngày)", 1, 30, 5, 1, key="fe_horizon")
                target_type = st.selectbox("Loại target", ["log_return","direction","tertile"], key="fe_target_type")
            with col_b:
                return_periods = st.multiselect("Log return periods", [1,3,5,10,20], default=[1,5,20], key="fe_return_periods")
                vol_windows = st.multiselect("Volatility windows", [5,10,21,63], default=[5,21], key="fe_vol_windows")
            with col_c:
                zscore_window = st.number_input("Z-score window", 21, 252, 63, 21, key="fe_zscore_window")
                corr_threshold = st.slider("Ngưỡng tương quan (báo cáo)", 0.70, 0.99, 0.85, 0.01, key="fe_corr_threshold")

            if target_type == "tertile":
                ct1, ct2 = st.columns(2)
                with ct1: thr_low = st.number_input("Down <", value=-1.0, step=0.5, format="%.1f", key="fe_thr_low") / 100
                with ct2: thr_high = st.number_input("Up >", value=1.0, step=0.5, format="%.1f", key="fe_thr_high") / 100
                target_thresholds = (thr_low, thr_high)
            else:
                target_thresholds = (-0.01, 0.01)

            st.markdown("---")
            if st.button("🚀 Chạy Feature Engineering", key="run_fe", type="primary"):
                try:
                    from feature_engineering import build_features, get_feature_cols, get_correlation_report
                    import plotly.express as px

                    if not return_periods:
                        st.error("Chọn ít nhất 1 return period.")
                        st.stop()

                    df_for_fe = st.session_state["raw_df"].copy()
                    df_for_fe.columns = df_for_fe.columns.str.lower()

                    with st.spinner("Đang tạo features..."):
                        df_fe = build_features(
                            df_for_fe,
                            return_periods=return_periods,
                            vol_windows=vol_windows,
                            atr_window=14,
                            zscore_window=zscore_window,
                            zscore_inplace=True,
                            horizon=horizon,
                            target_type=target_type,
                            target_thresholds=target_thresholds,
                            drop_na=False,
                        )

                    # Lấy danh sách features chuẩn
                    feature_cols = get_feature_cols(df_fe)
                    # Đảm bảo target có trong df_fe
                    if "target" not in df_fe.columns:
                        st.error("Lỗi: target không được tạo.")
                        st.stop()

                    # Lưu vào session_state
                    st.session_state["fe_df"] = df_fe
                    st.session_state["fe_done"] = True
                    st.session_state["feature_list"] = feature_cols   # <--- QUAN TRỌNG
                    st.session_state["feature_cols"] = feature_cols   # fallback
                    st.session_state["target_column"] = "target"
                    st.session_state["fe_horizon_saved"] = horizon
                    st.session_state["fe_target_type_saved"] = target_type
                    # Xoá các bước sau
                    for k in ["train_raw","val_raw","test_raw","train","val","test","scaler"]:
                        st.session_state.pop(k, None)

                    # Thống kê cơ bản
                    nan_rows = df_fe[feature_cols].isna().any(axis=1).sum() if feature_cols else 0
                    st.success(f"✅ FE hoàn tất — **{len(feature_cols)} features** | raw: {len(df_fe):,} dòng")

                    # Hiển thị danh sách features
                    with st.expander("📋 Danh sách features", expanded=False):
                        st.write(feature_cols)

                    # Báo cáo tương quan
                    with st.expander("📈 Ma trận tương quan giữa các features", expanded=True):
                        df_clean_corr = df_fe[feature_cols + ["target"]].dropna()
                        if len(df_clean_corr) < 10:
                            st.warning("Không đủ dữ liệu để tính tương quan (cần ít nhất 10 dòng sạch).")
                        else:
                            corr_mat, target_corr, high_pairs = get_correlation_report(
                                df_clean_corr, feature_cols, "target", threshold=corr_threshold
                            )
                            fig = px.imshow(
                                corr_mat,
                                text_auto=".2f",
                                aspect="auto",
                                color_continuous_scale="RdBu_r",
                                zmin=-1, zmax=1,
                                title=f"Ma trận tương quan (threshold = {corr_threshold})",
                                labels=dict(color="Corr")
                            )
                            fig.update_layout(height=600, font=dict(size=10))
                            st.plotly_chart(fig, use_container_width=True)

                            if high_pairs:
                                st.warning(f"⚠️ {len(high_pairs)} cặp feature có |corr| > {corr_threshold}")
                                for f1, f2, v in high_pairs[:10]:
                                    st.write(f"- **{f1}** ↔ **{f2}** : {v:.3f}")
                            else:
                                st.success(f"✅ Không có cặp feature nào vượt ngưỡng {corr_threshold}.")

                            st.markdown("**🎯 Tương quan với target:**")
                            st.dataframe(target_corr.to_frame("Corr").style.background_gradient(cmap="RdBu", vmin=-1, vmax=1))

                    # Hiển thị 10 dòng cuối
                    with st.expander("🔍 Xem dữ liệu sau FE (10 dòng cuối)", expanded=False):
                        show_cols = [c for c in ["date"] + feature_cols + ["target"] if c in df_fe.columns]
                        st.dataframe(df_fe[show_cols].tail(10), use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Lỗi FE: {e}")
                    import traceback
                    with st.expander("Chi tiết lỗi"):
                        st.code(traceback.format_exc())

        # ── BƯỚC 4: SPLIT ───────────────────────────────────────────
        with st.expander("4️⃣ Chia tập Train / Validation / Test", expanded=True):
            if not st.session_state.get("fe_done", False):
                st.error("❌ Chạy Feature Engineering (bước 3) trước!")
            else:
                df_fe_cur = st.session_state["fe_df"]
                feature_cols_ = st.session_state["feature_list"]  # lấy từ feature_list

                # Drop NaN trước split
                clean_mask = df_fe_cur[feature_cols_].notna().all(axis=1)
                if "target" in df_fe_cur.columns:
                    clean_mask = clean_mask & df_fe_cur["target"].notna()
                df_clean = df_fe_cur[clean_mask].reset_index(drop=True)

                n_dropped = len(df_fe_cur) - len(df_clean)
                st.info(f"📋 Input: {len(df_fe_cur):,} dòng → sau dropna: {len(df_clean):,} dòng (bỏ {n_dropped} dòng warm-up) × {len(feature_cols_)} features")

                col1,col2,col3 = st.columns(3)
                with col1: train_ratio = st.number_input("Train %", 0.0, 1.0, 0.70, 0.05, key="split_train")
                with col2: val_ratio   = st.number_input("Val %",   0.0, 1.0, 0.15, 0.05, key="split_val")
                with col3: test_ratio  = st.number_input("Test %",  0.0, 1.0, 0.15, 0.05, key="split_test")

                date_cols = [c for c in df_clean.columns if "date" in c.lower()]
                sort_col = st.selectbox("Cột thời gian", date_cols, key="sort_col_split") if date_cols else None

                if st.button("✂️ Chia tập", type="primary", key="btn_split"):
                    from data_preprocessing import train_val_test_split
                    train_raw, val_raw, test_raw = train_val_test_split(
                        df_clean, train_ratio, val_ratio, test_ratio, sort_by=sort_col
                    )
                    # Kiểm tra features còn đủ
                    missing_feat = [c for c in feature_cols_ if c not in train_raw.columns]
                    if missing_feat:
                        st.error(f"❌ Thiếu {len(missing_feat)} features trong train_raw: {missing_feat[:5]}")
                        st.stop()
                    st.session_state["train_raw"] = train_raw
                    st.session_state["val_raw"]   = val_raw
                    st.session_state["test_raw"]  = test_raw
                    # Đảm bảo feature_list vẫn tồn tại
                    if "feature_list" not in st.session_state:
                        st.session_state["feature_list"] = feature_cols_
                    # Xoá scale để chạy lại bước 5
                    for k in ["train","val","test","scaler"]:
                        st.session_state.pop(k, None)
                    st.success(f"✅ Split xong — Train: {len(train_raw):,} | Val: {len(val_raw):,} | Test: {len(test_raw):,}")

        # ── BƯỚC 5: CHUẨN HÓA (fit trên train, transform val/test) ──
        if "train_raw" in st.session_state:
            with st.expander("5️⃣ Chuẩn hóa (fit trên train, transform val/test)", expanded=False):
                train_raw_ = st.session_state["train_raw"]
                val_raw_   = st.session_state["val_raw"]
                test_raw_  = st.session_state["test_raw"]
                feature_list_ = st.session_state.get("feature_list", [])
                if not feature_list_:
                    st.error("❌ Không có feature_list. Hãy chạy lại bước 3 FE.")
                    st.stop()

                # Các cột đã được rolling z-score trong FE (không cần scale lại)
                # Lấy từ file feature_engineering, nhưng tạm thời dùng set mặc định
                already_zscored = {"macd_signal", "momentum_ratio"}  # tuỳ theo feature set hiện tại
                exclude_scale = {"date", "datetime", "target"} | already_zscored

                num_cols = [
                    c for c in feature_list_
                    if c in train_raw_.columns
                    and c not in exclude_scale
                    and np.issubdtype(train_raw_[c].dtype, np.number)
                ]

                if not num_cols:
                    st.info("ℹ️ Không có cột nào cần scale (tất cả đã được rolling z-score). Vẫn tiến hành lưu dữ liệu.")
                    # Lưu mà không scale
                    st.session_state["train"] = train_raw_.copy()
                    st.session_state["val"]   = val_raw_.copy()
                    st.session_state["test"]  = test_raw_.copy()
                    st.session_state["scaler"] = None
                    st.success("✅ Dữ liệu đã sẵn sàng (không cần scale thêm).")
                else:
                    norm_method = st.selectbox("Phương pháp", ["Standard","MinMax","Robust"], key="norm_method")
                    if st.button("✅ Chuẩn hóa", key="norm_apply", type="primary"):
                        from data_preprocessing import Normalizer
                        scaler = Normalizer(method=norm_method.lower())
                        train_norm = scaler.fit_transform(train_raw_, columns=num_cols)
                        val_norm   = scaler.transform(val_raw_)
                        test_norm  = scaler.transform(test_raw_)

                        # Giữ lại target và các cột không scale
                        for col in train_raw_.columns:
                            if col not in num_cols:
                                train_norm[col] = train_raw_[col]
                                val_norm[col]   = val_raw_[col]
                                test_norm[col]  = test_raw_[col]

                        st.session_state["train"]  = train_norm
                        st.session_state["val"]    = val_norm
                        st.session_state["test"]   = test_norm
                        st.session_state["scaler"] = scaler
                        st.success(f"✅ Đã scale {len(num_cols)} cột. Dữ liệu sẵn sàng.")
                        st.dataframe(train_norm[num_cols + (["target"] if "target" in train_norm else [])].head(), use_container_width=True)

        # ── PREVIEW DỮ LIỆU CUỐI ──
        st.markdown("---")
        st.subheader("📊 Dữ liệu sau tiền xử lý")
        if "train" in st.session_state:
            train_df = st.session_state["train"]
            val_df   = st.session_state["val"]
            test_df  = st.session_state["test"]
            st.success(f"✅ Train {train_df.shape} | Val {val_df.shape} | Test {test_df.shape}")
            tab_tr, tab_vl, tab_te = st.tabs(["🚂 Train","🛸 Validation","🚀 Test"])
            for tab, df_show, name in [(tab_tr,train_df,"train"),(tab_vl,val_df,"val"),(tab_te,test_df,"test")]:
                with tab:
                    st.dataframe(df_show.head(20), use_container_width=True)
                    st.caption(f"{len(df_show):,} dòng × {df_show.shape[1]} cột")
                    st.download_button(f"⬇️ Tải {name}", df_show.to_csv(index=False).encode("utf-8"), f"{name}_normalized.csv", "text/csv", key=f"dl_{name}")
        elif "train_raw" in st.session_state:
            st.info("✓ Đã split – chưa chuẩn hóa. Thực hiện bước 5.")
            tr = st.session_state["train_raw"]
            st.dataframe(tr.head(10), use_container_width=True)
        elif st.session_state.get("fe_done"):
            st.info("✓ FE xong – thực hiện bước 4 (split).")
        else:
            st.info("ℹ️ Thực hiện các bước từ 0 đến 5.")
            
with tab3:
    # ---------- KIỂM TRA DỮ LIỆU ----------
    if "train_raw" not in st.session_state:
        st.warning("⚠️ Chưa có dữ liệu train_raw. Hãy chạy Feature Engineering và Split ở tab trước.")
        st.stop()
    
    # Lấy feature_list
    feature_list = st.session_state.get("feature_list", None)
    if feature_list is None:
        st.error("❌ Không tìm thấy `feature_list`. Hãy chạy lại FE (bước 3).")
        st.stop()
    
    df = st.session_state["train_raw"].copy()
    # Lọc các feature có trong df
    feature_cols = [c for c in feature_list if c in df.columns and c != 'target']
    if len(feature_cols) == 0:
        st.error(f"❌ Không tìm thấy feature columns. feature_list: {feature_list[:5]}...")
        st.stop()
    
    # Xác định task
    if "target" in df.columns:
        if df["target"].dtype in ["float64", "float32"] and df["target"].nunique() > 10:
            task = "regression"
            n_classes = 1
        else:
            task = "classification"
            n_classes = df["target"].nunique()
    else:
        task = "regression"
        n_classes = 1
    
    st.subheader("🤖 Model Training Pipeline")
    st.markdown(f"**Data:** {len(df):,} dòng | **Features:** {len(feature_cols)} | **Task:** {task} | **Classes:** {n_classes}")
    
    # ---------- CẤU HÌNH WALK-FORWARD ----------
    with st.expander("⚙️ Walk‑forward CV Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_folds = st.number_input("Number of folds", 3, 10, 5, 1)
            min_train = st.number_input("Min train size (rows)", 100, 1000, 252, 50)
        with col2:
            gap = st.number_input("Gap (rows)", 1, 20, 5, 1, help="= horizon to avoid target leakage")
            val_size = st.number_input("Validation size (rows)", 20, 500, 126, 10, help="Nên ≥ seq_len + gap + 30")
        with col3:
            model_choice = st.selectbox("Model", ["LightGBM", "XGBoost", "LSTM", "Transformer", "Both (compare)"])
            primary_metric = "ic" if task == "regression" else "f1_macro"
            st.caption(f"Optimization metric: `{primary_metric}`")
        
        # Khởi tạo giá trị mặc định cho LSTM params (dùng chung cho cả LSTM và "Both")
        lstm_hidden = 32      # giảm từ 64 xuống 32
        lstm_layers = 1       # giảm từ 2 xuống 1
        lstm_seq_len = 30
        lstm_epochs = 50
        lstm_batch = 32
        lstm_lr = 0.001
        
        trans_d_model = 64
        trans_nhead = 4
        trans_num_layers = 2
        trans_dim_feedforward = 128
        trans_dropout = 0.1
        trans_seq_len = 30   
        trans_epochs = 50
        trans_batch = 32
        trans_lr = 0.001
        
        # Nếu chọn LSTM hoặc Both, hiển thị tuỳ chỉnh (và ghi đè giá trị mặc định)
        if model_choice in ("LSTM", "Both (compare)"):
            with st.expander("LSTM Hyperparameters (SAFE version)"):
                lstm_seq_len = st.number_input("Sequence length (seq_len)", 10, min(120, val_size-10), 30, 10,
                    help="Số ngày quá khứ. Phải nhỏ hơn val_size - gap")
                lstm_hidden = st.number_input("Hidden units", 16, 128, 32, 8)
                lstm_layers = st.number_input("Number of layers", 1, 3, 1, 1)
                lstm_epochs = st.number_input("Epochs", 20, 200, 50, 10)
                lstm_batch = st.number_input("Batch size", 16, 128, 32, 16)
                lstm_lr = st.number_input("Learning rate", 0.0001, 0.01, 0.001, format="%.4f")
                
        if model_choice in ("Transformer", "Both (compare)"):
            with st.expander("Transformer Hyperparameters (SAFE version)"):
                trans_seq_len = st.number_input("Transformer Sequence length", 10, min(120, val_size-10), 30, 10,
                    help="Số ngày quá khứ. Phải nhỏ hơn val_size - gap")
                trans_d_model = st.number_input("d_model (embedding dimension)", 32, 256, 64, 16)
                trans_nhead = st.number_input("Number of heads", 2, 8, 4, 1)
                trans_num_layers = st.number_input("Number of transformer layers", 1, 4, 2, 1)
                trans_dim_feedforward = st.number_input("Feedforward dimension", 64, 512, 128, 32)
                trans_dropout = st.number_input("Dropout", 0.0, 0.5, 0.1, 0.05, format="%.2f")
                trans_epochs = st.number_input("Transformer Epochs", 20, 200, 50, 10)
                trans_batch = st.number_input("Transformer Batch size", 16, 128, 32, 16)
                trans_lr = st.number_input("Transformer Learning rate", 0.0001, 0.01, 0.001, format="%.4f")
        
        from model import ModelConfig
        cfg = ModelConfig(
            task=task, n_classes=n_classes,
            n_folds=n_folds, min_train_size=min_train,
            gap=gap, val_size=val_size,
            feature_cols=feature_cols,
            target_col="target", date_col="date",
            primary_metric=primary_metric,
        )
    
    # ---------- BASELINE ----------
    st.markdown("### 1️⃣ Baseline (default params)")
    if st.button("🚀 Run Baseline", key="baseline_btn"):
        from model import walk_forward_evaluate, LGBM_DEFAULT, XGB_DEFAULT, lstm_walk_forward_evaluate, transformer_walk_forward_evaluate
        results = {}
        with st.spinner("Evaluating baseline..."):
            if model_choice in ("LightGBM", "Both (compare)"):
                res_lgb = walk_forward_evaluate(df, cfg, model_type="lgbm", params=LGBM_DEFAULT, verbose=False)
                results["LightGBM"] = res_lgb
            if model_choice in ("XGBoost", "Both (compare)"):
                res_xgb = walk_forward_evaluate(df, cfg, model_type="xgb", params=XGB_DEFAULT, verbose=False)
                results["XGBoost"] = res_xgb
            if model_choice in ("LSTM", "Both (compare)"):
                res_lstm = lstm_walk_forward_evaluate(
                    df, cfg, seq_len=lstm_seq_len, hidden_dim=lstm_hidden, num_layers=lstm_layers,
                    epochs=lstm_epochs, batch_size=lstm_batch, lr=lstm_lr, patience=5, verbose=False
                )
                results["LSTM"] = res_lstm
            if model_choice in ("Transformer", "Both (compare)"):
                res_trans = transformer_walk_forward_evaluate(
                    df, cfg, seq_len=trans_seq_len, d_model=trans_d_model, nhead=trans_nhead,
                    num_layers=trans_num_layers, dim_feedforward=trans_dim_feedforward,
                    dropout=trans_dropout, epochs=trans_epochs, batch_size=trans_batch,
                    lr=trans_lr, patience=5, verbose=False
                )
                results["Transformer"] = res_trans
        st.session_state["baseline_results"] = results
        st.session_state["model_cfg"] = cfg
        st.success("Baseline evaluation completed!")
    
    if "baseline_results" in st.session_state:
        for name, res in st.session_state["baseline_results"].items():
            ov = res["overall"]
            st.markdown(f"**{name}**")
            if task == "regression":
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("IC", f"{ov.get('ic',0):+.4f}")
                col2.metric("IC IR", f"{ov.get('ic_ir',0):+.3f}")
                col3.metric("Directional Acc", f"{ov.get('dir_acc',0):.1%}")
                col4.metric("RMSE", f"{ov.get('rmse',0):.4f}")
            else:
                # Classification: hiển thị tuỳ theo số lớp
                if n_classes == 3:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("F1-macro", f"{ov.get('f1_macro',0):.4f}")
                    col2.metric("Accuracy", f"{ov.get('accuracy',0):.1%}")
                    col3.metric("F1-weighted", f"{ov.get('f1_weighted',0):.4f}")
                    st.caption(f"📊 F1 Up: {ov.get('f1_up',0):.4f} | Side: {ov.get('f1_sideway',0):.4f} | Down: {ov.get('f1_down',0):.4f}")
                else:  # binary
                    col1, col2 = st.columns(2)
                    col1.metric("F1-macro", f"{ov.get('f1_macro',0):.4f}")
                    col2.metric("Accuracy", f"{ov.get('accuracy',0):.1%}")
            
            # Plot per fold
            fold_df = pd.DataFrame(res["fold_results"])
            if not fold_df.empty:
                if task == "regression":
                    st.line_chart(fold_df.set_index("fold")["ic"], height=250)
                else:
                    metric_col = "f1_macro" if "f1_macro" in fold_df.columns else "accuracy"
                    st.line_chart(fold_df.set_index("fold")[metric_col], height=250)
    
    # ---------- OPTUNA TUNING (chỉ cho LGBM/XGB) ----------
    if model_choice != "LSTM":
        st.markdown("### 2️⃣ Hyperparameter Tuning (Optuna)")
        if st.button("🔍 Tune with Optuna", key="tune_btn"):
            st.warning("Tuning sẽ chạy khoảng 1-2 phút. Đảm bảo đã cài `optuna`.")
            try:
                from model import tune_hyperparams
                with st.spinner("Optuna đang tìm params tối ưu..."):
                    tune_res = tune_hyperparams(df, cfg, model_type="lgbm", n_trials=30, timeout=90)
                st.session_state["tune_result"] = tune_res
                st.success(f"Best {primary_metric}: {tune_res['best_score']:.4f}")
                st.write("Best params:", tune_res["best_params"])
            except Exception as e:
                st.error(f"Lỗi tuning: {e}")
    
    # ---------- TRAIN FINAL MODEL ----------
    st.markdown("### 3️⃣ Train Final Model")
    final_model_type = st.selectbox("Select final model", ["LightGBM", "XGBoost", "LSTM"])
    use_tuned = st.checkbox("Dùng best params từ Optuna (nếu có)", value=False)
    if st.button("🏋️ Train Final Model", key="final_btn"):
        from model import train_final_model, train_final_lstm, train_final_transformer
        if final_model_type == "LSTM":
            with st.spinner("Training LSTM (SAFE version)..."):
                model, scaler = train_final_lstm(
                    df, cfg,
                    seq_len=lstm_seq_len,
                    gap=cfg.gap,                     
                    hidden_dim=lstm_hidden,
                    num_layers=lstm_layers,
                    epochs=lstm_epochs,
                    batch_size=lstm_batch,
                    lr=lstm_lr,
                    patience=10
                )
            st.session_state["final_model"] = model
            st.session_state["lstm_scaler"] = scaler
            st.success("✅ LSTM model trained")
            
        if final_model_type == "Transformer":
            with st.spinner("Training Transformer (SAFE version)..."):
                model, scaler = train_final_transformer(
                    df, cfg,
                    seq_len=trans_seq_len,
                    gap=cfg.gap,
                    d_model=trans_d_model,
                    nhead=trans_nhead,
                    num_layers=trans_num_layers,
                    dim_feedforward=trans_dim_feedforward,
                    dropout=trans_dropout,
                    epochs=trans_epochs,
                    batch_size=trans_batch,
                    lr=trans_lr,
                    patience=10
                )
            st.session_state["final_model"] = model
            st.session_state["transformer_scaler"] = scaler
            st.success("✅ Transformer model trained")
            
        else:
            mtype = "lgbm" if final_model_type == "LightGBM" else "xgb"
            params = None
            if use_tuned and "tune_result" in st.session_state:
                params = st.session_state["tune_result"]["best_params"]
            with st.spinner(f"Training {final_model_type}..."):
                model, fi = train_final_model(df, cfg, model_type=mtype, params=params)
            st.session_state["final_model"] = model
            st.session_state["final_model_fi"] = fi
            st.success(f"✅ Final {final_model_type} model trained")
            st.dataframe(fi.head(10), use_container_width=True)
            
with tab4:
    from evaluation import evaluation_tab_content
    evaluation_tab_content()
    
with tab5:
    st.subheader("💰 Backtest giao dịch với số vốn thực")
    if "baseline_results" not in st.session_state:
        st.info("Chưa có kết quả baseline. Hãy chạy Baseline ở tab Model Training trước.")
    else:
        results = st.session_state["baseline_results"]
        model_names = list(results.keys())
        selected_model = st.selectbox("Chọn mô hình để backtest", model_names, key="backtest_model")
        if selected_model:
            res = results[selected_model]
            y_true = np.array(res["all_trues"])
            y_pred = np.array(res["all_preds"])
            cfg = st.session_state.get("model_cfg", None)
            horizon = cfg.gap if cfg else 5
            task = cfg.task if cfg else "regression"
            
            col1, col2 = st.columns(2)
            with col1:
                initial_capital = st.number_input("💰 Số vốn ban đầu (USD)", min_value=100.0, value=10000.0, step=1000.0)
            with col2:
                transaction_cost = st.slider("💸 Phí giao dịch (%)", 0.0, 1.0, 0.1, 0.01) / 100
            
            if st.button("▶️ Chạy Backtest", type="primary"):
                from backtest import simulate_backtest, display_backtest_in_streamlit
                bt = simulate_backtest(
                    y_true, y_pred,
                    initial_capital=initial_capital,
                    transaction_cost=transaction_cost,
                    horizon=horizon,
                    task=task
                )
                st.session_state["backtest_result"] = bt
                display_backtest_in_streamlit(bt, initial_capital)
        
        if "backtest_result" in st.session_state and st.button("🔄 Xóa kết quả"):
            del st.session_state["backtest_result"]
            st.rerun()