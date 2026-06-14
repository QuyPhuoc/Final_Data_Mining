"""
evaluation.py
=============
Module đánh giá mô hình cho bài toán dự đoán giá cổ phiếu / coin.

Hỗ trợ:
  - Regression & Classification
  - Walk‑forward evaluation results (fold results)
  - Backtest (equity curve, drawdown)
  - Cảnh báo look‑ahead bias
  - Hiển thị trực quan trong Streamlit
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any

import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
# METRICS NÂNG CAO
# ═══════════════════════════════════════════════════════════════

def rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Rank IC (Spearman correlation) – same as IC but explicitly computed."""
    if len(y_true) < 2:
        return 0.0
    return stats.spearmanr(y_pred, y_true)[0]

def hit_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tỷ lệ dự đoán đúng chiều (dấu) – chỉ dùng cho regression."""
    return np.mean(np.sign(y_pred) == np.sign(y_true))

def sharpe_of_predictions(y_pred: np.ndarray, annualize: bool = True, periods_per_year: int = 252) -> float:
    """
    Sharpe ratio của bản thân dự đoán (coi như return forecast).
    Hữu ích để đánh giá độ ổn định của tín hiệu.
    """
    if len(y_pred) < 2:
        return 0.0
    mean_ret = np.mean(y_pred)
    std_ret = np.std(y_pred, ddof=1)
    if std_ret == 0:
        return 0.0
    sharpe = mean_ret / std_ret
    if annualize:
        sharpe = sharpe * np.sqrt(periods_per_year)
    return sharpe

def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "regression"
) -> Dict[str, float]:
    """Tính toàn bộ metrics cần thiết."""
    metrics = {}
    if task == "regression":
        # Cơ bản
        metrics["rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        metrics["mae"] = float(np.mean(np.abs(y_true - y_pred)))
        metrics["ic"] = rank_ic(y_true, y_pred)
        metrics["dir_acc"] = hit_ratio(y_true, y_pred)
        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics["r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        # Sharpe của dự đoán
        metrics["pred_sharpe"] = sharpe_of_predictions(y_pred, annualize=True)
    else:
        # Classification
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        # Confusion matrix based metrics
        cm = confusion_matrix(y_true, y_pred, labels=sorted(np.unique(y_true)))
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["precision"] = tp / (tp + fp) if (tp+fp) > 0 else 0.0
            metrics["recall"] = tp / (tp + fn) if (tp+fn) > 0 else 0.0
        else:
            # multiclass, skip for now
            pass
    return metrics

# ═══════════════════════════════════════════════════════════════
# BACKTEST SIMULATION (CHỈ DÙNG CHO REGRESSION VỚI LOG_RETURN)
# ═══════════════════════════════════════════════════════════════

def simulate_backtest(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001,
    horizon: int = 1,          # <- thêm tham số horizon (từ cfg)
) -> Dict[str, Any]:
    """
    Mô phỏng backtest với lợi nhuận hàng ngày.
    Nếu horizon > 1, chia lợi nhuận cho horizon để có lợi nhuận trung bình mỗi ngày.
    """
    if len(y_true) == 0:
        return {}
    signal = np.sign(y_pred)
    # Quy đổi lợi nhuận về daily return
    daily_return = y_true / horizon   # giả định phân bố đều
    gross_returns = signal * daily_return
    trades = np.abs(np.diff(signal, prepend=signal[0]))
    costs = trades * transaction_cost
    net_returns = gross_returns - costs
    equity = initial_capital * (1 + net_returns).cumprod()
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown = np.min(drawdown)
    mean_ret = np.mean(net_returns)
    std_ret = np.std(net_returns, ddof=1)
    sharpe_strategy = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0
    total_return = equity[-1] / initial_capital - 1
    return {
        "equity": equity,
        "drawdown": drawdown,
        "max_drawdown": float(max_drawdown),
        "sharpe_strategy": float(sharpe_strategy),
        "total_return": float(total_return),
        "num_trades": int(np.sum(trades)),
    }

# ═══════════════════════════════════════════════════════════════
# VIZUALIZATION (PLOTLY)
# ═══════════════════════════════════════════════════════════════

def plot_pred_vs_actual_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Dự đoán vs Thực tế"
) -> go.Figure:
    """Scatter plot với đường y=x."""
    fig = px.scatter(x=y_true, y=y_pred, labels={"x": "Actual", "y": "Predicted"},
                     title=title, opacity=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                             mode="lines", name="y=x", line=dict(dash="dash", color="red")))
    fig.update_layout(width=500, height=500)
    return fig

def plot_time_series_actual_pred(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Thực tế vs Dự đoán",
    show_ic: bool = True
) -> go.Figure:
    """Biểu đồ chuỗi thời gian thực tế và dự đoán."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_true, mode="lines", name="Actual", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=dates, y=y_pred, mode="lines", name="Predicted", line=dict(color="red", dash="dot")))
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Value", width=800, height=400)
    if show_ic:
        ic_val = rank_ic(y_true, y_pred)
        fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper", showarrow=False,
                           text=f"IC = {ic_val:.4f}", font=dict(size=12, color="darkgreen"))
    return fig

def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Phân phối sai số"
) -> go.Figure:
    """Histogram và Q-Q plot của sai số."""
    errors = y_true - y_pred
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogram", "Q-Q Plot"))
    fig.add_trace(go.Histogram(x=errors, nbinsx=50, name="Errors"), row=1, col=1)
    # Q-Q plot
    sorted_errors = np.sort(errors)
    theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_errors)))
    fig.add_trace(go.Scatter(x=theoretical, y=sorted_errors, mode="markers", name="Q-Q"), row=1, col=2)
    fig.add_trace(go.Scatter(x=theoretical, y=theoretical, mode="lines", name="Normal line", line=dict(dash="dash")), row=1, col=2)
    fig.update_layout(title=title, width=800, height=400)
    return fig

def plot_fold_metrics(fold_results: List[Dict], metric_name: str = "ic", title: str = "") -> go.Figure:
    """Biểu đồ bar metrics qua các fold."""
    folds = [r["fold"] for r in fold_results]
    values = [r[metric_name] for r in fold_results]
    fig = px.bar(x=folds, y=values, labels={"x": "Fold", "y": metric_name.upper()}, title=title)
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(width=600, height=400)
    return fig

def plot_calibration_curve(y_true, y_pred_proba, n_bins=10, title="Calibration Curve"):
    """Chỉ dùng cho classification binary (có xác suất)."""
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="Model"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Perfect", line=dict(dash="dash")))
    fig.update_layout(title=title, xaxis_title="Mean predicted probability", yaxis_title="Fraction of positives", width=500, height=500)
    return fig

def plot_equity_curve(backtest_dict: Dict, title: str = "Equity Curve") -> go.Figure:
    """Biểu đồ equity và drawdown từ backtest."""
    equity = backtest_dict.get("equity")
    drawdown = backtest_dict.get("drawdown")
    if equity is None:
        return go.Figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Equity Curve", "Drawdown"))
    fig.add_trace(go.Scatter(x=np.arange(len(equity)), y=equity, mode="lines", name="Equity"), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(drawdown)), y=drawdown, mode="lines", fill="tozeroy", name="Drawdown"), row=2, col=1)
    fig.update_layout(title=title, height=600, width=800)
    return fig

# ═══════════════════════════════════════════════════════════════
# TỔNG HỢP BÁO CÁO CHO STREAMLIT
# ═══════════════════════════════════════════════════════════════

def display_evaluation_report_streamlit(
    result_dict: Dict,
    task: str = "regression",
    show_backtest: bool = True,
) -> None:
    """
    Hiển thị báo cáo đánh giá trong Streamlit từ kết quả walk_forward_evaluate.
    result_dict: output của walk_forward_evaluate (chứa overall, fold_results, all_preds, all_trues, ...)
    """
    import streamlit as st
    
    overall = result_dict["overall"]
    fold_results = result_dict.get("fold_results", [])
    all_preds = result_dict.get("all_preds")
    all_trues = result_dict.get("all_trues")
    
    st.markdown("### 📊 Chi tiết đánh giá mô hình")
    
    # Cảnh báo look‑ahead nếu IC quá cao
    if task == "regression":
        ic = overall.get("ic", 0)
        if abs(ic) > 0.3:
            st.error(f"⚠️ IC = {ic:.4f} rất cao! Có thể có look‑ahead bias hoặc overfitting. Kiểm tra lại pipeline dữ liệu.")
        elif abs(ic) > 0.1:
            st.warning(f"IC = {ic:.4f} khá cao, hãy đảm bảo không có leakage.")
    else:
        f1 = overall.get("f1_macro", 0)
        if f1 > 0.95:
            st.error(f"⚠️ F1-macro = {f1:.4f} quá cao! Rất có thể look‑ahead bias.")
    
    # Metrics overall
    col_met1, col_met2 = st.columns(2)
    with col_met1:
        st.metric("IC (overall)", f"{overall.get('ic', 0):+.4f}" if task=="regression" else "N/A")
        if task == "regression":
            st.metric("Directional Accuracy", f"{overall.get('dir_acc', 0):.1%}")
            st.metric("RMSE", f"{overall.get('rmse', 0):.4f}")
        else:
            st.metric("Accuracy", f"{overall.get('accuracy', 0):.1%}")
            st.metric("F1-macro", f"{overall.get('f1_macro', 0):.4f}")
    with col_met2:
        if task == "regression":
            st.metric("IC IR", f"{overall.get('ic_ir', 0):+.3f}")
            st.metric("R²", f"{overall.get('r2', 0):.3f}")
            # Thêm Sharpe của dự đoán nếu có
            pred_sharpe = overall.get("pred_sharpe", None)
            if pred_sharpe is not None:
                st.metric("Pred Sharpe", f"{pred_sharpe:.2f}")
        else:
            st.metric("F1 Up", f"{overall.get('f1_up', 0):.4f}")
            st.metric("F1 Down", f"{overall.get('f1_down', 0):.4f}")
    
    # Biểu đồ metrics qua các fold
    if fold_results:
        st.markdown("**Performance qua các fold**")
        if task == "regression":
            fig = plot_fold_metrics(fold_results, "ic", "IC theo fold")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = plot_fold_metrics(fold_results, "f1_macro", "F1-macro theo fold")
            st.plotly_chart(fig, use_container_width=True)
    
    # Biểu đồ scatter và time series (nếu có đủ dữ liệu)
    if all_trues is not None and all_preds is not None and len(all_trues) > 0:
        st.markdown("**Visualizations trên toàn bộ validation set**")
        # Scatter
        fig_scatter = plot_pred_vs_actual_scatter(all_trues, all_preds, "Scatter plot")
        st.plotly_chart(fig_scatter, use_container_width=True)
        # Time series (cần date nếu có)
        dates = result_dict.get("validation_dates", None)
        if dates is None and "splits" in result_dict:
            # Tạo date giả từ split cuối
            last_split = result_dict["splits"][-1] if result_dict["splits"] else None
            if last_split is not None:
                val_idx = last_split[1]
                dates = np.arange(val_idx[0], val_idx[-1]+1)
        if dates is not None and len(dates) == len(all_trues):
            fig_ts = plot_time_series_actual_pred(dates, all_trues, all_preds, "Time series")
            st.plotly_chart(fig_ts, use_container_width=True)
        # Error distribution
        fig_err = plot_error_distribution(all_trues, all_preds, "Sai số")
        st.plotly_chart(fig_err, use_container_width=True)
    
    # Backtest (chỉ cho regression)
    if task == "regression" and show_backtest and all_trues is not None and all_preds is not None:
        st.markdown("**🧪 Backtest mô phỏng**")
        bt = simulate_backtest(all_trues, all_preds)
        if bt:
            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            col_b1.metric("Total Return", f"{bt['total_return']:.1%}")
            col_b2.metric("Max Drawdown", f"{bt['max_drawdown']:.1%}")
            col_b3.metric("Sharpe (strategy)", f"{bt['sharpe_strategy']:.2f}")
            col_b4.metric("Num Trades", bt['num_trades'])
            fig_eq = plot_equity_curve(bt, "Equity & Drawdown")
            st.plotly_chart(fig_eq, use_container_width=True)
        else:
            st.info("Không thể thực hiện backtest (dữ liệu không đủ).")
    
    st.markdown("---")

# ═══════════════════════════════════════════════════════════════
# TÍCH HỢP VÀO TAB4 (CÓ THỂ GỌI TỪ APP.PY)
# ═══════════════════════════════════════════════════════════════

def evaluation_tab_content():
    """Hàm này được gọi từ tab4 trong app.py để hiển thị báo cáo."""
    import streamlit as st
    
    st.subheader("📈 Đánh giá mô hình")
    if "baseline_results" not in st.session_state:
        st.info("Chưa có kết quả baseline. Hãy chạy baseline ở tab Model Training trước.")
        return
    
    results = st.session_state["baseline_results"]
    cfg = st.session_state.get("model_cfg", None)
    task = cfg.task if cfg else "regression"
    
    model_names = list(results.keys())
    selected_model = st.selectbox("Chọn mô hình để xem chi tiết", model_names)
    if selected_model:
        res = results[selected_model]
        display_evaluation_report_streamlit(res, task, show_backtest=True)

# ═══════════════════════════════════════════════════════════════
# DEMO (CHẠY STANDALONE)
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Tạo dữ liệu giả để test
    np.random.seed(42)
    n = 500
    y_true = np.random.normal(0, 0.02, n)
    y_pred = y_true + np.random.normal(0, 0.01, n)  # khá tốt
    metrics = calculate_all_metrics(y_true, y_pred, "regression")
    print("Metrics:", metrics)
    
    # Backtest
    bt = simulate_backtest(y_true, y_pred)
    print("Backtest:", {k: v for k, v in bt.items() if k not in ["equity", "drawdown"]})
    
    # Plot (sẽ hiện nếu chạy trong Jupyter, nhưng ở đây chỉ test)
    fig = plot_pred_vs_actual_scatter(y_true, y_pred)
    fig.show() if hasattr(fig, "show") else None