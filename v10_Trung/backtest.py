"""
backtest.py
===========
Module mô phỏng backtest cho chiến lược giao dịch dựa trên tín hiệu từ model.

Hỗ trợ:
  - Regression: tín hiệu dựa trên dấu của y_pred (long nếu >0, short nếu <0)
  - Classification: tín hiệu dựa trên lớp dự đoán (Up=1 => long, Down=-1 => short, Sideway=0 => out)
  - Tính toán các chỉ số: tổng lợi nhuận, tỷ lệ thắng, sharpe ratio, max drawdown, số giao dịch
  - Vẽ equity curve và drawdown
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional, Tuple
import streamlit as st


def simulate_backtest(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001,
    horizon: int = 1,
    task: str = "regression",
    class_mapping: Optional[Dict] = None,
) -> Dict:
    """
    Mô phỏng backtest cho chiến lược giao dịch đơn giản.

    Parameters
    ----------
    y_true : np.ndarray
        Lợi nhuận thực tế (daily return nếu horizon=1, hoặc multi-day return nếu horizon>1)
    y_pred : np.ndarray
        Dự đoán của model (giá trị hồi quy hoặc nhãn phân loại)
    initial_capital : float
        Vốn ban đầu
    transaction_cost : float
        Phí giao dịch mỗi lần (tỷ lệ, ví dụ 0.001 = 0.1%)
    horizon : int
        Số ngày mà y_true đại diện (thường là gap trong walk-forward). Backtest sẽ quy về daily.
    task : str
        "regression" hoặc "classification"
    class_mapping : dict, optional
        Ánh xạ từ nhãn phân loại sang tín hiệu (1: long, -1: short, 0: out).
        Mặc định: {1: 1, -1: -1, 0: 0} cho ternary classification.

    Returns
    -------
    dict
        Các chỉ số và dữ liệu backtest.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return {}

    # 1. Tín hiệu
    if task == "regression":
        signal = np.sign(y_pred)
    else:
        if class_mapping is None:
            class_mapping = {1: 1, -1: -1, 0: 0}
        signal = np.array([class_mapping.get(p, 0) for p in y_pred])

    # 2. Quy đổi về daily return (nếu horizon > 1, giả sử lợi nhuận phân bố đều)
    daily_return = y_true / horizon if horizon > 1 else y_true

    # 3. Lợi nhuận gộp hàng ngày
    gross_returns = signal * daily_return

    # 4. Phí giao dịch
    trade_changes = np.abs(np.diff(signal, prepend=signal[0]))
    costs = trade_changes * transaction_cost
    net_returns = gross_returns - costs

    # 5. Equity curve
    equity = initial_capital * (1 + net_returns).cumprod()
    # 6. Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown = np.min(drawdown)

    # 7. Các chỉ số tổng hợp
    total_return = equity[-1] / initial_capital - 1
    mean_ret = np.mean(net_returns)
    std_ret = np.std(net_returns, ddof=1)
    sharpe_ratio = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0

    # 8. Thống kê giao dịch
    # Tìm các lần vào lệnh (signal != 0 và trước đó signal == 0 hoặc đổi dấu)
    entry_indices = []
    exit_indices = []
    in_trade = False
    for i in range(len(signal)):
        if signal[i] != 0 and not in_trade:
            entry_indices.append(i)
            in_trade = True
        elif (signal[i] == 0 or i == len(signal)-1) and in_trade:
            exit_indices.append(i if signal[i]==0 else i)
            in_trade = False

    trades = []
    for entry, exit_ in zip(entry_indices, exit_indices):
        if exit_ > entry:
            trade_return = (1 + net_returns[entry+1:exit_+1]).prod() - 1
            trades.append(trade_return)

    num_trades = len(trades)
    if num_trades > 0:
        winning_trades = [r for r in trades if r > 0]
        win_rate = len(winning_trades) / num_trades
        avg_win = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = np.mean([r for r in trades if r <= 0]) if any(r <= 0 for r in trades) else 0.0
        profit_factor = abs(sum(winning_trades) / sum([r for r in trades if r < 0])) if any(r < 0 for r in trades) else np.inf
    else:
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0

    # 9. Dự đoán lợi nhuận kỳ vọng (dựa trên tín hiệu và lợi nhuận trung bình)
    expected_return_per_trade = np.mean(trades) if trades else 0.0

    return {
        "equity": equity,
        "drawdown": drawdown,
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "num_trades": int(num_trades),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor),
        "expected_return_per_trade": float(expected_return_per_trade),
        "net_returns": net_returns,
        "signal": signal,
    }


def plot_equity_curve(backtest_dict: Dict, title: str = "Equity Curve & Drawdown") -> go.Figure:
    """Biểu đồ equity và drawdown."""
    equity = backtest_dict.get("equity")
    drawdown = backtest_dict.get("drawdown")
    if equity is None or len(equity) == 0:
        return go.Figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Equity Curve", "Drawdown"))
    fig.add_trace(go.Scatter(x=np.arange(len(equity)), y=equity, mode="lines",
                             name="Equity", line=dict(color="#3fb950")), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(drawdown)), y=drawdown, mode="lines",
                             fill="tozeroy", name="Drawdown", line=dict(color="#f85149")), row=2, col=1)
    fig.update_layout(title=title, height=600, width=800, template="plotly_dark")
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", row=2, col=1, tickformat=".0%")
    return fig


def display_backtest_in_streamlit(backtest_dict: Dict, initial_capital: float) -> None:
    """Hiển thị kết quả backtest trong Streamlit."""
    if not backtest_dict:
        st.warning("Không có dữ liệu backtest.")
        return

    st.markdown("#### 📊 Kết quả Backtest")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng lợi nhuận", f"{backtest_dict['total_return']:.2%}")
    col2.metric("Sharpe Ratio (ann.)", f"{backtest_dict['sharpe_ratio']:.2f}")
    col3.metric("Max Drawdown", f"{backtest_dict['max_drawdown']:.2%}")
    col4.metric("Số giao dịch", f"{backtest_dict['num_trades']}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Tỷ lệ thắng", f"{backtest_dict['win_rate']:.2%}")
    col6.metric("Lợi nhuận TB/win", f"{backtest_dict['avg_win']:.2%}")
    col7.metric("Lỗ TB/loss", f"{backtest_dict['avg_loss']:.2%}")
    col8.metric("Profit Factor", f"{backtest_dict['profit_factor']:.2f}")

    st.markdown("#### 📈 Biểu đồ vốn và Drawdown")
    fig = plot_equity_curve(backtest_dict)
    st.plotly_chart(fig, use_container_width=True)

    # Tải báo cáo
    report_df = pd.DataFrame({
        "Metric": ["Total Return", "Sharpe Ratio", "Max Drawdown", "Num Trades",
                   "Win Rate", "Avg Win", "Avg Loss", "Profit Factor",
                   "Expected Return per Trade"],
        "Value": [
            f"{backtest_dict['total_return']:.2%}",
            f"{backtest_dict['sharpe_ratio']:.2f}",
            f"{backtest_dict['max_drawdown']:.2%}",
            backtest_dict['num_trades'],
            f"{backtest_dict['win_rate']:.2%}",
            f"{backtest_dict['avg_win']:.2%}",
            f"{backtest_dict['avg_loss']:.2%}",
            f"{backtest_dict['profit_factor']:.2f}",
            f"{backtest_dict['expected_return_per_trade']:.2%}"
        ]
    })
    st.download_button("⬇️ Tải báo cáo Backtest (CSV)", report_df.to_csv(index=False),
                       "backtest_report.csv", "text/csv")