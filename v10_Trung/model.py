"""
core/model.py
=============
Module huấn luyện model cho bài toán dự đoán giá cổ phiếu.

Hỗ trợ:
  - Task       : regression (log_return) | classification (Up/Sideway/Down)
  - Models     : LightGBM, XGBoost, Dummy (baseline), LSTM (fixed leakage), Transformer (new)
  - Validation : Walk-forward CV (có gap, không dùng TimeSeriesSplit)
  - Tuning     : Optuna (Bayesian) với early stopping (chỉ cho LightGBM/XGB)
"""

from __future__ import annotations

import warnings
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from scipy import stats
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    task: str = "regression"
    n_classes: int = 3
    n_folds: int = 5
    min_train_size: int = 252
    gap: int = 5
    val_size: int = 63
    n_trials: int = 50
    optuna_timeout: int = 120
    feature_cols: List[str] = field(default_factory=list)
    target_col: str = "target"
    date_col: str = "date"
    primary_metric: str = "ic"


# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD SPLITS
# ═══════════════════════════════════════════════════════════════

def walk_forward_splits(
    n: int,
    n_folds: int,
    min_train_size: int,
    gap: int,
    val_size: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits = []
    total_needed = min_train_size + gap + val_size
    if n < total_needed:
        raise ValueError(f"Không đủ data: cần {total_needed} dòng, chỉ có {n}.")
    last_val_end = n
    last_val_start = last_val_end - val_size
    last_train_end = last_val_start - gap
    available_range = last_train_end - min_train_size
    step = max(1, available_range // (n_folds - 1)) if n_folds > 1 else available_range
    for i in range(n_folds):
        train_end = min_train_size + i * step
        val_start = train_end + gap
        val_end = val_start + val_size
        if val_end > n:
            break
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        if len(train_idx) < min_train_size or len(val_idx) < 10:
            continue
        splits.append((train_idx, val_idx))
    # Loại duplicate
    seen = set()
    unique = []
    for tr, vl in splits:
        key = (tr[-1], vl[0], vl[-1])
        if key not in seen:
            seen.add(key)
            unique.append((tr, vl))
    return unique


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> Dict[str, float]:
    m = {}
    if task == "regression":
        m["rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        m["mae"] = float(np.mean(np.abs(y_true - y_pred)))
        ic, pval = stats.spearmanr(y_pred, y_true)
        m["ic"] = float(ic) if not np.isnan(ic) else 0.0
        m["pval"] = float(pval) if not np.isnan(pval) else 1.0
        m["dir_acc"] = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        m["r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        m["accuracy"] = float(accuracy_score(y_true, y_pred))
        m["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        m["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        for cls, label in [(-1, "down"), (0, "sideway"), (1, "up")]:
            mask = y_true == cls
            if mask.sum() == 0:
                continue
            m[f"precision_{label}"] = float(precision_score(y_true == cls, y_pred == cls, zero_division=0))
            m[f"recall_{label}"] = float(recall_score(y_true == cls, y_pred == cls, zero_division=0))
            m[f"f1_{label}"] = float(f1_score(y_true == cls, y_pred == cls, zero_division=0))
    return m


# ═══════════════════════════════════════════════════════════════
# MODEL BUILDERS (LGBM, XGB, DUMMY)
# ═══════════════════════════════════════════════════════════════

def _build_lgbm(params: Dict, task: str, n_classes: int, n_jobs: int = -1):
    import lightgbm as lgb
    base = dict(random_state=42, verbose=-1, n_jobs=n_jobs, importance_type="gain")
    if task == "regression":
        return lgb.LGBMRegressor(objective="regression", **base, **params)
    else:
        return lgb.LGBMClassifier(
            objective="multiclass" if n_classes > 2 else "binary",
            num_class=n_classes if n_classes > 2 else None,
            class_weight="balanced",
            **base, **params,
        )

def _build_xgb(params: Dict, task: str, n_classes: int, n_jobs: int = -1):
    import xgboost as xgb
    base = dict(random_state=42, n_jobs=n_jobs, verbosity=0, eval_metric=None)
    if task == "regression":
        return xgb.XGBRegressor(objective="reg:squarederror", **base, **params)
    else:
        obj = "multi:softmax" if n_classes > 2 else "binary:logistic"
        return xgb.XGBClassifier(objective=obj, num_class=n_classes if n_classes > 2 else None, **base, **params)

def _build_dummy(task: str, n_classes: int):
    if task == "regression":
        return DummyRegressor(strategy="mean")
    else:
        return DummyClassifier(strategy="stratified", random_state=42)


# ═══════════════════════════════════════════════════════════════
# DEFAULT PARAMS
# ═══════════════════════════════════════════════════════════════

LGBM_DEFAULT = dict(
    n_estimators=200, learning_rate=0.05, max_depth=4, num_leaves=15,
    min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, reg_alpha=0.1,
)

XGB_DEFAULT = dict(
    n_estimators=200, learning_rate=0.05, max_depth=4, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.1,
)


# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD EVALUATION (LGBM, XGB, DUMMY)
# ═══════════════════════════════════════════════════════════════

def walk_forward_evaluate(
    df: pd.DataFrame,
    cfg: ModelConfig,
    model_type: str = "lgbm",
    params: Optional[Dict] = None,
    verbose: bool = True,
) -> Dict:
    X = df[cfg.feature_cols].values
    y = df[cfg.target_col].values
    dates = df[cfg.date_col].values if cfg.date_col in df.columns else np.arange(len(df))
    splits = walk_forward_splits(len(df), cfg.n_folds, cfg.min_train_size, cfg.gap, cfg.val_size)
    if not splits:
        raise RuntimeError("Không tạo được fold nào.")
    if verbose:
        print(f"\n{'─'*55}\n  Walk-forward: {len(splits)} folds | model={model_type.upper()} | task={cfg.task}\n{'─'*55}")
    fold_results, all_preds, all_trues = [], [], []
    for fold_i, (tr_idx, vl_idx) in enumerate(splits):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_vl, y_vl = X[vl_idx], y[vl_idx]
        if model_type == "lgbm":
            model = _build_lgbm(params or LGBM_DEFAULT, cfg.task, cfg.n_classes)
        elif model_type == "xgb":
            model = _build_xgb(params or XGB_DEFAULT, cfg.task, cfg.n_classes)
        else:
            model = _build_dummy(cfg.task, cfg.n_classes)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_vl)
        m = compute_metrics(y_vl, preds, cfg.task)
        m.update({"fold": fold_i+1, "train_size": len(tr_idx), "val_size": len(vl_idx),
                  "val_start": str(dates[vl_idx[0]])[:10], "val_end": str(dates[vl_idx[-1]])[:10]})
        fold_results.append(m)
        all_preds.extend(preds)
        all_trues.extend(y_vl)
        if verbose:
            if cfg.task == "regression":
                print(f"  Fold {fold_i+1} | {m['val_start']} → {m['val_end']} | IC={m['ic']:+.4f} | DirAcc={m['dir_acc']:.1%} | RMSE={m['rmse']:.4f}")
            else:
                print(f"  Fold {fold_i+1} | {m['val_start']} → {m['val_end']} | F1={m['f1_macro']:.4f} | Acc={m['accuracy']:.1%}")
    all_trues = np.array(all_trues)
    all_preds = np.array(all_preds)
    overall = compute_metrics(all_trues, all_preds, cfg.task)
    if cfg.task == "regression":
        ic_list = [f["ic"] for f in fold_results]
        overall["ic_mean"] = np.mean(ic_list)
        overall["ic_std"] = np.std(ic_list)
        overall["ic_pos_folds"] = sum(1 for ic in ic_list if ic > 0)
        overall["ic_ir"] = overall["ic_mean"] / overall["ic_std"] if overall["ic_std"] > 0 else 0.0
    else:
        f1_list = [f["f1_macro"] for f in fold_results]
        overall["f1_mean"] = np.mean(f1_list)
        overall["f1_std"] = np.std(f1_list)
    if verbose:
        print(f"{'─'*55}")
        if cfg.task == "regression":
            print(f"  Overall IC  : {overall['ic']:+.4f}  (mean={overall['ic_mean']:+.4f} ± {overall['ic_std']:.4f})")
            print(f"  IC IR       : {overall['ic_ir']:+.3f}")
            print(f"  Dir acc     : {overall['dir_acc']:.1%}")
            print(f"  RMSE        : {overall['rmse']:.4f}")
        else:
            print(f"  F1-macro    : {overall['f1_macro']:.4f}  (mean={overall['f1_mean']:.4f} ± {overall['f1_std']:.4f})")
            print(f"  Accuracy    : {overall['accuracy']:.1%}")
        print(f"{'─'*55}\n")
    return {"overall": overall, "fold_results": fold_results, "all_preds": all_preds, "all_trues": all_trues,
            "splits": splits, "model_type": model_type, "params": params, "cfg": cfg}


# ═══════════════════════════════════════════════════════════════
# OPTUNA TUNING (LGBM/XGB)
# ═══════════════════════════════════════════════════════════════

def _objective_factory(df, cfg, model_type, splits):
    import optuna, lightgbm as lgb, xgboost as xgb
    X = df[cfg.feature_cols].values
    y = df[cfg.target_col].values
    def objective(trial):
        if model_type == "lgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "num_leaves": trial.suggest_int("num_leaves", 8, 63),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 5.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 2.0, log=True),
            }
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 5.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 2.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            }
        scores = []
        for tr_idx, vl_idx in splits:
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_vl, y_vl = X[vl_idx], y[vl_idx]
            if model_type == "lgbm":
                model = _build_lgbm(params, cfg.task, cfg.n_classes, n_jobs=1)
                model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], eval_metric="rmse" if cfg.task=="regression" else "multi_logloss",
                          callbacks=[lgb.early_stopping(20, verbose=False)])
            else:
                model = _build_xgb(params, cfg.task, cfg.n_classes, n_jobs=1)
                model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], early_stopping_rounds=20, verbose=False)
            preds = model.predict(X_vl)
            m = compute_metrics(y_vl, preds, cfg.task)
            scores.append(m["ic"] if cfg.task=="regression" else m["f1_macro"])
        return float(np.mean(scores))
    return objective

def tune_hyperparams(df, cfg, model_type="lgbm", n_trials=50, timeout=120, show_progress=True):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    splits = walk_forward_splits(len(df), cfg.n_folds, cfg.min_train_size, cfg.gap, cfg.val_size)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    objective = _objective_factory(df, cfg, model_type, splits)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=show_progress)
    best = study.best_trial
    return {"best_params": best.params, "best_score": best.value, "study": study}


# ═══════════════════════════════════════════════════════════════
# TRAIN FINAL MODEL (LGBM/XGB)
# ═══════════════════════════════════════════════════════════════

def train_final_model(df_train, cfg, model_type="lgbm", params=None, n_jobs=-1):
    if params is None:
        params = LGBM_DEFAULT if model_type=="lgbm" else XGB_DEFAULT
    X_tr = df_train[cfg.feature_cols].values
    y_tr = df_train[cfg.target_col].values
    if model_type == "lgbm":
        model = _build_lgbm(params, cfg.task, cfg.n_classes, n_jobs)
    else:
        model = _build_xgb(params, cfg.task, cfg.n_classes, n_jobs)
    model.fit(X_tr, y_tr)
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({"feature": cfg.feature_cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False).reset_index(drop=True)
    else:
        fi = pd.DataFrame()
    return model, fi


# ═══════════════════════════════════════════════════════════════
# LSTM (FIXED LEAKAGE)
# ═══════════════════════════════════════════════════════════════

class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def _build_lstm(input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
    return LSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout)

def create_sequences_safe(X: np.ndarray, y: np.ndarray, seq_len: int, gap: int = 5):
    """Tạo sequences tránh overlap và forward leak. Input: [t-seq_len:t], Target: y[t+gap]"""
    max_idx = len(X) - seq_len - gap
    if max_idx <= 0:
        raise ValueError(f"Không đủ dữ liệu: cần {seq_len + gap} rows, có {len(X)}")
    X_seq, y_seq = [], []
    for i in range(max_idx):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len+gap])   # Target sau gap
    return np.array(X_seq), np.array(y_seq)

def train_lstm_safe(model, X_tr, y_tr, X_val, y_val, epochs=50, batch_size=32, lr=0.001,
                    patience=5, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() if model.fc.out_features == 1 else nn.CrossEntropyLoss()
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=False)  # NO SHUFFLE
    best_loss, best_state, patience_counter = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t.to(device)), y_val_t.to(device)).item()
        if verbose and (epoch+1)%10==0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(loader):.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss, best_state, patience_counter = val_loss, model.state_dict().copy(), 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose: print(f"Early stopping at epoch {epoch+1}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def lstm_walk_forward_evaluate(
    df: pd.DataFrame,
    cfg: ModelConfig,
    seq_len: int = 30,
    hidden_dim: int = 32,
    num_layers: int = 1,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    patience: int = 5,
    verbose: bool = False,
) -> Dict:
    """
    Walk-forward evaluation cho LSTM - FIXED.
    - Tạo sequences với gap (không overlap)
    - NO SHUFFLE
    - Fit scaler per fold
    """
    X_raw = df[cfg.feature_cols].values
    y_raw = df[cfg.target_col].values
    dates = df[cfg.date_col].values if cfg.date_col in df.columns else np.arange(len(df))

    splits = walk_forward_splits(
        n=len(df),
        n_folds=cfg.n_folds,
        min_train_size=cfg.min_train_size,
        gap=cfg.gap,
        val_size=cfg.val_size,
    )

    if not splits:
        raise RuntimeError("Không tạo được fold nào.")

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  LSTM Walk-forward SAFE | {len(splits)} folds | seq_len={seq_len}")
        print(f"{'─'*55}")

    fold_results = []
    all_preds = []
    all_trues = []

    for fold_i, (tr_idx, val_idx) in enumerate(splits):
        # Lấy dữ liệu thô
        X_tr_raw = X_raw[tr_idx]
        y_tr_raw = y_raw[tr_idx]
        X_val_raw = X_raw[val_idx]
        y_val_raw = y_raw[val_idx]

        # Scale chỉ trên train
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_raw)
        X_val_scaled = scaler.transform(X_val_raw)

        # Tạo sequences với gap (tránh forward leak)
        try:
            X_tr_seq, y_tr_seq = create_sequences_safe(X_tr_scaled, y_tr_raw, seq_len, cfg.gap)
            X_val_seq, y_val_seq = create_sequences_safe(X_val_scaled, y_val_raw, seq_len, cfg.gap)
        except ValueError as e:
            if verbose:
                print(f"  ⚠️  Fold {fold_i+1}: {e}")
            continue

        if len(X_tr_seq) == 0 or len(X_val_seq) == 0:
            if verbose:
                print(f"  ⚠️  Fold {fold_i+1}: không tạo được sequences (train={len(X_tr_seq)}, val={len(X_val_seq)})")
            continue

        input_dim = X_tr_seq.shape[2]
        output_dim = 1 if cfg.task == 'regression' else cfg.n_classes
        model = _build_lstm(input_dim, hidden_dim, num_layers, output_dim)

        # Train với shuffle=False
        model = train_lstm_safe(model, X_tr_seq, y_tr_seq, X_val_seq, y_val_seq,
                                epochs, batch_size, lr, patience, verbose=False)

        # Dự đoán
        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val_seq, dtype=torch.float32)
            preds = model(X_val_t).numpy().flatten()

        if cfg.task == 'classification' and output_dim > 1:
            preds = np.argmax(preds, axis=1)

        m = compute_metrics(y_val_seq, preds, cfg.task)
        m["fold"] = fold_i + 1
        m["train_size"] = len(tr_idx)
        m["val_size"] = len(val_idx)
        m["val_start"] = str(dates[val_idx[0]])[:10]
        m["val_end"] = str(dates[val_idx[-1]])[:10]
        m["seq_samples"] = len(X_val_seq)
        fold_results.append(m)

        all_preds.extend(preds)
        all_trues.extend(y_val_seq)

        if verbose:
            if cfg.task == "regression":
                print(f"  Fold {fold_i+1} | {m['val_start']} → {m['val_end']} | Seq={len(X_val_seq)} | IC={m['ic']:+.4f} | RMSE={m['rmse']:.4f}")
            else:
                print(f"  Fold {fold_i+1} | {m['val_start']} → {m['val_end']} | Seq={len(X_val_seq)} | F1={m['f1_macro']:.4f}")

    if not fold_results:
        raise RuntimeError("Không có fold nào thành công. Giảm seq_len hoặc tăng min_train_size/val_size.")

    all_trues = np.array(all_trues)
    all_preds = np.array(all_preds)
    overall = compute_metrics(all_trues, all_preds, cfg.task)

    # Stability metrics
    if cfg.task == "regression":
        ic_list = [f["ic"] for f in fold_results]
        overall["ic_mean"] = float(np.mean(ic_list))
        overall["ic_std"] = float(np.std(ic_list))
        overall["ic_pos_folds"] = sum(1 for ic in ic_list if ic > 0)
        overall["ic_ir"] = (np.mean(ic_list) / np.std(ic_list) if np.std(ic_list) > 0 else 0.0)
    else:
        f1_list = [f["f1_macro"] for f in fold_results]
        overall["f1_mean"] = float(np.mean(f1_list))
        overall["f1_std"] = float(np.std(f1_list))

    if verbose:
        print(f"{'─'*55}")
        if cfg.task == "regression":
            print(f"  Overall IC  : {overall['ic']:+.4f}  (mean={overall['ic_mean']:+.4f} ± {overall['ic_std']:.4f})")
            print(f"  IC IR       : {overall['ic_ir']:+.3f}")
            print(f"  Dir acc     : {overall['dir_acc']:.1%}")
            print(f"  RMSE        : {overall['rmse']:.4f}")
        else:
            print(f"  F1-macro    : {overall['f1_macro']:.4f}  (mean={overall['f1_mean']:.4f} ± {overall['f1_std']:.4f})")
            print(f"  Accuracy    : {overall['accuracy']:.1%}")
        print(f"{'─'*55}\n")

    return {
        "overall": overall,
        "fold_results": fold_results,
        "all_preds": all_preds,
        "all_trues": all_trues,
        "splits": splits,
        "model_type": "lstm_safe",
        "params": {
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "patience": patience
        },
        "cfg": cfg,
    }

def train_final_lstm(df_train, cfg, seq_len=30, gap=5, hidden_dim=32, num_layers=1,
                     epochs=100, batch_size=32, lr=0.001, patience=10):
    X_all = df_train[cfg.feature_cols].values
    y_all = df_train[cfg.target_col].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    X_seq, y_seq = create_sequences_safe(X_scaled, y_all, seq_len, gap)
    input_dim = X_seq.shape[2]
    output_dim = 1 if cfg.task == 'regression' else cfg.n_classes
    model = _build_lstm(input_dim, hidden_dim, num_layers, output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() if output_dim == 1 else nn.CrossEntropyLoss()
    dataset = TensorDataset(torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32).view(-1,1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # NO SHUFFLE
    best_loss, best_state, patience_counter = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        if avg_loss < best_loss:
            best_loss, best_state, patience_counter = avg_loss, model.state_dict().copy(), 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler


# ═══════════════════════════════════════════════════════════════
# TRANSFORMER (NEW)
# ═══════════════════════════════════════════════════════════════

class TransformerModel(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.1, output_dim: int = 1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.pos_encoder, num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)                     # (batch, seq_len, d_model)
        x = self.transformer(x)                    # (batch, seq_len, d_model)
        x = x[:, -1, :]                            # lấy output cuối cùng
        x = self.dropout(x)
        return self.fc(x)

def _build_transformer(input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                       dim_feedforward: int = 128, dropout: float = 0.1, output_dim: int = 1) -> TransformerModel:
    return TransformerModel(input_dim, d_model, nhead, num_layers, dim_feedforward, dropout, output_dim)

def train_transformer_safe(model, X_tr, y_tr, X_val, y_val, epochs=50, batch_size=32, lr=0.001,
                           patience=5, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() if model.fc.out_features == 1 else nn.CrossEntropyLoss()
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=False)  # NO SHUFFLE
    best_loss, best_state, patience_counter = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t.to(device)), y_val_t.to(device)).item()
        if verbose and (epoch+1)%10==0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(loader):.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss, best_state, patience_counter = val_loss, model.state_dict().copy(), 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose: print(f"Early stopping at epoch {epoch+1}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def transformer_walk_forward_evaluate(
    df: pd.DataFrame,
    cfg: ModelConfig,
    seq_len: int = 30,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    patience: int = 5,
    verbose: bool = False,
) -> Dict:
    """
    Walk-forward evaluation cho Transformer.
    - Tạo sequences với gap (giống LSTM)
    - NO SHUFFLE
    - Fit scaler per fold
    """
    X_raw = df[cfg.feature_cols].values
    y_raw = df[cfg.target_col].values
    dates = df[cfg.date_col].values if cfg.date_col in df.columns else np.arange(len(df))

    splits = walk_forward_splits(
        n=len(df),
        n_folds=cfg.n_folds,
        min_train_size=cfg.min_train_size,
        gap=cfg.gap,
        val_size=cfg.val_size,
    )

    if not splits:
        raise RuntimeError("Không tạo được fold nào.")

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  Transformer Walk-forward SAFE | {len(splits)} folds | seq_len={seq_len}")
        print(f"{'─'*55}")

    fold_results = []
    all_preds = []
    all_trues = []

    for fold_i, (tr_idx, val_idx) in enumerate(splits):
        X_tr_raw = X_raw[tr_idx]
        y_tr_raw = y_raw[tr_idx]
        X_val_raw = X_raw[val_idx]
        y_val_raw = y_raw[val_idx]

        # Scale chỉ trên train
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_raw)
        X_val_scaled = scaler.transform(X_val_raw)

        try:
            X_tr_seq, y_tr_seq = create_sequences_safe(X_tr_scaled, y_tr_raw, seq_len, cfg.gap)
            X_val_seq, y_val_seq = create_sequences_safe(X_val_scaled, y_val_raw, seq_len, cfg.gap)
        except ValueError as e:
            if verbose:
                print(f"  ⚠️  Fold {fold_i+1}: {e}")
            continue

        if len(X_tr_seq) == 0 or len(X_val_seq) == 0:
            if verbose:
                print(f"  ⚠️  Fold {fold_i+1}: không tạo được sequences")
            continue

        input_dim = X_tr_seq.shape[2]
        output_dim = 1 if cfg.task == 'regression' else cfg.n_classes
        model = _build_transformer(input_dim, d_model, nhead, num_layers, dim_feedforward, dropout, output_dim)

        model = train_transformer_safe(model, X_tr_seq, y_tr_seq, X_val_seq, y_val_seq,
                                       epochs, batch_size, lr, patience, verbose=False)

        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val_seq, dtype=torch.float32)
            preds = model(X_val_t).numpy().flatten()

        if cfg.task == 'classification' and output_dim > 1:
            preds = np.argmax(preds, axis=1)

        m = compute_metrics(y_val_seq, preds, cfg.task)
        m["fold"] = fold_i + 1
        m["train_size"] = len(tr_idx)
        m["val_size"] = len(val_idx)
        m["val_start"] = str(dates[val_idx[0]])[:10]
        m["val_end"] = str(dates[val_idx[-1]])[:10]
        m["seq_samples"] = len(X_val_seq)
        fold_results.append(m)

        all_preds.extend(preds)
        all_trues.extend(y_val_seq)

        if verbose:
            if cfg.task == "regression":
                print(f"  Fold {fold_i+1} | {m['val_start']} → {m['val_end']} | Seq={len(X_val_seq)} | IC={m['ic']:+.4f} | RMSE={m['rmse']:.4f}")
            else:
                print(f"  Fold {fold_i+1} | {m['val_start']} → {m['val_end']} | Seq={len(X_val_seq)} | F1={m['f1_macro']:.4f}")

    if not fold_results:
        raise RuntimeError("Không có fold nào thành công. Giảm seq_len hoặc tăng min_train_size/val_size.")

    all_trues = np.array(all_trues)
    all_preds = np.array(all_preds)
    overall = compute_metrics(all_trues, all_preds, cfg.task)

    if cfg.task == "regression":
        ic_list = [f["ic"] for f in fold_results]
        overall["ic_mean"] = float(np.mean(ic_list))
        overall["ic_std"] = float(np.std(ic_list))
        overall["ic_pos_folds"] = sum(1 for ic in ic_list if ic > 0)
        overall["ic_ir"] = (np.mean(ic_list) / np.std(ic_list) if np.std(ic_list) > 0 else 0.0)
    else:
        f1_list = [f["f1_macro"] for f in fold_results]
        overall["f1_mean"] = float(np.mean(f1_list))
        overall["f1_std"] = float(np.std(f1_list))

    if verbose:
        print(f"{'─'*55}")
        if cfg.task == "regression":
            print(f"  Overall IC  : {overall['ic']:+.4f}  (mean={overall['ic_mean']:+.4f} ± {overall['ic_std']:.4f})")
            print(f"  IC IR       : {overall['ic_ir']:+.3f}")
            print(f"  Dir acc     : {overall['dir_acc']:.1%}")
            print(f"  RMSE        : {overall['rmse']:.4f}")
        else:
            print(f"  F1-macro    : {overall['f1_macro']:.4f}  (mean={overall['f1_mean']:.4f} ± {overall['f1_std']:.4f})")
            print(f"  Accuracy    : {overall['accuracy']:.1%}")
        print(f"{'─'*55}\n")

    return {
        "overall": overall,
        "fold_results": fold_results,
        "all_preds": all_preds,
        "all_trues": all_trues,
        "splits": splits,
        "model_type": "transformer",
        "params": {
            "seq_len": seq_len,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "patience": patience,
        },
        "cfg": cfg,
    }

def train_final_transformer(df_train, cfg, seq_len=30, gap=5, d_model=64, nhead=4,
                            num_layers=2, dim_feedforward=128, dropout=0.1,
                            epochs=100, batch_size=32, lr=0.001, patience=10):
    X_all = df_train[cfg.feature_cols].values
    y_all = df_train[cfg.target_col].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    X_seq, y_seq = create_sequences_safe(X_scaled, y_all, seq_len, gap)
    input_dim = X_seq.shape[2]
    output_dim = 1 if cfg.task == 'regression' else cfg.n_classes
    model = _build_transformer(input_dim, d_model, nhead, num_layers, dim_feedforward, dropout, output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() if output_dim == 1 else nn.CrossEntropyLoss()
    dataset = TensorDataset(torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32).view(-1,1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # NO SHUFFLE
    best_loss, best_state, patience_counter = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        if avg_loss < best_loss:
            best_loss, best_state, patience_counter = avg_loss, model.state_dict().copy(), 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler


# ═══════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════

def save_model(model, path: Path):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def compare_models(df, cfg, models_params=None):
    if models_params is None:
        models_params = {"LightGBM": ("lgbm", LGBM_DEFAULT), "XGBoost": ("xgb", XGB_DEFAULT), "Dummy": ("dummy", None)}
    rows = []
    for name, (mtype, params) in models_params.items():
        res = walk_forward_evaluate(df, cfg, model_type=mtype, params=params, verbose=False)
        ov = res["overall"]
        if cfg.task == "regression":
            rows.append({"Model": name, "IC": round(ov.get("ic",0),4), "IC mean±std": f"{ov.get('ic_mean',0):+.4f}±{ov.get('ic_std',0):.4f}",
                         "IC IR": round(ov.get("ic_ir",0),3), "Dir acc": f"{ov.get('dir_acc',0):.1%}", "RMSE": round(ov.get("rmse",0),4)})
        else:
            rows.append({"Model": name, "F1-macro": round(ov.get("f1_macro",0),4), "Accuracy": f"{ov.get('accuracy',0):.1%}",
                         "F1 Up": round(ov.get("f1_up",0),4), "F1 Side": round(ov.get("f1_sideway",0),4), "F1 Down": round(ov.get("f1_down",0),4)})
    return pd.DataFrame(rows)