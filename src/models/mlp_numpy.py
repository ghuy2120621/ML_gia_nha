"""MLP tự cài bằng NumPy với dropout, Adam và early stopping."""

from pathlib import Path

import numpy as np
import pandas as pd

from ..baseline import score
from ..features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, make_preprocessor


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "processed" / "splits"
RESULTS = ROOT / "reports" / "exp2_results.csv"
REPORT = ROOT / "reports" / "06_exp2_mlp.md"
SEED = 42
MAX_SAMPLE = 50_000
HIDDEN = 64
DROPOUT = 0.20
LEARNING_RATE = 0.001
BATCH_SIZE = 512
MAX_EPOCHS = 20
PATIENCE = 3


def forward(x, w1, b1, w2, b2, rng=None, training=False):
    hidden = np.maximum(x @ w1 + b1, 0.0)
    mask = None
    if training:
        mask = (rng.random(hidden.shape) >= DROPOUT).astype(np.float32) / (1.0 - DROPOUT)
        hidden = hidden * mask
    prediction = (hidden @ w2 + b2).ravel()
    return prediction, hidden, mask


def train_mlp(x_sparse, y, inner_x_sparse, inner_y, seed: int):
    """Huấn luyện MLP mini-batch; early stopping chỉ dùng inner tail của fit."""
    rng = np.random.default_rng(seed)
    input_size = x_sparse.shape[1]
    w1 = (rng.standard_normal((input_size, HIDDEN)) * np.sqrt(2.0 / input_size)).astype(np.float32)
    b1 = np.zeros(HIDDEN, dtype=np.float32)
    w2 = (rng.standard_normal((HIDDEN, 1)) / np.sqrt(HIDDEN)).astype(np.float32)
    b2 = np.zeros(1, dtype=np.float32)
    params = [w1, b1, w2, b2]
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    best_params = [p.copy() for p in params]
    best_loss = np.inf
    best_epoch = 0
    wait = 0
    step = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        order = rng.permutation(x_sparse.shape[0])
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            x = x_sparse[idx].toarray().astype(np.float32, copy=False)
            target = y[idx].astype(np.float32, copy=False)
            pred, hidden, mask = forward(x, *params, rng=rng, training=True)
            delta = (2.0 / len(idx)) * (pred - target)
            grad_w2 = hidden.T @ delta[:, None]
            grad_b2 = np.array([delta.sum()], dtype=np.float32)
            grad_hidden = delta[:, None] @ params[2].T
            grad_hidden *= mask
            grad_hidden *= ((x @ params[0] + params[1]) > 0.0)
            grad_w1 = x.T @ grad_hidden
            grad_b1 = grad_hidden.sum(axis=0)
            grads = [grad_w1, grad_b1, grad_w2, grad_b2]
            step += 1
            for i, (param, grad) in enumerate(zip(params, grads)):
                first[i] = 0.9 * first[i] + 0.1 * grad
                second[i] = 0.999 * second[i] + 0.001 * grad * grad
                m_hat = first[i] / (1.0 - 0.9**step)
                v_hat = second[i] / (1.0 - 0.999**step)
                param -= LEARNING_RATE * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Đánh giá inner validation theo batch để không tạo ma trận dense lớn.
        loss_sum = 0.0
        count = 0
        for start in range(0, inner_x_sparse.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, inner_x_sparse.shape[0])
            x = inner_x_sparse[start:end].toarray().astype(np.float32, copy=False)
            pred, _, _ = forward(x, *params)
            loss_sum += float(np.square(pred - inner_y[start:end]).sum())
            count += end - start
        loss = loss_sum / count
        if loss < best_loss - 1e-5:
            best_loss = loss
            best_params = [p.copy() for p in params]
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    return best_params, best_epoch, float(best_loss)


def predict(x_sparse, params):
    predictions = []
    for start in range(0, x_sparse.shape[0], BATCH_SIZE):
        x = x_sparse[start:start + BATCH_SIZE].toarray().astype(np.float32, copy=False)
        predictions.append(forward(x, *params)[0])
    return np.concatenate(predictions).astype(np.float64)


def evaluate(train: pd.DataFrame, valid: pd.DataFrame, evaluation: str, fold: int | None):
    """Fit preprocessing trên sample train và score trên inner/outer validation."""
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    rng = np.random.default_rng(SEED + (fold or 0))
    sampled = np.sort(rng.choice(len(train), size=min(MAX_SAMPLE, len(train)), replace=False))
    sampled_train = train.iloc[sampled]
    inner_start = max(1, int(len(sampled_train) * 0.85))
    fit_sample = sampled_train.iloc[:inner_start]
    inner = sampled_train.iloc[inner_start:]
    prep = make_preprocessor()
    x_fit = prep.fit_transform(fit_sample[features])
    x_inner = prep.transform(inner[features])
    x_valid = prep.transform(valid[features])
    y_fit = np.log1p(fit_sample[TARGET].to_numpy(dtype=np.float64)).astype(np.float32)
    y_inner = np.log1p(inner[TARGET].to_numpy(dtype=np.float64)).astype(np.float32)
    params, best_epoch, inner_loss = train_mlp(x_fit, y_fit, x_inner, y_inner, SEED + (fold or 0))
    log_fit = predict(x_fit, params)
    log_valid = predict(x_valid, params)
    fit_actual = fit_sample[TARGET].to_numpy(dtype=np.float64)
    valid_actual = valid[TARGET].to_numpy(dtype=np.float64)
    if evaluation == "validation":
        pd.DataFrame({
            "published_at": valid["published_at"].to_numpy(),
            "actual_price": valid_actual,
            "predicted_price": np.expm1(log_valid),
            "residual_log": np.log1p(valid_actual) - log_valid,
        }).to_csv(ROOT / "reports" / "mlp_validation_residuals.csv", index=False)
    return [
        {"model": "numpy_mlp", "evaluation": evaluation, "fold": fold, "partition": "fit_sample",
         "sample_rows": len(sampled), "best_epoch": best_epoch, "inner_val_rmse_log": np.sqrt(inner_loss),
         **score(fit_actual, log_fit, np.expm1(log_fit))},
        {"model": "numpy_mlp", "evaluation": evaluation, "fold": fold, "partition": "outer_validation",
         "sample_rows": len(sampled), "best_epoch": best_epoch, "inner_val_rmse_log": np.sqrt(inner_loss),
         **score(valid_actual, log_valid, np.expm1(log_valid))},
    ]


def main() -> None:
    train = pd.read_parquet(SPLITS / "train.parquet")
    validation = pd.read_parquet(SPLITS / "validation.parquet")
    fold_specs = pd.read_csv(SPLITS / "cv_folds.csv")
    rows = evaluate(train, validation, "validation", None)
    for fold in fold_specs.itertuples():
        fit = train.iloc[:fold.fit_end_row]
        valid = train.iloc[fold.validation_start_row:fold.validation_end_row]
        rows.extend(evaluate(fit, valid, "time_series_cv", int(fold.fold)))
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS, index=False)
    metrics = ["rmse_vnd", "mae_vnd", "mape_pct", "r2_vnd", "rmse_log", "mae_log", "r2_log"]
    def fmt(metric: str, value: float) -> str:
        return f"{value:,.0f}" if metric.endswith("_vnd") and not metric.startswith("r2") else f"{value:.3f}"
    validation = results[(results.evaluation == "validation") & (results.partition == "outer_validation")].iloc[0]
    cv = results[(results.evaluation == "time_series_cv") & (results.partition == "outer_validation")]
    val_table = " | ".join(fmt(m, float(validation[m])) for m in metrics)
    cv_table = " | ".join(f"{fmt(m, cv[m].mean())} ± {fmt(m, cv[m].std(ddof=1))}" for m in metrics)
    epoch_rows = ", ".join(f"fold {int(r.fold)}: {int(r.best_epoch)}" for r in cv.itertuples())
    REPORT.write_text(f"""# Bước 6 — TN2: MLP tự cài bằng NumPy

## Giả thuyết và cấu hình

- MLP một tầng ẩn (64 ReLU), dropout 0.20, Adam (lr=0.001), batch 512, mục tiêu `log1p(price)`.
- Early stopping dùng 15% cuối của sample fit theo thời gian (inner validation), patience=3, tối đa {MAX_EPOCHS} epoch.
- Do PyTorch chưa được cài và dữ liệu lớn, dùng NumPy MLP tự cài. Mỗi fold lấy sample ngẫu nhiên có seed={SEED}, tối đa {MAX_SAMPLE:,} dòng từ phần fit; preprocessing chỉ fit trên 85% đầu sample, không dùng outer validation để early stop.
- Bảng bên dưới đánh giá trên toàn bộ outer validation; test không được đọc.

## Kết quả

Metric: RMSE VND | MAE VND | MAPE % | R² VND | RMSE log | MAE log | R² log.

- Validation: {val_table}
- Time-series CV (mean ± std): {cv_table}
- Epoch tốt nhất theo fold: {epoch_rows}.

Chi tiết từng fold: `reports/exp2_results.csv`.

## Điểm tốt/chưa đạt và kết luận

- MLP học được quan hệ phi tuyến; so sánh log metrics với TN1/baseline trong bước tổng hợp.
- Sample 50.000 dòng/fold giảm chi phí tính toán nhưng có thể bỏ lỡ nhóm hiếm; kết quả không tương đương huấn luyện toàn bộ fit rows.
- MAPE/RMSE VND nhạy với giá thấp/cực cao; xem cả log metrics và phân tích residual ở bước chọn mô hình.
- Kết luận giữ/loại sẽ dựa trên so sánh chung; hiện chưa dùng test.
""", encoding="utf-8")
    print(f"Wrote {len(results)} MLP evaluation rows; test untouched.")


if __name__ == "__main__":
    main()
