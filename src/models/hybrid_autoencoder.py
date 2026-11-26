"""Autoencoder NumPy tự học biểu diễn, sau đó fit đầu hồi quy Ridge."""

from pathlib import Path

import numpy as np
import pandas as pd

from ..baseline import score
from ..features import NUMERIC_FEATURES, TARGET, make_hybrid_preprocessor


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "processed" / "splits"
RESULTS = ROOT / "reports" / "exp3_results.csv"
REPORT = ROOT / "reports" / "07_exp3_hybrid.md"
SEED = 42
MAX_SAMPLE = 50_000
LATENT = 32
BATCH_SIZE = 512
MAX_EPOCHS = 15
PATIENCE = 3
LEARNING_RATE = 0.001
RIDGE_ALPHA = 0.1


def train_autoencoder(x_train, x_inner, seed: int):
    """Fit encoder-decoder bằng MSE; chọn epoch theo reconstruction inner loss."""
    rng = np.random.default_rng(seed)
    input_size = x_train.shape[1]
    weights = [
        (rng.standard_normal((input_size, LATENT)) * np.sqrt(2 / input_size)).astype(np.float32),
        (rng.standard_normal((LATENT, input_size)) / np.sqrt(LATENT)).astype(np.float32),
    ]
    biases = [np.zeros(LATENT, dtype=np.float32), np.zeros(input_size, dtype=np.float32)]
    params = weights + biases
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    best = [p.copy() for p in params]
    best_loss, best_epoch, wait, step = np.inf, 0, 0, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        order = rng.permutation(x_train.shape[0])
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            x = x_train[idx].toarray().astype(np.float32, copy=False)
            hidden_pre = x @ params[0] + params[2]
            hidden = np.maximum(hidden_pre, 0.0)
            reconstructed = hidden @ params[1] + params[3]
            delta_out = (2.0 / (len(idx) * input_size)) * (reconstructed - x)
            grad_w_decoder = hidden.T @ delta_out
            grad_b_decoder = delta_out.sum(axis=0)
            delta_hidden = (delta_out @ params[1].T) * (hidden_pre > 0.0)
            grad_w_encoder = x.T @ delta_hidden
            grad_b_encoder = delta_hidden.sum(axis=0)
            gradients = [grad_w_encoder, grad_w_decoder, grad_b_encoder, grad_b_decoder]
            step += 1
            for i, (param, grad) in enumerate(zip(params, gradients)):
                first[i] = 0.9 * first[i] + 0.1 * grad
                second[i] = 0.999 * second[i] + 0.001 * grad * grad
                m = first[i] / (1 - 0.9**step)
                v = second[i] / (1 - 0.999**step)
                param -= LEARNING_RATE * m / (np.sqrt(v) + 1e-8)

        loss_sum, rows = 0.0, 0
        for start in range(0, x_inner.shape[0], BATCH_SIZE):
            x = x_inner[start:start + BATCH_SIZE].toarray().astype(np.float32, copy=False)
            hidden = np.maximum(x @ params[0] + params[2], 0.0)
            reconstructed = hidden @ params[1] + params[3]
            loss_sum += float(np.square(reconstructed - x).sum())
            rows += x.shape[0] * input_size
        loss = loss_sum / rows
        if loss < best_loss - 1e-6:
            best_loss, best_epoch = loss, epoch
            best = [p.copy() for p in params]
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    best_weights, best_biases = [best[0], best[1]], [best[2], best[3]]
    return best_weights, best_biases, best_epoch, float(best_loss)


def encode(x_sparse, weights, biases):
    codes = []
    for start in range(0, x_sparse.shape[0], BATCH_SIZE):
        x = x_sparse[start:start + BATCH_SIZE].toarray().astype(np.float32, copy=False)
        codes.append(np.maximum(x @ weights[0] + biases[0], 0.0))
    return np.concatenate(codes, axis=0).astype(np.float64)


def evaluate(train: pd.DataFrame, valid: pd.DataFrame, evaluation: str, fold: int | None):
    features = NUMERIC_FEATURES + ["province_name", "house_direction", "balcony_direction"]
    rng = np.random.default_rng(SEED + (fold or 0))
    sampled = np.sort(rng.choice(len(train), size=min(MAX_SAMPLE, len(train)), replace=False))
    sampled_train = train.iloc[sampled]
    inner_start = max(1, int(len(sampled_train) * 0.85))
    fit_sample, inner = sampled_train.iloc[:inner_start], sampled_train.iloc[inner_start:]
    preprocessor = make_hybrid_preprocessor()
    x_fit = preprocessor.fit_transform(fit_sample[features])
    x_inner = preprocessor.transform(inner[features])
    x_valid = preprocessor.transform(valid[features])
    encoder, decoder, best_epoch, inner_loss = train_autoencoder(x_fit, x_inner, SEED + (fold or 0))
    z_fit = encode(x_fit, encoder, decoder)
    z_valid = encode(x_valid, encoder, decoder)
    z_mean = z_fit.mean(axis=0)
    z_scale = z_fit.std(axis=0)
    z_scale[z_scale == 0] = 1.0
    z_fit = (z_fit - z_mean) / z_scale
    z_valid = (z_valid - z_mean) / z_scale
    design = np.column_stack((np.ones(len(z_fit)), z_fit))
    y_fit = np.log1p(fit_sample[TARGET].to_numpy(dtype=np.float64))
    penalty = np.eye(design.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    head = np.linalg.solve(design.T @ design / len(design) + penalty, design.T @ y_fit / len(design))
    log_fit = design @ head
    log_valid = np.column_stack((np.ones(len(z_valid)), z_valid)) @ head
    if evaluation == "validation":
        pd.DataFrame({
            "published_at": valid["published_at"].to_numpy(),
            "actual_price": valid[TARGET].to_numpy(dtype=np.float64),
            "predicted_price": np.expm1(log_valid),
            "residual_log": np.log1p(valid[TARGET].to_numpy(dtype=np.float64)) - log_valid,
        }).to_csv(ROOT / "reports" / "hybrid_validation_residuals.csv", index=False)
    return [
        {"model": "autoencoder_ridge_head", "evaluation": evaluation, "fold": fold, "partition": "fit_sample",
         "sample_rows": len(sampled), "best_epoch": best_epoch, "inner_reconstruction_mse": inner_loss,
         **score(fit_sample[TARGET].to_numpy(dtype=np.float64), log_fit, np.expm1(log_fit))},
        {"model": "autoencoder_ridge_head", "evaluation": evaluation, "fold": fold, "partition": "outer_validation",
         "sample_rows": len(sampled), "best_epoch": best_epoch, "inner_reconstruction_mse": inner_loss,
         **score(valid[TARGET].to_numpy(dtype=np.float64), log_valid, np.expm1(log_valid))},
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
    val = results[(results.evaluation == "validation") & (results.partition == "outer_validation")].iloc[0]
    cv = results[(results.evaluation == "time_series_cv") & (results.partition == "outer_validation")]
    val_values = " | ".join(fmt(m, float(val[m])) for m in metrics)
    cv_values = " | ".join(f"{fmt(m, cv[m].mean())} ± {fmt(m, cv[m].std(ddof=1))}" for m in metrics)
    epochs = ", ".join(f"fold {int(r.fold)}: {int(r.best_epoch)}" for r in cv.itertuples())
    REPORT.write_text(f"""# Bước 7 — TN3: Autoencoder + Ridge head tự cài

## Giả thuyết và cấu hình

- Autoencoder NumPy tự học latent code 32 chiều từ numeric + tỉnh + hướng nhà; đầu hồi quy Ridge (alpha={RIDGE_ALPHA}) dự đoán `log1p(price)` trên latent code.
- Adam (lr={LEARNING_RATE}), batch={BATCH_SIZE}, tối đa {MAX_EPOCHS} epoch; early stopping theo MSE tái tạo trên inner tail 15%, patience={PATIENCE}.
- Mỗi fold dùng sample tối đa {MAX_SAMPLE:,} dòng với seed={SEED}; preprocessing chỉ fit trên 85% đầu sample. Outer validation không dùng để dừng sớm; test không được đọc.
- Bỏ district/ward trong TN3 để giữ đầu vào gọn; TN1 dùng các biến đó nên đây không phải so sánh cùng bộ đặc trưng.

## Kết quả

Metric: RMSE VND | MAE VND | MAPE % | R² VND | RMSE log | MAE log | R² log.

- Validation: {val_values}
- Time-series CV (mean ± std): {cv_values}
- Epoch tốt nhất: {epochs}.

Chi tiết từng fold: `reports/exp3_results.csv`.

## Điểm tốt/chưa đạt và kết luận

- Biểu diễn latent được học không giám sát trên các đặc trưng của phần fit; đầu Ridge được huấn luyện riêng trên cùng phần fit.
- Sample giới hạn và bỏ district/ward làm giảm chi phí nhưng có thể làm yếu thông tin vị trí; so sánh trực tiếp với TN1 cần lưu ý khác biệt đặc trưng.
- Chọn giữ/loại dựa trên số liệu validation/CV và độ phức tạp trong bảng tổng hợp; chưa dùng test.
""", encoding="utf-8")
    print(f"Wrote {len(results)} hybrid evaluation rows; test untouched.")


if __name__ == "__main__":
    main()
