"""Baseline giá trung vị và hồi quy tuyến tính NumPy."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .features import TARGET


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "processed" / "splits"
FOLDS = SPLITS / "cv_folds.csv"
RESULTS = ROOT / "reports" / "baseline_results.csv"
REPORT = ROOT / "reports" / "04_baseline.md"
PREDICTIONS = ROOT / "reports" / "baseline_validation_residuals.csv"


def fit_linear(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit OLS trên log diện tích và thời gian, chỉ lấy thống kê từ train."""
    area = pd.to_numeric(train["area"], errors="coerce").to_numpy(dtype=np.float64)
    elapsed = train["months_since_start"].to_numpy(dtype=np.float64)
    x = np.column_stack((np.log1p(area), elapsed))
    medians = np.nanmedian(x, axis=0)
    x = np.where(np.isnan(x), medians, x)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales == 0] = 1.0
    x = (x - means) / scales
    design = np.column_stack((np.ones(len(x)), x))
    y = np.log1p(train[TARGET].to_numpy(dtype=np.float64))
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return medians, means, scales, coefficients


def predict_linear(frame: pd.DataFrame, params: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Dự đoán log-price và VND bằng OLS đã fit."""
    medians, means, scales, coefficients = params
    area = pd.to_numeric(frame["area"], errors="coerce").to_numpy(dtype=np.float64)
    elapsed = frame["months_since_start"].to_numpy(dtype=np.float64)
    x = np.column_stack((np.log1p(area), elapsed))
    x = np.where(np.isnan(x), medians, x)
    x = (x - means) / scales
    log_prediction = np.column_stack((np.ones(len(x)), x)) @ coefficients
    return log_prediction, np.expm1(log_prediction)


def score(y: np.ndarray, log_prediction: np.ndarray, prediction: np.ndarray) -> dict:
    """Tính metric trên cả giá gốc và log1p giá."""
    log_y = np.log1p(y)
    return {
        "rmse_vnd": float(np.sqrt(mean_squared_error(y, prediction))),
        "mae_vnd": float(mean_absolute_error(y, prediction)),
        "mape_pct": float(np.mean(np.abs((y - prediction) / y)) * 100),
        "r2_vnd": float(r2_score(y, prediction)),
        "rmse_log": float(np.sqrt(mean_squared_error(log_y, log_prediction))),
        "mae_log": float(mean_absolute_error(log_y, log_prediction)),
        "r2_log": float(r2_score(log_y, log_prediction)),
    }


def evaluate_pair(train: pd.DataFrame, valid: pd.DataFrame, label: str, fold: int | None) -> list[dict]:
    """Fit hai baseline trên train slice rồi chấm cả train và valid."""
    y_train = train[TARGET].to_numpy(dtype=np.float64)
    y = valid[TARGET].to_numpy(dtype=np.float64)
    rows = []
    median_price = float(train[TARGET].median())
    median_train_pred = np.full(len(train), median_price)
    median_pred = np.full(len(valid), median_price)
    for partition, actual, prediction in [
        ("fit", y_train, median_train_pred), ("validation", y, median_pred),
    ]:
        rows.append({"model": "train_median", "evaluation": label, "fold": fold, "partition": partition,
                     **score(actual, np.log1p(prediction), prediction)})
    params = fit_linear(train)
    log_train, pred_train = predict_linear(train, params)
    log_pred, pred = predict_linear(valid, params)
    rows.extend([
        {"model": "numpy_ols_log_area_time", "evaluation": label, "fold": fold, "partition": "fit",
         **score(y_train, log_train, pred_train)},
        {"model": "numpy_ols_log_area_time", "evaluation": label, "fold": fold, "partition": "validation",
         **score(y, log_pred, pred)},
    ])
    if label == "validation":
        pred_rows = []
        for model, pred_log, actual, prediction in [
            ("train_median", np.log1p(median_pred), y, median_pred),
            ("numpy_ols_log_area_time", log_pred, y, pred),
        ]:
            pred_rows.append(pd.DataFrame({
                "model": model,
                "published_at": valid["published_at"].to_numpy(),
                "actual_price": actual,
                "predicted_price": prediction,
                "residual_log": np.log1p(actual) - pred_log,
            }))
        pd.concat(pred_rows, ignore_index=True).to_csv(PREDICTIONS, index=False)
    return rows


def main() -> None:
    """Chạy baseline trên validation và 5 fold CV; không đọc test."""
    train = pd.read_parquet(SPLITS / "train.parquet")
    validation = pd.read_parquet(SPLITS / "validation.parquet")
    fold_specs = pd.read_csv(FOLDS)
    rows = evaluate_pair(train, validation, "validation", None)
    for fold in fold_specs.itertuples():
        fit = train.iloc[:fold.fit_end_row]
        valid = train.iloc[fold.validation_start_row:fold.validation_end_row]
        rows.extend(evaluate_pair(fit, valid, "time_series_cv", int(fold.fold)))
    results = pd.DataFrame(rows)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS, index=False)

    metrics = ["rmse_vnd", "mae_vnd", "mape_pct", "r2_vnd", "rmse_log", "mae_log", "r2_log"]
    validation_rows = results[(results["evaluation"] == "validation") & (results["partition"] == "validation")].set_index("model")
    cv = results[results["evaluation"] == "time_series_cv"]
    table = []
    for model in results["model"].unique():
        val = validation_rows.loc[model]
        cv_model = cv[(cv["model"] == model) & (cv["partition"] == "validation")]
        table.append((model, val, cv_model))
    def fmt(metric: str, value: float) -> str:
        if metric in {"mape_pct", "r2_vnd", "r2_log"}:
            return f"{value:.3f}"
        if metric.endswith("_vnd"):
            return f"{value:,.0f}"
        return f"{value:.3f}"

    val_table = "\n".join(
        "| " + model + " | " + " | ".join(fmt(m, float(val[m])) for m in metrics) + " |"
        for model, val, _ in table
    )
    cv_table = "\n".join(
        "| " + model + " | " + " | ".join(
            f"{fmt(m, float(cv_model[m].mean()))} ± {fmt(m, float(cv_model[m].std(ddof=1)))}"
            for m in metrics
        ) + " |"
        for model, _, cv_model in table
    )
    headers = "| Mô hình | RMSE VND | MAE VND | MAPE % | R² VND | RMSE log | MAE log | R² log |\n|---|---:|---:|---:|---:|---:|---:|---:|"
    REPORT.write_text(f"""# Bước 4 — Baseline

## Cấu hình

- **Train median:** luôn dự đoán trung vị giá trên phần fit.
- **NumPy OLS:** hồi quy tuyến tính tự cài bằng `numpy.linalg.lstsq` trên `log1p(area)` và `months_since_start`; mục tiêu `log1p(price)`. Chỉ fit hệ số trên phần fit của mỗi fold.
- Đánh giá trên validation chính thức và 5 expanding-window fold trong train. Không đọc tập test.
- MAPE tính trên giá VND dương; metric có thể bị chi phối bởi các tin giá thấp.

## Kết quả

### Validation

{headers}
{val_table}

### Time-series CV (mean ± std)

{headers}
{cv_table}

Chi tiết từng fold: `reports/baseline_results.csv`.

### Khoảng cách fit–validation (CV, log scale)

{chr(10).join(f"- `{model}`: Δ RMSE log = {cv[(cv.model == model) & (cv.partition == 'validation')].rmse_log.mean() - cv[(cv.model == model) & (cv.partition == 'fit')].rmse_log.mean():.3f}; Δ MAE log = {cv[(cv.model == model) & (cv.partition == 'validation')].mae_log.mean() - cv[(cv.model == model) & (cv.partition == 'fit')].mae_log.mean():.3f}." for model in results['model'].unique())}

## Nhận xét

- Đây là mốc đối chiếu; OLS chỉ dùng biến số, không có vị trí/biến phân loại.
- Chưa chọn mô hình từ kết quả baseline và chưa đánh giá test.
- OLS chỉ dùng diện tích và thời gian để giữ baseline gọn; metric VND được tính sau khi đổi ngược bằng `expm1`.
""", encoding="utf-8")
    print(f"Wrote {len(results)} baseline evaluation rows; test untouched.")


if __name__ == "__main__":
    main()
