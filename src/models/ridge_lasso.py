"""Ridge/Lasso tự cài bằng proximal gradient trên ma trận thưa."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from ..baseline import score
from ..features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, make_preprocessor


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "processed" / "splits"
RESULTS = ROOT / "reports" / "exp1_results.csv"
REPORT = ROOT / "reports" / "05_exp1_ridge_lasso.md"
SEED = 42
ITERATIONS = 200
TOLERANCE = 1e-4
CONFIGS = {"Ridge": 0.01, "Lasso": 0.001}


def add_intercept(x):
    """Thêm cột intercept vào ma trận CSR."""
    x = sparse.csr_matrix(x, dtype=np.float64)
    return sparse.hstack((np.ones((x.shape[0], 1)), x), format="csr")


def lipschitz_constant(x, iterations: int = 15) -> float:
    """Ước lượng trị riêng lớn nhất của X'X/n bằng power iteration."""
    rng = np.random.default_rng(SEED)
    vector = rng.normal(size=x.shape[1])
    vector /= np.linalg.norm(vector)
    for _ in range(iterations):
        next_vector = np.asarray(x.T @ (x @ vector)).ravel() / x.shape[0]
        norm = np.linalg.norm(next_vector)
        if norm == 0:
            return 1.0
        vector = next_vector / norm
    return max(float(vector @ (x.T @ (x @ vector)).ravel() / x.shape[0]), 1e-8)


def fit_numpy(x, y: np.ndarray, alpha: float, penalty: str) -> tuple[np.ndarray, int, float]:
    """Fit bình phương tối thiểu có regularization bằng FISTA."""
    x = add_intercept(x)
    y = np.asarray(y, dtype=np.float64)
    n, width = x.shape
    penalty_mask = np.ones(width)
    penalty_mask[0] = 0.0
    lipschitz = lipschitz_constant(x) + (alpha if penalty == "Ridge" else 0.0)
    weights = np.zeros(width)
    accelerated = weights.copy()
    momentum = 1.0
    relative_change = np.inf
    for iteration in range(1, ITERATIONS + 1):
        residual = x @ accelerated - y
        gradient = np.asarray(x.T @ residual).ravel() / n
        if penalty == "Ridge":
            gradient += alpha * accelerated * penalty_mask
        updated = accelerated - gradient / lipschitz
        if penalty == "Lasso":
            threshold = alpha / lipschitz
            updated[1:] = np.sign(updated[1:]) * np.maximum(np.abs(updated[1:]) - threshold, 0.0)
        relative_change = float(np.linalg.norm(updated - weights) / (np.linalg.norm(weights) + 1e-12))
        next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
        accelerated = updated + ((momentum - 1.0) / next_momentum) * (updated - weights)
        weights, momentum = updated, next_momentum
        if relative_change < TOLERANCE:
            break
    return weights, iteration, relative_change


def predict(x, weights: np.ndarray) -> np.ndarray:
    return np.asarray(add_intercept(x) @ weights).ravel()


def evaluate(train: pd.DataFrame, valid: pd.DataFrame, evaluation: str, fold: int | None) -> list[dict]:
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    y_train = np.log1p(train[TARGET].to_numpy(dtype=np.float64))
    y_valid = valid[TARGET].to_numpy(dtype=np.float64)
    prep = make_preprocessor()
    x_train = prep.fit_transform(train[columns])
    x_valid = prep.transform(valid[columns])
    rows = []
    for model, alpha in CONFIGS.items():
        weights, iterations, change = fit_numpy(x_train, y_train, alpha, model)
        log_train = predict(x_train, weights)
        log_valid = predict(x_valid, weights)
        rows.extend([
            {"model": model, "evaluation": evaluation, "fold": fold, "partition": "fit",
             "iterations": iterations, "relative_change": change,
             **score(np.expm1(y_train), log_train, np.expm1(log_train))},
            {"model": model, "evaluation": evaluation, "fold": fold, "partition": "validation",
             "iterations": iterations, "relative_change": change,
             **score(y_valid, log_valid, np.expm1(log_valid))},
        ])
    return rows


def main() -> None:
    """Chạy TN1 trên validation và CV; không dùng test."""
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

    tables = []
    for partition in ["fit", "validation"]:
        subset = results[(results.evaluation == "validation") & (results.partition == partition)]
        body = []
        for row in subset.itertuples():
            values = [getattr(row, metric) for metric in metrics]
            body.append("| " + row.model + " | " + " | ".join(fmt(m, v) for m, v in zip(metrics, values)) + " |")
        tables.append(f"### Validation run — {partition}\n\n| Mô hình | RMSE VND | MAE VND | MAPE % | R² VND | RMSE log | MAE log | R² log |\n|---|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(body))
    cv = results[results.evaluation == "time_series_cv"]
    cv_body = []
    for model in CONFIGS:
        for partition in ["fit", "validation"]:
            part = cv[(cv.model == model) & (cv.partition == partition)]
            values = [f"{fmt(m, part[m].mean())} ± {fmt(m, part[m].std(ddof=1))}" for m in metrics]
            cv_body.append(f"| {model} | {partition} | " + " | ".join(values) + " |")
    cv_table = "| Mô hình | Phần | RMSE VND | MAE VND | MAPE % | R² VND | RMSE log | MAE log | R² log |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(cv_body)
    convergence = results[results.evaluation == "time_series_cv"].groupby("model")["iterations"].agg(["min", "max"]).to_dict("index")
    REPORT.write_text(f"""# Bước 5 — TN1: Ridge/Lasso tự cài

## Giả thuyết và cấu hình

- Giả thuyết: regularization giúp mô hình tuyến tính dùng đặc trưng số và one-hot ổn định hơn baseline ít đặc trưng.
- Đầu vào: numeric đã median-impute/scale và categorical đã impute/one-hot bằng `make_preprocessor`; mỗi lần chỉ fit preprocessing trên phần fit tương ứng.
- Mục tiêu: `log1p(price)`; optimizer: FISTA tự viết với phép nhân CSR và vector NumPy; intercept không bị phạt.
- Cấu hình: clip numeric tại P0.5–P99.5 (ranh giới fit-only); Ridge alpha={CONFIGS['Ridge']}, Lasso alpha={CONFIGS['Lasso']}, tối đa {ITERATIONS} vòng lặp, tolerance={TOLERANCE}, seed={SEED} cho power iteration.
- Mỗi fold được đo trên fit và validation để quan sát khoảng cách tổng quát hóa. Test không được đọc.

{tables[0]}

{tables[1]}

### Time-series CV (mean ± std)

{cv_table}

Số vòng lặp thực chạy theo mô hình (min–max trên CV): `{convergence}`. Chi tiết fold trong `reports/exp1_results.csv`.

## Diễn giải

- So sánh với baseline trong `reports/04_baseline.md`; ưu tiên metric log cùng MAE/RMSE trên VND vì RMSE giá gốc rất nhạy với ngoại lai.
- Nếu đạt giới hạn {ITERATIONS} vòng trước tolerance {TOLERANCE}, xem là cấu hình giới hạn vòng lặp và không khẳng định nghiệm tối ưu.
- Kết luận giữ/loại cần dựa trên validation/CV và sẽ được ghi ở bảng so sánh; chưa dùng test.
""", encoding="utf-8")
    print(f"Wrote {len(results)} TN1 result rows; test untouched.")


if __name__ == "__main__":
    main()
