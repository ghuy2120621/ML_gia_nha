"""Đo đường học Lasso theo số vòng chỉ trên train/CV/validation."""

from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd

from ..baseline import score
from ..features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, make_preprocessor
from .ridge_lasso import ROOT, SPLITS, add_intercept, lipschitz_constant


ALPHA = 0.001
CHECKPOINTS = (50, 100, 150, 200, 300, 500, 750, 1000)


def evaluate_path(train: pd.DataFrame, valid: pd.DataFrame, fold: str, alpha: float) -> list[dict]:
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    prep = make_preprocessor()
    x_fit = add_intercept(prep.fit_transform(train[columns]))
    x_valid = add_intercept(prep.transform(valid[columns]))
    y_fit = np.log1p(train[TARGET].to_numpy(dtype=np.float64))
    y_valid = valid[TARGET].to_numpy(dtype=np.float64)
    n, width = x_fit.shape
    lipschitz = lipschitz_constant(x_fit)
    weights = np.zeros(width)
    accelerated = weights.copy()
    momentum = 1.0
    rows = []
    for iteration in range(1, CHECKPOINTS[-1] + 1):
        residual = x_fit @ accelerated - y_fit
        gradient = np.asarray(x_fit.T @ residual).ravel() / n
        updated = accelerated - gradient / lipschitz
        updated[1:] = np.sign(updated[1:]) * np.maximum(
            np.abs(updated[1:]) - alpha / lipschitz, 0.0
        )
        relative_change = float(
            np.linalg.norm(updated - weights) / (np.linalg.norm(weights) + 1e-12)
        )
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        accelerated = updated + ((momentum - 1) / next_momentum) * (updated - weights)
        weights, momentum = updated, next_momentum
        if iteration not in CHECKPOINTS:
            continue
        for partition, actual, x in [
            ("fit", np.expm1(y_fit), x_fit),
            ("validation", y_valid, x_valid),
        ]:
            log_prediction = np.asarray(x @ weights).ravel()
            rows.append({
                "fold": fold,
                "partition": partition,
                "iterations": iteration,
                "relative_change": relative_change,
                **score(actual, log_prediction, np.maximum(np.expm1(log_prediction), 0.0)),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Đo đường học Lasso theo số vòng.")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    alpha = parser.parse_args().alpha
    label = f"{alpha:g}".replace(".", "p")
    results_path = ROOT / "reports" / f"lasso_iteration_alpha_{label}.csv"
    report_path = ROOT / "reports" / f"15_lasso_iteration_alpha_{label}.md"
    start = time.perf_counter()
    train = pd.read_parquet(SPLITS / "train.parquet")
    validation = pd.read_parquet(SPLITS / "validation.parquet")
    folds = pd.read_csv(SPLITS / "cv_folds.csv")
    rows = evaluate_path(train, validation, "official_validation", alpha)
    for fold in folds.itertuples():
        rows.extend(evaluate_path(
            train.iloc[:fold.fit_end_row],
            train.iloc[fold.validation_start_row:fold.validation_end_row],
            f"cv_{fold.fold}", alpha,
        ))

    results = pd.DataFrame(rows)
    assert set(results.iterations.unique()) == set(CHECKPOINTS)
    assert len(results) == len(CHECKPOINTS) * 2 * 6
    results.to_csv(results_path, index=False)
    cv = results[(results.fold.str.startswith("cv_")) & (results.partition == "validation")]
    official = results[(results.fold == "official_validation") & (results.partition == "validation")]
    summary = cv.groupby("iterations").agg(
        rmse_mean=("rmse_log", "mean"), rmse_std=("rmse_log", "std"),
        mae_mean=("mae_log", "mean"), mae_std=("mae_log", "std"),
        relative_change_max=("relative_change", "max"),
    )
    best = int(summary.rmse_mean.idxmin())
    rows_md = []
    for iteration in CHECKPOINTS:
        cv_row, val_row = summary.loc[iteration], official[official.iterations == iteration].iloc[0]
        rows_md.append(
            f"| {iteration} | {cv_row.rmse_mean:.4f} ± {cv_row.rmse_std:.4f} | "
            f"{cv_row.mae_mean:.4f} ± {cv_row.mae_std:.4f} | {val_row.rmse_log:.4f} | "
            f"{val_row.mae_log:.4f} | {cv_row.relative_change_max:.6g} |"
        )
    elapsed = time.perf_counter() - start
    report_path.write_text(f"""# TN1 — chọn số vòng bằng đường học

Lasso alpha={alpha}, các đặc trưng và preprocessing giữ nguyên TN1. Mỗi fold fit preprocessing riêng; chạy một đường FISTA đến 1.000 vòng, ghi metric tại các checkpoint. Chỉ dùng train, 5 expanding-window CV folds và validation chính thức; test không được đọc. Thời gian chạy: {elapsed:.1f} giây.

| Vòng lặp | CV RMSE log mean ± std | CV MAE log mean ± std | Validation RMSE log | Validation MAE log | Relative step CV lớn nhất |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows_md)}

Số vòng có CV RMSE log trung bình thấp nhất: **{best}**. Đây là lựa chọn theo CV; validation được báo cáo như kiểm tra chuyển tiếp. Kết quả sweep không thay artifact cuối hoặc báo cáo test hiện tại. Không mở test để so sánh các checkpoint.

Chi tiết từng fold/checkpoint: `{results_path.relative_to(ROOT).as_posix()}`.
""", encoding="utf-8")
    print(f"Wrote Lasso iteration sweep for {len(results)} rows; test untouched ({elapsed:.1f}s).")


if __name__ == "__main__":
    main()
