"""Tìm alpha Lasso bằng CV thời gian, chỉ dùng train và validation."""

from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd

from ..baseline import score
from ..features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, make_preprocessor
from .ridge_lasso import ITERATIONS, ROOT, SPLITS, fit_numpy, predict


ALPHAS = (0.0001, 0.0003, 0.001, 0.003, 0.01)


def evaluate(train: pd.DataFrame, valid: pd.DataFrame, fold: str,
             alphas: tuple[float, ...]) -> list[dict]:
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    prep = make_preprocessor()
    x_train = prep.fit_transform(train[columns])
    x_valid = prep.transform(valid[columns])
    y_train = np.log1p(train[TARGET].to_numpy(dtype=np.float64))
    y_valid = valid[TARGET].to_numpy(dtype=np.float64)
    rows = []
    for alpha in alphas:
        weights, iterations, change = fit_numpy(x_train, y_train, alpha, "Lasso", ITERATIONS)
        for partition, actual, log_prediction in [
            ("fit", np.expm1(y_train), predict(x_train, weights)),
            ("validation", y_valid, predict(x_valid, weights)),
        ]:
            rows.append({
                "fold": fold,
                "alpha": alpha,
                "partition": partition,
                "iterations": iterations,
                "relative_change": change,
                **score(actual, log_prediction, np.maximum(np.expm1(log_prediction), 0.0)),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Tìm alpha Lasso bằng CV thời gian.")
    parser.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    alphas = tuple(parser.parse_args().alphas)
    refined = alphas != ALPHAS
    results_path = ROOT / "reports" / ("lasso_alpha_refinement.csv" if refined else "lasso_alpha_sweep.csv")
    report_path = ROOT / "reports" / ("16_lasso_alpha_refinement.md" if refined else "14_lasso_alpha_sweep.md")
    start = time.perf_counter()
    train = pd.read_parquet(SPLITS / "train.parquet")
    validation = pd.read_parquet(SPLITS / "validation.parquet")
    folds = pd.read_csv(SPLITS / "cv_folds.csv")
    rows = evaluate(train, validation, "official_validation", alphas)
    for fold in folds.itertuples():
        rows.extend(evaluate(
            train.iloc[:fold.fit_end_row],
            train.iloc[fold.validation_start_row:fold.validation_end_row],
            f"cv_{fold.fold}", alphas,
        ))
    results = pd.DataFrame(rows)
    assert len(results) == len(alphas) * 2 * 6
    assert set(results.alpha.unique()) == set(alphas)
    results.to_csv(results_path, index=False)
    cv = results[(results.fold.str.startswith("cv_")) & (results.partition == "validation")]
    official = results[(results.fold == "official_validation") & (results.partition == "validation")]
    summary = cv.groupby("alpha").agg(
        rmse_mean=("rmse_log", "mean"), rmse_std=("rmse_log", "std"),
        mae_mean=("mae_log", "mean"), mae_std=("mae_log", "std"),
        mape_mean=("mape_pct", "mean"), relative_change_max=("relative_change", "max"),
    )
    best = float(summary.rmse_mean.idxmin())
    val = official.set_index("alpha")
    table = "\n".join(
        f"| {alpha:g} | {summary.loc[alpha].rmse_mean:.4f} ± {summary.loc[alpha].rmse_std:.4f} | "
        f"{summary.loc[alpha].mae_mean:.4f} ± {summary.loc[alpha].mae_std:.4f} | "
        f"{summary.loc[alpha].mape_mean:.2f}% | {val.loc[alpha].rmse_log:.4f} | "
        f"{val.loc[alpha].mae_log:.4f} | {summary.loc[alpha].relative_change_max:.6g} |"
        for alpha in alphas
    )
    elapsed = time.perf_counter() - start
    report_path.write_text(f"""# TN1 — lưới alpha Lasso

Giữ preprocessing, đặc trưng và giới hạn {ITERATIONS} vòng như TN1; chỉ thay alpha. Chọn theo mean validation RMSE log trên 5 expanding-window fold trong train, sau đó xem validation chính thức. Không đọc test. Thời gian chạy: {elapsed:.1f} giây.

| Alpha | CV RMSE log mean ± std | CV MAE log mean ± std | CV MAPE mean | Val RMSE log | Val MAE log | Relative step CV lớn nhất |
|---:|---:|---:|---:|---:|---:|---:|
{table}

Alpha có CV RMSE log thấp nhất: **{best:g}**. Kết quả là ứng viên nghiên cứu trên CV/validation; không thay artifact đã chấm test. Nếu chọn cấu hình mới, cần đánh giá độc lập trên dữ liệu tương lai mới, không dùng lại test cũ.

Chi tiết từng fold và alpha: `{results_path.relative_to(ROOT).as_posix()}`.
""", encoding="utf-8")
    print(f"Wrote Lasso alpha sweep for {len(results)} rows; test untouched ({elapsed:.1f}s).")


if __name__ == "__main__":
    main()
