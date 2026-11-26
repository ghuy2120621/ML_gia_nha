"""Residual diagnostics cho ứng viên Lasso trên validation (không đọc test)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .baseline import score
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from .features import make_preprocessor
from .models.ridge_lasso import CONFIGS, TOLERANCE, fit_numpy, predict


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "processed" / "splits"
FIGURE = ROOT / "reports" / "figures" / "validation_residuals_lasso.png"
PREDICTIONS = ROOT / "reports" / "validation_ridge_lasso_residuals.csv"
REPORT = ROOT / "reports" / "08_validation_residuals.md"


def main() -> None:
    train = pd.read_parquet(SPLITS / "train.parquet")
    valid = pd.read_parquet(SPLITS / "validation.parquet")
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    prep = make_preprocessor()
    x_train = prep.fit_transform(train[columns])
    x_valid = prep.transform(valid[columns])
    y_train = np.log1p(train[TARGET].to_numpy(dtype=np.float64))
    actual = valid[TARGET].to_numpy(dtype=np.float64)
    log_actual = np.log1p(actual)
    predictions, summaries = [], {}
    for model, alpha in CONFIGS.items():
        weights, iterations, relative_change = fit_numpy(x_train, y_train, alpha, model)
        log_pred = predict(x_valid, weights)
        predicted = np.expm1(log_pred)
        pred = pd.DataFrame({
            "model": model,
            "published_at": valid["published_at"].to_numpy(),
            "actual_price": actual,
            "predicted_price": predicted,
            "residual_vnd": actual - predicted,
            "actual_log_price": log_actual,
            "predicted_log_price": log_pred,
            "residual_log": log_actual - log_pred,
        })
        predictions.append(pred)
        summaries[model] = {
            **score(actual, log_pred, predicted),
            "iterations": iterations,
            "relative_change": relative_change,
            "mean_log_residual": float((log_actual - log_pred).mean()),
            "corr_residual_actual_log": float(np.corrcoef(log_actual, log_actual - log_pred)[0, 1]),
        }
    pd.concat(predictions, ignore_index=True).to_csv(PREDICTIONS, index=False)
    pred = next(frame for frame in predictions if frame["model"].iloc[0] == "Lasso")
    log_pred = pred["predicted_log_price"].to_numpy()
    residual_log = pred["residual_log"].to_numpy()
    residual_vnd = pred["residual_vnd"].to_numpy()
    metrics = summaries["Lasso"]
    iterations = metrics["iterations"]
    relative_change = metrics["relative_change"]
    month = valid["published_at"].astype("string").str.slice(0, 7)
    by_month = pred.assign(month=month).groupby("month").agg(
        rows=("actual_price", "size"), mean_log_residual=("residual_log", "mean"),
        median_log_residual=("residual_log", "median"), mae_vnd=("residual_vnd", lambda x: np.abs(x).mean()),
    )
    by_month.to_csv(ROOT / "reports" / "validation_monthly_residuals.csv")
    pred["price_bin"] = pd.qcut(pred["actual_price"], q=4, duplicates="drop")
    by_price = pred.groupby("price_bin", observed=True).agg(
        rows=("actual_price", "size"), mae_log=("residual_log", lambda x: np.abs(x).mean()),
        mean_log_residual=("residual_log", "mean"), mae_vnd=("residual_vnd", lambda x: np.abs(x).mean()),
    )
    by_price.to_csv(ROOT / "reports" / "validation_price_band_residuals.csv")

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(pred), size=min(10_000, len(pred)), replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].scatter(pred.iloc[sample_idx]["actual_log_price"], pred.iloc[sample_idx]["predicted_log_price"], s=5, alpha=0.2)
    limits = [min(log_actual.min(), log_pred.min()), max(log_actual.max(), log_pred.max())]
    axes[0, 0].plot(limits, limits, color="black", linewidth=1)
    axes[0, 0].set(title="Giá log thực tế và dự đoán (mẫu 10k)", xlabel="log1p giá thực", ylabel="log1p giá dự đoán")
    axes[0, 1].scatter(pred.iloc[sample_idx]["predicted_log_price"], pred.iloc[sample_idx]["residual_log"], s=5, alpha=0.2)
    axes[0, 1].axhline(0, color="black", linewidth=1)
    axes[0, 1].set(title="Phần dư log theo giá dự đoán", xlabel="log1p giá dự đoán", ylabel="log dư (thực − dự đoán)")
    axes[1, 0].plot(by_month.index, by_month["mean_log_residual"], marker="o", label="Trung bình")
    axes[1, 0].plot(by_month.index, by_month["median_log_residual"], marker="o", label="Trung vị")
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].tick_params(axis="x", rotation=30)
    axes[1, 0].set(title="Phần dư log theo tháng đăng", ylabel="Thực − dự đoán")
    axes[1, 0].legend()
    axes[1, 1].bar(range(len(by_price)), by_price["mae_log"])
    axes[1, 1].set(title="MAE log theo tứ phân vị giá thực", xlabel="Nhóm giá (thấp → cao)", ylabel="MAE log")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=160)
    plt.close(fig)

    monthly_table = "\n".join(
        f"| {month} | {int(row.rows):,} | {row.mean_log_residual:.3f} | {row.median_log_residual:.3f} |"
        for month, row in by_month.iterrows()
    )
    price_table = "\n".join(
        f"| {str(group)} | {int(row.rows):,} | {row.mae_log:.3f} | {row.mean_log_residual:.3f} |"
        for group, row in by_price.iterrows()
    )
    REPORT.write_text(f"""# Validation residuals — Lasso candidate

- Fit preprocessing/model on train only; validation is the later period 2025-12-17–2026-02-08. Test is not read.
- Metrics: RMSE VND **{metrics['rmse_vnd']:,.0f}**, MAE VND **{metrics['mae_vnd']:,.0f}**, MAPE **{metrics['mape_pct']:.2f}%**, R² VND **{metrics['r2_vnd']:.3f}**; RMSE log **{metrics['rmse_log']:.3f}**, MAE log **{metrics['mae_log']:.3f}**, R² log **{metrics['r2_log']:.3f}**.
- FISTA stopped at {iterations} iterations; final relative change **{relative_change:.5f}**, convergence threshold {TOLERANCE}.
- Mean log residual (actual − predicted): **{residual_log.mean():.3f}**; correlation with actual log-price: **{np.corrcoef(log_actual, residual_log)[0,1]:.3f}**. Positive residual means underprediction.

## Residual by month

| Month | Tin | Mean log residual | Median log residual |
|---|---:|---:|---:|
{monthly_table}

## Residual by actual price band

| Price band (VND) | Tin | MAE log | Mean log residual |
|---|---:|---:|---:|
{price_table}

![Residual diagnostics](figures/validation_residuals_lasso.png)

Ridge/Lasso validation predictions are saved in `reports/validation_ridge_lasso_residuals.csv`. Price/month aggregates above describe Lasso. Its corresponding metrics are compared with Ridge below.

- Ridge validation: RMSE log **{summaries['Ridge']['rmse_log']:.3f}**, MAE log **{summaries['Ridge']['mae_log']:.3f}**, MAPE **{summaries['Ridge']['mape_pct']:.2f}%**, mean residual **{summaries['Ridge']['mean_log_residual']:.3f}**, corr(residual, actual log price) **{summaries['Ridge']['corr_residual_actual_log']:.3f}**.
- Lasso validation: RMSE log **{summaries['Lasso']['rmse_log']:.3f}**, MAE log **{summaries['Lasso']['mae_log']:.3f}**, MAPE **{summaries['Lasso']['mape_pct']:.2f}%**, mean residual **{summaries['Lasso']['mean_log_residual']:.3f}**, corr(residual, actual log price) **{summaries['Lasso']['corr_residual_actual_log']:.3f}**.
""", encoding="utf-8")
    print(f"Wrote validation residual diagnostics for {len(valid):,} rows; test untouched.")


if __name__ == "__main__":
    main()
