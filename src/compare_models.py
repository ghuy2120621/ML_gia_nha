"""Tổng hợp metric, độ ổn định, residual và rubric trọng số; tuyệt đối không đọc test."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "model_comparison.csv"
REPORT = REPORTS / "09_model_comparison.md"
WEIGHTS = {"accuracy": 0.30, "stability": 0.25, "generalization": 0.20, "residual": 0.15, "interpretability": 0.10}
COMPLEXITY = {
    "train_median": 5,
    "numpy_ols_log_area_time": 5,
    "Ridge": 4,
    "Lasso": 4,
    "numpy_mlp": 2,
    "autoencoder_ridge_head": 2,
}
NAMES = {
    "train_median": "Median baseline",
    "numpy_ols_log_area_time": "NumPy OLS",
    "Ridge": "TN1 Ridge",
    "Lasso": "TN1 Lasso",
    "numpy_mlp": "TN2 NumPy MLP",
    "autoencoder_ridge_head": "TN3 Autoencoder + Ridge",
}


def rank_score(values: pd.Series, lower_is_better: bool) -> pd.Series:
    rank = values.rank(method="average", ascending=lower_is_better)
    if len(values) == 1:
        return pd.Series(3.0, index=values.index)
    return 5.0 - (rank - 1.0) * (4.0 / (len(values) - 1.0))


def load_results() -> pd.DataFrame:
    baseline = pd.read_csv(REPORTS / "baseline_results.csv")
    exp1 = pd.read_csv(REPORTS / "exp1_results.csv")
    exp2 = pd.read_csv(REPORTS / "exp2_results.csv")
    exp3 = pd.read_csv(REPORTS / "exp3_results.csv")
    exp2["partition"] = exp2["partition"].replace({"fit_sample": "fit", "outer_validation": "validation"})
    exp3["partition"] = exp3["partition"].replace({"fit_sample": "fit", "outer_validation": "validation"})
    return pd.concat([baseline, exp1, exp2, exp3], ignore_index=True, sort=False)


def load_residuals() -> pd.DataFrame:
    baseline = pd.read_csv(REPORTS / "baseline_validation_residuals.csv")
    baseline["model"] = baseline["model"].replace({
        "train_median": "train_median", "numpy_ols_log_area_time": "numpy_ols_log_area_time",
    })
    ridge_lasso = pd.read_csv(REPORTS / "validation_ridge_lasso_residuals.csv")
    mlp = pd.read_csv(REPORTS / "mlp_validation_residuals.csv")
    mlp["model"] = "numpy_mlp"
    hybrid = pd.read_csv(REPORTS / "hybrid_validation_residuals.csv")
    hybrid["model"] = "autoencoder_ridge_head"
    parts = [baseline, ridge_lasso, mlp, hybrid]
    for part in parts:
        if "actual_log_price" not in part:
            part["actual_log_price"] = np.log1p(part["actual_price"])
        if "residual_log" not in part:
            part["residual_log"] = part["actual_log_price"] - np.log1p(part["predicted_price"])
    return pd.concat(parts, ignore_index=True, sort=False)


def main() -> None:
    results = load_results()
    residuals = load_residuals()
    models = list(COMPLEXITY)
    records = []
    for model in models:
        val = results[(results.model == model) & (results.evaluation == "validation") & (results.partition == "validation")]
        cv_val = results[(results.model == model) & (results.evaluation == "time_series_cv") & (results.partition == "validation")]
        cv_fit = results[(results.model == model) & (results.evaluation == "time_series_cv") & (results.partition == "fit")]
        if val.empty or cv_val.empty or cv_fit.empty:
            raise ValueError(f"Thiếu kết quả cần so sánh cho {model}.")
        residual = residuals[residuals.model == model].copy()
        residual["month"] = residual["published_at"].astype("string").str.slice(0, 7)
        monthly_bias = residual.groupby("month")["residual_log"].mean()
        records.append({
            "model": model,
            **{f"val_{key}": float(val.iloc[0][key]) for key in ["rmse_vnd", "mae_vnd", "mape_pct", "r2_vnd", "rmse_log", "mae_log", "r2_log"]},
            **{f"cv_{key}_mean": float(cv_val[key].mean()) for key in ["rmse_vnd", "mae_vnd", "mape_pct", "r2_vnd", "rmse_log", "mae_log", "r2_log"]},
            **{f"cv_{key}_std": float(cv_val[key].std(ddof=1)) for key in ["rmse_vnd", "mae_vnd", "mape_pct", "r2_vnd", "rmse_log", "mae_log", "r2_log"]},
            "generalization_gap_rmse_log": float(cv_val.rmse_log.mean() - cv_fit.rmse_log.mean()),
            "generalization_gap_mae_log": float(cv_val.mae_log.mean() - cv_fit.mae_log.mean()),
            "residual_mean_log": float(residual.residual_log.mean()),
            "residual_corr_actual_log": float(np.corrcoef(residual.actual_log_price, residual.residual_log)[0, 1]),
            "max_abs_monthly_log_bias": float(monthly_bias.abs().max()),
            "interpretability_score": COMPLEXITY[model],
        })
    table = pd.DataFrame(records).set_index("model")

    accuracy_metrics = [
        ("val_rmse_log", True), ("val_mae_log", True),
        ("cv_rmse_log_mean", True), ("cv_mae_log_mean", True),
        ("val_mape_pct", True), ("cv_mape_pct_mean", True),
        ("val_mae_vnd", True), ("cv_mae_vnd_mean", True),
        ("val_r2_log", False), ("cv_r2_log_mean", False),
        ("val_r2_vnd", False), ("cv_r2_vnd_mean", False),
    ]
    accuracy_rank = pd.concat([rank_score(table[col], lower) for col, lower in accuracy_metrics], axis=1).mean(axis=1)
    stability_value = table[["cv_rmse_log_std", "cv_mae_log_std", "cv_mape_pct_std"]].rank(method="average").mean(axis=1)
    stability = 5.0 - (stability_value - 1.0) * (4.0 / (len(table) - 1.0))
    generalization_value = table[["generalization_gap_rmse_log", "generalization_gap_mae_log"]].abs().rank(method="average").mean(axis=1)
    generalization = 5.0 - (generalization_value - 1.0) * (4.0 / (len(table) - 1.0))
    residual_index = (
        table.residual_mean_log.abs() * 0.35
        + table.residual_corr_actual_log.abs() * 0.35
        + table.max_abs_monthly_log_bias * 0.30
    )
    residual = rank_score(residual_index, True)
    table["accuracy_score"] = accuracy_rank
    table["stability_score"] = stability
    table["generalization_score"] = generalization
    table["residual_score"] = residual
    table["weighted_score"] = sum(table[f"{name}_score"] * weight for name, weight in WEIGHTS.items())
    table["model_name"] = [NAMES[m] for m in table.index]
    table = table.sort_values("weighted_score", ascending=False)
    table.to_csv(OUTPUT)

    score_rows = "\n".join(
        f"| {row.model_name} | {row.accuracy_score:.1f} | {row.stability_score:.1f} | "
        f"{row.generalization_score:.1f} | {row.residual_score:.1f} | {row.interpretability_score:.1f} | {row.weighted_score:.2f} |"
        for row in table.itertuples()
    )
    metric_rows = "\n".join(
        f"| {NAMES[model]} | {row.val_rmse_log:.3f} | {row.val_mae_log:.3f} | {row.cv_rmse_log_mean:.3f} ± {row.cv_rmse_log_std:.3f} | "
        f"{row.cv_mae_log_mean:.3f} ± {row.cv_mae_log_std:.3f} | {row.val_mape_pct:.1f}% | {row.val_mae_vnd/1e9:.2f}B |"
        for model, row in table.iterrows()
    )
    winner = table.iloc[0]
    REPORT.write_text(f"""# Bước 8 — So sánh và chọn ứng viên

## Kết quả validation và time-series CV

| Mô hình | Val RMSE log | Val MAE log | CV RMSE log mean ± std | CV MAE log mean ± std | Val MAPE | Val MAE VND |
|---|---:|---:|---:|---:|---:|---:|
{metric_rows}

Các metric gốc đầy đủ, gồm RMSE/MAE/MAPE/R² trên VND và log, nằm trong `reports/model_comparison.csv`.

## Bảng điểm có trọng số

Điểm 1–5 là **thứ hạng tương đối giữa sáu mô hình**, không phải xác suất hay chứng nhận tuyệt đối. Accuracy xếp hạng đồng thời validation/CV log errors, MAPE, MAE VND và R²; Stability dùng độ lệch chuẩn CV; Generalization dùng khoảng cách fit–validation trên log metrics; Residual dùng độ lệch residual trung bình, tương quan residual với giá log thực và drift lớn nhất theo tháng. Complexity/interpretability được đánh giá thủ công theo cấu trúc mô hình.

| Mô hình | Accuracy 30% | Stability 25% | Generalization 20% | Residual 15% | Diễn giải 10% | Tổng |
|---|---:|---:|---:|---:|---:|---:|
{score_rows}

Tổng điểm cao nhất theo rubric hiện tại: **{winner.model_name} ({winner.weighted_score:.2f}/5)**.

## Residual và độ bền thời gian

Lasso có validation residual diagnostics ở `reports/08_validation_residuals.md` và hình `reports/figures/validation_residuals_lasso.png`. Residual có tương quan với giá thực; mô hình có xu hướng dự đoán thấp ở nhóm giá cao và tháng 02/2026. Đây là hạn chế đáng kể dù log errors tốt nhất.

CV bao phủ các cửa sổ 09–12/2025; validation chính thức phủ 12/2025–02/2026. Số liệu cho thấy khả năng chuyển tiếp trong giai đoạn ngắn của bộ tin đăng này, chưa xác nhận khả năng bền qua chu kỳ dài.

## Quyết định trước test

- **Ứng viên được chọn theo rubric:** {winner.model_name}. Chỉ sau khi khóa mô hình/cấu hình mới được dùng test một lần.
- TN1 Lasso có độ chính xác log tốt nhất; TN2 MLP gần hơn trong CV nhưng kém trên validation; TN3 hybrid kém accuracy và còn bỏ district/ward; baseline median/OLS yếu hơn về accuracy nhưng đơn giản.
- Lasso FISTA dừng ở 200 vòng, relative change cuối khoảng 0.00348, cao hơn tolerance 0.0001. Ứng viên được khóa theo cấu hình này; kết quả là nghiệm gần đúng, chưa hội tụ theo tiêu chí đã đặt.
- RMSE VND rất nhạy với một số giá cực cao; dataset có ít nhất một tin validation 67.999 nghìn tỷ VND cho diện tích 100 m². Cần giữ rõ kết quả thô và không loại các điểm này âm thầm.

**Test chưa được chấm trong bước so sánh này.**
""", encoding="utf-8")
    print(f"Wrote model comparison for {len(table)} candidates; test untouched.")


if __name__ == "__main__":
    main()
