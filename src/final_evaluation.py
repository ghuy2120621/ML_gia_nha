"""Fit ứng viên đã khóa trên train+validation và chấm test đúng một lần."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .baseline import score
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, make_preprocessor
from .models.ridge_lasso import CONFIGS, ITERATIONS, TOLERANCE, fit_numpy, predict


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "processed" / "splits"
REPORTS = ROOT / "reports"
ARTIFACT = ROOT / "models" / "vietnam_house_lasso.pkl"


def main() -> None:
    train = pd.read_parquet(SPLITS / "train.parquet")
    validation = pd.read_parquet(SPLITS / "validation.parquet")
    fit = pd.concat([train, validation], ignore_index=True)
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    prep = make_preprocessor()
    x_fit = prep.fit_transform(fit[columns])
    y_fit = np.log1p(fit[TARGET].to_numpy(dtype=np.float64))
    weights, iterations, change = fit_numpy(x_fit, y_fit, CONFIGS["Lasso"], "Lasso")

    # Chỉ mở test tại điểm đánh giá cuối, sau khi cấu hình Lasso đã khóa.
    test = pd.read_parquet(SPLITS / "test.parquet")
    x_test = prep.transform(test[columns])
    log_prediction = predict(x_test, weights)
    prediction = np.maximum(np.expm1(log_prediction), 0.0)
    actual = test[TARGET].to_numpy(dtype=np.float64)
    metrics = score(actual, log_prediction, prediction)

    pred = pd.DataFrame({
        "published_at": test["published_at"].astype("string").to_numpy(),
        "actual_price": actual,
        "predicted_price": prediction,
        "actual_log_price": np.log1p(actual),
        "residual_log": np.log1p(actual) - log_prediction,
    })
    pred["month"] = pred["published_at"].str.slice(0, 7)
    pred["price_band"] = pd.qcut(pred.actual_price, 4, duplicates="drop")
    pred.to_csv(REPORTS / "test_predictions.csv", index=False)
    monthly = pred.groupby("month", observed=True).agg(
        rows=("actual_price", "size"), mean_log_residual=("residual_log", "mean"),
        mae_vnd=("actual_price", lambda y: np.mean(np.abs(y - pred.loc[y.index, "predicted_price"]))),
    )
    bands = pred.groupby("price_band", observed=True).agg(
        rows=("actual_price", "size"),
        mae_log=("residual_log", lambda x: np.mean(np.abs(x))),
        mean_log_residual=("residual_log", "mean"),
    )
    monthly.to_csv(REPORTS / "test_residuals_by_month.csv")
    bands.to_csv(REPORTS / "test_residuals_by_price_band.csv")

    names = prep.get_feature_names_out()
    importance = pd.DataFrame({"feature": names, "coefficient": weights[1:]})
    importance["abs_coefficient"] = importance.coefficient.abs()
    importance.sort_values("abs_coefficient", ascending=False).to_csv(
        REPORTS / "test_feature_importance.csv", index=False
    )
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT.open("wb") as file:
        pickle.dump({"preprocessor": prep, "weights": weights, "model": "self_implemented_lasso",
                     "alpha": CONFIGS["Lasso"], "iterations": ITERATIONS,
                     "tolerance": TOLERANCE, "features": columns}, file)

    metric_table = "\n".join(f"| {key} | {value:,.4f} |" for key, value in metrics.items())
    monthly_table = "| Tháng | Số tin | Residual log TB | MAE VND |\n|---|---:|---:|---:|\n" + "\n".join(
        f"| {month} | {row.rows:,} | {row.mean_log_residual:.3f} | {row.mae_vnd:,.0f} |"
        for month, row in monthly.iterrows()
    )
    band_table = "| Khoảng giá thực | Số tin | MAE log | Residual log TB |\n|---|---:|---:|---:|\n" + "\n".join(
        f"| {band} | {row.rows:,} | {row.mae_log:.3f} | {row.mean_log_residual:.3f} |"
        for band, row in bands.iterrows()
    )
    top = "| Đặc trưng | Hệ số | Trị tuyệt đối |\n|---|---:|---:|\n" + "\n".join(
        f"| {row.feature} | {row.coefficient:.5f} | {row.abs_coefficient:.5f} |"
        for row in importance.head(15).itertuples()
    )
    (REPORTS / "10_final_test.md").write_text(f"""# Bước 9 — Đánh giá cuối trên test

Ứng viên TN1 Lasso và siêu tham số đã được chọn bằng train/CV/validation trước khi mở test. Fit lại pipeline trên train+validation ({len(fit):,} dòng), sau đó đọc và chấm test một lần ({len(test):,} dòng; {test.published_at.min()} đến {test.published_at.max()}). Không điều chỉnh mô hình theo kết quả test.

## Metric test

| Metric | Giá trị |
|---|---:|
{metric_table}

## Phần dư theo thời gian

Residual log = log giá thực − log giá dự đoán; dương nghĩa là dự đoán thấp.

{monthly_table}

## Phần dư theo tứ phân vị giá thực

{band_table}

## 15 hệ số có trị tuyệt đối lớn nhất

Hệ số được tính trên đặc trưng đã chuẩn hóa/mã hóa; đây là độ lớn liên hệ trong mô hình tuyến tính, không phải quan hệ nhân quả.

{top}

## Hạn chế

- FISTA dừng ở {iterations} vòng, relative change {change:.6g} (ngưỡng {TOLERANCE}); chưa đạt ngưỡng hội tụ nên hệ số và metric có thể còn cải thiện nếu tối ưu tiếp, nhưng không chỉnh sau khi xem test.
- Giá là giá niêm yết trong tin, không phải giá giao dịch hoàn tất. Phạm vi thời gian dữ liệu ngắn, nên kết quả không chứng minh khả năng dự báo qua chu kỳ dài.
- RMSE giá gốc nhạy với tin giá cực đoan/nhập sai; cần đọc cùng MAE và metric log.

Dự đoán lưu tại `reports/test_predictions.csv`; bảng phần dư theo tháng/giá và hệ số ở cùng thư mục. Mô hình và tiền xử lý lưu tại `models/vietnam_house_lasso.pkl`.
""", encoding="utf-8")
    print(f"Scored held-out test once; wrote metrics, diagnostics and {ARTIFACT}.")


if __name__ == "__main__":
    main()
