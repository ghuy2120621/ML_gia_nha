"""Đánh giá cấu hình TN1 tối đa 1.000 vòng trên train/CV/validation, không đọc test."""

from pathlib import Path
import time

import pandas as pd

from .ridge_lasso import ROOT, SPLITS, evaluate


ITERATIONS = 1000
RESULTS = ROOT / "reports" / "lasso_1000_results.csv"
REPORT = ROOT / "reports" / "12_lasso_1000_validation.md"


def main() -> None:
    start = time.perf_counter()
    train = pd.read_parquet(SPLITS / "train.parquet")
    validation = pd.read_parquet(SPLITS / "validation.parquet")
    folds = pd.read_csv(SPLITS / "cv_folds.csv")
    rows = evaluate(train, validation, "validation", None, ITERATIONS)
    for fold in folds.itertuples():
        fit = train.iloc[:fold.fit_end_row]
        valid = train.iloc[fold.validation_start_row:fold.validation_end_row]
        rows.extend(evaluate(fit, valid, "time_series_cv", int(fold.fold), ITERATIONS))

    results = pd.DataFrame(rows)
    assert len(results) == 24 and results["fold"].isna().sum() == 4
    results.to_csv(RESULTS, index=False)
    previous = pd.read_csv(ROOT / "reports" / "exp1_results.csv")
    elapsed = time.perf_counter() - start

    def metric_table(frame: pd.DataFrame, evaluation: str, partition: str) -> str:
        part = frame[(frame.evaluation == evaluation) & (frame.partition == partition)]
        rows_md = []
        for model in ["Ridge", "Lasso"]:
            vals = part[part.model == model]
            if evaluation == "time_series_cv":
                fmt = lambda key: f"{vals[key].mean():.4f} ± {vals[key].std(ddof=1):.4f}"
            else:
                fmt = lambda key: f"{vals.iloc[0][key]:.4f}"
            rows_md.append(f"| {model} | {fmt('rmse_log')} | {fmt('mae_log')} | "
                           f"{fmt('mape_pct')} | {int(vals.iterations.max())} | "
                           f"{vals.relative_change.max():.6g} |")
        return "\n".join(rows_md)

    report = f"""# TN1 — ứng viên giới hạn 1.000 vòng

Đánh giá lại trên train, năm expanding-window fold trong train và validation chính thức. Preprocessing được fit lại trên fit slice tương ứng. Test không được đọc. Thời gian chạy tổng: {elapsed:.1f} giây.

## Validation

| Mô hình | RMSE log | MAE log | MAPE % | Vòng thực tế | Relative step cuối |
|---|---:|---:|---:|---:|---:|
{metric_table(results, 'validation', 'validation')}

## Time-series CV (mean ± std)

| Mô hình | RMSE log | MAE log | MAPE % | Vòng thực tế tối đa | Relative step lớn nhất |
|---|---:|---:|---:|---:|---:|
{metric_table(results, 'time_series_cv', 'validation')}

## So sánh Lasso với cấu hình 200 vòng

| Tập | Cấu hình | RMSE log | MAE log | MAPE % | Relative step |
|---|---|---:|---:|---:|---:|
"""
    for evaluation, partition, label in [
        ("validation", "validation", "validation"),
        ("time_series_cv", "validation", "CV validation mean"),
    ]:
        for frame, config in [(previous, "200 vòng"), (results, "tối đa 1.000 vòng")]:
            subset = frame[(frame.model == "Lasso") & (frame.evaluation == evaluation)
                           & (frame.partition == partition)]
            if evaluation == "time_series_cv":
                rmse, mae, mape = (subset[key].mean() for key in ["rmse_log", "mae_log", "mape_pct"])
            else:
                rmse, mae, mape = (subset.iloc[0][key] for key in ["rmse_log", "mae_log", "mape_pct"])
            report += f"| {label} | {config} | {rmse:.4f} | {mae:.4f} | {mape:.2f} | {subset.relative_change.max():.6g} |\n"
    report += """
## Diễn giải

Lasso đạt tolerance trước 1.000 vòng. Tuy vậy, cấu hình 200 vòng tốt hơn nhẹ ở validation và cả năm CV validation window (RMSE log 0,5087 so với 0,5098 trên validation; CV mean 0,5288 so với 0,5300). Điều này phù hợp với early stopping đang có tác dụng regularization. Giữ artifact 200 vòng làm kết quả đã chấm; ghi rõ đây là lựa chọn theo validation/CV, còn optimizer chưa hội tụ theo tolerance.

Kết quả 1.000 vòng không thay báo cáo đánh giá cuối và không dùng metric test cũ để chọn cấu hình. Chi tiết từng fold lưu tại `reports/lasso_1000_results.csv`.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(f"Wrote 1,000-iteration candidate results; test untouched ({elapsed:.1f}s).")


if __name__ == "__main__":
    main()
