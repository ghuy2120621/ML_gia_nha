# Bước 4 — Baseline

## Cấu hình

- **Train median:** luôn dự đoán trung vị giá trên phần fit.
- **NumPy OLS:** hồi quy tuyến tính tự cài bằng `numpy.linalg.lstsq` trên `log1p(area)` và `months_since_start`; mục tiêu `log1p(price)`. Chỉ fit hệ số trên phần fit của mỗi fold.
- Đánh giá trên validation chính thức và 5 expanding-window fold trong train. Không đọc tập test.
- MAPE tính trên giá VND dương; metric có thể bị chi phối bởi các tin giá thấp.

## Kết quả

### Validation

| Mô hình | RMSE VND | MAE VND | MAPE % | R² VND | RMSE log | MAE log | R² log |
|---|---:|---:|---:|---:|---:|---:|---:|
| train_median | 163,673,933,373 | 11,531,631,259 | 68.890 | -0.003 | 0.851 | 0.611 | -0.009 |
| numpy_ols_log_area_time | 162,191,572,901 | 10,370,072,855 | 76.829 | 0.016 | 0.701 | 0.529 | 0.316 |

### Time-series CV (mean ± std)

| Mô hình | RMSE VND | MAE VND | MAPE % | R² VND | RMSE log | MAE log | R² log |
|---|---:|---:|---:|---:|---:|---:|---:|
| train_median | 3,241,933,614,435 ± 3,778,197,764,296 | 23,356,979,210 ± 15,331,061,660 | 73.812 ± 8.384 | -0.001 ± 0.001 | 0.855 ± 0.011 | 0.625 ± 0.013 | -0.013 ± 0.010 |
| numpy_ols_log_area_time | 3,241,515,613,269 ± 3,778,430,011,704 | 22,244,442,583 ± 15,246,787,072 | 81.441 ± 10.006 | 0.002 ± 0.003 | 0.727 ± 0.020 | 0.543 ± 0.011 | 0.268 ± 0.045 |

Chi tiết từng fold: `reports/baseline_results.csv`.

### Khoảng cách fit–validation (CV, log scale)

- `train_median`: Δ RMSE log = -0.077; Δ MAE log = -0.060.
- `numpy_ols_log_area_time`: Δ RMSE log = -0.053; Δ MAE log = -0.033.

## Nhận xét

- Đây là mốc đối chiếu; OLS chỉ dùng biến số, không có vị trí/biến phân loại.
- Chưa chọn mô hình từ kết quả baseline và chưa đánh giá test.
- OLS chỉ dùng diện tích và thời gian để giữ baseline gọn; metric VND được tính sau khi đổi ngược bằng `expm1`.
