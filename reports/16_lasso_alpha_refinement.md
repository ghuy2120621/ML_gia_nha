# TN1 — lưới alpha Lasso

Giữ preprocessing, đặc trưng và giới hạn 200 vòng như TN1; chỉ thay alpha. Chọn theo mean validation RMSE log trên 5 expanding-window fold trong train, sau đó xem validation chính thức. Không đọc test. Thời gian chạy: 375.0 giây.

| Alpha | CV RMSE log mean ± std | CV MAE log mean ± std | CV MAPE mean | Val RMSE log | Val MAE log | Relative step CV lớn nhất |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-05 | 0.4965 ± 0.0160 | 0.3557 ± 0.0111 | 44.74% | 0.4775 | 0.3402 | 0.00145673 |
| 3e-05 | 0.4970 ± 0.0160 | 0.3559 ± 0.0111 | 44.77% | 0.4780 | 0.3404 | 0.00137483 |
| 0.0001 | 0.4997 ± 0.0161 | 0.3576 ± 0.0112 | 44.99% | 0.4806 | 0.3420 | 0.00128882 |
| 0.0003 | 0.5088 ± 0.0158 | 0.3645 ± 0.0114 | 46.05% | 0.4890 | 0.3477 | 0.00156354 |
| 0.001 | 0.5288 ± 0.0155 | 0.3792 ± 0.0123 | 48.11% | 0.5087 | 0.3605 | 0.00360029 |

Alpha có CV RMSE log thấp nhất: **1e-05**. Kết quả là ứng viên nghiên cứu trên CV/validation; không thay artifact đã chấm test. Nếu chọn cấu hình mới, cần đánh giá độc lập trên dữ liệu tương lai mới, không dùng lại test cũ.

Chi tiết từng fold và alpha: `reports/lasso_alpha_refinement.csv`.
