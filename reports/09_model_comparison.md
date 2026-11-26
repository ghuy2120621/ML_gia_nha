# Bước 8 — So sánh và chọn ứng viên

## Kết quả validation và time-series CV

| Mô hình | Val RMSE log | Val MAE log | CV RMSE log mean ± std | CV MAE log mean ± std | Val MAPE | Val MAE VND |
|---|---:|---:|---:|---:|---:|---:|
| TN1 Lasso | 0.509 | 0.360 | 0.529 ± 0.015 | 0.379 ± 0.012 | 47.2% | 8.72B |
| TN1 Ridge | 0.525 | 0.369 | 0.547 ± 0.015 | 0.388 ± 0.012 | 50.7% | 8.79B |
| TN2 NumPy MLP | 0.798 | 0.614 | 0.584 ± 0.020 | 0.389 ± 0.012 | 47.9% | 9.83B |
| NumPy OLS | 0.701 | 0.529 | 0.727 ± 0.020 | 0.543 ± 0.011 | 76.8% | 10.37B |
| TN3 Autoencoder + Ridge | 0.650 | 0.490 | 0.633 ± 0.018 | 0.461 ± 0.015 | 77.7% | 10.70B |
| Median baseline | 0.851 | 0.611 | 0.855 ± 0.011 | 0.625 ± 0.013 | 68.9% | 11.53B |

Các metric gốc đầy đủ, gồm RMSE/MAE/MAPE/R² trên VND và log, nằm trong `reports/model_comparison.csv`.

## Bảng điểm có trọng số

Điểm 1–5 là **thứ hạng tương đối giữa sáu mô hình**, không phải xác suất hay chứng nhận tuyệt đối. Accuracy xếp hạng đồng thời validation/CV log errors, MAPE, MAE VND và R²; Stability dùng độ lệch chuẩn CV; Generalization dùng khoảng cách fit–validation trên log metrics; Residual dùng độ lệch residual trung bình, tương quan residual với giá log thực và drift lớn nhất theo tháng. Complexity/interpretability được đánh giá thủ công theo cấu trúc mô hình.

| Mô hình | Accuracy 30% | Stability 25% | Generalization 20% | Residual 15% | Diễn giải 10% | Tổng |
|---|---:|---:|---:|---:|---:|---:|
| TN1 Lasso | 4.8 | 3.7 | 4.2 | 5.0 | 4.0 | 4.35 |
| TN1 Ridge | 4.1 | 3.7 | 2.6 | 4.2 | 4.0 | 3.69 |
| TN2 NumPy MLP | 3.1 | 3.4 | 5.0 | 1.0 | 2.0 | 3.14 |
| NumPy OLS | 2.2 | 2.3 | 3.4 | 3.4 | 5.0 | 2.93 |
| TN3 Autoencoder + Ridge | 2.5 | 2.1 | 1.8 | 2.6 | 2.0 | 2.23 |
| Median baseline | 1.3 | 2.9 | 1.0 | 1.8 | 5.0 | 2.07 |

Tổng điểm cao nhất theo rubric hiện tại: **TN1 Lasso (4.35/5)**.

## Residual và độ bền thời gian

Lasso có validation residual diagnostics ở `reports/08_validation_residuals.md` và hình `reports/figures/validation_residuals_lasso.png`. Residual có tương quan với giá thực; mô hình có xu hướng dự đoán thấp ở nhóm giá cao và tháng 02/2026. Đây là hạn chế đáng kể dù log errors tốt nhất.

CV bao phủ các cửa sổ 09–12/2025; validation chính thức phủ 12/2025–02/2026. Số liệu cho thấy khả năng chuyển tiếp trong giai đoạn ngắn của bộ tin đăng này, chưa xác nhận khả năng bền qua chu kỳ dài.

## Quyết định trước test

- **Ứng viên được chọn theo rubric:** TN1 Lasso. Chỉ sau khi khóa mô hình/cấu hình mới được dùng test một lần.
- TN1 Lasso có độ chính xác log tốt nhất; TN2 MLP gần hơn trong CV nhưng kém trên validation; TN3 hybrid kém accuracy và còn bỏ district/ward; baseline median/OLS yếu hơn về accuracy nhưng đơn giản.
- Lasso FISTA dừng ở 200 vòng, relative change cuối khoảng 0.00348, cao hơn tolerance 0.0001. Ứng viên được khóa theo cấu hình này; kết quả là nghiệm gần đúng, chưa hội tụ theo tiêu chí đã đặt.
- RMSE VND rất nhạy với một số giá cực cao; dataset có ít nhất một tin validation 67.999 nghìn tỷ VND cho diện tích 100 m². Cần giữ rõ kết quả thô và không loại các điểm này âm thầm.

**Test chưa được chấm trong bước so sánh này.**
