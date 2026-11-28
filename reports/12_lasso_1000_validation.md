# TN1 — ứng viên giới hạn 1.000 vòng

Đánh giá lại trên train, năm expanding-window fold trong train và validation chính thức. Preprocessing được fit lại trên fit slice tương ứng. Test không được đọc. Thời gian chạy tổng: 410.9 giây.

## Validation

| Mô hình | RMSE log | MAE log | MAPE % | Vòng thực tế | Relative step cuối |
|---|---:|---:|---:|---:|---:|
| Ridge | 0.5249 | 0.3689 | 50.5381 | 290 | 9.60078e-05 |
| Lasso | 0.5098 | 0.3608 | 47.5973 | 670 | 9.81847e-05 |

## Time-series CV (mean ± std)

| Mô hình | RMSE log | MAE log | MAPE % | Vòng thực tế tối đa | Relative step lớn nhất |
|---|---:|---:|---:|---:|---:|
| Ridge | 0.5460 ± 0.0149 | 0.3879 ± 0.0124 | 51.1820 ± 3.5748 | 289 | 9.49488e-05 |
| Lasso | 0.5300 ± 0.0156 | 0.3796 ± 0.0124 | 48.3791 ± 3.1454 | 654 | 9.97912e-05 |

## So sánh Lasso với cấu hình 200 vòng

| Tập | Cấu hình | RMSE log | MAE log | MAPE % | Relative step |
|---|---|---:|---:|---:|---:|
| validation | 200 vòng | 0.5087 | 0.3605 | 47.22 | 0.00347987 |
| validation | tối đa 1.000 vòng | 0.5098 | 0.3608 | 47.60 | 9.81847e-05 |
| CV validation mean | 200 vòng | 0.5288 | 0.3792 | 48.11 | 0.00360029 |
| CV validation mean | tối đa 1.000 vòng | 0.5300 | 0.3796 | 48.38 | 9.97912e-05 |

## Diễn giải

Lasso đạt tolerance trước 1.000 vòng. Tuy vậy, cấu hình 200 vòng tốt hơn nhẹ ở validation và cả năm CV validation window (RMSE log 0,5087 so với 0,5098 trên validation; CV mean 0,5288 so với 0,5300). Điều này phù hợp với early stopping đang có tác dụng regularization. Giữ artifact 200 vòng làm kết quả đã chấm; ghi rõ đây là lựa chọn theo validation/CV, còn optimizer chưa hội tụ theo tolerance.

Kết quả 1.000 vòng không thay báo cáo đánh giá cuối và không dùng metric test cũ để chọn cấu hình. Chi tiết từng fold lưu tại `reports/lasso_1000_results.csv`.
