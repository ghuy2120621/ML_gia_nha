# Bước 5 — TN1: Ridge/Lasso tự cài

## Giả thuyết và cấu hình

- Giả thuyết: regularization giúp mô hình tuyến tính dùng đặc trưng số và one-hot ổn định hơn baseline ít đặc trưng.
- Đầu vào: numeric đã median-impute/scale và categorical đã impute/one-hot bằng `make_preprocessor`; mỗi lần chỉ fit preprocessing trên phần fit tương ứng.
- Mục tiêu: `log1p(price)`; optimizer: FISTA tự viết với phép nhân CSR và vector NumPy; intercept không bị phạt.
- Cấu hình: clip numeric tại P0.5–P99.5 (ranh giới fit-only); Ridge alpha=0.01, Lasso alpha=0.001, tối đa 200 vòng lặp, tolerance=0.0001, seed=42 cho power iteration.
- Mỗi fold được đo trên fit và validation để quan sát khoảng cách tổng quát hóa. Test không được đọc.

### Validation run — fit

| Mô hình | RMSE VND | MAE VND | MAPE % | R² VND | RMSE log | MAE log | R² log |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge | 3,361,291,206,982 | 17,116,825,748 | 82.401 | 0.001 | 0.581 | 0.409 | 0.582 |
| Lasso | 3,360,959,513,573 | 17,105,993,684 | 78.845 | 0.001 | 0.561 | 0.397 | 0.610 |

### Validation run — validation

| Mô hình | RMSE VND | MAE VND | MAPE % | R² VND | RMSE log | MAE log | R² log |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge | 160,340,485,834 | 8,792,640,923 | 50.717 | 0.038 | 0.525 | 0.369 | 0.615 |
| Lasso | 160,235,503,953 | 8,715,084,461 | 47.220 | 0.039 | 0.509 | 0.360 | 0.639 |

### Time-series CV (mean ± std)

| Mô hình | Phần | RMSE VND | MAE VND | MAPE % | R² VND | RMSE log | MAE log | R² log |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge | fit | 1,897,310,738,169 ± 942,556,026,510 | 14,217,721,675 ± 1,038,600,769 | 97.630 ± 10.618 | 0.001 ± 0.000 | 0.601 ± 0.012 | 0.423 ± 0.008 | 0.576 ± 0.002 |
| Ridge | validation | 3,240,709,664,941 ± 3,777,873,918,956 | 20,902,418,706 ± 15,221,896,278 | 51.403 ± 3.614 | 0.003 ± 0.004 | 0.547 ± 0.015 | 0.388 ± 0.012 | 0.586 ± 0.015 |
| Lasso | fit | 1,897,055,560,017 ± 942,218,462,825 | 14,207,340,760 ± 1,037,628,921 | 94.069 ± 10.641 | 0.001 ± 0.001 | 0.581 ± 0.012 | 0.411 ± 0.008 | 0.603 ± 0.003 |
| Lasso | validation | 3,240,474,902,949 ± 3,777,467,730,480 | 20,832,988,003 ± 15,247,601,022 | 48.109 ± 3.069 | 0.003 ± 0.004 | 0.529 ± 0.015 | 0.379 ± 0.012 | 0.612 ± 0.015 |

Số vòng lặp thực chạy theo mô hình (min–max trên CV): `{'Lasso': {'min': 200, 'max': 200}, 'Ridge': {'min': 200, 'max': 200}}`. Chi tiết fold trong `reports/exp1_results.csv`.

## Diễn giải

- So sánh với baseline trong `reports/04_baseline.md`; ưu tiên metric log cùng MAE/RMSE trên VND vì RMSE giá gốc rất nhạy với ngoại lai.
- Nếu đạt giới hạn 200 vòng trước tolerance 0.0001, xem là cấu hình giới hạn vòng lặp và không khẳng định nghiệm tối ưu.
- Kết luận giữ/loại cần dựa trên validation/CV và sẽ được ghi ở bảng so sánh; chưa dùng test.
