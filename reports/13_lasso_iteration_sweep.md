# TN1 — chọn số vòng bằng đường học

Lasso alpha=0.001, các đặc trưng và preprocessing giữ nguyên TN1. Mỗi fold fit preprocessing riêng; chạy một đường FISTA đến 1.000 vòng, ghi metric tại các checkpoint. Chỉ dùng train, 5 expanding-window CV folds và validation chính thức; test không được đọc. Thời gian chạy: 378.3 giây.

| Vòng lặp | CV RMSE log mean ± std | CV MAE log mean ± std | Validation RMSE log | Validation MAE log | Relative step CV lớn nhất |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.5617 ± 0.0154 | 0.4019 ± 0.0118 | 0.5437 | 0.3866 | 0.00621444 |
| 100 | 0.5314 ± 0.0159 | 0.3807 ± 0.0122 | 0.5119 | 0.3629 | 0.00331262 |
| 150 | 0.5297 ± 0.0153 | 0.3800 ± 0.0121 | 0.5102 | 0.3619 | 0.00308292 |
| 200 | 0.5288 ± 0.0155 | 0.3792 ± 0.0123 | 0.5087 | 0.3605 | 0.00360029 |
| 300 | 0.5305 ± 0.0154 | 0.3800 ± 0.0123 | 0.5101 | 0.3612 | 0.00171962 |
| 500 | 0.5299 ± 0.0156 | 0.3796 ± 0.0124 | 0.5097 | 0.3609 | 0.000343836 |
| 750 | 0.5301 ± 0.0156 | 0.3797 ± 0.0124 | 0.5098 | 0.3608 | 7.98575e-05 |
| 1000 | 0.5301 ± 0.0156 | 0.3797 ± 0.0124 | 0.5097 | 0.3608 | 4.14162e-05 |

Số vòng có CV RMSE log trung bình thấp nhất: **200**. Đây là lựa chọn theo CV; validation được báo cáo như kiểm tra chuyển tiếp. Kết quả sweep không thay artifact cuối hoặc báo cáo test hiện tại. Không mở test để so sánh các checkpoint.

Chi tiết từng fold/checkpoint: `reports/lasso_iteration_sweep.csv`.
