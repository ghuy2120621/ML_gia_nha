# TN1 — chọn số vòng bằng đường học

Lasso alpha=1e-05, các đặc trưng và preprocessing giữ nguyên TN1. Mỗi fold fit preprocessing riêng; chạy một đường FISTA đến 1.000 vòng, ghi metric tại các checkpoint. Chỉ dùng train, 5 expanding-window CV folds và validation chính thức; test không được đọc. Thời gian chạy: 479.3 giây.

| Vòng lặp | CV RMSE log mean ± std | CV MAE log mean ± std | Validation RMSE log | Validation MAE log | Relative step CV lớn nhất |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.5532 ± 0.0153 | 0.3971 ± 0.0111 | 0.5361 | 0.3840 | 0.00615844 |
| 100 | 0.5103 ± 0.0157 | 0.3646 ± 0.0109 | 0.4915 | 0.3494 | 0.00275734 |
| 150 | 0.5004 ± 0.0158 | 0.3581 ± 0.0108 | 0.4818 | 0.3429 | 0.00196096 |
| 200 | 0.4965 ± 0.0160 | 0.3557 ± 0.0111 | 0.4775 | 0.3402 | 0.00145673 |
| 300 | 0.4941 ± 0.0156 | 0.3536 ± 0.0109 | 0.4754 | 0.3385 | 0.000992415 |
| 500 | 0.4924 ± 0.0156 | 0.3524 ± 0.0108 | 0.4733 | 0.3372 | 0.000613335 |
| 750 | 0.4917 ± 0.0156 | 0.3519 ± 0.0109 | 0.4726 | 0.3366 | 0.000412624 |
| 1000 | 0.4916 ± 0.0157 | 0.3518 ± 0.0109 | 0.4722 | 0.3364 | 0.000308422 |

Số vòng có CV RMSE log trung bình thấp nhất: **1000**. Đây là lựa chọn theo CV; validation được báo cáo như kiểm tra chuyển tiếp. Kết quả sweep không thay artifact cuối hoặc báo cáo test hiện tại. Không mở test để so sánh các checkpoint.

Chi tiết từng fold/checkpoint: `reports/lasso_iteration_alpha_1e-05.csv`.
