# TN1 — chọn số vòng bằng đường học

Lasso alpha=0.0001, các đặc trưng và preprocessing giữ nguyên TN1. Mỗi fold fit preprocessing riêng; chạy một đường FISTA đến 1.000 vòng, ghi metric tại các checkpoint. Chỉ dùng train, 5 expanding-window CV folds và validation chính thức; test không được đọc. Thời gian chạy: 437.3 giây.

| Vòng lặp | CV RMSE log mean ± std | CV MAE log mean ± std | Validation RMSE log | Validation MAE log | Relative step CV lớn nhất |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.5540 ± 0.0154 | 0.3975 ± 0.0112 | 0.5367 | 0.3841 | 0.0061102 |
| 100 | 0.5123 ± 0.0158 | 0.3659 ± 0.0110 | 0.4932 | 0.3503 | 0.00259707 |
| 150 | 0.5032 ± 0.0158 | 0.3601 ± 0.0109 | 0.4845 | 0.3445 | 0.0017552 |
| 200 | 0.4997 ± 0.0161 | 0.3576 ± 0.0112 | 0.4806 | 0.3420 | 0.00128882 |
| 300 | 0.4981 ± 0.0158 | 0.3562 ± 0.0110 | 0.4791 | 0.3407 | 0.000934551 |
| 500 | 0.4979 ± 0.0159 | 0.3561 ± 0.0111 | 0.4785 | 0.3404 | 0.000949695 |
| 750 | 0.4980 ± 0.0159 | 0.3562 ± 0.0111 | 0.4787 | 0.3405 | 0.000322161 |
| 1000 | 0.4981 ± 0.0159 | 0.3562 ± 0.0111 | 0.4786 | 0.3405 | 0.000260571 |

Số vòng có CV RMSE log trung bình thấp nhất: **500**. Đây là lựa chọn theo CV; validation được báo cáo như kiểm tra chuyển tiếp. Kết quả sweep không thay artifact cuối hoặc báo cáo test hiện tại. Không mở test để so sánh các checkpoint.

Chi tiết từng fold/checkpoint: `reports/lasso_iteration_alpha_0p0001.csv`.
