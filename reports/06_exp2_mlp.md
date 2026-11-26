# Bước 6 — TN2: MLP tự cài bằng NumPy

## Giả thuyết và cấu hình

- MLP một tầng ẩn (64 ReLU), dropout 0.20, Adam (lr=0.001), batch 512, mục tiêu `log1p(price)`.
- Early stopping dùng 15% cuối của sample fit theo thời gian (inner validation), patience=3, tối đa 20 epoch.
- Do PyTorch chưa được cài và dữ liệu lớn, dùng NumPy MLP tự cài. Mỗi fold lấy sample ngẫu nhiên có seed=42, tối đa 50,000 dòng từ phần fit; preprocessing chỉ fit trên 85% đầu sample, không dùng outer validation để early stop.
- Bảng bên dưới đánh giá trên toàn bộ outer validation; test không được đọc.

## Kết quả

Metric: RMSE VND | MAE VND | MAPE % | R² VND | RMSE log | MAE log | R² log.

- Validation: 160,537,942,392 | 9,834,516,688 | 47.938 | 0.036 | 0.798 | 0.614 | 0.112
- Time-series CV (mean ± std): 3,241,030,586,130 ± 3,777,326,196,465 | 20,328,000,136 ± 14,943,602,918 | 42.156 ± 3.319 | 0.000 ± 0.002 | 0.584 ± 0.020 | 0.389 ± 0.012 | 0.527 ± 0.037
- Epoch tốt nhất theo fold: fold 1: 20, fold 2: 20, fold 3: 16, fold 4: 20, fold 5: 20.

Chi tiết từng fold: `reports/exp2_results.csv`.

## Điểm tốt/chưa đạt và kết luận

- MLP học được quan hệ phi tuyến; so sánh log metrics với TN1/baseline trong bước tổng hợp.
- Sample 50.000 dòng/fold giảm chi phí tính toán nhưng có thể bỏ lỡ nhóm hiếm; kết quả không tương đương huấn luyện toàn bộ fit rows.
- MAPE/RMSE VND nhạy với giá thấp/cực cao; xem cả log metrics và phân tích residual ở bước chọn mô hình.
- Kết luận giữ/loại sẽ dựa trên so sánh chung; hiện chưa dùng test.
