# Bước 7 — TN3: Autoencoder + Ridge head tự cài

## Giả thuyết và cấu hình

- Autoencoder NumPy tự học latent code 32 chiều từ numeric + tỉnh + hướng nhà; đầu hồi quy Ridge (alpha=0.1) dự đoán `log1p(price)` trên latent code.
- Adam (lr=0.001), batch=512, tối đa 15 epoch; early stopping theo MSE tái tạo trên inner tail 15%, patience=3.
- Mỗi fold dùng sample tối đa 50,000 dòng với seed=42; preprocessing chỉ fit trên 85% đầu sample. Outer validation không dùng để dừng sớm; test không được đọc.
- Bỏ district/ward trong TN3 để giữ đầu vào gọn; TN1 dùng các biến đó nên đây không phải so sánh cùng bộ đặc trưng.

## Kết quả

Metric: RMSE VND | MAE VND | MAPE % | R² VND | RMSE log | MAE log | R² log.

- Validation: 162,608,759,071 | 10,696,918,515 | 77.736 | 0.011 | 0.650 | 0.490 | 0.410
- Time-series CV (mean ± std): 3,241,210,659,769 ± 3,777,810,950,022 | 21,898,369,418 ± 15,227,038,287 | 62.507 ± 4.265 | 0.001 ± 0.002 | 0.633 ± 0.018 | 0.461 ± 0.015 | 0.445 ± 0.023
- Epoch tốt nhất: fold 1: 15, fold 2: 15, fold 3: 15, fold 4: 15, fold 5: 15.

Chi tiết từng fold: `reports/exp3_results.csv`.

## Điểm tốt/chưa đạt và kết luận

- Biểu diễn latent được học không giám sát trên các đặc trưng của phần fit; đầu Ridge được huấn luyện riêng trên cùng phần fit.
- Sample giới hạn và bỏ district/ward làm giảm chi phí nhưng có thể làm yếu thông tin vị trí; so sánh trực tiếp với TN1 cần lưu ý khác biệt đặc trưng.
- Chọn giữ/loại dựa trên số liệu validation/CV và độ phức tạp trong bảng tổng hợp; chưa dùng test.
