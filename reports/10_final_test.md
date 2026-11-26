# Bước 9 — Đánh giá cuối trên test

Ứng viên TN1 Lasso và siêu tham số đã được chọn bằng train/CV/validation trước khi mở test. Fit lại pipeline trên train+validation (1,270,311 dòng), sau đó đọc và chấm test một lần (223,677 dòng; 2026-02-09T00:00:13 đến 2026-03-30T23:58:16.386000). Không điều chỉnh mô hình theo kết quả test.

## Metric test

| Metric | Giá trị |
|---|---:|
| rmse_vnd | 355,849,790,313.1684 |
| mae_vnd | 12,552,147,018.4026 |
| mape_pct | 2,045.1978 |
| r2_vnd | 0.0127 |
| rmse_log | 0.5625 |
| mae_log | 0.3963 |
| r2_log | 0.6301 |

## Phần dư theo thời gian

Residual log = log giá thực − log giá dự đoán; dương nghĩa là dự đoán thấp.

| Tháng | Số tin | Residual log TB | MAE VND |
|---|---:|---:|---:|
| 2026-02 | 52,274.0 | 0.099 | 14,779,712,054 |
| 2026-03 | 171,403.0 | 0.035 | 11,872,790,562 |

## Phần dư theo tứ phân vị giá thực

| Khoảng giá thực | Số tin | MAE log | Residual log TB |
|---|---:|---:|---:|
| (999.999, 6000000000.0] | 56,358.0 | 0.443 | -0.371 |
| (6000000000.0, 8900000000.0] | 56,850.0 | 0.215 | -0.086 |
| (8900000000.0, 16660000000.0] | 54,550.0 | 0.256 | 0.105 |
| (16660000000.0, 129000000000000.0] | 55,919.0 | 0.670 | 0.560 |

## 15 hệ số có trị tuyệt đối lớn nhất

Hệ số được tính trên đặc trưng đã chuẩn hóa/mã hóa; đây là độ lớn liên hệ trong mô hình tuyến tính, không phải quan hệ nhân quả.

| Đặc trưng | Hệ số | Trị tuyệt đối |
|---|---:|---:|
| numeric__area | 0.46208 | 0.46208 |
| numeric__floor_count | 0.10989 | 0.10989 |
| numeric__frontage_width | 0.05164 | 0.05164 |
| numeric__house_depth | -0.00628 | 0.00628 |
| numeric__road_width | 0.05873 | 0.05873 |
| numeric__bedroom_count | 0.11155 | 0.11155 |
| numeric__bathroom_count | -0.04637 | 0.04637 |
| numeric__year | 0.00000 | 0.00000 |
| numeric__month | 0.00000 | 0.00000 |
| numeric__quarter | -0.00353 | 0.00353 |
| numeric__months_since_start | 0.00707 | 0.00707 |
| categorical__province_name_An Giang | -0.00000 | 0.00000 |
| categorical__province_name_Bà Rịa - Vũng Tàu | 0.02824 | 0.02824 |
| categorical__province_name_Bình Dương | -0.01864 | 0.01864 |
| categorical__province_name_Bình Phước | -0.00000 | 0.00000 |

## Hạn chế

- FISTA dừng ở 200 vòng, relative change 0.00346282 (ngưỡng 0.0001); chưa đạt ngưỡng hội tụ nên hệ số và metric có thể còn cải thiện nếu tối ưu tiếp, nhưng không chỉnh sau khi xem test.
- Giá là giá niêm yết trong tin, không phải giá giao dịch hoàn tất. Phạm vi thời gian dữ liệu ngắn, nên kết quả không chứng minh khả năng dự báo qua chu kỳ dài.
- RMSE giá gốc nhạy với tin giá cực đoan/nhập sai; cần đọc cùng MAE và metric log.

Dự đoán lưu tại `reports/test_predictions.csv`; bảng phần dư theo tháng/giá và hệ số ở cùng thư mục. Mô hình và tiền xử lý lưu tại `models/vietnam_house_lasso.pkl`.
