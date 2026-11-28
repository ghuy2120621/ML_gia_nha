# Chẩn đoán hội tụ TN1 — tập con train, không dùng test

## Phạm vi

Đây là chẩn đoán cục bộ để tìm nguyên nhân FISTA chưa đạt tolerance sau 200 vòng, không phải đánh giá chọn mô hình. Dùng 30.000 dòng đầu của `train.parquet`, fit tiền xử lý trên chính lát train này và mục tiêu `log1p(price)`. Không đọc validation hoặc test trong phép chẩn đoán này.

## Kiểm tra bước lặp

- Lipschitz estimate dùng trong code (15 bước power iteration): **3,0543006385**.
- Tham chiếu trị riêng lớn nhất của `X'X/n` trên cùng ma trận qua `scipy.sparse.linalg.eigsh`: **3,0543007911**.
- Estimate thấp hơn tham chiếu khoảng **0,000005%** trên lát này; sai lệch estimate không giải thích được việc chưa hội tụ.
- Với Lasso alpha=0,001, relative step và objective trên lát train:

| Vòng lặp | Relative step | Objective train |
|---:|---:|---:|
| 200 | 0,003097718 | 0,294409268 |
| 500 | 0,000532050 | 0,283107120 |
| 1.000 | 0,000064003 | 0,282393215 |
| 2.000 | 0,000010573 | 0,282391043 |

Ngưỡng đặt trong dự án là `1e-4`. Trên lát nhỏ này, ngưỡng được đạt khoảng 1.000 vòng; objective giảm mạnh từ vòng 200 đến 1.000 rồi gần như ổn định.

## Kết luận và giới hạn

Chẩn đoán gợi ý giới hạn 200 vòng là nguyên nhân trực tiếp dễ thấy hơn sai số power iteration. Có thể thử 1.000 vòng như cấu hình nghiên cứu, nhưng cần chạy lại đầy đủ 5 fold CV và validation trước khi tuyên bố cải thiện. Kết quả lát 30.000 dòng không thay thế đánh giá toàn bộ dữ liệu.

Artifact Lasso hiện tại và báo cáo test vẫn thuộc cấu hình 200 vòng. Không thay artifact hoặc gán metric test cũ cho cấu hình 1.000 vòng; test đã được dùng và không được mở lại để chọn cấu hình.
