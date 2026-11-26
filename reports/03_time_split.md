# Bước 3 — Chia dữ liệu theo thời gian

## Train / validation / test

Chia theo thứ tự `published_at`, với ranh giới căn theo **ngày đầy đủ** để không tách tin cùng ngày sang hai tập. Tỷ lệ mục tiêu là 70/15/15; sai khác nhỏ phát sinh do căn ranh giới theo ngày.

| Tập | Số dòng | Tỷ lệ thực tế | Ngày đầu | Ngày cuối |
|---|---:|---:|---|---|
| train | 1,043,965 | 69.88% | 2025-06-01 | 2025-12-16 |
| validation | 226,346 | 15.15% | 2025-12-17 | 2026-02-08 |
| test | 223,677 | 14.97% | 2026-02-09 | 2026-03-30 |

Các tập được lưu trong `data/processed/splits/`. `test.parquet` chỉ được giữ lại cho bước đánh giá cuối; ở bước này chưa tính metric hoặc xem phân phối mục tiêu trên test.

## Expanding-window cross-validation

Năm fold chỉ sử dụng phần train. Cửa sổ fit bắt đầu ở khoảng 50% train và mở rộng; mỗi fold kiểm tra trên đoạn thời gian kế tiếp, không xáo trộn và không chồng lấn validation.

| Fold | Số dòng fit | Khoảng fit | Số dòng validation | Khoảng validation |
|---:|---:|---|---:|---|
| 1 | 520,893 | 2025-06-01 → 2025-09-19 | 105,794 | 2025-09-20 → 2025-10-08 |
| 2 | 626,687 | 2025-06-01 → 2025-10-08 | 106,445 | 2025-10-09 → 2025-10-25 |
| 3 | 733,132 | 2025-06-01 → 2025-10-25 | 104,893 | 2025-10-26 → 2025-11-12 |
| 4 | 838,025 | 2025-06-01 → 2025-11-12 | 104,420 | 2025-11-13 → 2025-11-29 |
| 5 | 942,445 | 2025-06-01 → 2025-11-29 | 101,520 | 2025-12-01 → 2025-12-16 |

Ranh giới theo hàng (zero-based, end-exclusive) và ngày được lưu trong `data/processed/splits/cv_folds.csv`; tỷ lệ và khoảng ngày của ba tập nằm trong `split_summary.csv`.

## Chống rò rỉ

- Mọi phép fit imputation, scaling và encoding phải được fit riêng trên train/fold-fit rồi mới transform validation.
- Các fold CV không dùng validation chính thức hoặc test.
- Tập test chưa được dùng để chọn đặc trưng, siêu tham số hay mô hình.
