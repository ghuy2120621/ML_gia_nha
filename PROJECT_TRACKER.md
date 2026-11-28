# Theo dõi dự án: Dự đoán giá nhà theo thời điểm

## Thông tin dự án

- **Bài toán hiện tại:** Hồi quy giá niêm yết bất động sản từ thuộc tính tin đăng và thời điểm đăng tin.
- **Dataset đã chọn:** [tinixai/vietnam-real-estates](https://huggingface.co/datasets/tinixai/vietnam-real-estates), Parquet, 3.500.744 tin đăng; giấy phép CC BY-NC 4.0.
- **Diễn giải mục tiêu:** `price` là giá niêm yết trong tin, không phải giá giao dịch đã chốt; `published_at` là ngày đăng tin.
- **Ràng buộc:** Mô hình cuối do dự án tự xây dựng và huấn luyện; chia dữ liệu theo thời gian; chỉ fit tiền xử lý trên train; chỉ đánh giá test một lần ở cuối; cố định random seed.
- **Trạng thái tổng thể:** Nghiên cứu ban đầu và CLI đã hoàn thành; đang theo dõi một ứng viên Lasso hội tụ kỹ hơn, chưa thay artifact.
- **Cập nhật gần nhất:** 2026-11-28 — bổ sung CLI/tài liệu; chẩn đoán FISTA trên lát train, chưa đổi artifact.

## Cấu trúc thư mục

```text
data/
  raw/                  # Dữ liệu gốc, không chỉnh sửa
  processed/            # Dữ liệu đã xử lý (nếu cần lưu)
src/
  models/                # Mô hình tự cài và huấn luyện
reports/
  figures/               # Biểu đồ và hình trong báo cáo
examples/                # JSON input mẫu cho CLI dự đoán
models/                  # Artifact Lasso + preprocessing
PROJECT_TRACKER.md
TEAM_TASKS.md
README.md
requirements.txt
```

## Tiến độ và tiêu chí hoàn thành

| Bước | Công việc | Trạng thái | Điều kiện hoàn thành / ghi chú |
|---|---|---|---|
| 0 | Chọn, tải và mô tả dataset | ✅ Hoàn thành | Đã tải và xác nhận đủ 10 shard Parquet, tổng 3.500.744 dòng; nguồn, giấy phép, schema và `published_at` được ghi trong `README.md`/báo cáo. King County CSV được giữ lại nhưng không dùng. |
| 1 | EDA | ✅ Hoàn thành | Đã chạy `src/eda.py` trên 3.500.744 dòng. Báo cáo số liệu, bảng tháng và 3 biểu đồ ở `reports/01_eda.md` và `reports/figures/`. Phân tích dùng giá đăng tin, không phải giá giao dịch. |
| 2 | Tiền xử lý và tạo đặc trưng | ✅ Hoàn thành | Chốt chỉ loại `Nhà`; còn 1.493.988 tin có giá dương/ngày hợp lệ. Đã tạo đặc trưng thời gian, lưu bảng đã sắp xếp và định nghĩa pipeline impute/scale/encode. Chưa fit transformer để tránh rò rỉ trước khi chia thời gian. |
| 3 | Chia dữ liệu theo thời gian | ✅ Hoàn thành | Căn theo ngày: train 69,88% (2025-06-01–2025-12-16), validation 15,15% (2025-12-17–2026-02-08), test 14,97% (2026-02-09–2026-03-30). Đã tạo 5 expanding-window folds chỉ trong train. Tập test được giữ riêng đến bước 9. |
| 4 | Baseline | ✅ Hoàn thành | Median và NumPy OLS trên log diện tích + thời gian; validation và 5-fold CV trong `reports/04_baseline.md`. Chưa đọc test. |
| 5 | Thử nghiệm 1: Ridge/Lasso NumPy | ✅ Hoàn thành | FISTA trên CSR, preprocessing fit theo fold; Lasso giữ làm ứng viên theo rubric. Giới hạn 200 vòng vẫn chưa đạt tolerance; nghiệm gần đúng. |
| 6 | Thử nghiệm 2: MLP | ✅ Hoàn thành | MLP NumPy, dropout 0,20, Adam, early stopping; sample tối đa 50.000/fold. Kém Lasso trên log error, giữ làm đối chứng phi tuyến; xem `reports/06_exp2_mlp.md`. |
| 7 | Thử nghiệm 3: mô hình nâng cao | ✅ Hoàn thành | Autoencoder NumPy + Ridge head; kém Lasso trên validation/CV; xem `reports/07_exp3_hybrid.md`. |
| 8 | So sánh và chọn mô hình | ✅ Hoàn thành | Rubric trọng số chọn Lasso (4,35/5); xem `reports/09_model_comparison.md`. Test chưa dùng ở bước chọn. |
| 9 | Đánh giá cuối | ✅ Hoàn thành | Đánh giá test một lần; báo cáo metric, phần dư theo tháng/nhóm giá và hệ số ở `reports/10_final_test.md`. |
| 10 | Đóng gói và báo cáo | ✅ Hoàn thành | Pipeline và Lasso tự cài ở `models/vietnam_house_lasso.pkl`; hướng dẫn chạy ở README. Demo Streamlit/Gradio được bỏ qua vì là tùy chọn. |

## Nhật ký thử nghiệm

| ID | Ngày | Thử nghiệm / thay đổi | Cấu hình và dữ liệu | Kết quả validation | Time-series CV (mean ± std) | Quyết định / lý do |
|---|---|---|---|---|---|---|
| 1 | 2026-11-25 | Baseline median + NumPy OLS | Train theo thời gian; OLS dùng log diện tích + thời gian | OLS RMSE/MAE log 0,701/0,529 | 0,727 ± 0,020 / 0,543 ± 0,011 | Mốc đối chiếu; yếu hơn Lasso |
| 2 | 2026-11-25 | TN1 Ridge/Lasso NumPy | FISTA, alpha Ridge 0,01/Lasso 0,001; giới hạn ban đầu 200 vòng | Lasso RMSE/MAE log 0,509/0,360 | 0,529 ± 0,015 / 0,379 ± 0,012 | Thắng rubric hiện tại; chưa hội tụ ở 200 vòng |
| 3 | 2026-11-25 | TN2 NumPy MLP | 64 ReLU, dropout 0,2, tối đa 50.000 mẫu/fold | RMSE/MAE log 0,798/0,614 | 0,584 ± 0,020 / 0,389 ± 0,012 | Giữ làm đối chứng phi tuyến; yếu hơn Lasso ở validation |
| 4 | 2026-11-25 | TN3 autoencoder + Ridge head | Biểu diễn tự học, sample tối đa 50.000/fold | RMSE/MAE log 0,650/0,490 | 0,633 ± 0,018 / 0,461 ± 0,015 | Loại khỏi ứng viên thắng |
| 5 | 2026-11-26 | CLI dự đoán một căn nhà | Artifact Lasso hiện hành + JSON input mẫu | Smoke check mẫu: 18.824.395.878 VND | Không áp dụng | Luồng CLI chạy được; đây là giá niêm yết |
| 6 | 2026-11-28 | Chẩn đoán FISTA trên 30.000 dòng train | So sánh power iteration/eigsh; chạy đến 2.000 vòng | 1.000 vòng đạt relative step 0,000064 trên lát train | Chưa áp dụng | Sai số power iteration không đáng kể; giới hạn vòng là điểm chính |
| 7 | 2026-11-28 | TN1 ứng viên tối đa 1.000 vòng | Đủ train, 5 expanding CV fold và validation; không đọc test | RMSE/MAE log 0,5098/0,3608; relative step 9,82e-5 | RMSE 0,5300 ± 0,0156; MAE 0,3796 ± 0,0124 | 200 vòng nhỉnh hơn nhẹ trên validation và cả 5 fold; giữ artifact 200 vòng theo generalization, công khai chưa hội tụ |

## Mẫu ghi cho mỗi thử nghiệm

1. **Giả thuyết và cấu hình:** đặc trưng, kiến trúc/siêu tham số, seed và cách huấn luyện.
2. **Kết quả:** validation và time-series CV; nêu metric, thang đo, mean ± std.
3. **Điểm tốt:** chỉ ghi điều được số liệu/biểu đồ hỗ trợ.
4. **Điểm chưa đạt:** nêu số liệu, biểu đồ hoặc dấu hiệu overfit/drift liên quan.
5. **Kết luận:** giữ hay loại và lý do.

## Quy tắc chống rò rỉ và tái lập

- Sắp xếp theo thời điểm giao dịch trước khi chia; không chia ngẫu nhiên.
- Scaler, encoder, thống kê theo khu vực và mọi phép biến đổi có bước `fit` chỉ học từ phần train của fold tương ứng.
- Không dùng thông tin tương lai để tạo đặc trưng; kiểm tra riêng đặc trưng giá/m² theo khu vực.
- Không điều chỉnh siêu tham số hoặc chọn mô hình dựa trên test.
- Ghi seed, phiên bản dữ liệu, cấu hình và metric cho mỗi lần chạy.
- Dùng tập test duy nhất một lần trong bước đánh giá cuối; ghi ngày và commit/config đã khóa.

## Rủi ro cần theo dõi

- Rò rỉ dữ liệu qua chia ngẫu nhiên, tiền xử lý fit toàn bộ dữ liệu hoặc thống kê khu vực.
- Mốc thời gian/định dạng ngày không nhất quán hoặc dataset không đủ các giai đoạn.
- Ngoại lai giá nhà làm metric nhạy; cần báo cáo cả nhiều metric và phân tích theo nhóm.
- MLP/mô hình phức tạp có thể overfit hoặc kém ổn định giữa các fold.
- MAPE không ổn định khi giá thật bằng hoặc gần 0; ghi rõ cách xử lý nếu gặp trường hợp này.

## Quyết định và vấn đề cần xác nhận

- Dataset cuối cùng: **tinixai/vietnam-real-estates**; file cần có trong `data/raw/vietnam-real-estates/`.
- Trường thời gian: **`published_at` (ngày đăng tin), không phải ngày giao dịch**.
- Phạm vi loại tài sản cho mô hình: **Chỉ `Nhà` (theo lựa chọn của người dùng).**
- Dữ liệu đã chuẩn bị: **1.493.988 tin; `data/processed/houses.parquet`, đã sắp xếp theo `published_at`.**
- Random seed: **42** cho power iteration và các bước lấy mẫu thử nghiệm.
- Trạng thái lấy dữ liệu: **Đủ 10 shard; 3.500.744 dòng Parquet đã được kiểm tra metadata.**
- Kết quả EDA thực đo: **Đã ghi tại `reports/01_eda.md`; đây là giá niêm yết, phạm vi thời gian 2025-06-01 đến 2026-03-30.**
