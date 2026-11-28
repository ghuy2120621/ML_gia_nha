# Báo cáo tổng kết — Dự đoán giá niêm yết nhà ở Việt Nam

## Tóm tắt

Dự án xây dựng mô hình hồi quy tự cài bằng NumPy để ước lượng giá niêm yết của nhà từ đặc điểm căn nhà, địa điểm và tháng đăng tin. Lasso tự cài bằng FISTA được chọn theo kết quả train/CV/validation và rubric nhiều tiêu chí. Trên tập test đã khóa, mô hình đạt RMSE log **0,563**, MAE log **0,396**, R² log **0,630** và MAE giá gốc khoảng **12,55 tỷ VND**.

Kết quả này chỉ mô tả giá chào bán trong dataset. MAPE test **2.045%** và R² trên VND **0,013** cho thấy các metric trên giá gốc rất nhạy với các tin giá cực nhỏ/cực lớn; mô hình còn đánh giá thấp đáng kể nhóm nhà đắt tiền. Đây chưa phải hệ thống định giá sản phẩm.

## 1. Bài toán và dữ liệu

- Dataset: [`tinixai/vietnam-real-estates`](https://huggingface.co/datasets/tinixai/vietnam-real-estates), giấy phép CC BY-NC 4.0.
- Nguồn có 3.500.744 tin đăng trong 10 Parquet shard, từ 2025-06-01 đến 2026-03-30.
- Phạm vi mô hình: chỉ loại `Nhà`; giữ 1.493.988 tin có giá dương và ngày đăng hợp lệ.
- Mục tiêu `price` là giá niêm yết, không phải giá giao dịch đã chốt. Thời gian `published_at` là ngày đăng tin.
- Giá lệch phải mạnh; trung vị giá dương toàn nguồn là 7,5 tỷ VND và skewness khoảng 1.733. Không loại giá cực đoan âm thầm.

## 2. Tiền xử lý và đánh giá theo thời gian

Đặc trưng gồm diện tích, số tầng/phòng, kích thước mặt tiền/nhà/đường, địa giới, hướng nhà và các đặc trưng năm/tháng/quý/số tháng từ 2025-06. Dữ liệu thiếu được impute; số liệu được clip ở percentile 0,5–99,5 rồi scale; biến phân loại được one-hot encode. Mỗi imputer, clipper, scaler và encoder chỉ fit trên phần fit tương ứng.

Chia theo ngày, không xáo trộn:

| Tập | Số tin | Khoảng ngày |
|---|---:|---|
| Train | 1.043.965 | 2025-06-01 đến 2025-12-16 |
| Validation | 226.346 | 2025-12-17 đến 2026-02-08 |
| Test | 223.677 | 2026-02-09 đến 2026-03-30 |

Train dùng 5 expanding-window fold. Test chỉ dùng cho đánh giá cuối sau khi cấu hình 200 vòng được khóa.

## 3. Thử nghiệm và lựa chọn

Các RMSE/MAE trong bảng dưới là trên `log1p(price)`; CV là mean ± std của năm fold.

| Mô hình | Validation RMSE / MAE | CV RMSE / MAE | Kết luận |
|---|---:|---:|---|
| Median baseline | 0,851 / 0,611 | 0,855 ± 0,011 / 0,625 ± 0,013 | Mốc đơn giản, sai số cao |
| NumPy OLS | 0,701 / 0,529 | 0,727 ± 0,020 / 0,543 ± 0,011 | Cải thiện baseline, chưa đủ đặc trưng |
| TN1 Ridge | 0,525 / 0,369 | 0,547 ± 0,015 / 0,388 ± 0,012 | Kém Lasso nhẹ |
| **TN1 Lasso, 200 vòng** | **0,509 / 0,360** | **0,529 ± 0,015 / 0,379 ± 0,012** | **Đứng đầu rubric, được đánh giá test** |
| TN2 NumPy MLP | 0,798 / 0,614 | 0,584 ± 0,020 / 0,389 ± 0,012 | Dùng tối đa 50.000 mẫu/fold; kém ở validation |
| TN3 Autoencoder + Ridge | 0,650 / 0,490 | 0,633 ± 0,018 / 0,461 ± 0,015 | Dùng tối đa 50.000 mẫu/fold, bỏ district/ward; kém Lasso |

Rubric trọng số cho Lasso 4,35/5. Điểm là xếp hạng tương đối giữa sáu ứng viên, không phải độ tin cậy tuyệt đối.

### Tinh chỉnh train/CV/validation sau đánh giá ban đầu

Sau khi khóa và đánh giá cấu hình ban đầu, đã chạy thêm lưới alpha và đường học; các kết quả này **không dùng test** và chưa thay artifact đã đánh giá:

- Lưới alpha tinh chỉnh chọn 0,00001 ở 200 vòng: CV RMSE/MAE log 0,4965 ± 0,0160 / 0,3557 ± 0,0111; validation 0,4775 / 0,3402.
- Đường học alpha 0,00001 tiếp tục cải thiện đến mốc khảo sát 1.000 vòng: CV RMSE/MAE log 0,4916 ± 0,0157 / 0,3518 ± 0,0109; validation 0,4722 / 0,3364.
- Relative step CV lớn nhất ở 1.000 vòng là 0,000308, cao hơn tolerance 0,0001; metric vẫn cải thiện ở biên cuối của lưới. Đây là ứng viên nghiên cứu, chưa hội tụ và chưa có holdout độc lập. Việc chọn alpha/vòng lặp tuần tự trên cùng CV có thể lạc quan.

Vì vậy, các số liệu test bên dưới chỉ thuộc artifact cũ alpha 0,001/200 vòng. Artifact và CLI hiện vẫn dùng cấu hình đó; không được gán test score cũ cho ứng viên alpha 0,00001/1.000 vòng.

### Kiểm tra hội tụ TN1

Lasso 200 vòng có relative step khoảng 0,00348, cao hơn tolerance 0,0001. Ứng viên tối đa 1.000 vòng đã chạy đủ train/CV/validation và đạt tolerance (validation RMSE/MAE log 0,5098/0,3608; CV 0,5300 ± 0,0156 / 0,3796 ± 0,0124). Cấu hình 200 vòng có lỗi thấp hơn nhẹ trên validation và cả năm CV validation window, nên artifact/test report hiện hành vẫn dùng cấu hình 200 vòng. Chi tiết ở `reports/12_lasso_1000_validation.md`.

Trên dữ liệu tổng hợp, hệ số Ridge/Lasso FISTA được đối chiếu với nghiệm tham chiếu sklearn bằng `python -m src.models.check_optimizer`; self-check đạt.

## 4. Đánh giá test cuối

Tập test được đọc/chấm một lần với artifact Lasso 200 vòng đã khóa. Không điều chỉnh mô hình sau kết quả này.

| Metric | Test |
|---|---:|
| RMSE VND | 355,85 tỷ |
| MAE VND | 12,55 tỷ |
| MAPE | 2.045,20% |
| R² VND | 0,013 |
| RMSE log | 0,563 |
| MAE log | 0,396 |
| R² log | 0,630 |

Phần dư log trung bình là dương ở tháng 02/2026 (0,099) và tháng 03/2026 (0,035), biểu hiện xu hướng dự đoán thấp. Theo tứ phân vị giá thực, residual log trung bình chuyển từ **−0,371** ở nhóm giá thấp nhất sang **+0,560** ở nhóm cao nhất; MAE log nhóm cao nhất là **0,670**. Mô hình có xu hướng dự đoán cao ở nhà giá thấp và thấp ở nhà giá cao.

MAPE đặc biệt kém ổn định do tập test có giá niêm yết rất nhỏ cùng với ngoại lai tới 129 nghìn tỷ VND. Vì thế không nên chỉ dùng MAPE hoặc RMSE giá gốc để diễn giải hiệu quả.

## 5. Sử dụng mô hình

Sửa JSON mẫu `examples/house_input.json`, rồi chạy:

```powershell
python -m src.predict_house examples/house_input.json
```

`area` và `published_at` bắt buộc; các trường khác có thể bỏ trống. Địa danh nên khớp chính tả trong tập huấn luyện. CLI trả giá niêm yết ước lượng bằng VND; ngày ngoài 2025-06 đến 2026-03 là ngoại suy.

## 6. Giới hạn và việc tiếp theo

- Dataset chỉ bao phủ khoảng 10 tháng và là tin rao; kết quả không chứng minh khả năng dự báo chu kỳ dài hoặc giá giao dịch.
- Lasso 200 vòng cho khả năng tổng quát hóa hơi tốt hơn trong các cửa sổ đã đo nhưng chưa hội tụ theo tolerance; đây là đánh đổi giữa early stopping và nghiệm tối ưu.
- Sweep sau đó tìm thấy ứng viên alpha 0,00001/1.000 vòng tốt hơn trên CV/validation nhưng vẫn cải thiện ở lưới cuối và chưa hội tụ; chưa thay mô hình đã lưu.
- MLP/autoencoder dùng sample giới hạn; TN3 còn bỏ một phần đặc trưng vị trí nên không so sánh hoàn toàn cùng bộ đầu vào.
- Công việc tiếp theo là thu thập thêm dữ liệu theo thời gian, khóa quy trình chọn mô hình bằng train/CV/validation, rồi đánh giá ứng viên alpha 0,0001/500 vòng trên holdout tương lai mới đúng một lần.
- Docker và giao diện web không cần cho bản nghiên cứu/CLI hiện tại.

## Tệp liên quan

- EDA: `reports/01_eda.md`
- Chia thời gian: `reports/03_time_split.md`
- Bảng so sánh: `reports/09_model_comparison.md`
- Đánh giá test: `reports/10_final_test.md`
- Mã nguồn: `src/`; artifact: `models/vietnam_house_lasso.pkl`
