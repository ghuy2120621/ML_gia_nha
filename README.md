# Dự án dự đoán giá bất động sản tại Việt Nam

Dataset được chọn là [`tinixai/vietnam-real-estates`](https://huggingface.co/datasets/tinixai/vietnam-real-estates) trên Hugging Face. Đây là dữ liệu tin đăng, nên `price` là giá niêm yết chứ không phải giá giao dịch đã chốt. Dataset dùng giấy phép CC BY-NC 4.0; cần ghi nguồn và không dùng cho mục đích thương mại nếu chưa được cấp phép.

## EDA

Đặt các tệp `shard_*.parquet` trong `data/raw/vietnam-real-estates/`, cài phụ thuộc rồi chạy:

```powershell
python -m pip install -r requirements.txt
python src/eda.py
```

Báo cáo được lưu tại `reports/01_eda.md`, biểu đồ tại `reports/figures/`.

## Chuẩn bị dữ liệu nhà

Sau EDA, dự án chỉ giữ loại `Nhà` và tạo bảng đã sắp xếp theo `published_at`:

```powershell
python -m src.prepare_houses
```

Kết quả nằm ở `data/processed/houses.parquet`; pipeline tiền xử lý được định nghĩa tại `src/features.py`. Chỉ fit imputer, scaler và encoder trên train sau khi chia dữ liệu theo thời gian.

## Chia theo thời gian

```powershell
python -m src.split_data
```

Lệnh tạo `train.parquet`, `validation.parquet`, `test.parquet`, cùng `split_summary.csv` và `cv_folds.csv` trong `data/processed/splits/`. Ranh giới được căn theo ngày để các tin cùng ngày ở chung một tập. Năm fold CV chỉ dùng phần train.

King County CSV tải trong lúc khởi tạo được giữ trong `data/raw/` làm dữ liệu chưa dùng; dự án hiện không phân tích tệp đó.

## Đánh giá và artifact cuối

Sau khi đã chạy baseline và ba thử nghiệm, chạy so sánh rồi chấm test một lần:

```powershell
python -m src.compare_models
python -m src.final_evaluation
```

`final_evaluation` khóa ứng viên Lasso theo validation/CV, fit lại trên train+validation, rồi mới đọc test. Báo cáo nằm tại `reports/10_final_test.md`; artifact gồm pipeline tiền xử lý và hệ số mô hình ở `models/vietnam_house_lasso.pkl`. Các tệp dự đoán test chỉ phục vụ phân tích báo cáo cuối, không dùng để điều chỉnh mô hình.
