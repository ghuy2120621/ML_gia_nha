# Dự án dự đoán giá bất động sản tại Việt Nam

Dataset được chọn là [`tinixai/vietnam-real-estates`](https://huggingface.co/datasets/tinixai/vietnam-real-estates) trên Hugging Face. Đây là dữ liệu tin đăng, nên `price` là giá niêm yết chứ không phải giá giao dịch đã chốt. Dataset dùng giấy phép CC BY-NC 4.0; cần ghi nguồn và không dùng cho mục đích thương mại nếu chưa được cấp phép.

Báo cáo tổng kết dự án: [`reports/final_project_report.md`](reports/final_project_report.md). Kế hoạch hiện được một mình chủ dự án phụ trách: [`PROJECT_PLAN.md`](PROJECT_PLAN.md).

## Cài đặt và tải dữ liệu

Artifact pickle được tạo bằng scikit-learn 1.8.0; phiên bản này được khóa trong `requirements.txt` để tránh cảnh báo/khác biệt khi nạp pipeline. Các thư viện còn lại chưa khóa phiên bản. Môi trường ban đầu: Python 3.12.9, NumPy 2.4.4, pandas 3.0.4, SciPy 1.17.1, PyArrow 26.0.1 và Matplotlib 3.10.8.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
hf download tinixai/vietnam-real-estates --repo-type dataset --include "shard_*.parquet" --local-dir data/raw/vietnam-real-estates
```

Dataset có 10 shard Parquet, tổng khoảng 1,67 GB. Không commit dữ liệu lên Git; `.gitignore` đã loại `data/`. Nguồn và giấy phép CC BY-NC 4.0: [Hugging Face dataset card](https://huggingface.co/datasets/tinixai/vietnam-real-estates). Nếu PowerShell chặn activate môi trường ảo, có thể gọi trực tiếp `.venv\Scripts\python.exe -m pip ...` và `.venv\Scripts\hf.exe ...`.

## Chạy pipeline và thí nghiệm

Chạy từ thư mục gốc repository bằng `python -m`:

```powershell
python -m src.eda
python -m src.prepare_houses
python -m src.split_data
python -m src.baseline
python -m src.models.ridge_lasso
python -m src.models.mlp_numpy
python -m src.models.hybrid_autoencoder
python -m src.analyze_validation_residuals
python -m src.compare_models
```

Đối chiếu optimizer TN1 trên bài toán tổng hợp nhỏ (sklearn chỉ làm nghiệm tham chiếu):

```powershell
python -m src.models.check_optimizer
```

EDA sinh `reports/01_eda.md` và biểu đồ; các bước sau tạo dữ liệu trong `data/processed/` cùng báo cáo/metric tương ứng trong `reports/`. Chạy một bước sau khi bước trước tạo đủ đầu vào.

Quy trình lọc giữ tin loại `Nhà`, giá dương và ngày đăng hợp lệ, sau đó tạo `data/processed/houses.parquet`. Chia dữ liệu theo `published_at` thành train, validation và test, căn ranh giới theo ngày; năm expanding-window fold chỉ nằm trong train. Transformer chỉ fit trên phần train của fold tương ứng.

Chi tiết phân chia nằm trong `reports/03_time_split.md`. Tập test đã được đánh giá và báo cáo ở `reports/10_final_test.md`; không dùng lại để chọn mô hình hoặc siêu tham số.

King County CSV tải trong lúc khởi tạo được giữ trong `data/raw/` làm dữ liệu chưa dùng; dự án hiện không phân tích tệp đó.

Mô hình đã chọn và pipeline tiền xử lý nằm trong `models/vietnam_house_lasso.pkl`. Các bước tạo lại baseline/thử nghiệm không cần mở tập test. Báo cáo đánh giá cuối đã lưu tại `reports/10_final_test.md`.

## Dự đoán cho một căn nhà

Sửa `examples/house_input.json` rồi chạy:

```powershell
python -m src.predict_house examples/house_input.json
```

`area` và `published_at` là bắt buộc. Các thuộc tính còn lại có thể bỏ qua hoặc đặt `null`; pipeline sẽ điền giá trị thiếu theo thống kê đã học lúc fit. Địa danh nên dùng đúng cách viết trong dữ liệu huấn luyện (ví dụ `province_name: "Hồ Chí Minh"`, `district_name: "1"`), nếu không OneHotEncoder sẽ coi đó là nhóm chưa thấy. Kết quả là giá niêm yết ước tính bằng VND. Dự đoán ngoài khoảng thời gian dữ liệu (2025-06 đến 2026-03) là ngoại suy, độ tin cậy chưa được xác nhận.

Mô hình được lưu bằng pickle; chỉ nạp artifact đáng tin cậy do dự án tạo ra, không dùng file pickle tải từ nguồn lạ.
