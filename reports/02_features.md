# Bước 2 — Lọc dữ liệu và tạo đặc trưng

## Phạm vi và lọc dữ liệu

- Chỉ giữ `property_type_name == Nhà`.
- Tập nguồn: 3,500,744 tin; giữ lại **1,493,988** tin Nhà có giá dương và `published_at` đúng định dạng.
- Không xóa ngoại lai hoặc dòng trùng; giá và diện tích chưa biến đổi.
- Bảng đã sắp xếp theo `published_at` tăng dần để sẵn sàng cho chia thời gian ở Bước 3.
- Tệp đầu ra: `data/processed/houses.parquet`.

## Đặc trưng

- Số: `area`, `floor_count`, `frontage_width`, `house_depth`, `road_width`, `bedroom_count`, `bathroom_count`, `year`, `month`, `quarter`, `months_since_start`.
- Phân loại: `province_name`, `district_name`, `ward_name`, `house_direction`, `balcony_direction`.
- Mục tiêu: `price` (VND, giá niêm yết).
- Giữ `published_at` để chia theo thời gian; không đưa cột này trực tiếp vào mô hình.
- Tạo `year`, `month`, `quarter`, `months_since_start` từ thời điểm đăng. Không tạo tuổi nhà vì dataset không có năm xây dựng.
- Không tạo giá/m² theo khu vực từ `price`, vì sẽ đưa mục tiêu vào đặc trưng.

## Thiếu dữ liệu trong các đặc trưng

| Cột | Số thiếu |
|---|---:|
| `house_depth` | 1,470,177 |
| `balcony_direction` | 1,342,609 |
| `house_direction` | 1,207,718 |
| `road_width` | 1,180,675 |
| `floor_count` | 894,374 |
| `bathroom_count` | 613,089 |
| `frontage_width` | 578,478 |
| `bedroom_count` | 529,684 |
| `ward_name` | 160,212 |
| `district_name` | 16,532 |
| `year` | 0 |
| `area` | 0 |
| `province_name` | 0 |
| `months_since_start` | 0 |
| `quarter` | 0 |
| `month` | 0 |

Các giá trị thiếu sẽ được điền bằng trung vị (số) hoặc giá trị phổ biến nhất (phân loại) trong transformer. Scaler và encoder chưa được fit ở bước này; chỉ fit trên train sau khi hoàn tất chia thời gian.

## Pipeline tiền xử lý

`src/features.py` định nghĩa `ColumnTransformer`: median imputation, clip số tại percentile 0,5–99,5 rồi `StandardScaler`; most-frequent imputation + `OneHotEncoder(handle_unknown="ignore", min_frequency=25)` cho phân loại. Thống kê điền thiếu, ngưỡng clip, scale và nhóm category hiếm chỉ học từ train/fold-fit.

Chưa chia train/validation/test ở bước này; sẽ thực hiện riêng tại Bước 3.
