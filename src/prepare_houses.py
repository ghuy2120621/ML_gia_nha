"""Lọc tin đăng loại Nhà và tạo bảng đặc trưng trước khi chia thời gian."""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, TIME_COLUMN


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "vietnam-real-estates"
OUTPUT = ROOT / "data" / "processed" / "houses.parquet"
REPORT = ROOT / "reports" / "02_features.md"
SOURCE_COLUMNS = [
    "property_type_name", "province_name", "district_name", "ward_name",
    "price", "area", "floor_count", "frontage_width", "house_depth",
    "road_width", "bedroom_count", "bathroom_count", "house_direction",
    "balcony_direction", "published_at",
]


def main() -> None:
    """Đọc shard, lọc nhà có giá/thời điểm hợp lệ, tạo biến thời gian và lưu Parquet."""
    files = sorted(RAW_DIR.glob("shard_[0-9][0-9][0-9][0-9].parquet"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy shard Parquet trong {RAW_DIR}")
    for path in files:
        pq.ParquetFile(path)  # Dừng rõ ràng nếu có shard chưa tải hoàn chỉnh.

    df = pd.concat(
        [pd.read_parquet(path, columns=SOURCE_COLUMNS) for path in files],
        ignore_index=True,
    )
    source_rows = len(df)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    published = df[TIME_COLUMN].astype("string")
    valid_date = published.str.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:", na=False)
    keep = df["property_type_name"].eq("Nhà") & df[TARGET].gt(0) & valid_date
    df = df.loc[keep, SOURCE_COLUMNS].copy()
    df[TIME_COLUMN] = df[TIME_COLUMN].astype("string")
    year = pd.to_numeric(df[TIME_COLUMN].str.slice(0, 4), errors="raise").astype("int16")
    month = pd.to_numeric(df[TIME_COLUMN].str.slice(5, 7), errors="raise").astype("int8")
    df["year"] = year
    df["month"] = month
    df["quarter"] = ((month - 1) // 3 + 1).astype("int8")
    df["months_since_start"] = (year * 12 + month - (2025 * 12 + 6)).astype("int16")
    df = df.sort_values(TIME_COLUMN, kind="stable", ignore_index=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)
    missing = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].isna().sum().sort_values(ascending=False)
    missing_rows = "\n".join(f"| `{name}` | {int(count):,} |" for name, count in missing.items())
    REPORT.write_text(f"""# Bước 2 — Lọc dữ liệu và tạo đặc trưng

## Phạm vi và lọc dữ liệu

- Chỉ giữ `property_type_name == Nhà`.
- Tập nguồn: {source_rows:,} tin; giữ lại **{len(df):,}** tin Nhà có giá dương và `published_at` đúng định dạng.
- Không xóa ngoại lai hoặc dòng trùng; giá và diện tích chưa biến đổi.
- Bảng đã sắp xếp theo `published_at` tăng dần để sẵn sàng cho chia thời gian ở Bước 3.
- Tệp đầu ra: `data/processed/houses.parquet`.

## Đặc trưng

- Số: {", ".join(f"`{c}`" for c in NUMERIC_FEATURES)}.
- Phân loại: {", ".join(f"`{c}`" for c in CATEGORICAL_FEATURES)}.
- Mục tiêu: `price` (VND, giá niêm yết).
- Giữ `published_at` để chia theo thời gian; không đưa cột này trực tiếp vào mô hình.
- Tạo `year`, `month`, `quarter`, `months_since_start` từ thời điểm đăng. Không tạo tuổi nhà vì dataset không có năm xây dựng.
- Không tạo giá/m² theo khu vực từ `price`, vì sẽ đưa mục tiêu vào đặc trưng.

## Thiếu dữ liệu trong các đặc trưng

| Cột | Số thiếu |
|---|---:|
{missing_rows}

Các giá trị thiếu sẽ được điền bằng trung vị (số) hoặc giá trị phổ biến nhất (phân loại) trong transformer. Scaler và encoder chưa được fit ở bước này; chỉ fit trên train sau khi hoàn tất chia thời gian.

## Pipeline tiền xử lý

`src/features.py` định nghĩa `ColumnTransformer`: median imputation + `StandardScaler` cho số; most-frequent imputation + `OneHotEncoder(handle_unknown="ignore", min_frequency=25)` cho phân loại. Các thống kê điền thiếu, scale và nhóm category hiếm đều học từ train.

Chưa chia train/validation/test ở bước này; sẽ thực hiện riêng tại Bước 3.
""", encoding="utf-8")
    print(f"Prepared {len(df):,} rows in {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
