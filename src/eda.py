"""EDA cho tin đăng bất động sản Việt Nam trên Hugging Face."""

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "vietnam-real-estates"
FIGURES = ROOT / "reports" / "figures"
REPORT = ROOT / "reports" / "01_eda.md"
COLUMNS = [
    "property_type_name", "province_name", "district_name", "ward_name",
    "price", "area", "floor_count", "frontage_width", "house_depth",
    "road_width", "bedroom_count", "bathroom_count", "house_direction",
    "balcony_direction", "published_at",
]


def main() -> None:
    """Đọc các shard Parquet, tạo thống kê và lưu biểu đồ EDA."""
    candidates = sorted(DATA_DIR.glob("shard_*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"Chưa có đủ shard Parquet hợp lệ trong {DATA_DIR}. "
            "Tải dataset từ https://huggingface.co/datasets/tinixai/vietnam-real-estates rồi chạy lại."
        )
    grouped = {}
    for path in candidates:
        match = re.fullmatch(r"shard_(\d{4})(?:\.[^.]+)?\.parquet", path.name)
        if match:
            grouped.setdefault(match.group(1), []).append(path)
    complete_files = []
    for shard_id, paths in sorted(grouped.items()):
        paths.sort(key=lambda path: path.name != f"shard_{shard_id}.parquet")
        for path in paths:
            try:
                pq.ParquetFile(path)
                complete_files.append(path)
                break
            except (OSError, ValueError):
                continue
    if not complete_files:
        raise FileNotFoundError("Chưa có shard Parquet nào tải hoàn chỉnh.")
    try:
        frames = [pd.read_parquet(path, columns=COLUMNS) for path in complete_files]
    except ImportError as exc:
        raise RuntimeError("Cần cài pyarrow để đọc Parquet: python -m pip install pyarrow") from exc
    df = pd.concat(frames, ignore_index=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["published_at"] = df["published_at"].astype("string")
    date_text = df["published_at"].dropna()
    bad_dates = ~date_text.str.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:")
    price = df["price"]
    valid_price = price[price > 0]
    if valid_price.empty:
        raise ValueError("Không có giá dương hợp lệ trong cột price.")

    FIGURES.mkdir(parents=True, exist_ok=True)
    missing = df.isna().sum().sort_values(ascending=False)
    dupes = int(df.duplicated().sum())
    q1, q3 = valid_price.quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = int(((valid_price < q1 - 1.5 * iqr) | (valid_price > q3 + 1.5 * iqr)).sum())
    by_month = (
        df.loc[price > 0, ["published_at", "price"]]
        .dropna()
        .assign(month=lambda x: x["published_at"].str.slice(0, 7))
        .groupby("month")["price"].agg(["median", "count"])
    )
    numeric = df.select_dtypes(include="number").drop(columns=["price"], errors="ignore")
    numeric["log1p_price"] = np.log1p(price.where(price > 0))
    corr = numeric.replace([np.inf, -np.inf], np.nan).corr()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(valid_price, bins=70, range=(0, valid_price.quantile(0.99)), color="#3977a8")
    axes[0].set(title="Giá đăng tin (cắt trục tại P99)", xlabel="Giá (VND)", ylabel="Số tin")
    axes[1].hist(np.log1p(valid_price), bins=70, color="#d17b49")
    axes[1].set(title="Phân phối log1p(giá)", xlabel="log1p(giá VND)", ylabel="Số tin")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_vn_price_distribution.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    axes[0].plot(by_month.index, by_month["count"], marker="o", color="#d17b49")
    axes[0].set(title="Lượng tin và giá đăng trung vị theo tháng", ylabel="Số tin có giá")
    axes[1].plot(by_month.index, by_month["median"], marker="o", color="#3977a8")
    axes[1].set(xlabel="Tháng đăng tin", ylabel="Trung vị giá (VND)")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_vn_monthly_price.png", dpi=160)
    plt.close(fig)

    if "log1p_price" in corr:
        related = corr["log1p_price"].drop("log1p_price").abs().sort_values(ascending=False).head(10)
        plot_columns = related.index.tolist() + ["log1p_price"]
        corr = corr.loc[plot_columns, plot_columns]
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=70, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    ax.set_title("Tương quan Pearson (các biến số)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_vn_correlation.png", dpi=160)
    plt.close(fig)

    missing_rows = "\n".join(f"| `{name}` | {int(count):,} |" for name, count in missing.items())
    type_rows = df["property_type_name"].value_counts(dropna=False).head(12)
    type_table = "\n".join(f"| {kind if pd.notna(kind) else '(thiếu)'} | {int(count):,} |" for kind, count in type_rows.items())
    monthly_table = "\n".join(
        f"| {month} | {int(row['count']):,} | {row['median']:,.0f} |"
        for month, row in by_month.iterrows()
    )
    related = corr["log1p_price"].drop("log1p_price").dropna().abs().sort_values(ascending=False).head(10)
    correlation_table = "\n".join(
        f"| `{name}` | {corr.loc[name, 'log1p_price']:.3f} |" for name in related.index
    ) or "| Không tính được tương quan hữu hạn | — |"
    schema_table = "\n".join(f"| `{name}` | `{dtype}` |" for name, dtype in df.dtypes.items())
    date_min = date_text.min() if len(date_text) else "không có"
    date_max = date_text.max() if len(date_text) else "không có"
    REPORT.write_text(f"""# Bước 1 — EDA: Tin đăng bất động sản Việt Nam

## Nguồn và phạm vi

- Dataset: [tinixai/vietnam-real-estates](https://huggingface.co/datasets/tinixai/vietnam-real-estates), giấy phép CC BY-NC 4.0.
- Đọc {len(complete_files)} shard Parquet hoàn chỉnh, tổng **{len(df):,} dòng** (khoảng {len(df) / 3_500_744:.1%} số dòng dataset theo card); bỏ qua {len(candidates) - len(complete_files)} tệp dở/nhân bản. Mỗi shard chỉ dùng một bản hoàn chỉnh. Trường thời gian là `published_at` (ngày đăng tin).
- Khoảng ngày theo chuỗi ISO: **{date_min} → {date_max}**; ngày thiếu: **{int(df['published_at'].isna().sum()):,}**, sai định dạng: **{int(bad_dates.sum()):,}**.
- Đây là **giá niêm yết trong tin đăng**, không phải giá giao dịch đã chốt. Kết quả chỉ phản ánh giá chào bán.
- Phân tích dùng các cột cấu trúc, địa lý, giá và thời gian; bỏ qua văn bản `name`/`description` ở vòng EDA này.

## Kiểu dữ liệu sau khi đọc

| Cột | Kiểu |
|---|---|
{schema_table}

## Chất lượng dữ liệu

- Giá thiếu/không hợp lệ: **{int(price.isna().sum()):,}** thiếu; **{int((price <= 0).sum()):,}** không dương.
- Tin trùng hoàn toàn trên các cột đã nạp: **{dupes:,}** (không có mã tin duy nhất trong schema; đây là phép kiểm tra trên các cột EDA).
- Giá hợp lệ (price > 0): **{len(valid_price):,}**; trung vị **{valid_price.median():,.0f} VND**, trung bình **{valid_price.mean():,.0f} VND**, P1 **{valid_price.quantile(.01):,.0f}**, P99 **{valid_price.quantile(.99):,.0f}**, skewness **{valid_price.skew():.2f}**.
- Ngoài hàng rào 1.5×IQR trên giá dương: **{outliers:,}**; chưa loại bỏ vì có thể là bất động sản thật ở phân khúc cao.

### Giá trị thiếu theo cột

| Cột | Số thiếu |
|---|---:|
{missing_rows}

### Loại bất động sản phổ biến

| Loại | Số tin |
|---|---:|
{type_table}

## Lượng tin và giá đăng theo tháng

| Tháng | Số tin có giá dương | Giá trung vị (VND) |
|---|---:|---:|
{monthly_table}

Giá trung vị theo tháng dao động; lượng tin thay đổi đáng kể. Vì vậy cần dùng chia theo thời gian, và không diễn giải thay đổi trong tin đăng thành biến động giá giao dịch.

## Tương quan tuyến tính

| Biến số | Pearson r với log1p(giá) |
|---|---:|
{correlation_table}

Tương quan Pearson chỉ xét quan hệ tuyến tính trên các biến số; các cột địa lý/phân loại không được thể hiện trong bảng này.

## Biểu đồ

![Phân phối giá](figures/eda_vn_price_distribution.png)

![Giá và lượng tin theo tháng](figures/eda_vn_monthly_price.png)

![Tương quan](figures/eda_vn_correlation.png)

## Diễn giải và giới hạn

- Giá thường lệch phải; đối chiếu thêm `log1p(price)` trong mô hình và báo cáo lỗi cả trên thang log lẫn VND.
- Trung vị theo tháng mô tả giá chào bán của các tin được đăng trong tháng; không diễn giải thành chỉ số tăng/giảm giá giao dịch.
- Dataset card mô tả khoảng thời gian 06/2025–03/2026. Đây là khoảng ngắn, nên đánh giá theo thời gian chỉ đo khả năng dự đoán giai đoạn kế tiếp trong dữ liệu, không chứng minh độ bền qua chu kỳ dài.
- Trước mô hình hóa cần chốt phạm vi loại tài sản (ví dụ chỉ `Nhà`, hay gồm cả căn hộ/đất); giá đất và giá nhà không cùng một nhóm mục tiêu.
- Nếu chưa đủ toàn bộ shard, các thống kê bên trên chỉ mô tả phần dữ liệu đã tải, không đại diện chắc chắn cho toàn bộ dataset.
- Cần xác minh trong thực nghiệm các listing trùng/đăng lại và các giá được suy ra/chuẩn hóa từ nội dung tin.
""", encoding="utf-8")
    print(f"Wrote {REPORT.relative_to(ROOT)} and 3 charts from {len(df):,} rows.")


if __name__ == "__main__":
    main()
