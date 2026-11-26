"""Chia theo ngày đăng tin và lập fold expanding-window trong train."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "houses.parquet"
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
REPORT = ROOT / "reports" / "03_time_split.md"


def nearest_day_boundary(cumulative: np.ndarray, fraction: float) -> int:
    """Chọn ranh giới ngày có số dòng gần tỷ lệ mục tiêu nhất."""
    total = int(cumulative[-1])
    target = total * fraction
    index = min(int(np.searchsorted(cumulative, target, side="left")), len(cumulative) - 1)
    after = int(cumulative[index])
    before = int(cumulative[index - 1]) if index else 0
    return before if target - before < after - target else after


def date_range(frame: pd.DataFrame, start: int, end: int) -> tuple[str, str]:
    """Trả ngày đầu/cuối trong lát cắt [start, end)."""
    dates = frame["published_at"].astype("string").str.slice(0, 10)
    return str(dates.iloc[start]), str(dates.iloc[end - 1])


def main() -> None:
    """Lưu các partition theo thời gian và metadata của CV."""
    df = pd.read_parquet(SOURCE)
    df["published_at"] = df["published_at"].astype("string")
    df = df.sort_values("published_at", kind="stable", ignore_index=True)
    day = df["published_at"].str.slice(0, 10)
    daily_counts = day.value_counts().sort_index()
    cumulative = daily_counts.cumsum().to_numpy()

    train_end = nearest_day_boundary(cumulative, 0.70)
    validation_end = nearest_day_boundary(cumulative, 0.85)
    total = len(df)
    if not 0 < train_end < validation_end < total:
        raise ValueError("Ranh giới thời gian không tạo được ba tập khác rỗng.")

    partitions = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, total),
    }
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, (start, end) in partitions.items():
        part = df.iloc[start:end]
        part.to_parquet(SPLIT_DIR / f"{name}.parquet", index=False)
        first_day, last_day = date_range(df, start, end)
        summary.append({
            "split": name,
            "rows": end - start,
            "fraction": (end - start) / total,
            "start_date": first_day,
            "end_date": last_day,
        })
    summary_df = pd.DataFrame(summary)

    train = df.iloc[:train_end]
    train_days = train["published_at"].str.slice(0, 10).value_counts().sort_index()
    cv_cumulative = train_days.cumsum().to_numpy()
    cv_boundaries = [nearest_day_boundary(cv_cumulative, part) for part in np.linspace(0.50, 1.00, 6)]
    if any(left >= right for left, right in zip(cv_boundaries, cv_boundaries[1:])):
        raise ValueError("Các fold CV bị rỗng; cần điều chỉnh số fold hoặc kích thước cửa sổ.")

    folds = []
    for fold_no in range(5):
        fit_end = cv_boundaries[fold_no]
        val_end = cv_boundaries[fold_no + 1]
        fit_start_day, fit_end_day = date_range(train, 0, fit_end)
        val_start_day, val_end_day = date_range(train, fit_end, val_end)
        folds.append({
            "fold": fold_no + 1,
            "fit_start_row": 0,
            "fit_end_row": fit_end,
            "fit_rows": fit_end,
            "fit_start_date": fit_start_day,
            "fit_end_date": fit_end_day,
            "validation_start_row": fit_end,
            "validation_end_row": val_end,
            "validation_rows": val_end - fit_end,
            "validation_start_date": val_start_day,
            "validation_end_date": val_end_day,
        })
    folds_df = pd.DataFrame(folds)

    # Tự kiểm tra ranh giới thời gian và số dòng trước khi chốt manifest.
    assert summary_df["rows"].sum() == total
    assert summary_df["end_date"].iloc[0] < summary_df["start_date"].iloc[1]
    assert summary_df["end_date"].iloc[1] < summary_df["start_date"].iloc[2]
    assert folds_df["fit_end_row"].iloc[-1] < train_end
    assert (folds_df["fit_end_row"].to_numpy() == folds_df["validation_start_row"].to_numpy()).all()

    summary_df.to_csv(SPLIT_DIR / "split_summary.csv", index=False)
    folds_df.to_csv(SPLIT_DIR / "cv_folds.csv", index=False)
    split_table = "\n".join(
        f"| {r.split} | {r.rows:,} | {r.fraction:.2%} | {r.start_date} | {r.end_date} |"
        for r in summary_df.itertuples()
    )
    fold_table = "\n".join(
        f"| {r.fold} | {r.fit_rows:,} | {r.fit_start_date} → {r.fit_end_date} | "
        f"{r.validation_rows:,} | {r.validation_start_date} → {r.validation_end_date} |"
        for r in folds_df.itertuples()
    )
    REPORT.write_text(f"""# Bước 3 — Chia dữ liệu theo thời gian

## Train / validation / test

Chia theo thứ tự `published_at`, với ranh giới căn theo **ngày đầy đủ** để không tách tin cùng ngày sang hai tập. Tỷ lệ mục tiêu là 70/15/15; sai khác nhỏ phát sinh do căn ranh giới theo ngày.

| Tập | Số dòng | Tỷ lệ thực tế | Ngày đầu | Ngày cuối |
|---|---:|---:|---|---|
{split_table}

Các tập được lưu trong `data/processed/splits/`. `test.parquet` chỉ được giữ lại cho bước đánh giá cuối; ở bước này chưa tính metric hoặc xem phân phối mục tiêu trên test.

## Expanding-window cross-validation

Năm fold chỉ sử dụng phần train. Cửa sổ fit bắt đầu ở khoảng 50% train và mở rộng; mỗi fold kiểm tra trên đoạn thời gian kế tiếp, không xáo trộn và không chồng lấn validation.

| Fold | Số dòng fit | Khoảng fit | Số dòng validation | Khoảng validation |
|---:|---:|---|---:|---|
{fold_table}

Ranh giới theo hàng (zero-based, end-exclusive) và ngày được lưu trong `data/processed/splits/cv_folds.csv`; tỷ lệ và khoảng ngày của ba tập nằm trong `split_summary.csv`.

## Chống rò rỉ

- Mọi phép fit imputation, scaling và encoding phải được fit riêng trên train/fold-fit rồi mới transform validation.
- Các fold CV không dùng validation chính thức hoặc test.
- Tập test chưa được dùng để chọn đặc trưng, siêu tham số hay mô hình.
""", encoding="utf-8")
    print(f"Wrote 3 temporal splits and {len(folds_df)} expanding-window folds.")


if __name__ == "__main__":
    main()
