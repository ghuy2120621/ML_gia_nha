"""Dự đoán giá niêm yết cho một căn nhà từ JSON đầu vào."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from .models.ridge_lasso import predict


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "models" / "vietnam_house_lasso.pkl"
REQUIRED = {"area", "published_at"}
ALLOWED = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["published_at"])


def make_feature_row(payload: dict) -> pd.DataFrame:
    """Kiểm tra input và tạo các trường thời gian giống pipeline huấn luyện."""
    if not isinstance(payload, dict):
        raise ValueError("Input JSON phải là một object.")
    unknown = set(payload) - ALLOWED
    missing = REQUIRED - set(payload)
    if unknown:
        raise ValueError(f"Trường không được hỗ trợ: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"Thiếu trường bắt buộc: {', '.join(sorted(missing))}")

    try:
        date = pd.Timestamp(payload["published_at"])
    except (TypeError, ValueError) as error:
        raise ValueError("'published_at' phải là ngày ISO, ví dụ 2026-03-15.") from error
    if pd.isna(date):
        raise ValueError("'published_at' phải có giá trị ngày hợp lệ.")
    numeric = {}
    for name in NUMERIC_FEATURES:
        if name in {"year", "month", "quarter", "months_since_start"}:
            continue
        value = payload.get(name)
        if value is None:
            numeric[name] = np.nan
        elif isinstance(value, bool):
            raise ValueError(f"'{name}' phải là số hoặc null.")
        else:
            try:
                numeric[name] = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"'{name}' phải là số hoặc null.") from error
            if not np.isfinite(numeric[name]):
                raise ValueError(f"'{name}' phải là số hữu hạn hoặc null.")
    if not np.isfinite(numeric["area"]) or numeric["area"] <= 0:
        raise ValueError("'area' phải là số lớn hơn 0.")

    for name in CATEGORICAL_FEATURES:
        value = payload.get(name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"'{name}' phải là chuỗi hoặc null.")

    year, month = date.year, date.month
    row = {
        **numeric,
        "year": year,
        "month": month,
        "quarter": (month - 1) // 3 + 1,
        "months_since_start": year * 12 + month - (2025 * 12 + 6),
        **{name: payload.get(name) for name in CATEGORICAL_FEATURES},
    }
    return pd.DataFrame([row], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def predict_price(payload: dict, artifact_path: Path = ARTIFACT) -> float:
    """Trả về giá niêm yết dự đoán bằng VND."""
    with artifact_path.open("rb") as file:
        artifact = pickle.load(file)
    if artifact.get("model") != "self_implemented_lasso":
        raise ValueError("Artifact không phải mô hình Lasso của dự án.")
    row = make_feature_row(payload)
    features = artifact["features"]
    x = artifact["preprocessor"].transform(row[features])
    log_price = predict(x, artifact["weights"])[0]
    return max(0.0, float(np.expm1(log_price)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dự đoán giá niêm yết nhà ở Việt Nam.")
    parser.add_argument("input_json", type=Path, help="Đường dẫn đến JSON thông tin căn nhà")
    args = parser.parse_args()
    try:
        payload = json.loads(args.input_json.read_text(encoding="utf-8"))
        price = predict_price(payload)
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({
        "predicted_listing_price_vnd": round(price),
        "formatted": f"{price:,.0f} VND",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
