"""Đặc trưng và tiền xử lý cho mô hình giá niêm yết nhà riêng."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "area", "floor_count", "frontage_width", "house_depth", "road_width",
    "bedroom_count", "bathroom_count", "year", "month", "quarter",
    "months_since_start",
]
CATEGORICAL_FEATURES = [
    "province_name", "district_name", "ward_name", "house_direction",
    "balcony_direction",
]
TARGET = "price"
TIME_COLUMN = "published_at"


class FeatureClipper(BaseEstimator, TransformerMixin):
    """Giới hạn ngoại lai theo percentile được học riêng từ dữ liệu fit."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, x, y=None):
        values = np.asarray(x, dtype=np.float64)
        self.lower_bounds_ = np.quantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.quantile(values, self.upper, axis=0)
        return self

    def transform(self, x):
        return np.clip(np.asarray(x, dtype=np.float64), self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def make_preprocessor() -> ColumnTransformer:
    """Tạo transformer; chỉ gọi fit trên phần train của từng fold."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("clipper", FeatureClipper()),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=25)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def make_hybrid_preprocessor() -> ColumnTransformer:
    """Preprocessing cho TN3; giữ cấp tỉnh/hướng nhà, bỏ ward/district độ cao."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("clipper", FeatureClipper()),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=25)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, ["province_name", "house_direction", "balcony_direction"]),
    ])
