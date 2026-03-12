from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

DISTRICT_MAP = {
    "Алматинский": "Алматы",
    "Байконурский": "Байконур",
    "Есильский": "Есиль",
    "Нуринский": "Нура",
    "Сарыаркинский": "Сарыарка",
    "Сарайшыкский": "Сарайшык",
}

RAW_FEATURE_COLUMNS = [
    "district",
    "tip_doma",
    "jk",
    "year",
    "floor",
    "area",
    "toilet",
    "parking",
    "status",
    "num_rooms",
]

NUMERIC_FEATURES = ["year", "floor", "area", "num_rooms", "area_per_room", "is_new_build", "log_area"]
CATEGORICAL_FEATURES = ["district", "tip_doma", "jk", "toilet", "parking", "status"]


def prepare_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().drop_duplicates()

    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace("〒", "", regex=False)
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df["area"] = df["area"].astype(str).str.extract(r"([\d\.]+)")[0]
    df["area"] = pd.to_numeric(df["area"], errors="coerce")

    df["floor"] = pd.to_numeric(df["floor"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["num_rooms"] = pd.to_numeric(df["num_rooms"], errors="coerce")

    df["district"] = df["district"].replace(DISTRICT_MAP)

    df = df.dropna(subset=["price"]).reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["district"] = df["district"].replace(DISTRICT_MAP)

    categorical_cols = ["district", "tip_doma", "jk", "toilet", "parking", "status"]
    for col in categorical_cols:
        df[col] = df[col].fillna("No data").astype(str).str.strip()
        df.loc[df[col].isin(["nan", "None", ""]), col] = "No data"

    numeric_cols = ["year", "floor", "area", "num_rooms"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["area_per_room"] = df["area"] / df["num_rooms"].replace(0, np.nan)
    df["is_new_build"] = (df["year"] >= 2023).astype(float)
    df["log_area"] = np.log1p(df["area"].clip(lower=0))

    return df


def build_preprocessor(min_frequency: int = 15, scale_numeric: bool = False) -> Pipeline:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    return Pipeline(
        steps=[
            ("feature_builder", FunctionTransformer(add_features, validate=False)),
            (
                "column_transformer",
                ColumnTransformer(
                    transformers=[
                        ("num", Pipeline(num_steps), NUMERIC_FEATURES),
                        (
                            "cat",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="constant", fill_value="No data")),
                                    (
                                        "onehot",
                                        OneHotEncoder(
                                            handle_unknown="infrequent_if_exist",
                                            min_frequency=min_frequency,
                                        ),
                                    ),
                                ]
                            ),
                            CATEGORICAL_FEATURES,
                        ),
                    ]
                ),
            ),
        ]
    )


def make_pipeline_log(model, min_frequency: int = 15, scale_numeric: bool = False) -> Pipeline:
    preprocessor = build_preprocessor(
        min_frequency=min_frequency,
        scale_numeric=scale_numeric,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                TransformedTargetRegressor(
                    regressor=model,
                    func=np.log1p,
                    inverse_func=np.expm1,
                ),
            ),
        ]
    )


def make_pipeline(model, min_frequency: int = 15, scale_numeric: bool = False) -> Pipeline:
    preprocessor = build_preprocessor(
        min_frequency=min_frequency,
        scale_numeric=scale_numeric,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def build_app_metadata(df: pd.DataFrame, model_family: str, selected_model_params: Dict) -> Dict:
    return {
        "raw_data_file_expected": "data/krisha_data.csv",
        "feature_columns_raw": RAW_FEATURE_COLUMNS,
        "district_options": sorted(df["district"].dropna().astype(str).unique().tolist()),
        "tip_doma_options": sorted(df["tip_doma"].fillna("No data").astype(str).replace({"nan": "No data"}).unique().tolist()),
        "toilet_options": sorted(df["toilet"].fillna("No data").astype(str).replace({"nan": "No data"}).unique().tolist()),
        "parking_options": sorted(df["parking"].fillna("No data").astype(str).replace({"nan": "No data"}).unique().tolist()),
        "status_options": sorted(df["status"].fillna("No data").astype(str).replace({"nan": "No data"}).unique().tolist()),
        "example_jk_values": df["jk"].fillna("No data").astype(str).value_counts().head(12).index.tolist(),
        "training_rows_after_cleaning": int(len(df)),
        "model_family": model_family,
        "selected_model_params": selected_model_params,
    }
