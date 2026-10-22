"""Explicit feature allowlists: no IDs, outcomes, site or external classifier scores."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CLINICAL_NUMERIC = ["age_approx", "clin_size_long_diam_mm"]
CLINICAL_CATEGORICAL = ["sex", "anatom_site_general"]
MORPHOMETRY_NUMERIC = [
    "tbp_lv_A",
    "tbp_lv_Aext",
    "tbp_lv_B",
    "tbp_lv_Bext",
    "tbp_lv_C",
    "tbp_lv_Cext",
    "tbp_lv_H",
    "tbp_lv_Hext",
    "tbp_lv_L",
    "tbp_lv_Lext",
    "tbp_lv_areaMM2",
    "tbp_lv_area_perim_ratio",
    "tbp_lv_color_std_mean",
    "tbp_lv_deltaA",
    "tbp_lv_deltaB",
    "tbp_lv_deltaL",
    "tbp_lv_deltaLBnorm",
    "tbp_lv_eccentricity",
    "tbp_lv_minorAxisMM",
    "tbp_lv_norm_border",
    "tbp_lv_norm_color",
    "tbp_lv_perimeterMM",
    "tbp_lv_radial_color_std_max",
    "tbp_lv_stdL",
    "tbp_lv_stdLExt",
    "tbp_lv_symm_2axis",
]


def feature_columns(feature_set: str) -> tuple[list[str], list[str]]:
    if feature_set not in {"clinical", "morphometry"}:
        raise ValueError(f"Unknown feature set: {feature_set}")
    numeric = CLINICAL_NUMERIC.copy()
    if feature_set == "morphometry":
        numeric += MORPHOMETRY_NUMERIC
    return numeric, CLINICAL_CATEGORICAL.copy()


def select_features(frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    numeric, categorical = feature_columns(feature_set)
    missing = set(numeric + categorical) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")
    result = frame[numeric + categorical].copy()
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result[numeric] = result[numeric].replace([np.inf, -np.inf], np.nan)
    for column in categorical:
        result[column] = result[column].fillna("missing").astype(str)
    return result


def make_preprocessor(feature_set: str) -> ColumnTransformer:
    numeric, categorical = feature_columns(feature_set)
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median", add_indicator=True, keep_empty_features=True
                            ),
                        ),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
