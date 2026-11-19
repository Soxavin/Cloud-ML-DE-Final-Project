"""Builds the preprocessing part of the model pipeline.

The notebook, the tests and (through the saved model) the model service all
use this one function, so the cleaning steps can't drift apart.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from bankml.features import add_contact_flag


def build_preprocessor(categorical: list[str], numeric: list[str], log_numeric: list[str]) -> Pipeline:
    """Flag step first, then encode and scale.

    `numeric` uses the column names *after* the flag step, so it should
    include `previously_contacted`. Columns not listed in any group are dropped.
    """
    log_then_scale = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])

    encode = ColumnTransformer(
        transformers=[
            # dense output: HistGradientBoosting and GaussianNB can't take sparse input
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("log_numeric", log_then_scale, log_numeric),
            ("numeric", StandardScaler(), numeric),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("flags", FunctionTransformer(add_contact_flag)),
        ("encode", encode),
    ])
