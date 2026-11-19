"""Train a quick placeholder model so the backend can be built before the real one exists.

It uses the same data, cleaning, split and feature lists as the notebook, but a
plain logistic regression instead of the stacking model. The real model from
the notebook replaces it as v1.

Run from the repo root:  python scripts/train_placeholder.py
"""
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from bankml.config import CATEGORICAL, LOG_NUMERIC, MODEL_INPUTS, NUMERIC  # noqa: E402
from bankml.pipeline import build_preprocessor  # noqa: E402

CSV_PATH = REPO_DIR / "data" / "bank-additional" / "bank-additional-full.csv"
MODEL_DIR = REPO_DIR / "model"


def main():
    df = (
        pd.read_csv(CSV_PATH, sep=";")
        .drop_duplicates()
        .reset_index(drop=True)
    )
    X = df[MODEL_INPUTS]
    y = (df["y"] == "yes").astype(int)

    # same split as the notebook; only the training part is used
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = Pipeline([
        ("prep", build_preprocessor(CATEGORICAL, NUMERIC, LOG_NUMERIC)),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ]).fit(X_train, y_train)

    categories = {}
    for col in CATEGORICAL:
        categories[col] = sorted(X_train[col].unique().tolist())

    metadata = {
        "model_version": "v0-placeholder",
        "features": MODEL_INPUTS,
        "categories": categories,
        "threshold": 0.5,
        "base_models": [],
        "meta_model": "LogisticRegression (placeholder, not the final model)",
        "test_metrics": None,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib", compress=3)
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Saved placeholder model to", MODEL_DIR)


if __name__ == "__main__":
    main()
