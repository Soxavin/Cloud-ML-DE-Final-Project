from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_rows() -> pd.DataFrame:
    """~200 real rows from the raw dataset (duration already dropped), target still attached."""
    return pd.read_csv(FIXTURES / "sample_rows.csv")


@pytest.fixture
def feature_lists(sample_rows):
    """Column groups for build_preprocessor, worked out from the sample itself.

    The real lists get frozen in the notebook (Task 3); tests only need something
    shaped the same way.
    """
    X = sample_rows.drop(columns=["y"])
    categorical = list(X.select_dtypes(exclude="number").columns)
    log_numeric = ["campaign"]
    numeric = []
    for col in X.select_dtypes(include="number").columns:
        if col not in ("pdays", "campaign"):
            numeric.append(col)
    numeric.append("previously_contacted")  # created by add_contact_flag
    return categorical, numeric, log_numeric
