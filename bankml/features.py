"""Custom feature steps used inside the saved pipeline.

Keep every custom function the pipeline needs in here, not in a notebook
cell. A pickled pipeline only stores a *reference* to each function, so a
function defined in Colab would be saved as `__main__.<name>` and the model
service would fail to load it.
"""
import pandas as pd

# In this dataset pdays = 999 is a code for "never contacted before",
# not an actual number of days.
NEVER_CONTACTED = 999


def add_contact_flag(X: pd.DataFrame) -> pd.DataFrame:
    """Replace pdays with a 0/1 `previously_contacted` column."""
    return (
        X
        .assign(previously_contacted=(X["pdays"] != NEVER_CONTACTED).astype(int))
        .drop(columns=["pdays"])
    )
