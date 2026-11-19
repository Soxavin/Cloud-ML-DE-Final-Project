import numpy as np

from bankml.pipeline import build_preprocessor


def test_preprocessor_output_is_dense_with_no_nans(sample_rows, feature_lists):
    categorical, numeric, log_numeric = feature_lists
    X = sample_rows.drop(columns=["y"])

    output = build_preprocessor(categorical, numeric, log_numeric).fit_transform(X)

    # dense on purpose: HistGradientBoosting and GaussianNB refuse sparse input
    assert isinstance(output, np.ndarray)
    assert not np.isnan(output).any()
    assert output.shape[0] == len(X)


def test_unseen_category_does_not_raise(sample_rows, feature_lists):
    categorical, numeric, log_numeric = feature_lists
    X = sample_rows.drop(columns=["y"])
    preprocessor = build_preprocessor(categorical, numeric, log_numeric).fit(X)

    new_row = X.head(1).copy()
    new_row["job"] = "astronaut"  # never seen in training

    output = preprocessor.transform(new_row)
    assert output.shape == (1, preprocessor.transform(X.head(1)).shape[1])
