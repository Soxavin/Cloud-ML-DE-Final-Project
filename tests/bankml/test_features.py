import pandas as pd

from bankml.features import add_contact_flag


def make_frame():
    return pd.DataFrame({"age": [30, 41, 55], "pdays": [999, 6, 0]})


def test_flag_is_0_for_999_and_1_for_real_days():
    result = add_contact_flag(make_frame())
    assert result["previously_contacted"].tolist() == [0, 1, 1]


def test_pdays_column_is_removed():
    result = add_contact_flag(make_frame())
    assert "pdays" not in result.columns


def test_input_frame_is_not_modified():
    original = make_frame()
    add_contact_flag(original)
    assert list(original.columns) == ["age", "pdays"]
    assert original["pdays"].tolist() == [999, 6, 0]


def test_other_columns_are_unchanged():
    result = add_contact_flag(make_frame())
    assert result["age"].tolist() == [30, 41, 55]
