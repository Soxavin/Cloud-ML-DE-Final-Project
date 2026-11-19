from bankml.config import CATEGORICAL, LOG_NUMERIC, MODEL_INPUTS, NUMERIC


def test_model_inputs_are_the_frozen_17():
    assert len(MODEL_INPUTS) == 17
    assert len(set(MODEL_INPUTS)) == 17


def test_dropped_columns_are_not_inputs():
    for col in ["housing", "loan", "duration", "y"]:
        assert col not in MODEL_INPUTS


def test_feature_groups_cover_exactly_the_model_inputs():
    # pdays comes in raw and leaves the flag step as previously_contacted
    after_flag_step = set(MODEL_INPUTS) - {"pdays"} | {"previously_contacted"}
    assert set(CATEGORICAL) | set(NUMERIC) | set(LOG_NUMERIC) == after_flag_step
