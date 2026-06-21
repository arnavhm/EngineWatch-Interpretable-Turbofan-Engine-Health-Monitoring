import pandas as pd

from app.utils.nl_parser import handle_nl_query


def make_dummy_df(units):
    return pd.DataFrame({"unit": units})


def test_handle_nl_single_override():
    df = make_dummy_df([1, 2, 3, 14])
    session = {}
    ok, msg, selection = handle_nl_query("state of engine 14 in FD001", df, session)
    assert ok is True
    assert selection == ("FD001", 14)
    assert session["last_dataset_id"] == "FD001"
    assert session["select_engine_override_FD001"] == 14


def test_handle_nl_range_override():
    df = make_dummy_df(list(range(1, 21)))
    session = {}
    ok, msg, selection = handle_nl_query("show engines 5 to 10 in FD001", df, session)
    assert ok is True
    assert selection == ("FD001", 5)
    assert session["select_engine_override_FD001"] in range(5, 11)
