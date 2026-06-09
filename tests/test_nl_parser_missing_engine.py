import pandas as pd
from app.utils.nl_parser import handle_nl_query


def test_missing_engine_auto_select_single():
    # Only Unit ID 1 exists in this minimal dataset
    df = pd.DataFrame({"unit": [1]})
    session = {}
    ok, msg, selection = handle_nl_query("state of engine 14 in FD001", df, session)
    assert ok is True
    assert selection == ("FD001", 1)
    assert session["select_engine_override_FD001"] == 1
    assert "Only Unit ID 1 available" in msg
