import pandas as pd

from app.utils.nl_parser import parse_engine_query


def test_engine_override_flow(tmp_path, monkeypatch):
    # simulate a small df with engine ids
    df = pd.DataFrame({"unit": [1, 2, 3, 14]})
    parsed = parse_engine_query("state of engine 14 in FD001")
    assert parsed == ("FD001", 14)
    dataset_hint, engine_hint = parsed
    assert engine_hint in list(df["unit"])
