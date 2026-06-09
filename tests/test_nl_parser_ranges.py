from app.utils.nl_parser import parse_engine_query


def test_parse_single_engine():
    assert parse_engine_query("state of engine 14 in FD001") == ("FD001", 14)


def test_parse_range_engines():
    ds, engines = parse_engine_query("engines 5-8 in FD002")
    assert ds == "FD002"
    assert isinstance(engines, list)
    assert engines[0] == 5 and engines[-1] == 8 and len(engines) == 4


def test_parse_fuzzy_dataset():
    assert parse_engine_query("engine 3 in fd 003") == ("FD003", 3)


def test_parse_short_synonym():
    assert parse_engine_query("eng 7") == ("FD001", 7)
