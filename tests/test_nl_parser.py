import pytest
from app.utils.nl_parser import parse_engine_query


@pytest.mark.parametrize(
    "text,expected",
    [
        ("state of engine 14 in FD001", ("FD001", 14)),
        ("FD001 engine 14", ("FD001", 14)),
        ("engine 7 in FD002", ("FD002", 7)),
        ("what is the status of unit 3", ("FD001", 3)),
        ("FD 003 engine 12", ("FD003", 12)),
    ],
)
def test_parse_engine_query_success(text, expected):
    assert parse_engine_query(text) == expected


def test_parse_engine_query_fail():
    assert parse_engine_query("") is None
    assert parse_engine_query("no numbers here") is None
