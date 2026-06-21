"""Fetch BTS Form 41 Schedule P-5.2 and T-2 and produce benchmarks CSV.

This standalone script attempts to download Form 41 Schedule P-5.2
(operating expenses) and Schedule T-2 (traffic & capacity) from the
US DOT BTS TranStats site, compute engine maintenance cost per block
hour for selected aircraft types, and write results to
`data/bts_benchmarks.csv`.

Usage:
    python scripts/fetch_bts_benchmarks.py

The script is resilient: if BTS is unreachable it writes fallback
values with source="BTS_FALLBACK_{year}".
"""

from __future__ import annotations

import io
import logging
from typing import Dict, List

import pandas as pd
import requests

# Configuration dictionary — no other hardcoded constants
CONFIG: Dict = {
    "year": 2023,
    "aircraft_codes": {
        612: {"name": "B737-800", "category": "narrowbody"},
        634: {"name": "A320", "category": "narrowbody"},
        667: {"name": "B777-200", "category": "widebody"},
        672: {"name": "B787-9", "category": "widebody"},
    },
    "accounts": [5245, 5246, 5247],
    "fallback": {"narrowbody": 180.0, "widebody": 420.0},
    "output_csv": "data/bts_benchmarks.csv",
    "sources_md": "data/sources.md",
    "bts_timeout": 15,
}


logger = logging.getLogger("fetch_bts_benchmarks")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def fetch_table_csv(candidate_urls: List[str]) -> pd.DataFrame:
    """Attempt to fetch a CSV from a list of candidate URLs.

    Tries each URL in order and returns the first successful CSV parsed
    into a DataFrame.

    Inputs:
        candidate_urls: List of full URLs expected to return CSV content.

    Output:
        Parsed pandas DataFrame.

    Failure:
        Raises requests.RequestException or ValueError if all attempts fail.
    """
    last_exc = None
    for url in candidate_urls:
        try:
            logger.info("Attempting to download: %s", url)
            resp = requests.get(url, timeout=CONFIG["bts_timeout"])
            resp.raise_for_status()
            # Some BTS endpoints return non-UTF8; pandas can read from buffer
            buf = io.StringIO(resp.text)
            df = pd.read_csv(buf)
            logger.info(
                "Downloaded %d rows from %s",
                len(df),
            )
            return df
        except Exception as exc:  # pragma: no cover - network I/O
            logger.debug("Failed to fetch %s: %s", url, exc)
            last_exc = exc
    raise last_exc  # type: ignore


def parse_p5_operating_expenses(df_p5: pd.DataFrame) -> pd.DataFrame:
    """Normalize P-5.2 operating expenses into a table with required fields.

    Input:
        df_p5: Raw DataFrame from P-5.2 download.

    Output:
        DataFrame with columns: CARRIER, AIRCRAFT_TYPE (int), YEAR, QUARTER,
        ENGINE_LABOR, ENGINE_MATERIALS, ENGINE_OUTSOURCED

    The function is defensive: it attempts to detect common column names
    and pivot account-based layouts into the expected schema.
    """
    df = df_p5.copy()

    # Normalize column names to upper for easier matching
    df.columns = [c.upper().strip() for c in df.columns]

    # If any are missing, try to guess common variants
    if "UNIT" in df.columns and "AIRCRAFT_TYPE" not in df.columns:
        df.rename(columns={"UNIT": "AIRCRAFT_TYPE"}, inplace=True)

    # If the file is in long format with ACCOUNT and AMOUNT
    if "ACCOUNT" in df.columns and ("AMOUNT" in df.columns or "VALUE" in df.columns):
        amount_col = "AMOUNT" if "AMOUNT" in df.columns else "VALUE"
        # Keep only our accounts
        df = df[df["ACCOUNT"].isin(CONFIG["accounts"])].copy()
        # Convert AMOUNT to numeric
        df[amount_col] = pd.to_numeric(
            df[amount_col].astype(str).str.replace(",", ""), errors="coerce"
        ).fillna(0)
        # Pivot to wide: each account becomes a column
        pivot = (
            df.groupby(["CARRIER", "AIRCRAFT_TYPE", "YEAR", "QUARTER", "ACCOUNT"])
            .agg({amount_col: "sum"})
            .reset_index()
            .pivot_table(
                index=["CARRIER", "AIRCRAFT_TYPE", "YEAR", "QUARTER"],
                columns="ACCOUNT",
                values=amount_col,
                aggfunc="sum",
            )
            .reset_index()
        )
        # Rename account columns to friendly names
        rename_map = {}
        acct_names = {
            5245: "ENGINE_LABOR",
            5246: "ENGINE_MATERIALS",
            5247: "ENGINE_OUTSOURCED",
        }
        for acct, name in acct_names.items():
            if acct in pivot.columns:
                rename_map[acct] = name
        pivot.rename(columns=rename_map, inplace=True)
        # Ensure missing cols exist
        for col in ["ENGINE_LABOR", "ENGINE_MATERIALS", "ENGINE_OUTSOURCED"]:
            if col not in pivot.columns:
                pivot[col] = 0.0
        return pivot[
            [
                "CARRIER",
                "AIRCRAFT_TYPE",
                "YEAR",
                "QUARTER",
                "ENGINE_LABOR",
                "ENGINE_MATERIALS",
                "ENGINE_OUTSOURCED",
            ]
        ]

    # If file already has explicit columns named as needed, try to select them

    if set(["CARRIER", "AIRCRAFT_TYPE", "YEAR", "QUARTER"]).issubset(
        df.columns
    ) and any(
        x in df.columns
        for x in ["ENGINE_LABOR", "ENGINE_MATERIALS", "ENGINE_OUTSOURCED"]
    ):
        # fill missing cost columns with zeros
        for col in ["ENGINE_LABOR", "ENGINE_MATERIALS", "ENGINE_OUTSOURCED"]:
            if col not in df.columns:
                df[col] = 0.0
        return df[
            [
                "CARRIER",
                "AIRCRAFT_TYPE",
                "YEAR",
                "QUARTER",
                "ENGINE_LABOR",
                "ENGINE_MATERIALS",
                "ENGINE_OUTSOURCED",
            ]
        ]

    raise ValueError("Unrecognized P-5.2 CSV layout — cannot parse")


def parse_t2_traffic(df_t2: pd.DataFrame) -> pd.DataFrame:
    """Normalize T-2 traffic table to have BLOCK_HOURS.

    Input:
        df_t2: Raw DataFrame from Schedule T-2.

    Output:
        DataFrame with columns: CARRIER, AIRCRAFT_TYPE, YEAR, QUARTER, BLOCK_HOURS
    """
    df = df_t2.copy()
    df.columns = [c.upper().strip() for c in df.columns]

    if "BLOCK_HOURS" in df.columns:
        return df[["CARRIER", "AIRCRAFT_TYPE", "YEAR", "QUARTER", "BLOCK_HOURS"]]

    # Some exports use 'BLOCKHOURS' or 'BLOCK HRS'
    for candidate in ("BLOCKHOURS", "BLOCK HRS", "BLOCK_HR"):
        if candidate in df.columns:
            df.rename(columns={candidate: "BLOCK_HOURS"}, inplace=True)
            return df[["CARRIER", "AIRCRAFT_TYPE", "YEAR", "QUARTER", "BLOCK_HOURS"]]

    raise ValueError("Unrecognized T-2 CSV layout — BLOCK_HOURS column not found")


def compute_benchmarks(
    df_costs: pd.DataFrame, df_traffic: pd.DataFrame, year: int
) -> pd.DataFrame:
    """Compute engine cost per block hour per aircraft type for a given year.

    Aggregates across carriers, filters to configured aircraft types, and
    returns a DataFrame with code, name, category, value, year, source.
    """
    # Merge costs and traffic on carrier/aircraft/year/quarter
    # Ensure numerical columns
    df_costs = df_costs.copy()
    df_traffic = df_traffic.copy()
    for col in ["ENGINE_LABOR", "ENGINE_MATERIALS", "ENGINE_OUTSOURCED"]:
        df_costs[col] = pd.to_numeric(df_costs[col], errors="coerce").fillna(0.0)
    df_traffic["BLOCK_HOURS"] = pd.to_numeric(
        df_traffic["BLOCK_HOURS"], errors="coerce"
    ).fillna(0.0)

    # Sum the engine cost fields
    df_costs["ENGINE_COST_TOTAL"] = df_costs[
        ["ENGINE_LABOR", "ENGINE_MATERIALS", "ENGINE_OUTSOURCED"]
    ].sum(axis=1)

    # Filter to desired year
    df_costs = df_costs[df_costs["YEAR"] == year]
    df_traffic = df_traffic[df_traffic["YEAR"] == year]

    # Aggregate by AIRCRAFT_TYPE across carriers and quarters for the year
    costs_agg = (
        df_costs.groupby("AIRCRAFT_TYPE")
        .agg({"ENGINE_COST_TOTAL": "sum"})
        .rename(columns={"ENGINE_COST_TOTAL": "ENGINE_COST_TOTAL_SUM"})
    )
    blocks_agg = (
        df_traffic.groupby("AIRCRAFT_TYPE")
        .agg({"BLOCK_HOURS": "sum"})
        .rename(columns={"BLOCK_HOURS": "BLOCK_HOURS_SUM"})
    )

    df_join = costs_agg.join(blocks_agg, how="inner").reset_index()

    # Keep only configured aircraft codes
    allowed_codes = set(CONFIG["aircraft_codes"].keys())
    df_join = df_join[df_join["AIRCRAFT_TYPE"].isin(allowed_codes)].copy()

    # Compute cost per block-hour
    df_join["ENGINE_COST_PER_BLOCK_HOUR"] = df_join["ENGINE_COST_TOTAL_SUM"] / df_join[
        "BLOCK_HOURS_SUM"
    ].replace({0: pd.NA})

    # Map code -> name, category
    rows = []
    for _, row in df_join.iterrows():
        code = int(row["AIRCRAFT_TYPE"])
        meta = CONFIG["aircraft_codes"].get(code)
        if not meta:
            continue
        rows.append(
            {
                "aircraft_type_code": code,
                "aircraft_type_name": meta["name"],
                "category": meta["category"],
                "engine_cost_per_block_hour_usd": float(
                    row["ENGINE_COST_PER_BLOCK_HOUR"]
                ),
                "year": year,
                "source": "BTS_FORM41_P5.2_T2",
            }
        )

    return pd.DataFrame(rows)


def write_output(df_out: pd.DataFrame) -> None:
    """Write output DataFrame to CSV path defined in CONFIG.

    Inputs:
        df_out: DataFrame with benchmark rows.
    """
    df_out.to_csv(CONFIG["output_csv"], index=False)
    logger.info("Wrote %d benchmark rows to %s", len(df_out), CONFIG["output_csv"])


def write_fallback() -> pd.DataFrame:
    """Write fallback rows when BTS cannot be reached.

    The fallback populates each configured aircraft code with the
    category-level fallback value.
    """
    rows = []
    for code, meta in CONFIG["aircraft_codes"].items():
        cat = meta["category"]
        val = CONFIG["fallback"][cat]
        rows.append(
            {
                "aircraft_type_code": code,
                "aircraft_type_name": meta["name"],
                "category": cat,
                "engine_cost_per_block_hour_usd": float(val),
                "year": CONFIG["year"],
                "source": f"BTS_FALLBACK_{CONFIG['year']}",
            }
        )
    df_fallback = pd.DataFrame(rows)
    write_output(df_fallback)
    logger.warning(
        "BTS unreachable — wrote fallback benchmarks to %s", CONFIG["output_csv"]
    )
    return df_fallback


def append_sources_md() -> None:
    """Append a sources note to data/sources.md per requirements.

    If the file does not exist it will be created.
    """
    text = (
        "BTS Form 41 Schedule P-5.2 + T-2, US DOT transtats.bts.gov,\n"
        "accessed May 2026, FY2023 full year, accounts 5245+5246+5247\n"
    )
    try:
        with open(CONFIG["sources_md"], "a", encoding="utf8") as fh:
            fh.write(text)
        logger.info("Appended BTS source note to %s", CONFIG["sources_md"])
    except Exception as exc:
        logger.warning("Failed to write sources.md: %s", exc)


def main() -> None:
    """Main entrypoint: attempt BTS fetch, compute benchmarks, write CSV and print summary.

    Behavior:
        - Try to fetch P-5.2 and T-2 using public download endpoints.
        - If any step fails, fall back to hardcoded values as required.
        - Print narrowbody & widebody averages to console.
    """
    # Candidate URLs — these are best-effort. If BTS layout changes the
    # script will gracefully fall back.
    p5_candidate_urls = [
        # Common TranStats download pattern (may need adaptation in future)
        "https://www.transtats.bts.gov/DownLoad_Table.asp?Table_ID=4018&Download=csv",
        "https://www.transtats.bts.gov/DownLoad_Table.asp?Table_ID=4039&Download=csv",
    ]

    t2_candidate_urls = [
        "https://www.transtats.bts.gov/DownLoad_Table.asp?Table_ID=240&Download=csv",
        "https://www.transtats.bts.gov/DownLoad_Table.asp?Table_ID=241&Download=csv",
    ]

    try:
        df_p5 = fetch_table_csv(p5_candidate_urls)
        df_t2 = fetch_table_csv(t2_candidate_urls)
        df_costs = parse_p5_operating_expenses(df_p5)
        df_traffic = parse_t2_traffic(df_t2)
        df_bench = compute_benchmarks(df_costs, df_traffic, CONFIG["year"])
        if df_bench.empty:
            logger.warning("No benchmarks computed from BTS data — falling back")
            df_bench = write_fallback()
        else:
            write_output(df_bench)
    except Exception as exc:  # pragma: no cover - network and unpredictable parsing
        logger.warning("Encountered error fetching/parsing BTS data: %s", exc)
        df_bench = write_fallback()

    # Print narrowbody and widebody averages
    nb = df_bench[df_bench["category"] == "narrowbody"][
        "engine_cost_per_block_hour_usd"
    ].mean()
    wb = df_bench[df_bench["category"] == "widebody"][
        "engine_cost_per_block_hour_usd"
    ].mean()
    print(f"Narrowbody average engine cost per block-hour: ${nb:.2f}")
    print(f"Widebody average engine cost per block-hour: ${wb:.2f}")

    # Append sources note
    append_sources_md()


if __name__ == "__main__":
    main()
