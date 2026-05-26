"""
Housing / Rental Market Momentum Screener

Purpose:
- Load Zillow-style wide CSV files for home values and rents.
- Calculate 3, 6, and 9 month moving averages.
- Calculate first derivative: month-over-month percentage change.
- Calculate second derivative: acceleration / change in monthly percentage change.
- Rank markets by price momentum, rental growth, and rent-to-price fundamentals.

Data sources to download manually:
- Zillow ZHVI home value CSV
- Zillow ZORI rent CSV

Example Zillow files:
- Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv
- Metro_zori_sm_month.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def load_zillow_wide_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a Zillow Research wide-format CSV and convert it to long format.

    Zillow files usually have metadata columns followed by monthly date columns.

    Expected format:
        RegionID, SizeRank, RegionName, RegionType, StateName, ..., 2000-01-31, 2000-02-29, ...
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path}")

    df = pd.read_csv(path)

    id_cols: list[str] = []
    date_cols: list[str] = []

    for col in df.columns:
        try:
            pd.to_datetime(col)
            date_cols.append(col)
        except Exception:
            id_cols.append(col)

    if not date_cols:
        raise ValueError("No date columns found. Make sure this is a Zillow wide-format CSV.")

    long_df = df.melt(
        id_vars=id_cols,
        value_vars=date_cols,
        var_name="date",
        value_name="value",
    )

    long_df["date"] = pd.to_datetime(long_df["date"])
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

    return long_df


def _region_key(df: pd.DataFrame) -> list[str]:
    """Return the best available region identifier columns for merging/grouping."""
    preferred = ["RegionID", "RegionName", "StateName"]
    return [col for col in preferred if col in df.columns]


def add_market_momentum_features(
    df: pd.DataFrame,
    group_cols: str | Iterable[str] = "RegionName",
    value_col: str = "value",
) -> pd.DataFrame:
    """Add moving averages, first derivative, second derivative, and trend signals."""
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    group_cols = list(group_cols)

    df = df.sort_values(group_cols + ["date"])
    g = df.groupby(group_cols, group_keys=False)

    df["ma_3m"] = g[value_col].transform(lambda x: x.rolling(3, min_periods=3).mean())
    df["ma_6m"] = g[value_col].transform(lambda x: x.rolling(6, min_periods=6).mean())
    df["ma_9m"] = g[value_col].transform(lambda x: x.rolling(9, min_periods=9).mean())

    # First derivative: monthly percent change.
    df["mom_pct"] = g[value_col].transform(lambda x: x.pct_change())
    df["mom_pct_3m_avg"] = g["mom_pct"].transform(lambda x: x.rolling(3, min_periods=3).mean())

    # Second derivative: change in monthly percent change.
    df["acceleration"] = g["mom_pct"].transform(lambda x: x.diff())
    df["acceleration_3m_avg"] = g["acceleration"].transform(lambda x: x.rolling(3, min_periods=3).mean())

    df["yoy_pct"] = g[value_col].transform(lambda x: x.pct_change(12))

    df["bullish_ma_stack"] = (df["ma_3m"] > df["ma_6m"]) & (df["ma_6m"] > df["ma_9m"])
    df["recovery_signal"] = (
        (df["mom_pct_3m_avg"] > 0)
        & (df["acceleration_3m_avg"] > 0)
        & (df["ma_3m"] > df["ma_6m"])
    )

    return df


def add_ma_crossovers(df: pd.DataFrame, group_cols: str | Iterable[str] = "RegionName") -> pd.DataFrame:
    """Add moving-average crossover signals to identify potential turning points."""
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    group_cols = list(group_cols)

    df = df.sort_values(group_cols + ["date"])
    g = df.groupby(group_cols, group_keys=False)

    df["ma_3_above_6"] = df["ma_3m"] > df["ma_6m"]
    df["ma_3_above_6_prev"] = g["ma_3_above_6"].transform(lambda x: x.shift(1))
    df["bullish_3_6_cross"] = (df["ma_3_above_6"] == True) & (df["ma_3_above_6_prev"] == False)

    df["ma_6_above_9"] = df["ma_6m"] > df["ma_9m"]
    df["bullish_3_6_9_stack"] = (df["ma_3m"] > df["ma_6m"]) & (df["ma_6m"] > df["ma_9m"])

    return df


def latest_market_snapshot(df: pd.DataFrame, group_cols: str | Iterable[str] = "RegionName") -> pd.DataFrame:
    """Keep only the latest available month for each market."""
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    group_cols = list(group_cols)

    df = df.sort_values(group_cols + ["date"])
    return df.groupby(group_cols, as_index=False).tail(1)


def add_ranking_score(df: pd.DataFrame) -> pd.DataFrame:
    """Create a simple home-price momentum score."""
    df = df.copy()

    df["bullish_ma_score"] = df["bullish_ma_stack"].fillna(False).astype(int)
    df["recovery_score"] = df["recovery_signal"].fillna(False).astype(int)

    df["velocity_rank"] = df["mom_pct_3m_avg"].rank(pct=True)
    df["acceleration_rank"] = df["acceleration_3m_avg"].rank(pct=True)
    df["yoy_rank"] = df["yoy_pct"].rank(pct=True)

    # Penalize markets that may already be overheated.
    df["overheated_penalty"] = np.where(df["yoy_pct"] > 0.12, 0.15, 0)

    df["momentum_score"] = (
        0.30 * df["velocity_rank"]
        + 0.30 * df["acceleration_rank"]
        + 0.20 * df["yoy_rank"]
        + 0.10 * df["bullish_ma_score"]
        + 0.10 * df["recovery_score"]
        - df["overheated_penalty"]
    )

    return df


def merge_home_and_rent_data(home_df: pd.DataFrame, rent_df: pd.DataFrame) -> pd.DataFrame:
    """Merge long-format Zillow home value and rent data on region/date."""
    home = home_df.rename(columns={"value": "home_value"})
    rent = rent_df.rename(columns={"value": "rent_value"})

    merge_cols = [col for col in ["RegionID", "RegionName", "StateName", "date"] if col in home.columns and col in rent.columns]
    if "date" not in merge_cols:
        raise ValueError("Both home_df and rent_df need a date column.")

    return home.merge(rent[merge_cols + ["rent_value"]], on=merge_cols, how="left")


def add_rent_features(df: pd.DataFrame, group_cols: str | Iterable[str] = "RegionName") -> pd.DataFrame:
    """Add rent growth and rent-to-price metrics."""
    df = df.copy()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    group_cols = list(group_cols)

    df = df.sort_values(group_cols + ["date"])
    g = df.groupby(group_cols, group_keys=False)

    df["rent_mom_pct"] = g["rent_value"].transform(lambda x: x.pct_change())
    df["rent_yoy_pct"] = g["rent_value"].transform(lambda x: x.pct_change(12))
    df["gross_rent_yield"] = (df["rent_value"] * 12) / df["home_value"]

    return df


def add_combined_investment_score(df: pd.DataFrame) -> pd.DataFrame:
    """Combine housing momentum with rent fundamentals."""
    df = df.copy()

    df["rent_growth_rank"] = df["rent_yoy_pct"].rank(pct=True)
    df["gross_yield_rank"] = df["gross_rent_yield"].rank(pct=True)

    df["investment_score"] = (
        0.50 * df["momentum_score"]
        + 0.25 * df["rent_growth_rank"]
        + 0.25 * df["gross_yield_rank"]
    )

    return df


def add_pickup_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Identify markets that may be in early recovery rather than already overheated."""
    df = df.copy()

    df["early_pickup_signal"] = (
        (df["ma_3m"] > df["ma_6m"])
        & (df["mom_pct_3m_avg"] > 0)
        & (df["acceleration_3m_avg"] > 0)
        & (df["yoy_pct"] > -0.03)
        & (df["yoy_pct"] < 0.08)
        & (df["rent_yoy_pct"] > 0)
        & (df["gross_rent_yield"] > 0.055)
    )

    df["pickup_score"] = (
        0.25 * df["velocity_rank"]
        + 0.25 * df["acceleration_rank"]
        + 0.20 * df["rent_growth_rank"]
        + 0.20 * df["gross_yield_rank"]
        + 0.10 * df["early_pickup_signal"].fillna(False).astype(int)
    )

    return df


def run_screener(home_file: str | Path, rent_file: str | Path, output_file: str | Path = "top_markets.csv") -> pd.DataFrame:
    """Run the full market screener and save the ranked output."""
    home_long = load_zillow_wide_csv(home_file)
    rent_long = load_zillow_wide_csv(rent_file)

    merged = merge_home_and_rent_data(home_long, rent_long)
    group_cols = _region_key(merged)
    if not group_cols:
        group_cols = ["RegionName"]

    merged = add_market_momentum_features(merged, group_cols=group_cols, value_col="home_value")
    merged = add_ma_crossovers(merged, group_cols=group_cols)
    merged = add_rent_features(merged, group_cols=group_cols)

    latest = latest_market_snapshot(merged, group_cols=group_cols)
    latest = add_ranking_score(latest)
    latest = add_combined_investment_score(latest)
    latest = add_pickup_signal(latest)

    ranked = latest.sort_values("pickup_score", ascending=False)

    preferred_cols = [
        "RegionName",
        "StateName",
        "date",
        "home_value",
        "rent_value",
        "ma_3m",
        "ma_6m",
        "ma_9m",
        "mom_pct_3m_avg",
        "acceleration_3m_avg",
        "yoy_pct",
        "rent_yoy_pct",
        "gross_rent_yield",
        "momentum_score",
        "investment_score",
        "pickup_score",
        "bullish_ma_stack",
        "recovery_signal",
        "early_pickup_signal",
        "bullish_3_6_cross",
    ]
    cols = [col for col in preferred_cols if col in ranked.columns]
    ranked[cols].to_csv(output_file, index=False)
    return ranked[cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank US housing/rental markets by momentum and rental fundamentals.")
    parser.add_argument("--home-file", required=True, help="Path to Zillow ZHVI wide CSV")
    parser.add_argument("--rent-file", required=True, help="Path to Zillow ZORI wide CSV")
    parser.add_argument("--output-file", default="top_markets.csv", help="Output CSV path")
    parser.add_argument("--top-n", type=int, default=25, help="Number of top markets to print")
    args = parser.parse_args()

    ranked = run_screener(args.home_file, args.rent_file, args.output_file)
    print(ranked.head(args.top_n).to_string(index=False))
    print(f"\nSaved ranked markets to: {args.output_file}")


if __name__ == "__main__":
    main()
