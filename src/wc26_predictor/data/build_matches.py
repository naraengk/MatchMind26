from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

import pandas as pd

from wc26_predictor.data.loading import canonical_team_name, read_qualified_teams


RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
DEFAULT_START_DATE = "2016-06-10"
DEFAULT_END_DATE = "2026-06-10"

OUTPUT_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral",
]


def build_recent_qualified_team_matches(
    qualified_teams_path: Union[str, Path],
    output_path: Union[str, Path],
    raw_output_path: Union[str, Path, None] = None,
    source_url: str = RESULTS_URL,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    require_all_teams: bool = True,
) -> pd.DataFrame:
    qualified = read_qualified_teams(qualified_teams_path)
    qualified_names = set(qualified["team"])

    # Download the full international results dataset (or read a local copy).
    results = pd.read_csv(source_url, parse_dates=["date"])
    if raw_output_path is not None:
        raw_output = Path(raw_output_path)
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(raw_output, index=False)

    # Use our consistent team spellings so names line up everywhere.
    results["home_team"] = results["home_team"].map(canonical_team_name)
    results["away_team"] = results["away_team"].map(canonical_team_name)

    # Keep only matches inside the date window...
    recent = results[
        (results["date"] >= pd.Timestamp(start_date))
        & (results["date"] <= pd.Timestamp(end_date))
    ].copy()
    # ...and only those involving at least one team that qualified for 2026.
    involves_qualified_team = recent["home_team"].isin(qualified_names) | recent[
        "away_team"
    ].isin(qualified_names)
    filtered = recent.loc[involves_qualified_team, OUTPUT_COLUMNS].sort_values("date")
    filtered = filtered.dropna(subset=["home_score", "away_score"])
    filtered[["home_score", "away_score"]] = filtered[["home_score", "away_score"]].astype(int)

    missing_qualified_teams = sorted(
        team
        for team in qualified_names
        if not ((filtered["home_team"] == team) | (filtered["away_team"] == team)).any()
    )
    if require_all_teams and missing_qualified_teams:
        raise ValueError(
            "No matches found for qualified teams after alias normalization: "
            + ", ".join(missing_qualified_teams)
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output, index=False, date_format="%Y-%m-%d")
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and filter recent international matches for 2026 qualified teams."
    )
    parser.add_argument(
        "--source-url",
        default=RESULTS_URL,
        help="URL or local path for the full international results CSV.",
    )
    parser.add_argument(
        "--qualified-teams",
        default="data/sample/qualified_teams.csv",
        help="CSV containing the 48 qualified WC 2026 teams.",
    )
    parser.add_argument(
        "--output",
        default="data/sample/matches.csv",
        help="Filtered match CSV used by the model.",
    )
    parser.add_argument(
        "--raw-output",
        default="data/raw/international_results.csv",
        help="Optional path to store the full downloaded source dataset.",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="Inclusive lower date bound. Default is the last 10 years from 2026-05-24.",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help="Inclusive upper date bound. Default is today in this project context.",
    )
    args = parser.parse_args()

    frame = build_recent_qualified_team_matches(
        qualified_teams_path=args.qualified_teams,
        source_url=args.source_url,
        output_path=args.output,
        raw_output_path=args.raw_output,
        start_date=args.start_date,
        end_date=args.end_date,
        require_all_teams=True,
    )
    print(f"Wrote {len(frame):,} matches to {args.output}")
    print(f"Date range: {frame['date'].min().date()} to {frame['date'].max().date()}")
    print(f"Tournaments: {frame['tournament'].nunique():,}")


if __name__ == "__main__":
    main()
