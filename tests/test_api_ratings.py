from pathlib import Path

import pandas as pd

from wc26_predictor.api.main import (
    clean_full_squads,
    clean_squads,
    roster_name_key,
    squad_core,
)
from wc26_predictor.data.loading import read_squads


ROOT = Path(__file__).resolve().parents[1]


def test_core_and_roster_use_the_same_whole_number_ratings() -> None:
    core = clean_squads(read_squads(ROOT / "data/sample/squads.csv"))
    roster = clean_full_squads(
        pd.read_csv(ROOT / "data/sample/worldcup_squads_26.csv"),
        core,
    )

    for team in core["team"].unique():
        core_rows = squad_core(roster, team)
        roster_lookup = {
            roster_name_key(str(row.player)): row
            for row in roster[roster["team"] == team].itertuples(index=False)
        }
        for player in core_rows:
            matching_roster_player = roster_lookup[roster_name_key(player["player"])]
            assert player["club_strength"] == matching_roster_player.club_rating
            assert player["overall_rating"] == matching_roster_player.player_rating
            assert isinstance(player["club_strength"], int)
            assert isinstance(player["overall_rating"], int)


def test_recalibrated_rating_ranges_are_bounded() -> None:
    core = clean_squads(read_squads(ROOT / "data/sample/squads.csv"))
    roster = clean_full_squads(
        pd.read_csv(ROOT / "data/sample/worldcup_squads_26.csv"),
        core,
    )

    assert roster["club_rating"].between(60, 87).all()
    assert roster["player_rating"].between(60, 89).all()
    assert not roster["club"].isin(["", "Club not listed"]).any()
