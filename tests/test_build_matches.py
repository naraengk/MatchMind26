from pathlib import Path

from wc26_predictor.data.build_matches import build_recent_qualified_team_matches
from wc26_predictor.data.loading import read_matches


ROOT = Path(__file__).resolve().parents[1]


def test_read_matches_normalizes_source_team_names(tmp_path: Path) -> None:
    source = tmp_path / "matches.csv"
    source.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,neutral\n"
        "2024-03-26,Turkey,Iran,1,0,Friendly,False\n"
        "2024-06-08,Cape Verde,Curaçao,2,2,Friendly,True\n"
    )

    matches = read_matches(source)

    assert list(matches["home_team"]) == ["Turkiye", "Cabo Verde"]
    assert list(matches["away_team"]) == ["IR Iran", "Curacao"]


def test_build_recent_matches_filters_to_qualified_teams(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2015-01-01,England,France,1,0,Friendly,London,England,False\n"
        "2020-01-01,Iran,Japan,2,1,Friendly,Tehran,Iran,False\n"
        "2021-01-01,Italy,Wales,1,1,Friendly,Rome,Italy,False\n"
        "2022-01-01,Turkey,Czech Republic,2,2,Friendly,Istanbul,Turkey,False\n"
        "2026-06-01,England,France,1,1,Friendly,London,England,False\n"
    )
    output = tmp_path / "matches.csv"

    frame = build_recent_qualified_team_matches(
        qualified_teams_path=ROOT / "data/sample/qualified_teams.csv",
        output_path=output,
        raw_output_path=None,
        source_url=str(source),
        start_date="2016-05-24",
        end_date="2026-05-24",
        require_all_teams=False,
    )

    assert output.exists()
    assert len(frame) == 2
    assert "IR Iran" in set(frame["home_team"])
    assert "Turkiye" in set(frame["home_team"])
    assert "Czechia" in set(frame["away_team"])
