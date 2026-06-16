from __future__ import annotations

import os
import re
import threading
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from wc26_predictor.data.loading import read_squads
from wc26_predictor.features.build_features import build_prediction_row


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = Path(os.getenv("WC26_MODEL_PATH", PROJECT_ROOT / "artifacts/model.joblib"))
MATCHES_PATH = PROJECT_ROOT / "data/sample/matches.csv"
QUALIFIED_TEAMS_PATH = PROJECT_ROOT / "data/sample/qualified_teams.csv"
SQUADS_PATH = PROJECT_ROOT / "data/sample/squads.csv"
TEAM_METADATA_PATH = PROJECT_ROOT / "data/sample/team_metadata.csv"
FIXTURES_PATH = PROJECT_ROOT / "data/sample/group_stage_fixtures.csv"
FULL_SQUADS_PATH = PROJECT_ROOT / "data/sample/worldcup_squads_26.csv"
ASSETS_PATH = PROJECT_ROOT / "assets"
WEB_DIR = Path(__file__).resolve().parent / "web"

HOST_COUNTRIES = {"Canada": "Canada", "Mexico": "Mexico", "United States": "United States"}

CITY_COUNTRIES = {
    "Atlanta": "United States",
    "Arlington": "United States",
    "East Rutherford": "United States",
    "Foxborough": "United States",
    "Guadalupe": "Mexico",
    "Houston": "United States",
    "Inglewood": "United States",
    "Kansas City": "United States",
    "Los Angeles": "United States",
    "Miami Gardens": "United States",
    "Mexico City": "Mexico",
    "Philadelphia": "United States",
    "Santa Clara": "United States",
    "Seattle": "United States",
    "Toronto": "Canada",
    "Vancouver": "Canada",
    "Zapopan": "Mexico",
}

TEAM_FLAGS = {
    "Algeria": "🇩🇿",
    "Argentina": "🇦🇷",
    "Australia": "🇦🇺",
    "Austria": "🇦🇹",
    "Belgium": "🇧🇪",
    "Bosnia and Herzegovina": "🇧🇦",
    "Brazil": "🇧🇷",
    "Cabo Verde": "🇨🇻",
    "Canada": "🇨🇦",
    "Colombia": "🇨🇴",
    "Cote d'Ivoire": "🇨🇮",
    "Croatia": "🇭🇷",
    "Curacao": "🇨🇼",
    "Czechia": "🇨🇿",
    "DR Congo": "🇨🇩",
    "Ecuador": "🇪🇨",
    "Egypt": "🇪🇬",
    "England": "flag-england",
    "France": "🇫🇷",
    "Germany": "🇩🇪",
    "Ghana": "🇬🇭",
    "Haiti": "🇭🇹",
    "IR Iran": "🇮🇷",
    "Iraq": "🇮🇶",
    "Japan": "🇯🇵",
    "Jordan": "🇯🇴",
    "Mexico": "🇲🇽",
    "Morocco": "🇲🇦",
    "Netherlands": "🇳🇱",
    "New Zealand": "🇳🇿",
    "Norway": "🇳🇴",
    "Panama": "🇵🇦",
    "Paraguay": "🇵🇾",
    "Portugal": "🇵🇹",
    "Qatar": "🇶🇦",
    "Saudi Arabia": "🇸🇦",
    "Scotland": "flag-scotland",
    "Senegal": "🇸🇳",
    "South Africa": "🇿🇦",
    "South Korea": "🇰🇷",
    "Spain": "🇪🇸",
    "Sweden": "🇸🇪",
    "Switzerland": "🇨🇭",
    "Tunisia": "🇹🇳",
    "Turkiye": "🇹🇷",
    "United States": "🇺🇸",
    "Uruguay": "🇺🇾",
    "Uzbekistan": "🇺🇿",
    "Afghanistan": "🇦🇫",
    "Albania": "🇦🇱",
    "Andorra": "🇦🇩",
    "Angola": "🇦🇴",
    "Anguilla": "🇦🇮",
    "Antigua and Barbuda": "🇦🇬",
    "Armenia": "🇦🇲",
    "Aruba": "🇦🇼",
    "Azerbaijan": "🇦🇿",
    "Bahrain": "🇧🇭",
    "Bangladesh": "🇧🇩",
    "Barbados": "🇧🇧",
    "Basque Country": "🏳",
    "Belarus": "🇧🇾",
    "Belize": "🇧🇿",
    "Benin": "🇧🇯",
    "Bermuda": "🇧🇲",
    "Bolivia": "🇧🇴",
    "Botswana": "🇧🇼",
    "British Virgin Islands": "🇻🇬",
    "Bulgaria": "🇧🇬",
    "Burkina Faso": "🇧🇫",
    "Burundi": "🇧🇮",
    "Cambodia": "🇰🇭",
    "Cameroon": "🇨🇲",
    "Catalonia": "🏳",
    "Cayman Islands": "🇰🇾",
    "Central African Republic": "🇨🇫",
    "Chad": "🇹🇩",
    "Chile": "🇨🇱",
    "China": "🇨🇳",
    "Comoros": "🇰🇲",
    "Congo": "🇨🇬",
    "Costa Rica": "🇨🇷",
    "Cuba": "🇨🇺",
    "Cyprus": "🇨🇾",
    "Denmark": "🇩🇰",
    "Djibouti": "🇩🇯",
    "Dominica": "🇩🇲",
    "Dominican Republic": "🇩🇴",
    "El Salvador": "🇸🇻",
    "Equatorial Guinea": "🇬🇶",
    "Estonia": "🇪🇪",
    "Eswatini": "🇸🇿",
    "Ethiopia": "🇪🇹",
    "Faroe Islands": "🇫🇴",
    "Fiji": "🇫🇯",
    "Finland": "🇫🇮",
    "French Guiana": "🇬🇫",
    "Gabon": "🇬🇦",
    "Galicia": "🏳",
    "Gambia": "🇬🇲",
    "Georgia": "🇬🇪",
    "Gibraltar": "🇬🇮",
    "Greece": "🇬🇷",
    "Grenada": "🇬🇩",
    "Guadeloupe": "🇬🇵",
    "Guatemala": "🇬🇹",
    "Guinea": "🇬🇳",
    "Guinea-Bissau": "🇬🇼",
    "Guyana": "🇬🇾",
    "Honduras": "🇭🇳",
    "Hong Kong": "🇭🇰",
    "Hungary": "🇭🇺",
    "Iceland": "🇮🇸",
    "India": "🇮🇳",
    "Indonesia": "🇮🇩",
    "Israel": "🇮🇱",
    "Italy": "🇮🇹",
    "Jamaica": "🇯🇲",
    "Kazakhstan": "🇰🇿",
    "Kenya": "🇰🇪",
    "Kosovo": "🇽🇰",
    "Kuwait": "🇰🇼",
    "Kyrgyzstan": "🇰🇬",
    "Laos": "🇱🇦",
    "Latvia": "🇱🇻",
    "Lebanon": "🇱🇧",
    "Lesotho": "🇱🇸",
    "Liberia": "🇱🇷",
    "Libya": "🇱🇾",
    "Liechtenstein": "🇱🇮",
    "Lithuania": "🇱🇹",
    "Luxembourg": "🇱🇺",
    "Madagascar": "🇲🇬",
    "Malawi": "🇲🇼",
    "Malaysia": "🇲🇾",
    "Maldives": "🇲🇻",
    "Mali": "🇲🇱",
    "Malta": "🇲🇹",
    "Martinique": "🇲🇶",
    "Mauritania": "🇲🇷",
    "Mauritius": "🇲🇺",
    "Moldova": "🇲🇩",
    "Mongolia": "🇲🇳",
    "Montenegro": "🇲🇪",
    "Montserrat": "🇲🇸",
    "Mozambique": "🇲🇿",
    "Myanmar": "🇲🇲",
    "Namibia": "🇳🇦",
    "Nepal": "🇳🇵",
    "New Caledonia": "🇳🇨",
    "Nicaragua": "🇳🇮",
    "Niger": "🇳🇪",
    "Nigeria": "🇳🇬",
    "North Korea": "🇰🇵",
    "North Macedonia": "🇲🇰",
    "Northern Ireland": "🏳",
    "Oman": "🇴🇲",
    "Pakistan": "🇵🇰",
    "Palestine": "🇵🇸",
    "Papua New Guinea": "🇵🇬",
    "Peru": "🇵🇪",
    "Philippines": "🇵🇭",
    "Poland": "🇵🇱",
    "Puerto Rico": "🇵🇷",
    "Republic of Ireland": "🇮🇪",
    "Romania": "🇷🇴",
    "Russia": "🇷🇺",
    "Rwanda": "🇷🇼",
    "Saint Kitts and Nevis": "🇰🇳",
    "Saint Lucia": "🇱🇨",
    "Saint Martin": "🇲🇫",
    "Saint Vincent and the Grenadines": "🇻🇨",
    "Samoa": "🇼🇸",
    "San Marino": "🇸🇲",
    "Serbia": "🇷🇸",
    "Seychelles": "🇸🇨",
    "Sierra Leone": "🇸🇱",
    "Singapore": "🇸🇬",
    "Sint Maarten": "🇸🇽",
    "Slovakia": "🇸🇰",
    "Slovenia": "🇸🇮",
    "Solomon Islands": "🇸🇧",
    "Somalia": "🇸🇴",
    "South Sudan": "🇸🇸",
    "Sri Lanka": "🇱🇰",
    "Sudan": "🇸🇩",
    "Suriname": "🇸🇷",
    "Syria": "🇸🇾",
    "São Tomé and Príncipe": "🇸🇹",
    "Tahiti": "🇵🇫",
    "Taiwan": "🇹🇼",
    "Tajikistan": "🇹🇯",
    "Tanzania": "🇹🇿",
    "Thailand": "🇹🇭",
    "Togo": "🇹🇬",
    "Trinidad and Tobago": "🇹🇹",
    "Turkmenistan": "🇹🇲",
    "Turks and Caicos Islands": "🇹🇨",
    "Uganda": "🇺🇬",
    "Ukraine": "🇺🇦",
    "United Arab Emirates": "🇦🇪",
    "United States Virgin Islands": "🇻🇮",
    "Vanuatu": "🇻🇺",
    "Venezuela": "🇻🇪",
    "Vietnam": "🇻🇳",
    "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿",
    "Yemen": "🇾🇪",
    "Zambia": "🇿🇲",
    "Zimbabwe": "🇿🇼",
}

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Build the slow tournament projection once at startup, in the background.

    The 10,000-run simulation takes a few seconds, so we run it on a side thread
    when the server boots. The site stays responsive, and by the time someone
    opens the Knockout page the result is already cached and loads instantly.
    """

    def _warm() -> None:
        try:
            load_context()
            tournament_payload()
        except Exception:  # noqa: BLE001 - warming is best-effort, never fatal
            pass

    threading.Thread(target=_warm, name="cache-warm", daemon=True).start()
    yield


app = FastAPI(title="WC 2026 Predictor", version="0.2.0", lifespan=lifespan)
if ASSETS_PATH.exists():
    app.mount("/assets", StaticFiles(directory=ASSETS_PATH), name="assets")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    neutral: bool = True
    tournament: str = "FIFA World Cup"


def _jsonable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: _jsonable(value) for key, value in row.items()} for row in df.to_dict("records")]


def calibrated_club_rating(value: Any) -> int:
    # Rescale raw club strength onto a familiar 60-87 "FIFA-style" range for the UI.
    rating = float(value)
    return round(max(60, min(87, 60 + (rating - 63) * 0.85)))


def calibrated_player_rating(value: Any, club_rating: Any | None = None) -> int:
    # Same idea for players (60-89), with a small bump for playing at a stronger club.
    rating = float(value)
    club_adjustment = 0.0 if club_rating is None else (float(club_rating) - 74) * 0.10
    return round(max(60, min(89, 60 + (rating - 62) * 0.80 + club_adjustment)))


def core_player_rating(row: pd.Series) -> int:
    # Build an overall rating by weighting attack/defense/club/experience by position.
    position = str(row["position"])
    club = float(row["club_strength"])
    attack = float(row["attacking_value"])
    defense = float(row["defensive_value"])
    experience = 55 + min(float(row["caps"]), 100) * 0.30
    # Weights per position: forwards lean on attack, defenders/keepers on defense.
    weights = {
        "FW": (0.65, 0.05, 0.20, 0.10),
        "MF": (0.38, 0.27, 0.25, 0.10),
        "DF": (0.10, 0.55, 0.25, 0.10),
        "GK": (0.02, 0.58, 0.30, 0.10),
    }
    attack_weight, defense_weight, club_weight, experience_weight = weights.get(
        position, weights["MF"]
    )
    rating = (
        attack * attack_weight
        + defense * defense_weight
        + club * club_weight
        + experience * experience_weight
    )
    return round(max(60, min(89, rating)))


def clean_squads(squads: pd.DataFrame) -> pd.DataFrame:
    squads = squads.copy()
    squads["club_strength"] = squads["club_strength"].map(calibrated_club_rating)
    squads["overall_rating"] = squads.apply(core_player_rating, axis=1)
    return squads


def clean_full_squads(squads: pd.DataFrame, core_squads: pd.DataFrame) -> pd.DataFrame:
    squads = squads.copy()
    squads["club"] = squads["club"].fillna("").astype(str).str.strip()
    squads.loc[squads["club"] == "", "club"] = "Club not listed"
    squads["club_rating"] = squads["club_rating"].map(calibrated_club_rating).astype(int)
    squads["player_rating"] = squads.apply(
        lambda row: calibrated_player_rating(row["player_rating"], row["club_rating"]), axis=1
    ).astype(int)

    core_lookup = {
        (str(row.team), roster_name_key(str(row.player))): int(row.overall_rating)
        for row in core_squads.itertuples(index=False)
    }
    for index, row in squads.iterrows():
        core_rating = core_lookup.get((str(row["team"]), roster_name_key(str(row["player"]))))
        if core_rating is not None:
            squads.at[index, "player_rating"] = max(int(row["player_rating"]), core_rating)
    return squads


@lru_cache(maxsize=1)
def load_context() -> dict[str, Any]:
    # Load the trained model, fixtures, squads and metadata once and reuse them
    # (lru_cache means this only runs on the first request).
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")

    fixtures = pd.read_csv(FIXTURES_PATH)
    fixtures["date"] = pd.to_datetime(fixtures["date"], format="mixed")
    fixtures = fixtures.reset_index(drop=True)
    fixtures["fixture_id"] = fixtures.index.astype(int)
    fixtures["label"] = fixtures.apply(
        lambda row: (
            f"Group {row['group']} | {row['date'].strftime('%b %d')} | "
            f"{row['team_one']} vs {row['team_two']} | {row['city']}"
        ),
        axis=1,
    )

    matches = pd.read_csv(MATCHES_PATH)
    matches["date"] = pd.to_datetime(matches["date"], format="mixed")

    core_squads = clean_squads(read_squads(SQUADS_PATH))
    full_squads = clean_full_squads(pd.read_csv(FULL_SQUADS_PATH), core_squads)
    return {
        "artifact": joblib.load(MODEL_PATH),
        "matches": matches.sort_values("date"),
        "qualified": pd.read_csv(QUALIFIED_TEAMS_PATH),
        "squads": core_squads,
        "full_squads": full_squads,
        "metadata": pd.read_csv(TEAM_METADATA_PATH),
        "fixtures": fixtures,
    }


def metadata_for(metadata: pd.DataFrame, team: str) -> dict[str, str]:
    row = metadata[metadata["team"] == team]
    if row.empty:
        return {
            "team": team,
            "group": "TBD",
            "manager": "Not listed",
            "qualification_path": "Qualified team",
            "confederation": "World Cup field",
            "scouting_note": "Metadata can be updated as official information changes.",
        }
    return row.iloc[0].fillna("Not listed").to_dict()


def team_record(matches: pd.DataFrame, team: str) -> dict[str, int | float]:
    played = matches[(matches["home_team"] == team) | (matches["away_team"] == team)]
    wins = draws = losses = goals_for = goals_against = 0
    for row in played.itertuples(index=False):
        if row.home_team == team:
            scored, conceded = row.home_score, row.away_score
        else:
            scored, conceded = row.away_score, row.home_score
        goals_for += int(scored)
        goals_against += int(conceded)
        if scored > conceded:
            wins += 1
        elif scored == conceded:
            draws += 1
        else:
            losses += 1
    played_count = max(len(played), 1)
    return {
        "played": int(len(played)),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "points_per_match": round((wins * 3 + draws) / played_count, 2),
        "goal_diff_per_match": round((goals_for - goals_against) / played_count, 2),
    }


def venue_edge(team: str, venue_country: str) -> float:
    # Small home boost: 6% if a host plays in its own country, 2.5% if in a neighbouring host.
    team_country = HOST_COUNTRIES.get(team)
    if not team_country:
        return 0.0
    if team_country == venue_country:
        return 0.06
    if venue_country in HOST_COUNTRIES.values():
        return 0.025
    return 0.0


def venue_context(team_one: str, team_two: str, city: str) -> dict[str, Any]:
    venue_country = CITY_COUNTRIES.get(city, "Neutral")
    team_one_edge = venue_edge(team_one, venue_country)
    team_two_edge = venue_edge(team_two, venue_country)
    if team_one_edge == 0 and team_two_edge == 0:
        if venue_country == "Neutral":
            label = "Neutral venue with no host nation advantage."
        else:
            label = f"No host nation advantage in {venue_country}."
    else:
        edges = []
        if team_one_edge:
            edges.append(f"{team_one} +{team_one_edge * 100:.1f}% home advantage")
        if team_two_edge:
            edges.append(f"{team_two} +{team_two_edge * 100:.1f}% home advantage")
        label = f"{city}, {venue_country}: " + "; ".join(edges)
    return {
        "venue_country": venue_country,
        "team_one_edge": team_one_edge,
        "team_two_edge": team_two_edge,
        "venue_label": label,
    }


def apply_venue_adjustment(
    result: dict[str, float | str], team_one: str, team_two: str, city: str | None
) -> dict[str, float | str]:
    # Nudge the win probabilities for a host playing at home, then renormalize to 100%.
    if not city:
        return result
    context = venue_context(team_one, team_two, city)
    team_one_multiplier = 1 + float(context["team_one_edge"])
    team_two_multiplier = 1 + float(context["team_two_edge"])
    draw_multiplier = 0.985 if team_one_multiplier > 1 or team_two_multiplier > 1 else 1.0
    home_win = float(result["home_win"]) * team_one_multiplier
    draw = float(result["draw"]) * draw_multiplier
    away_win = float(result["away_win"]) * team_two_multiplier
    total = max(home_win + draw + away_win, 0.001)
    adjusted = {
        **result,
        "home_win": home_win / total,
        "draw": draw / total,
        "away_win": away_win / total,
        "venue_country": str(context["venue_country"]),
        "venue_label": str(context["venue_label"]),
        "team_one_venue_edge": float(context["team_one_edge"]),
        "team_two_venue_edge": float(context["team_two_edge"]),
    }
    adjusted["predicted_outcome"] = max(
        ["home_win", "draw", "away_win"], key=lambda key: float(adjusted[key])
    )
    return adjusted


def predict_from_artifact(
    artifact: dict[str, Any], team_one: str, team_two: str, neutral: bool = True, city: str | None = None
) -> dict[str, float | str]:
    # Run one match through the model: build its feature row, get probabilities and a
    # projected score, then apply the venue boost and pick the final result.
    model = artifact["model"]
    row = build_prediction_row(
        home_team=team_one,
        away_team=team_two,
        team_profiles=artifact["team_profiles"],
        neutral=neutral,
        tournament="FIFA World Cup",
    )
    probabilities = model.predict_proba(row)[0]
    class_probs = dict(zip(model.classes_, probabilities))
    expected_home, expected_away = model.expected_goals(row)
    result: dict[str, float | str] = {
        "team_one": team_one,
        "team_two": team_two,
        "home_team": team_one,
        "away_team": team_two,
        "predicted_outcome": str(model.predict(row)[0]),
        "home_win": float(class_probs.get("home_win", 0)),
        "draw": float(class_probs.get("draw", 0)),
        "away_win": float(class_probs.get("away_win", 0)),
        "expected_home_goals": float(expected_home[0]),
        "expected_away_goals": float(expected_away[0]),
        "venue_country": "Neutral",
        "venue_label": "Neutral venue with no host nation advantage.",
        "team_one_venue_edge": 0.0,
        "team_two_venue_edge": 0.0,
    }
    adjusted = apply_venue_adjustment(result, team_one, team_two, city)
    draw_threshold = float(getattr(model, "draw_threshold", 1.0))
    if float(adjusted["draw"]) >= draw_threshold:
        adjusted["predicted_outcome"] = "draw"
    else:
        adjusted["predicted_outcome"] = max(
            ["home_win", "away_win"], key=lambda key: float(adjusted[key])
        )
    return adjusted


def prediction_title(result: dict[str, Any]) -> str:
    outcome = str(result["predicted_outcome"])
    if outcome == "draw":
        return "Draw"
    winner = result["home_team"] if outcome == "home_win" else result["away_team"]
    return f"{winner} Win"


def outcome_sentence(result: dict[str, Any]) -> str:
    outcome = str(result["predicted_outcome"])
    if outcome == "draw":
        return "The win, draw, and loss chances are close, so the model sees this as an even game."
    winner = result["home_team"] if outcome == "home_win" else result["away_team"]
    return f"{winner} comes out ahead once team strength, recent form, scoring, squad quality, and home advantage are weighed together."


def feature_evidence(
    home: dict[str, Any], away: dict[str, Any], home_team: str, away_team: str, result: dict[str, Any]
) -> list[dict[str, Any]]:
    specs = [
        (
            "Elo Strength",
            "elo",
            24,
            "Historical team strength updated match by match, weighted by competition, "
            "winning margin, and home advantage.",
            "Qualified teams span roughly 1550-2250; 1900+ is strong, 2050+ elite.",
        ),
        (
            "Recent Points",
            "recent_points",
            14,
            "Average recent win/draw/loss points over the form window.",
            "0.0 to 3.0 points per match; 2.0+ is strong.",
        ),
        (
            "Recent Goal Difference",
            "recent_goal_diff",
            12,
            "Recent goals scored minus goals conceded, averaged by match.",
            "Negative means being outscored, 0 is even, +1.0 is strong.",
        ),
        (
            "Squad Attack",
            "squad_attack",
            14,
            "Sum of attacking-value ratings from listed squad players.",
            "Larger totals indicate more attacking quality and depth.",
        ),
        (
            "Squad Defense",
            "squad_defense",
            12,
            "Sum of defensive-value ratings from listed squad players.",
            "Larger totals indicate more defensive quality and depth.",
        ),
        (
            "Experience",
            "squad_total_caps",
            8,
            "Total senior international caps across listed squad players.",
            "Higher totals indicate more national-team experience.",
        ),
        (
            "Club Strength",
            "squad_club_strength",
            12,
            "Average club-quality rating across listed players.",
            "60s is modest, 70s solid, 80s strong, 90s elite.",
        ),
    ]
    rows = []
    for label, key, weight, formula, scale in specs:
        h = float(home.get(key, 0))
        a = float(away.get(key, 0))
        if key == "squad_club_strength":
            h = calibrated_club_rating(h)
            a = calibrated_club_rating(a)
        rows.append(
            {
                "signal": label,
                "team_one": round(h, 2),
                "team_two": round(a, 2),
                "delta": round(h - a, 2),
                "edge": home_team if h > a else away_team if a > h else "Even",
                "weight": weight,
                "formula": formula,
                "scale": scale,
            }
        )
    venue_delta = float(result.get("team_one_venue_edge", 0)) - float(
        result.get("team_two_venue_edge", 0)
    )
    rows.append(
        {
            "signal": "Venue Advantage",
            "team_one": round(float(result.get("team_one_venue_edge", 0)) * 100, 2),
            "team_two": round(float(result.get("team_two_venue_edge", 0)) * 100, 2),
            "delta": round(venue_delta * 100, 2),
            "edge": home_team if venue_delta > 0 else away_team if venue_delta < 0 else "Even",
            "weight": 4,
            "formula": str(result.get("venue_label", "Neutral venue with no host nation advantage.")),
            "scale": "A small bonus for the three host nations when they play at home. It is a minor factor.",
        }
    )
    return rows


def recent_results(matches: pd.DataFrame, team: str, limit: int = 6) -> list[dict[str, Any]]:
    played = matches[(matches["home_team"] == team) | (matches["away_team"] == team)].tail(limit)
    rows = []
    for row in played.itertuples(index=False):
        if row.home_team == team:
            opponent = row.away_team
            scored, conceded = row.home_score, row.away_score
        else:
            opponent = row.home_team
            scored, conceded = row.away_score, row.home_score
        result = "W" if scored > conceded else "D" if scored == conceded else "L"
        rows.append(
            {
                "date": row.date.strftime("%Y-%m-%d"),
                "opponent": opponent,
                "opponent_flag": TEAM_FLAGS.get(opponent, ""),
                "result": result,
                "score": f"{scored}-{conceded}",
                "competition": row.tournament,
            }
        )
    return rows


def head_to_head(matches: pd.DataFrame, team_one: str, team_two: str) -> list[dict[str, Any]]:
    h2h = matches[
        ((matches["home_team"] == team_one) & (matches["away_team"] == team_two))
        | ((matches["home_team"] == team_two) & (matches["away_team"] == team_one))
    ].tail(8)
    return [
        {
            "date": row.date.strftime("%Y-%m-%d"),
            "home_team": row.home_team,
            "home_flag": TEAM_FLAGS.get(row.home_team, ""),
            "home_score": int(row.home_score),
            "away_team": row.away_team,
            "away_flag": TEAM_FLAGS.get(row.away_team, ""),
            "away_score": int(row.away_score),
            "match": f"{row.home_team} {row.home_score}-{row.away_score} {row.away_team}",
            "competition": row.tournament,
        }
        for row in h2h.itertuples(index=False)
    ]


def roster_name_key(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(value).lower())
    normalized_tokens = []
    for token in tokens:
        if token == "al":
            continue
        if token.startswith("al") and len(token) > 5:
            token = token[2:]
        token = re.sub(r"(.)\1+", r"\1", token)
        normalized_tokens.append(token)
    return "".join(sorted(normalized_tokens))


def squad_core(full_squads: pd.DataFrame, team: str) -> list[dict[str, Any]]:
    roster = full_squads[full_squads["team"] == team].copy()
    if roster.empty:
        return []
    core = roster.sort_values(
        ["player_rating", "club_rating", "shirt_number"],
        ascending=[False, False, True],
    ).head(4)
    core = core.rename(
        columns={
            "club_rating": "club_strength",
            "player_rating": "overall_rating",
        }
    )
    core["caps"] = None
    core["attacking_value"] = None
    core["defensive_value"] = None
    cols = [
        "player",
        "position",
        "age",
        "caps",
        "club",
        "club_strength",
        "attacking_value",
        "defensive_value",
        "overall_rating",
    ]
    return records(core[cols])


def full_roster(full_squads: pd.DataFrame, team: str) -> list[dict[str, Any]]:
    roster = full_squads[full_squads["team"] == team].copy()
    if roster.empty:
        return []
    cols = ["shirt_number", "player", "position", "age", "club", "club_rating", "player_rating"]
    return records(roster[cols].sort_values("shirt_number"))


def group_projection(fixtures: pd.DataFrame, artifact: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Build a quick "expected" group table: predict every group game and give each
    # team its average points (win chance * 3 + draw chance), then rank the groups.
    standings: dict[str, dict[str, Any]] = {}
    match_rows = []
    for row in fixtures.sort_values(["group", "date"]).itertuples(index=False):
        team_one = str(row.team_one)
        team_two = str(row.team_two)
        group = str(row.group)
        for team in [team_one, team_two]:
            standings.setdefault(
                team,
                {
                    "Group": group,
                    "Team": team,
                    "Expected Points": 0.0,
                    "Win Probability Sum": 0.0,
                    "Goal-Difference Proxy": 0.0,
                    "Elo": float(artifact["team_profiles"][team].get("elo", 1500)),
                },
            )
        probs = predict_from_artifact(artifact, team_one, team_two, neutral=True, city=str(row.city))
        team_one_win = float(probs["home_win"])
        draw = float(probs["draw"])
        team_two_win = float(probs["away_win"])
        # Expected points = how many points a team would average given the odds.
        standings[team_one]["Expected Points"] += team_one_win * 3 + draw
        standings[team_two]["Expected Points"] += team_two_win * 3 + draw
        standings[team_one]["Win Probability Sum"] += team_one_win
        standings[team_two]["Win Probability Sum"] += team_two_win
        standings[team_one]["Goal-Difference Proxy"] += team_one_win - team_two_win
        standings[team_two]["Goal-Difference Proxy"] += team_two_win - team_one_win
        outcomes = {team_one: team_one_win, "Draw": draw, team_two: team_two_win}
        winner = max(outcomes, key=outcomes.get)
        match_rows.append(
            {
                "Group": group,
                "Match": f"{team_one} vs {team_two}",
                "Projected Result": winner,
                "Team One Win": round(team_one_win * 100, 1),
                "Draw": round(draw * 100, 1),
                "Team Two Win": round(team_two_win * 100, 1),
            }
        )
    table = pd.DataFrame(standings.values())
    table = table.sort_values(
        ["Group", "Expected Points", "Goal-Difference Proxy", "Win Probability Sum", "Elo"],
        ascending=[True, False, False, False, False],
    )
    table["Group Rank"] = table.groupby("Group").cumcount() + 1
    for column in ["Expected Points", "Win Probability Sum", "Goal-Difference Proxy", "Elo"]:
        table[column] = table[column].round(2)
    return table, pd.DataFrame(match_rows)


def tournament_qualifiers(group_table: pd.DataFrame) -> pd.DataFrame:
    # 32 teams reach the next round: the top two from each group, plus the eight
    # best third-placed teams. They are then seeded 1-32 for the bracket.
    automatic = group_table[group_table["Group Rank"] <= 2].copy()
    third_place = group_table[group_table["Group Rank"] == 3].sort_values(
        ["Expected Points", "Goal-Difference Proxy", "Win Probability Sum", "Elo"],
        ascending=False,
    ).head(8)
    qualifiers = pd.concat([automatic, third_place], ignore_index=True)
    qualifiers = qualifiers.sort_values(
        ["Group Rank", "Expected Points", "Goal-Difference Proxy", "Win Probability Sum", "Elo"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)
    qualifiers["Seed"] = qualifiers.index + 1
    qualifiers["Path"] = qualifiers["Group Rank"].map(
        {1: "Group Winner", 2: "Group Runner-Up", 3: "Best Third Place"}
    )
    return qualifiers


def projected_knockout_score(
    team_one: str,
    team_two: str,
    winner: str,
    confidence: float,
    artifact: dict[str, Any],
    expected_home_goals: float | None = None,
    expected_away_goals: float | None = None,
) -> tuple[int, int]:
    # Make up a sensible scoreline for a knockout tie (there are no draws here, so
    # the projected winner is always given at least one more goal).
    if expected_home_goals is not None and expected_away_goals is not None:
        team_one_goals = max(0, min(4, int(round(expected_home_goals))))
        team_two_goals = max(0, min(4, int(round(expected_away_goals))))
        if team_one_goals == team_two_goals:
            if winner == team_one:
                team_one_goals += 1
            else:
                team_two_goals += 1
        elif winner == team_one and team_one_goals < team_two_goals:
            team_one_goals = team_two_goals + 1
        elif winner == team_two and team_two_goals < team_one_goals:
            team_two_goals = team_one_goals + 1
        return team_one_goals, team_two_goals

    profiles = artifact["team_profiles"]
    elo_gap = abs(float(profiles[team_one].get("elo", 1500)) - float(profiles[team_two].get("elo", 1500)))
    loser_goals = 0 if confidence >= 0.78 and elo_gap > 120 else 1
    if confidence >= 0.84:
        winner_goals = 3
    elif confidence >= 0.55:
        winner_goals = 2
    else:
        winner_goals = 1
        loser_goals = 0
    if winner == team_one:
        return winner_goals, loser_goals
    return loser_goals, winner_goals


# Which seeds meet in the Round of 32 (1 plays 32, and so on down the bracket).
SEED_PAIRS = [
    (1, 32), (16, 17), (8, 25), (9, 24), (4, 29), (13, 20), (5, 28), (12, 21),
    (2, 31), (15, 18), (7, 26), (10, 23), (3, 30), (14, 19), (6, 27), (11, 22),
]


def knockout_projection(qualifiers: pd.DataFrame, artifact: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    # The single "most likely" bracket: in each tie the more-likely team advances,
    # round by round, until one champion is left. (The odds come from the sim below.)
    seed_lookup = qualifiers.set_index("Seed")["Team"].to_dict()
    current_matches = [(seed_lookup[a], seed_lookup[b]) for a, b in SEED_PAIRS]
    round_names = ["Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final"]
    rows = []
    progress: dict[str, Any] = {"Round of 32": list(qualifiers["Team"])}
    champion = ""
    for round_name in round_names:
        winners = []
        for match_number, (team_one, team_two) in enumerate(current_matches, start=1):
            probs = predict_from_artifact(artifact, team_one, team_two, neutral=True)
            team_one_win = float(probs["home_win"])
            team_two_win = float(probs["away_win"])
            winner = team_one if team_one_win >= team_two_win else team_two
            confidence = max(team_one_win, team_two_win) / max(team_one_win + team_two_win, 0.01)
            team_one_score, team_two_score = projected_knockout_score(
                team_one,
                team_two,
                winner,
                confidence,
                artifact,
                float(probs.get("expected_home_goals", 0)),
                float(probs.get("expected_away_goals", 0)),
            )
            winners.append(winner)
            rows.append(
                {
                    "Round": round_name,
                    "Match": match_number,
                    "Team One": team_one,
                    "Team Two": team_two,
                    "Projected Score": f"{team_one_score}-{team_two_score}",
                    "Projected Winner": winner,
                    "Win Confidence": round(confidence * 100, 1),
                    "Team One Flag": TEAM_FLAGS.get(team_one, ""),
                    "Team Two Flag": TEAM_FLAGS.get(team_two, ""),
                    "Winner Flag": TEAM_FLAGS.get(winner, ""),
                }
            )
        if round_name == "Final":
            champion = winners[0]
            break
        progress[round_names[round_names.index(round_name) + 1]] = winners
        current_matches = list(zip(winners[::2], winners[1::2]))
    progress["Champion"] = champion
    return pd.DataFrame(rows), progress


def monte_carlo_tournament(
    fixtures: pd.DataFrame,
    artifact: dict[str, Any],
    simulations: int = 10_000,
    seed: int = 2026,
) -> dict[str, Any]:
    # Play the whole tournament 10,000 times with random results drawn from the
    # model's probabilities, then count how often each team reaches each round.
    # A fixed seed makes the run reproducible (same odds every time).
    rng = np.random.default_rng(seed)
    teams = sorted(set(fixtures["team_one"]) | set(fixtures["team_two"]))
    team_index = {team: index for index, team in enumerate(teams)}
    profiles = artifact["team_profiles"]
    elo = np.array([float(profiles[team].get("elo", 1500)) for team in teams])
    groups = {
        str(group): sorted(set(frame["team_one"]) | set(frame["team_two"]))
        for group, frame in fixtures.groupby("group")
    }
    group_matches = []
    for row in fixtures.sort_values(["group", "date"]).itertuples(index=False):
        probs = predict_from_artifact(
            artifact, str(row.team_one), str(row.team_two), neutral=True, city=str(row.city)
        )
        group_matches.append(
            (
                str(row.group),
                team_index[str(row.team_one)],
                team_index[str(row.team_two)],
                np.array(
                    [float(probs["home_win"]), float(probs["draw"]), float(probs["away_win"])]
                ),
            )
        )

    stage_names = [
        "Round of 32",
        "Round of 16",
        "Quarterfinals",
        "Semifinals",
        "Final",
        "Champion",
    ]
    stage_counts = {stage: np.zeros(len(teams), dtype=np.int32) for stage in stage_names}
    rank_counts = {rank: np.zeros(len(teams), dtype=np.int32) for rank in range(1, 5)}
    final_matchups: dict[tuple[str, str], int] = {}
    knockout_probability_cache: dict[tuple[str, str], float] = {}
    pair_keys = [(team_one, team_two) for team_one in teams for team_two in teams if team_one != team_two]
    pair_rows = pd.concat(
        [
            build_prediction_row(
                home_team=team_one,
                away_team=team_two,
                team_profiles=profiles,
                neutral=True,
                tournament="FIFA World Cup",
            )
            for team_one, team_two in pair_keys
        ],
        ignore_index=True,
    )
    pair_probabilities = artifact["model"].predict_proba(pair_rows)
    class_positions = {
        label: index for index, label in enumerate(artifact["model"].classes_)
    }
    for key, probabilities in zip(pair_keys, pair_probabilities):
        team_one_win = float(probabilities[class_positions["home_win"]])
        team_two_win = float(probabilities[class_positions["away_win"]])
        knockout_probability_cache[key] = team_one_win / max(team_one_win + team_two_win, 0.001)

    def knockout_win_probability(team_one: str, team_two: str) -> float:
        return knockout_probability_cache[(team_one, team_two)]

    group_outcome_draws = rng.random((simulations, len(group_matches)))
    group_margin_draws = rng.random((simulations, len(group_matches)))
    for simulation_index in range(simulations):
        points = np.zeros(len(teams), dtype=np.int16)
        goal_difference = np.zeros(len(teams), dtype=np.int16)
        wins = np.zeros(len(teams), dtype=np.int8)
        for match_index, (_, team_one, team_two, probabilities) in enumerate(group_matches):
            # Roll a random number and pick the result by the model's odds:
            # below P(away) -> away win, next slice -> draw, otherwise home win.
            draw = group_outcome_draws[simulation_index, match_index]
            outcome = (
                0
                if draw < probabilities[0]
                else 1
                if draw < probabilities[0] + probabilities[1]
                else 2
            )
            if outcome == 0:
                margin = (
                    2
                    if group_margin_draws[simulation_index, match_index]
                    < probabilities[0] * 0.35
                    else 1
                )
                points[team_one] += 3
                wins[team_one] += 1
                goal_difference[team_one] += margin
                goal_difference[team_two] -= margin
            elif outcome == 1:
                points[team_one] += 1
                points[team_two] += 1
            else:
                margin = (
                    2
                    if group_margin_draws[simulation_index, match_index]
                    < probabilities[2] * 0.35
                    else 1
                )
                points[team_two] += 3
                wins[team_two] += 1
                goal_difference[team_two] += margin
                goal_difference[team_one] -= margin

        # Rank each group by points, then goal difference, wins, Elo, and finally a
        # random tiebreak so exact ties don't always fall the same way.
        ranked_groups: dict[str, list[int]] = {}
        for group, group_teams in groups.items():
            indices = [team_index[team] for team in group_teams]
            ranked = sorted(
                indices,
                key=lambda index: (
                    points[index],
                    goal_difference[index],
                    wins[index],
                    elo[index],
                    rng.random(),
                ),
                reverse=True,
            )
            ranked_groups[group] = ranked
            for rank, index in enumerate(ranked, start=1):
                rank_counts[rank][index] += 1

        automatic = [
            (index, rank)
            for ranked in ranked_groups.values()
            for rank, index in enumerate(ranked[:2], start=1)
        ]
        third_place = sorted(
            [(ranked[2], 3) for ranked in ranked_groups.values()],
            key=lambda item: (
                points[item[0]],
                goal_difference[item[0]],
                wins[item[0]],
                elo[item[0]],
                rng.random(),
            ),
            reverse=True,
        )[:8]
        qualifiers = sorted(
            automatic + third_place,
            key=lambda item: (
                -item[1],
                points[item[0]],
                goal_difference[item[0]],
                wins[item[0]],
                elo[item[0]],
                rng.random(),
            ),
            reverse=True,
        )
        seeded_teams = [teams[index] for index, _ in qualifiers]
        for team in seeded_teams:
            stage_counts["Round of 32"][team_index[team]] += 1

        current_matches = [(seeded_teams[a - 1], seeded_teams[b - 1]) for a, b in SEED_PAIRS]
        knockout_rounds = ["Round of 16", "Quarterfinals", "Semifinals", "Final", "Champion"]
        for stage in knockout_rounds:
            winners = []
            if stage == "Champion":
                final_key = tuple(sorted(current_matches[0]))
                final_matchups[final_key] = final_matchups.get(final_key, 0) + 1
            for team_one, team_two in current_matches:
                # Random draw decides the winner, weighted by their head-to-head odds.
                probability = knockout_win_probability(team_one, team_two)
                winner = team_one if rng.random() < probability else team_two
                winners.append(winner)
                stage_counts[stage][team_index[winner]] += 1
            if stage != "Champion":
                # Winners pair up for the next round.
                current_matches = list(zip(winners[::2], winners[1::2]))

    probability_rows = []
    for index, team in enumerate(teams):
        row: dict[str, Any] = {
            "Team": team,
            "Flag": TEAM_FLAGS.get(team, ""),
        }
        for stage in stage_names:
            row[stage] = round(float(stage_counts[stage][index]) * 100 / simulations, 1)
        for rank in range(1, 5):
            row[f"Group Rank {rank}"] = round(float(rank_counts[rank][index]) * 100 / simulations, 1)
        probability_rows.append(row)
    probability_rows.sort(key=lambda row: (-float(row["Champion"]), -float(row["Final"])))
    common_final = max(final_matchups.items(), key=lambda item: item[1])
    return {
        "simulations": simulations,
        "seed": seed,
        "probabilities": probability_rows,
        "most_common_final": {
            "Team One": common_final[0][0],
            "Team Two": common_final[0][1],
            "Probability": round(common_final[1] * 100 / simulations, 1),
            "Team One Flag": TEAM_FLAGS.get(common_final[0][0], ""),
            "Team Two Flag": TEAM_FLAGS.get(common_final[0][1], ""),
        },
        "method": (
            "Each run plays all 72 group matches by drawing a win, draw, or loss from the "
            "model's probabilities, then ranks each group on points with a simulated goal "
            "difference as the tie-breaker. The top two teams in every group plus the eight "
            "best third-place teams advance, and each knockout round picks a winner from the "
            "two teams' win chances."
        ),
    }


def get_context_or_500() -> dict[str, Any]:
    try:
        return load_context()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/bootstrap")
def bootstrap() -> dict[str, Any]:
    ctx = get_context_or_500()
    fixtures = ctx["fixtures"].copy()
    qualified = ctx["qualified"]
    split = ctx["artifact"].get("metrics", {}).get("split", {})
    training_matches = int(split.get("train_samples", 0)) + int(split.get("validation_samples", 0))
    return {
        "facts": {
            "teams": int(qualified["team"].nunique()),
            "groups": int(fixtures["group"].nunique()),
            "fixtures": int(len(fixtures)),
            "training_matches": training_matches,
            "dates": "June 11 - July 19, 2026",
            "hosts": "Canada, Mexico, USA",
        },
        "groups": ["All", *sorted(fixtures["group"].unique().tolist())],
        "fixtures": records(
            fixtures[
                ["fixture_id", "group", "date", "team_one", "team_two", "city", "stadium", "label"]
            ]
        ),
    }


@app.post("/predict")
def predict(request: MatchRequest) -> dict[str, Any]:
    ctx = get_context_or_500()
    try:
        result = predict_from_artifact(
            ctx["artifact"], request.home_team, request.away_team, neutral=request.neutral
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "home_team": request.home_team,
        "away_team": request.away_team,
        "predicted_outcome": result["predicted_outcome"],
        "home_win": round(float(result["home_win"]), 4),
        "draw": round(float(result["draw"]), 4),
        "away_win": round(float(result["away_win"]), 4),
        "expected_home_goals": round(float(result["expected_home_goals"]), 2),
        "expected_away_goals": round(float(result["expected_away_goals"]), 2),
    }


@app.get("/api/match/{fixture_id}")
def match_report(fixture_id: int) -> dict[str, Any]:
    ctx = get_context_or_500()
    fixtures = ctx["fixtures"]
    selected = fixtures[fixtures["fixture_id"] == fixture_id]
    if selected.empty:
        raise HTTPException(status_code=404, detail="Fixture not found")
    fixture = selected.iloc[0]
    team_one = str(fixture["team_one"])
    team_two = str(fixture["team_two"])
    result = predict_from_artifact(ctx["artifact"], team_one, team_two, city=str(fixture["city"]))
    profiles = ctx["artifact"]["team_profiles"]
    home_profile = profiles[team_one]
    away_profile = profiles[team_two]

    group_schedule = fixtures[fixtures["group"] == fixture["group"]].copy()
    group_schedule["date"] = group_schedule["date"].dt.strftime("%Y-%m-%d")
    group_schedule["team_one_flag"] = group_schedule["team_one"].map(TEAM_FLAGS).fillna("")
    group_schedule["team_two_flag"] = group_schedule["team_two"].map(TEAM_FLAGS).fillna("")

    return {
        "fixture": {key: _jsonable(value) for key, value in fixture.to_dict().items()},
        "prediction": {
            **{key: _jsonable(value) for key, value in result.items()},
            "title": prediction_title(result),
            "summary": outcome_sentence(result),
            "home_win_pct": round(float(result["home_win"]) * 100, 1),
            "draw_pct": round(float(result["draw"]) * 100, 1),
            "away_win_pct": round(float(result["away_win"]) * 100, 1),
            "confidence": round(
                max(float(result["home_win"]), float(result["draw"]), float(result["away_win"])) * 100, 1
            ),
        },
        "teams": [
            {
                "name": team_one,
                "flag": TEAM_FLAGS.get(team_one, ""),
                "profile": {key: round(float(value), 2) for key, value in home_profile.items()},
                "record": team_record(ctx["matches"], team_one),
                "metadata": metadata_for(ctx["metadata"], team_one),
                "squad_core": squad_core(ctx["full_squads"], team_one),
                "roster": full_roster(ctx["full_squads"], team_one),
                "recent": recent_results(ctx["matches"], team_one),
            },
            {
                "name": team_two,
                "flag": TEAM_FLAGS.get(team_two, ""),
                "profile": {key: round(float(value), 2) for key, value in away_profile.items()},
                "record": team_record(ctx["matches"], team_two),
                "metadata": metadata_for(ctx["metadata"], team_two),
                "squad_core": squad_core(ctx["full_squads"], team_two),
                "roster": full_roster(ctx["full_squads"], team_two),
                "recent": recent_results(ctx["matches"], team_two),
            },
        ],
        "evidence": feature_evidence(home_profile, away_profile, team_one, team_two, result),
        "head_to_head": head_to_head(ctx["matches"], team_one, team_two),
        "group_schedule": records(
            group_schedule[
                ["date", "team_one", "team_one_flag", "team_two", "team_two_flag", "city", "stadium"]
            ]
        ),
        "metrics": ctx["artifact"].get("metrics", {}),
    }


@lru_cache(maxsize=1)
def tournament_payload() -> dict[str, Any]:
    ctx = get_context_or_500()
    group_table, group_matches = group_projection(ctx["fixtures"], ctx["artifact"])
    qualifiers = tournament_qualifiers(group_table)
    group_table["Flag"] = group_table["Team"].map(TEAM_FLAGS).fillna("")
    qualifiers["Flag"] = qualifiers["Team"].map(TEAM_FLAGS).fillna("")
    bracket, progress = knockout_projection(qualifiers, ctx["artifact"])
    final = bracket[bracket["Round"] == "Final"].iloc[0]
    round_confidence = bracket.groupby("Round", as_index=False)["Win Confidence"].mean()
    monte_carlo = monte_carlo_tournament(ctx["fixtures"], ctx["artifact"])
    return {
        "champion": str(progress["Champion"]),
        "champion_flag": TEAM_FLAGS.get(str(progress["Champion"]), ""),
        "final_confidence": round(float(final["Win Confidence"]), 1),
        "round_confidence": records(round_confidence.round(1)),
        "groups": records(group_table),
        "qualifiers": records(qualifiers),
        "bracket": records(bracket),
        "group_matches": records(group_matches),
        "monte_carlo": monte_carlo,
        "method": (
            "Group standings use the expected points from every scheduled group-stage match. "
            "The top two teams in each group, plus the eight best third-place teams, reach the "
            "Round of 32. In the knockout rounds each tie compares the two teams' win chances, and "
            "confidence is the projected winner's share of those two chances. Projected scores are "
            "single best-guess scorelines based on that confidence and the Elo gap."
        ),
    }


@app.get("/api/tournament")
def tournament() -> dict[str, Any]:
    return tournament_payload()
