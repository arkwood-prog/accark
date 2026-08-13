"""Load your own fixtures and prices from a CSV.

Use this when you have odds the free feed does not carry (BTTS, extra
over/under lines, an exchange price you can actually get on).

Required columns: ``date``, ``home``, ``away``.
Optional: ``league``, ``time``.

Price columns are named either by canonical selection id (``1X2:H``,
``OU2.5:O``, ``BTTS:Y``) or by one of the friendly aliases below. Prefix a
column with ``sharp_`` to mark it as the sharp reference line rather than the
best price you can take:

    date,home,away,H,D,A,O2.5,U2.5,BTTS_Y,BTTS_N,sharp_H,sharp_D,sharp_A
    2026-08-15,Arsenal,Chelsea,2.10,3.50,3.60,1.85,1.95,1.70,2.10,2.05,3.40,3.55
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from .schema import Fixture, Selection

ALIASES: dict[str, str] = {
    "h": "1X2:H", "home": "1X2:H", "1": "1X2:H", "homewin": "1X2:H",
    "d": "1X2:D", "draw": "1X2:D", "x": "1X2:D",
    "a": "1X2:A", "away": "1X2:A", "2": "1X2:A", "awaywin": "1X2:A",
    "btts": "BTTS:Y", "btts_y": "BTTS:Y", "bttsy": "BTTS:Y", "btts_yes": "BTTS:Y",
    "btts_n": "BTTS:N", "bttsn": "BTTS:N", "btts_no": "BTTS:N",
}

for _line in ("0.5", "1.5", "2.5", "3.5", "4.5"):
    for _prefix in ("o", "over", ">"):
        ALIASES[f"{_prefix}{_line}"] = f"OU{_line}:O"
        ALIASES[f"{_prefix}_{_line}"] = f"OU{_line}:O"
    for _prefix in ("u", "under", "<"):
        ALIASES[f"{_prefix}{_line}"] = f"OU{_line}:U"
        ALIASES[f"{_prefix}_{_line}"] = f"OU{_line}:U"


def _selection_for(column: str) -> str | None:
    name = column.strip()
    if ":" in name:
        try:
            return Selection.parse(name).id
        except ValueError:
            return None
    return ALIASES.get(name.lower().replace(" ", ""))


def _parse_date(raw: str):
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised date {raw!r} — use YYYY-MM-DD")


def load_fixtures_csv(path: str | Path, default_league: str = "CSV") -> list[Fixture]:
    fixtures: list[Fixture] = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lowered = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            if not lowered.get("home") or not lowered.get("away"):
                continue
            odds: dict[str, float] = {}
            sharp: dict[str, float] = {}
            counts: dict[str, int] = {}
            for column, value in row.items():
                if column is None or not (value or "").strip():
                    continue
                key = column.strip()
                target = sharp if key.lower().startswith("sharp_") else odds
                if key.lower().startswith("sharp_"):
                    key = key[len("sharp_"):]
                selection_id = _selection_for(key)
                if selection_id is None:
                    continue
                try:
                    price = float(value)
                except ValueError:
                    continue
                if price > 1.0:
                    target[selection_id] = price
                    counts.setdefault(selection_id, 1)
            if not odds:
                continue
            fixtures.append(
                Fixture(
                    date=_parse_date(lowered["date"]),
                    league=lowered.get("league") or default_league,
                    home=lowered["home"],
                    away=lowered["away"],
                    kickoff=lowered.get("time") or None,
                    odds=odds,
                    sharp_odds=sharp,
                    book_counts=counts,
                )
            )
    fixtures.sort(key=lambda f: (f.date, f.home))
    return fixtures


def load_results_csv(path: str | Path, default_league: str = "CSV"):
    """Load completed matches from a simple CSV.

    Required columns: ``date``, ``home``, ``away``, ``home_goals``,
    ``away_goals``. Price columns are read exactly as for fixtures.
    """
    from .schema import Match

    matches: list[Match] = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lowered = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            try:
                home_goals = int(float(lowered.get("home_goals") or lowered["fthg"]))
                away_goals = int(float(lowered.get("away_goals") or lowered["ftag"]))
            except (KeyError, ValueError):
                continue
            home = lowered.get("home") or lowered.get("hometeam")
            away = lowered.get("away") or lowered.get("awayteam")
            if not home or not away:
                continue
            odds: dict[str, float] = {}
            for column, value in row.items():
                if column is None or not (value or "").strip():
                    continue
                selection_id = _selection_for(column.strip())
                if selection_id is None:
                    continue
                try:
                    price = float(value)
                except ValueError:
                    continue
                if price > 1.0:
                    odds[selection_id] = price
            matches.append(
                Match(
                    date=_parse_date(lowered["date"]),
                    league=lowered.get("league") or default_league,
                    home=home,
                    away=away,
                    home_goals=home_goals,
                    away_goals=away_goals,
                    closing_odds=dict(odds),
                    best_odds=dict(odds),
                )
            )
    matches.sort(key=lambda m: m.date)
    return matches
