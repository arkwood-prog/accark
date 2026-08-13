"""Loader for football-data.co.uk — free historical results *with* bookmaker odds.

This is the workhorse source for this project because it carries both the
result and the closing prices for the same match, which is what makes honest
backtesting (and closing-line comparison) possible at all.

Two endpoints are used:

* ``mmz4281/{season}/{league}.csv``  - completed matches, one file per season
* ``fixtures.csv``                   - this week's fixtures with opening prices

Files are cached on disk so repeated runs do not hammer the site.
"""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Sequence

from .schema import Fixture, Match

BASE_URL = "https://www.football-data.co.uk"

LEAGUES: dict[str, str] = {
    "E0": "England Premier League",
    "E1": "England Championship",
    "E2": "England League One",
    "E3": "England League Two",
    "EC": "England National League",
    "SC0": "Scotland Premiership",
    "D1": "Germany Bundesliga",
    "D2": "Germany 2. Bundesliga",
    "I1": "Italy Serie A",
    "I2": "Italy Serie B",
    "SP1": "Spain La Liga",
    "SP2": "Spain Segunda",
    "F1": "France Ligue 1",
    "F2": "France Ligue 2",
    "N1": "Netherlands Eredivisie",
    "B1": "Belgium Pro League",
    "P1": "Portugal Primeira Liga",
    "T1": "Turkey Super Lig",
    "G1": "Greece Super League",
}

DEFAULT_CACHE = Path(os.environ.get("BETTINGEDGE_CACHE", Path.home() / ".cache" / "bettingedge"))

# --------------------------------------------------------------------------
# Price snapshots
# --------------------------------------------------------------------------
# Each file carries two snapshots per market: a pre-closing price (B365H, PSH,
# MaxH, AvgH) and a closing price (B365CH, PSCH, MaxCH, AvgCH). Which pair you
# read decides what a backtest actually measures, so it is an explicit choice
# rather than a buried default.
#
# Each entry below is a preference chain — the first group where every price is
# present wins. The chains matter because seasons before ~2019 have no closing
# columns at all, and a long backtest has to degrade gracefully.

_BEST_CLOSING_1X2 = [("MaxCH", "MaxCD", "MaxCA")]
_BEST_EARLY_1X2 = [("MaxH", "MaxD", "MaxA"), ("B365H", "B365D", "B365A")]
_SHARP_CLOSING_1X2 = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA")]
_SHARP_EARLY_1X2 = [("PSH", "PSD", "PSA"), ("AvgH", "AvgD", "AvgA"),
                    ("B365H", "B365D", "B365A")]

_BEST_CLOSING_OU = [("MaxC>2.5", "MaxC<2.5")]
_BEST_EARLY_OU = [("Max>2.5", "Max<2.5"), ("B365>2.5", "B365<2.5")]
_SHARP_CLOSING_OU = [("PC>2.5", "PC<2.5"), ("AvgC>2.5", "AvgC<2.5")]
_SHARP_EARLY_OU = [("P>2.5", "P<2.5"), ("Avg>2.5", "Avg<2.5"),
                   ("B365>2.5", "B365<2.5")]


@dataclass(frozen=True)
class PriceMode:
    """Which columns supply the price you bet at, and the reference to beat."""

    key: str
    label: str
    description: str
    measures: str
    best_1x2: tuple[tuple[str, ...], ...]
    sharp_1x2: tuple[tuple[str, ...], ...]
    best_ou: tuple[tuple[str, ...], ...]
    sharp_ou: tuple[tuple[str, ...], ...]


PRICE_MODES: dict[str, PriceMode] = {
    "best-closing": PriceMode(
        key="best-closing",
        label="best closing price",
        description=(
            "Bet at the best closing price across every book; reference is "
            "Pinnacle's closing line."
        ),
        measures=(
            "Optimistic. Both sides are closing prices, so the reported closing-line "
            "value measures how good your price shopping was, not whether the market "
            "moved toward you."
        ),
        best_1x2=tuple(_BEST_CLOSING_1X2 + _BEST_EARLY_1X2 + _SHARP_CLOSING_1X2
                       + _SHARP_EARLY_1X2),
        sharp_1x2=tuple(_SHARP_CLOSING_1X2 + _SHARP_EARLY_1X2),
        best_ou=tuple(_BEST_CLOSING_OU + _BEST_EARLY_OU + _SHARP_CLOSING_OU
                      + _SHARP_EARLY_OU),
        sharp_ou=tuple(_SHARP_CLOSING_OU + _SHARP_EARLY_OU),
    ),
    "early": PriceMode(
        key="early",
        label="pre-closing price vs the closing line",
        description=(
            "Bet at the best price available *before* the close; reference is "
            "Pinnacle's closing line."
        ),
        measures=(
            "The honest closing-line-value test. Positive CLV here means the market "
            "moved toward your bet after you placed it, which is the strongest "
            "available evidence of a real edge."
        ),
        best_1x2=tuple(_BEST_EARLY_1X2),
        sharp_1x2=tuple(_SHARP_CLOSING_1X2),
        best_ou=tuple(_BEST_EARLY_OU),
        sharp_ou=tuple(_SHARP_CLOSING_OU),
    ),
    "sharp-only": PriceMode(
        key="sharp-only",
        label="sharp closing price only",
        description=(
            "Assume you can only ever get Pinnacle's closing line — no shopping, "
            "no early price."
        ),
        measures=(
            "Pessimistic. Strips out every penny of price-shopping edge, leaving "
            "only what the model itself contributes. If an edge survives this, it "
            "is a model edge."
        ),
        best_1x2=tuple(_SHARP_CLOSING_1X2 + _SHARP_EARLY_1X2),
        sharp_1x2=tuple(_SHARP_CLOSING_1X2 + _SHARP_EARLY_1X2),
        best_ou=tuple(_SHARP_CLOSING_OU + _SHARP_EARLY_OU),
        sharp_ou=tuple(_SHARP_CLOSING_OU + _SHARP_EARLY_OU),
    ),
}

DEFAULT_PRICE_MODE = "best-closing"


def resolve_price_mode(mode: str | PriceMode | None) -> PriceMode:
    if isinstance(mode, PriceMode):
        return mode
    key = mode or DEFAULT_PRICE_MODE
    if key not in PRICE_MODES:
        raise ValueError(
            f"unknown price mode {key!r}; choose one of {', '.join(PRICE_MODES)}"
        )
    return PRICE_MODES[key]


def _to_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 1.0 else None


def _first_complete(row: dict[str, str], groups: Sequence[tuple[str, ...]]) -> tuple[float, ...] | None:
    """Return the first column group where every price is present and valid."""
    for group in groups:
        values = tuple(_to_float(row.get(col)) for col in group)
        if all(v is not None for v in values):
            return values  # type: ignore[return-value]
    return None


def _count_books(row: dict[str, str], suffixes: Sequence[str]) -> int:
    """Rough count of how many books quoted this market (a liquidity proxy)."""
    prefixes = ["B365", "BW", "BF", "PS", "WH", "1XB", "VC", "IW", "LB", "SJ", "P"]
    count = 0
    for prefix in prefixes:
        if all(_to_float(row.get(f"{prefix}{suffix}")) for suffix in suffixes):
            count += 1
    return count


def _parse_date(raw: str) -> date | None:
    raw = (raw or "").strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _odds_from_row(
    row: dict[str, str], mode: PriceMode
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    """Extract best / sharp prices and book counts for the markets available."""
    best: dict[str, float] = {}
    sharp: dict[str, float] = {}
    counts: dict[str, int] = {}

    sharp_1x2 = _first_complete(row, mode.sharp_1x2)
    best_1x2 = _first_complete(row, mode.best_1x2)
    if sharp_1x2:
        for sel, price in zip(("1X2:H", "1X2:D", "1X2:A"), sharp_1x2):
            sharp[sel] = price
    if best_1x2:
        n_books = _count_books(row, ("H", "D", "A"))
        for sel, price in zip(("1X2:H", "1X2:D", "1X2:A"), best_1x2):
            best[sel] = price
            counts[sel] = n_books

    sharp_ou = _first_complete(row, mode.sharp_ou)
    best_ou = _first_complete(row, mode.best_ou)
    if sharp_ou:
        for sel, price in zip(("OU2.5:O", "OU2.5:U"), sharp_ou):
            sharp[sel] = price
    if best_ou:
        n_books = _count_books(row, (">2.5", "<2.5"))
        for sel, price in zip(("OU2.5:O", "OU2.5:U"), best_ou):
            best[sel] = price
            counts[sel] = n_books

    return best, sharp, counts


def _row_teams(row: dict[str, str]) -> tuple[str, str]:
    home = (row.get("HomeTeam") or row.get("Home") or "").strip()
    away = (row.get("AwayTeam") or row.get("Away") or "").strip()
    return home, away


def parse_results_csv(text: str, league: str,
                      price_mode: str | PriceMode | None = None,
                      strict_league: bool = True) -> list[Match]:
    """Parse a football-data.co.uk season file into Match objects.

    ``strict_league`` rejects rows whose ``Div`` column disagrees with the
    league that was requested. This is not paranoia: the site has served
    another division's data under a league's URL for a season that had not
    started yet, and without this check those matches are silently absorbed
    into the wrong model.
    """
    mode = resolve_price_mode(price_mode)
    matches: list[Match] = []
    wrong_division = 0
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    for row in reader:
        if not row:
            continue
        division = (row.get("Div") or "").strip()
        if strict_league and division and division.upper() != league.upper():
            wrong_division += 1
            continue
        match_date = _parse_date(row.get("Date", ""))
        home, away = _row_teams(row)
        raw_hg = row.get("FTHG") or row.get("HG")
        raw_ag = row.get("FTAG") or row.get("AG")
        if not (match_date and home and away and raw_hg and raw_ag):
            continue
        try:
            home_goals, away_goals = int(float(raw_hg)), int(float(raw_ag))
        except ValueError:
            continue
        best, sharp, _ = _odds_from_row(row, mode)
        matches.append(
            Match(
                date=match_date,
                league=(row.get("Div") or league).strip(),
                home=home,
                away=away,
                home_goals=home_goals,
                away_goals=away_goals,
                closing_odds=sharp,
                best_odds=best,
            )
        )
    if wrong_division:
        print(f"  ! {wrong_division} row(s) in the {league} file are labelled as a "
              f"different division and were rejected")
    matches.sort(key=lambda m: m.date)
    return matches


def parse_fixtures_csv(text: str, leagues: Sequence[str] | None = None,
                       price_mode: str | PriceMode | None = None) -> list[Fixture]:
    """Parse the upcoming-fixtures file (all leagues in one CSV).

    The fixtures feed only ever carries pre-closing prices, so the closing
    columns a price mode may ask for simply are not there; the preference
    chains fall through to what exists.
    """
    mode = resolve_price_mode(price_mode)
    wanted = {code.upper() for code in leagues} if leagues else None
    fixtures: list[Fixture] = []
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    for row in reader:
        div = (row.get("Div") or "").strip()
        if wanted and div.upper() not in wanted:
            continue
        fixture_date = _parse_date(row.get("Date", ""))
        home, away = _row_teams(row)
        if not (fixture_date and home and away):
            continue
        best, sharp, counts = _odds_from_row(row, mode)
        if not best:
            continue
        fixtures.append(
            Fixture(
                date=fixture_date,
                league=div,
                home=home,
                away=away,
                kickoff=(row.get("Time") or "").strip() or None,
                odds=best,
                sharp_odds=sharp,
                book_counts=counts,
            )
        )
    fixtures.sort(key=lambda f: (f.date, f.league, f.home))
    return fixtures


def season_code(start_year: int) -> str:
    """2024 -> '2425' (the 2024/25 season)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def recent_seasons(count: int, today: date | None = None) -> list[str]:
    """Season codes for the `count` most recent seasons, newest last."""
    today = today or date.today()
    # A season is labelled by the calendar year it starts in; European
    # seasons roll over in July.
    current_start = today.year if today.month >= 7 else today.year - 1
    return [season_code(year) for year in range(current_start - count + 1, current_start + 1)]


class FootballDataUK:
    """Downloads (and caches) results and fixtures."""

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE, timeout: int = 30,
                 offline: bool = False, price_mode: str | PriceMode | None = None):
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.offline = offline
        self.price_mode = resolve_price_mode(price_mode)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get(self, url: str, cache_name: str, max_age_hours: float | None = None) -> str:
        cache_file = self.cache_dir / cache_name
        if cache_file.exists():
            fresh = True
            if max_age_hours is not None:
                age_hours = (datetime.now().timestamp() - cache_file.stat().st_mtime) / 3600
                fresh = age_hours < max_age_hours
            if fresh or self.offline:
                # utf-8-sig: these files carry a BOM, which otherwise ends up
                # glued to the first column name.
                return cache_file.read_text(encoding="utf-8-sig", errors="replace")
        if self.offline:
            raise FileNotFoundError(f"offline mode and no cached copy of {cache_name}")

        import requests  # imported lazily so offline use needs no network stack

        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        text = response.content.decode("utf-8-sig", errors="replace")
        cache_file.write_text(text, encoding="utf-8")
        return text

    def results(self, league: str, seasons: Iterable[str]) -> list[Match]:
        matches: list[Match] = []
        for season in seasons:
            url = f"{BASE_URL}/mmz4281/{season}/{league}.csv"
            try:
                text = self._get(url, f"{league}_{season}.csv", max_age_hours=12)
            except Exception as exc:  # a missing season should not kill the run
                print(f"  ! could not load {league} {season}: {exc}")
                continue
            matches.extend(parse_results_csv(text, league, self.price_mode))
        matches.sort(key=lambda m: m.date)
        return matches

    def fixtures(self, leagues: Sequence[str] | None = None) -> list[Fixture]:
        text = self._get(f"{BASE_URL}/fixtures.csv", "fixtures.csv", max_age_hours=3)
        return parse_fixtures_csv(text, leagues, self.price_mode)
