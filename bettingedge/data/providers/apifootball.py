"""API-Football (api-football.com, also served through RapidAPI).

A second live source, useful as redundancy when The Odds API's monthly quota
runs out, and because it covers far more leagues. Free tier is roughly 100
requests a day.

    export API_FOOTBALL_KEY=...
    bettingedge recommend --league E0 --odds-provider apifootball

Fetching odds costs two calls per run: one for the fixture list, one for the
odds. Both are on the same date window, and results are cached by the caller.

This provider's payload is more deeply nested than The Odds API's and its
market naming varies by bookmaker, so parsing is deliberately forgiving —
anything unrecognised is skipped rather than guessed at. Use ``--dump-raw``
when something looks wrong.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from ..schema import Fixture
from .base import (
    BookQuote,
    ProviderError,
    api_key_for,
    assemble_fixture,
    parse_iso_datetime,
    request_json,
)

DIRECT_URL = "https://v3.football.api-sports.io"
RAPIDAPI_URL = "https://api-football-v1.p.rapidapi.com/v3"

# football-data.co.uk league code -> API-Football numeric league id.
LEAGUE_IDS: dict[str, int] = {
    "E0": 39, "E1": 40, "E2": 41, "E3": 42, "EC": 43,
    "SC0": 179,
    "D1": 78, "D2": 79,
    "I1": 135, "I2": 136,
    "SP1": 140, "SP2": 141,
    "F1": 61, "F2": 62,
    "N1": 88,
    "B1": 144,
    "P1": 94,
    "T1": 203,
    "G1": 197,
}

# Bet names used for 1X2 and over/under differ between bookmakers.
_MATCH_WINNER_BETS = {"match winner", "1x2", "full time result", "home/away"}
_TOTALS_BETS = {"goals over/under", "over/under", "total goals", "goals over under"}
_BTTS_BETS = {"both teams score", "both teams to score", "btts"}

SUPPORTED_TOTALS = {0.5, 1.5, 2.5, 3.5, 4.5}


class APIFootball:
    """Client for API-Football v3."""

    name = "apifootball"

    def __init__(self, api_key: str | None = None, use_rapidapi: bool = False,
                 timeout: int = 25, dump_raw: str | Path | None = None):
        # Resolved lazily so the parser works on a captured payload with no key.
        self._api_key = api_key
        self._resolved_key: str | None = None
        self.use_rapidapi = use_rapidapi
        self.timeout = timeout
        self.dump_raw = Path(dump_raw) if dump_raw else None

    @property
    def api_key(self) -> str:
        if self._resolved_key is None:
            self._resolved_key = api_key_for(
                ["API_FOOTBALL_KEY", "APIFOOTBALL_KEY", "RAPIDAPI_KEY"], self._api_key
            )
        return self._resolved_key

    @property
    def _base(self) -> str:
        return RAPIDAPI_URL if self.use_rapidapi else DIRECT_URL

    @property
    def _headers(self) -> dict[str, str]:
        if self.use_rapidapi:
            return {"x-rapidapi-key": self.api_key,
                    "x-rapidapi-host": "api-football-v1.p.rapidapi.com"}
        return {"x-apisports-key": self.api_key}

    def _get(self, path: str, params: dict[str, Any]) -> list:
        payload = request_json(f"{self._base}/{path}", params, timeout=self.timeout,
                               headers=self._headers)
        if isinstance(payload, dict):
            # The API reports auth and quota problems in the body, not the status.
            errors = payload.get("errors")
            if errors:
                raise ProviderError(f"API-Football returned errors: {errors}")
            return payload.get("response") or []
        raise ProviderError("unexpected payload from API-Football; try --dump-raw")

    def _league_id(self, league: str) -> int:
        league_id = LEAGUE_IDS.get(league.upper())
        if not league_id:
            raise ProviderError(
                f"no API-Football league id is mapped for {league!r}. "
                f"Mapped leagues: {', '.join(sorted(LEAGUE_IDS))}."
            )
        return league_id

    # -- the interface --------------------------------------------------
    def fixtures(self, league: str, days_ahead: int = 7) -> list[Fixture]:
        league_id = self._league_id(league)
        today = date.today()
        window = {
            "league": league_id,
            "season": today.year if today.month >= 7 else today.year - 1,
            "from": today.isoformat(),
            "to": (today + timedelta(days=days_ahead)).isoformat(),
        }

        raw_fixtures = self._get("fixtures", window)
        odds_pages = self._get("odds", {"league": league_id,
                                        "season": window["season"]})
        if self.dump_raw:
            self.dump_raw.write_text(
                json.dumps({"fixtures": raw_fixtures, "odds": odds_pages}, indent=2),
                encoding="utf-8",
            )

        odds_by_fixture = self._index_odds(odds_pages)

        fixtures: list[Fixture] = []
        for entry in raw_fixtures:
            if not isinstance(entry, dict):
                continue
            info = entry.get("fixture") or {}
            teams = entry.get("teams") or {}
            home = ((teams.get("home") or {}).get("name") or "").strip()
            away = ((teams.get("away") or {}).get("name") or "").strip()
            starts = parse_iso_datetime(info.get("date", ""))
            fixture_id = info.get("id")
            if not (home and away and starts and fixture_id):
                continue

            quotes = odds_by_fixture.get(fixture_id)
            if not quotes:
                continue
            local = starts.astimezone()
            built = assemble_fixture(
                match_date=local.date(), league=league, home=home, away=away,
                quotes=quotes, kickoff=local.strftime("%H:%M"),
            )
            if built:
                fixtures.append(built)
        fixtures.sort(key=lambda f: (f.date, f.home))
        return fixtures

    # -- parsing --------------------------------------------------------
    def _index_odds(self, pages: list) -> dict[int, list[BookQuote]]:
        indexed: dict[int, list[BookQuote]] = {}
        for page in pages:
            if not isinstance(page, dict):
                continue
            fixture_id = (page.get("fixture") or {}).get("id")
            if fixture_id is None:
                continue
            quotes: list[BookQuote] = []
            for bookmaker in page.get("bookmakers") or []:
                if not isinstance(bookmaker, dict):
                    continue
                prices = self._book_prices(bookmaker)
                if prices:
                    quotes.append(BookQuote(
                        book=str(bookmaker.get("name") or "unknown").lower(),
                        prices=prices,
                    ))
            if quotes:
                indexed[fixture_id] = quotes
        return indexed

    def _book_prices(self, bookmaker: dict) -> dict[str, float]:
        prices: dict[str, float] = {}
        for bet in bookmaker.get("bets") or []:
            if not isinstance(bet, dict):
                continue
            label = str(bet.get("name") or "").strip().lower()
            values = bet.get("values") or []
            if label in _MATCH_WINNER_BETS:
                prices.update(self._match_winner(values))
            elif label in _TOTALS_BETS:
                prices.update(self._totals(values))
            elif label in _BTTS_BETS:
                prices.update(self._btts(values))
        return prices

    @staticmethod
    def _odd(entry: Any) -> float | None:
        try:
            value = float(entry.get("odd"))
        except (TypeError, ValueError, AttributeError):
            return None
        return value if value > 1.0 else None

    def _match_winner(self, values: list) -> dict[str, float]:
        mapping = {"home": "1X2:H", "1": "1X2:H",
                   "draw": "1X2:D", "x": "1X2:D",
                   "away": "1X2:A", "2": "1X2:A"}
        prices: dict[str, float] = {}
        for entry in values:
            if not isinstance(entry, dict):
                continue
            key = mapping.get(str(entry.get("value") or "").strip().lower())
            price = self._odd(entry)
            if key and price:
                prices[key] = price
        return prices

    def _totals(self, values: list) -> dict[str, float]:
        prices: dict[str, float] = {}
        for entry in values:
            if not isinstance(entry, dict):
                continue
            raw = str(entry.get("value") or "").strip().lower()
            price = self._odd(entry)
            if price is None or " " not in raw:
                continue
            side, _, line_text = raw.partition(" ")
            try:
                line = float(line_text)
            except ValueError:
                continue
            if line not in SUPPORTED_TOTALS:
                continue
            if side == "over":
                prices[f"OU{line:g}:O"] = price
            elif side == "under":
                prices[f"OU{line:g}:U"] = price
        return prices

    def _btts(self, values: list) -> dict[str, float]:
        prices: dict[str, float] = {}
        for entry in values:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("value") or "").strip().lower()
            price = self._odd(entry)
            if price is None:
                continue
            if name == "yes":
                prices["BTTS:Y"] = price
            elif name == "no":
                prices["BTTS:N"] = price
        return prices
