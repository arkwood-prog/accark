"""The Odds API (the-odds-api.com).

The best free option for live prices: a few hundred requests a month at no
cost, dozens of bookmakers per fixture including Pinnacle and the exchanges,
and it covers every league this project models.

    export ODDS_API_KEY=...
    bettingedge recommend --league E0 --odds-provider theoddsapi

Because it returns *every* book's price for a fixture, it is a better fit for
this engine than the free CSV feed: best price and sharp reference come from
the same snapshot instead of being approximated by column choice.

Response shapes at any provider drift over time. Everything here is parsed
defensively and `--dump-raw` will write the untouched payload so a mismatch is
diagnosable rather than mysterious.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
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

BASE_URL = "https://api.the-odds-api.com/v4"

# football-data.co.uk league code -> The Odds API sport key.
SPORT_KEYS: dict[str, str] = {
    "E0": "soccer_epl",
    "E1": "soccer_efl_champ",
    "E2": "soccer_england_league1",
    "E3": "soccer_england_league2",
    "EC": "soccer_england_efl_cup",
    "SC0": "soccer_spl",
    "D1": "soccer_germany_bundesliga",
    "D2": "soccer_germany_bundesliga2",
    "I1": "soccer_italy_serie_a",
    "I2": "soccer_italy_serie_b",
    "SP1": "soccer_spain_la_liga",
    "SP2": "soccer_spain_segunda_division",
    "F1": "soccer_france_ligue_one",
    "F2": "soccer_france_ligue_two",
    "N1": "soccer_netherlands_eredivisie",
    "B1": "soccer_belgium_first_div",
    "P1": "soccer_portugal_primeira_liga",
    "T1": "soccer_turkey_super_league",
    "G1": "soccer_greece_super_league",
}

# Over/under lines the engine prices; anything else in the payload is ignored.
SUPPORTED_TOTALS = {0.5, 1.5, 2.5, 3.5, 4.5}


class TheOddsAPI:
    """Client for The Odds API v4."""

    name = "theoddsapi"

    def __init__(self, api_key: str | None = None, regions: str = "uk,eu",
                 timeout: int = 25, dump_raw: str | Path | None = None):
        # Resolved lazily so the parser can be used on a captured payload with
        # no key present — see providers/replay.py.
        self._api_key = api_key
        self._resolved_key: str | None = None
        self.regions = regions
        self.timeout = timeout
        self.dump_raw = Path(dump_raw) if dump_raw else None
        self.quota_remaining: str | None = None

    @property
    def api_key(self) -> str:
        if self._resolved_key is None:
            self._resolved_key = api_key_for(["ODDS_API_KEY", "THE_ODDS_API_KEY"],
                                             self._api_key)
        return self._resolved_key

    # -- helpers --------------------------------------------------------
    def sports(self) -> list[dict]:
        """Every sport key the account can access — useful when a code is wrong."""
        return request_json(f"{BASE_URL}/sports", {"apiKey": self.api_key},
                            timeout=self.timeout)

    def _sport_key(self, league: str) -> str:
        key = SPORT_KEYS.get(league.upper())
        if not key:
            raise ProviderError(
                f"no The Odds API sport key is mapped for league {league!r}. "
                f"Mapped leagues: {', '.join(sorted(SPORT_KEYS))}. "
                "Run `bettingedge providers --list-sports` to see what your key covers."
            )
        return key

    # -- the interface --------------------------------------------------
    def fixtures(self, league: str, days_ahead: int = 7) -> list[Fixture]:
        sport = self._sport_key(league)
        payload = request_json(
            f"{BASE_URL}/sports/{sport}/odds",
            {
                "apiKey": self.api_key,
                "regions": self.regions,
                "markets": "h2h,totals",
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
            timeout=self.timeout,
        )
        if self.dump_raw:
            self.dump_raw.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if not isinstance(payload, list):
            raise ProviderError(
                f"expected a list of events, got {type(payload).__name__}. "
                "Re-run with --dump-raw to inspect the payload."
            )

        cutoff = datetime.now(timezone.utc) + timedelta(days=days_ahead)
        fixtures: list[Fixture] = []
        for event in payload:
            fixture = self._event_to_fixture(event, league, cutoff)
            if fixture:
                fixtures.append(fixture)
        fixtures.sort(key=lambda f: (f.date, f.home))
        return fixtures

    # -- parsing --------------------------------------------------------
    def _event_to_fixture(self, event: Any, league: str,
                          cutoff: datetime) -> Fixture | None:
        if not isinstance(event, dict):
            return None
        home = (event.get("home_team") or "").strip()
        away = (event.get("away_team") or "").strip()
        starts = parse_iso_datetime(event.get("commence_time", ""))
        if not (home and away and starts):
            return None
        if starts > cutoff:
            return None

        quotes: list[BookQuote] = []
        for bookmaker in event.get("bookmakers") or []:
            prices = self._book_prices(bookmaker, home, away)
            if prices:
                quotes.append(BookQuote(book=str(bookmaker.get("key") or "unknown"),
                                        prices=prices))
        if not quotes:
            return None

        local = starts.astimezone()
        return assemble_fixture(
            match_date=local.date(),
            league=league,
            home=home,
            away=away,
            quotes=quotes,
            kickoff=local.strftime("%H:%M"),
        )

    def _book_prices(self, bookmaker: Any, home: str, away: str) -> dict[str, float]:
        if not isinstance(bookmaker, dict):
            return {}
        prices: dict[str, float] = {}
        for market in bookmaker.get("markets") or []:
            if not isinstance(market, dict):
                continue
            key = market.get("key")
            outcomes = market.get("outcomes") or []
            if key in ("h2h", "h2h_3_way"):
                prices.update(self._h2h(outcomes, home, away))
            elif key == "totals":
                prices.update(self._totals(outcomes))
            elif key == "btts":
                prices.update(self._btts(outcomes))
        return prices

    @staticmethod
    def _price(outcome: Any) -> float | None:
        try:
            value = float(outcome.get("price"))
        except (TypeError, ValueError, AttributeError):
            return None
        return value if value > 1.0 else None

    def _h2h(self, outcomes: list, home: str, away: str) -> dict[str, float]:
        prices: dict[str, float] = {}
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            name = (outcome.get("name") or "").strip()
            price = self._price(outcome)
            if price is None:
                continue
            # Outcomes are named by team, so match against this event's teams
            # rather than assuming an order.
            if name.lower() == "draw":
                prices["1X2:D"] = price
            elif name == home:
                prices["1X2:H"] = price
            elif name == away:
                prices["1X2:A"] = price
        return prices

    def _totals(self, outcomes: list) -> dict[str, float]:
        prices: dict[str, float] = {}
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            price = self._price(outcome)
            try:
                line = float(outcome.get("point"))
            except (TypeError, ValueError):
                continue
            if price is None or line not in SUPPORTED_TOTALS:
                continue
            side = (outcome.get("name") or "").strip().lower()
            if side == "over":
                prices[f"OU{line:g}:O"] = price
            elif side == "under":
                prices[f"OU{line:g}:U"] = price
        return prices

    def _btts(self, outcomes: list) -> dict[str, float]:
        prices: dict[str, float] = {}
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            price = self._price(outcome)
            name = (outcome.get("name") or "").strip().lower()
            if price is None:
                continue
            if name in ("yes", "both teams to score"):
                prices["BTTS:Y"] = price
            elif name in ("no", "not both teams to score"):
                prices["BTTS:N"] = price
        return prices
