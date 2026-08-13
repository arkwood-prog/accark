"""Replay a captured provider payload, with no network at all.

Written for a specific, common situation: the machine running the analysis
cannot reach the odds API. A locked-down CI runner, a corporate egress policy,
an air-gapped box, or an agent sandbox whose network policy allows package
registries and nothing else.

The workflow is two steps:

    # on a machine that CAN reach the API
    bettingedge capture --league E0 --odds-provider theoddsapi --out payload.json

    # anywhere, including with no network
    bettingedge recommend --league E0 --replay-raw payload.json

The captured file is the provider's untouched response. It contains **no API
key** — keys travel in the request, not the reply — so it is safe to commit or
to hand to someone else for debugging. Check before sharing anyway.

Replay is for verifying that parsing, team matching and pricing work against
genuine data. It is not for betting: a captured payload is a snapshot, and its
prices go stale within minutes.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..schema import Fixture
from .apifootball import APIFootball
from .base import ProviderError
from .theoddsapi import TheOddsAPI


def detect_shape(payload: Any) -> str:
    """Work out which provider produced a captured payload."""
    if isinstance(payload, list):
        for event in payload:
            if isinstance(event, dict) and "home_team" in event and "bookmakers" in event:
                return "theoddsapi"
        if not payload:
            raise ProviderError("the captured payload is an empty list — the provider "
                                "returned no fixtures when it was captured.")
    if isinstance(payload, dict):
        if "fixtures" in payload and "odds" in payload:
            return "apifootball"
        # A single Odds API event, or a wrapper someone saved by hand.
        if "home_team" in payload and "bookmakers" in payload:
            return "theoddsapi-single"
        if payload.get("errors"):
            raise ProviderError(f"the captured payload is an API error: "
                                f"{payload['errors']}")
    raise ProviderError(
        "could not tell which provider produced this payload. Expected either a "
        "list of Odds API events, or an API-Football capture with 'fixtures' and "
        "'odds' keys."
    )


class ReplayProvider:
    """Reparses a captured payload through the real provider's parser."""

    name = "replay"

    def __init__(self, path: str | Path, shape: str | None = None,
                 ignore_dates: bool = True):
        self.path = Path(path)
        if not self.path.exists():
            raise ProviderError(f"no captured payload at {self.path}")
        try:
            self.payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ProviderError(f"could not read {self.path}: {exc}") from exc
        self.shape = shape or detect_shape(self.payload)
        # A capture is by definition in the past by the time it is replayed, so
        # the "next N days" window is meaningless. Keep every event instead.
        self.ignore_dates = ignore_dates

    def fixtures(self, league: str, days_ahead: int = 7) -> list[Fixture]:
        if self.shape.startswith("theoddsapi"):
            return self._replay_odds_api(league, days_ahead)
        return self._replay_api_football(league)

    # -- per-provider replay --------------------------------------------
    def _replay_odds_api(self, league: str, days_ahead: int) -> list[Fixture]:
        client = TheOddsAPI(api_key="replay")   # never used; no request is made
        events = self.payload if isinstance(self.payload, list) else [self.payload]
        cutoff = (datetime.max.replace(tzinfo=timezone.utc) if self.ignore_dates
                  else datetime.now(timezone.utc) + timedelta(days=days_ahead))

        fixtures = []
        for event in events:
            built = client._event_to_fixture(event, league, cutoff)
            if built:
                fixtures.append(built)
        fixtures.sort(key=lambda f: (f.date, f.home))
        return fixtures

    def _replay_api_football(self, league: str) -> list[Fixture]:
        from .base import assemble_fixture, parse_iso_datetime

        client = APIFootball(api_key="replay")
        raw_fixtures = self.payload.get("fixtures") or []
        odds_by_fixture = client._index_odds(self.payload.get("odds") or [])

        fixtures = []
        for entry in raw_fixtures:
            if not isinstance(entry, dict):
                continue
            info = entry.get("fixture") or {}
            teams = entry.get("teams") or {}
            home = ((teams.get("home") or {}).get("name") or "").strip()
            away = ((teams.get("away") or {}).get("name") or "").strip()
            starts = parse_iso_datetime(info.get("date", ""))
            quotes = odds_by_fixture.get(info.get("id"))
            if not (home and away and starts and quotes):
                continue
            local = starts.astimezone()
            built = assemble_fixture(match_date=local.date(), league=league, home=home,
                                     away=away, quotes=quotes,
                                     kickoff=local.strftime("%H:%M"))
            if built:
                fixtures.append(built)
        fixtures.sort(key=lambda f: (f.date, f.home))
        return fixtures


def describe_capture(fixtures: list[Fixture]) -> str:
    """A summary of what a capture actually contains.

    Worth reading before anything else: it shows the team names exactly as the
    provider spells them, which is what team matching has to cope with, and
    which markets survived parsing.
    """
    if not fixtures:
        return ("No fixtures parsed out of this payload. Either the provider returned "
                "nothing (out of season, or a window with no games), or the response "
                "shape has changed. Inspect the raw file.")

    markets: dict[str, int] = {}
    books = 0
    for fixture in fixtures:
        for selection in fixture.odds:
            markets[selection] = markets.get(selection, 0) + 1
        books = max(books, max(fixture.book_counts.values(), default=0))

    lines = [
        f"{len(fixtures)} fixture(s) parsed, up to {books} book(s) per selection.",
        "",
        "Markets found (selection: fixtures quoting it):",
    ]
    lines += [f"    {selection:<12} {count}"
              for selection, count in sorted(markets.items())]
    lines += ["", "Team names exactly as the provider spells them:"]
    for fixture in fixtures[:12]:
        lines.append(f"    {fixture.date}  {fixture.home}  v  {fixture.away}")
    if len(fixtures) > 12:
        lines.append(f"    ... and {len(fixtures) - 12} more")
    return "\n".join(lines)
