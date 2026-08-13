"""Live odds providers.

    from bettingedge.data.providers import get_provider
    provider = get_provider("theoddsapi")
    fixtures = provider.fixtures("E0", days_ahead=7)

Adding another source means implementing one method — see
:class:`~bettingedge.data.providers.base.OddsProvider`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .apifootball import APIFootball
from .base import BookQuote, OddsProvider, ProviderError, assemble_fixture, pick_sharp_book
from .theoddsapi import TheOddsAPI


@dataclass(frozen=True)
class ProviderInfo:
    key: str
    title: str
    url: str
    env_vars: tuple[str, ...]
    free_tier: str
    markets: str
    notes: str


PROVIDER_INFO: dict[str, ProviderInfo] = {
    "footballdata": ProviderInfo(
        key="footballdata",
        title="football-data.co.uk (default)",
        url="https://www.football-data.co.uk",
        env_vars=(),
        free_tier="unlimited, no key",
        markets="1X2, over/under 2.5",
        notes=("The only free source carrying results and closing prices together, so "
               "it stays the source for history and backtesting. Its fixtures feed "
               "covers a few days and is empty between seasons."),
    ),
    "theoddsapi": ProviderInfo(
        key="theoddsapi",
        title="The Odds API",
        url="https://the-odds-api.com",
        env_vars=("ODDS_API_KEY", "THE_ODDS_API_KEY"),
        free_tier="about 500 requests/month",
        markets="1X2, over/under (all lines), BTTS on some plans",
        notes=("Best free option for live prices. Returns every book's price per "
               "fixture, so best price and sharp reference come from one snapshot."),
    ),
    "apifootball": ProviderInfo(
        key="apifootball",
        title="API-Football",
        url="https://www.api-football.com",
        env_vars=("API_FOOTBALL_KEY", "APIFOOTBALL_KEY", "RAPIDAPI_KEY"),
        free_tier="about 100 requests/day",
        markets="1X2, over/under, BTTS",
        notes=("Wider league coverage and a useful fallback when The Odds API's "
               "monthly quota runs out. Costs two requests per run."),
    ),
}

_FACTORIES = {
    "theoddsapi": TheOddsAPI,
    "apifootball": APIFootball,
}


def available_providers() -> list[str]:
    return ["footballdata", *sorted(_FACTORIES)]


def get_provider(name: str, **kwargs: Any) -> OddsProvider:
    """Build a live odds provider by name."""
    key = (name or "").lower()
    if key not in _FACTORIES:
        raise ProviderError(
            f"unknown odds provider {key!r}; choose one of "
            f"{', '.join(available_providers())}"
        )
    return _FACTORIES[key](**kwargs)


__all__ = [
    "APIFootball",
    "TheOddsAPI",
    "BookQuote",
    "OddsProvider",
    "ProviderError",
    "ProviderInfo",
    "PROVIDER_INFO",
    "assemble_fixture",
    "available_providers",
    "get_provider",
    "pick_sharp_book",
]
