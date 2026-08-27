"""The odds-provider interface.

A provider's only job is to return :class:`Fixture` objects with prices on
them. Everything downstream — modelling, devigging, staking — is source
agnostic, so adding a new bookmaker feed means implementing one method.

Two conventions every provider must honour, because the engine's maths
depends on them:

* ``odds`` is the **best price you could actually take** across the books
  the provider covers.
* ``sharp_odds`` is the **sharpest single book's** price, used as the market's
  honest opinion once the margin is stripped. Pinnacle or a betting exchange
  if available, otherwise the book with the thinnest overround.

Mixing those up silently inverts the edge calculation, so
:func:`assemble_fixture` does the work in one place rather than leaving it to
each client.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Iterable, Protocol, Sequence

from ..schema import Fixture

# Books believed to price sharply, best first. Used to pick the reference line.
SHARP_BOOKS = ["pinnacle", "betfair_ex_eu", "betfair_ex_uk", "betfair", "smarkets",
               "matchbook", "circasports", "bookmaker"]


class ProviderError(RuntimeError):
    """A provider could not return usable data."""


@dataclass
class BookQuote:
    """One book's prices for one fixture, keyed by canonical selection id."""

    book: str
    prices: dict[str, float]


class OddsProvider(Protocol):
    """Anything that can supply upcoming fixtures with prices."""

    name: str

    def fixtures(self, league: str, days_ahead: int = 7) -> list[Fixture]:
        ...


def _overround(prices: dict[str, float], group: Sequence[str]) -> float | None:
    if not all(sel in prices for sel in group):
        return None
    return sum(1.0 / prices[sel] for sel in group) - 1.0


def pick_sharp_book(quotes: Sequence[BookQuote]) -> BookQuote | None:
    """Prefer a known-sharp book; otherwise the thinnest 1X2 margin."""
    by_name = {q.book.lower(): q for q in quotes}
    for candidate in SHARP_BOOKS:
        if candidate in by_name:
            return by_name[candidate]

    group = ("1X2:H", "1X2:D", "1X2:A")
    priced = [(q, _overround(q.prices, group)) for q in quotes]
    priced = [(q, margin) for q, margin in priced if margin is not None]
    if not priced:
        return quotes[0] if quotes else None
    return min(priced, key=lambda pair: pair[1])[0]


def assemble_fixture(
    match_date: date,
    league: str,
    home: str,
    away: str,
    quotes: Sequence[BookQuote],
    kickoff: str | None = None,
) -> Fixture | None:
    """Collapse many books' prices into one Fixture.

    Best price per selection, the sharp book's line as the reference, and a
    count of how many books quoted each selection (the engine's liquidity
    proxy).
    """
    if not quotes:
        return None

    best: dict[str, float] = {}
    counts: dict[str, int] = {}
    for quote in quotes:
        for selection, price in quote.prices.items():
            if price <= 1.0:
                continue
            counts[selection] = counts.get(selection, 0) + 1
            if price > best.get(selection, 0.0):
                best[selection] = price
    if not best:
        return None

    sharp_quote = pick_sharp_book(quotes)
    sharp = dict(sharp_quote.prices) if sharp_quote else {}
    # Never report a best price worse than the reference we are comparing to.
    for selection, price in sharp.items():
        if price > best.get(selection, 0.0):
            best[selection] = price

    return Fixture(
        date=match_date,
        league=league,
        home=home,
        away=away,
        kickoff=kickoff,
        odds=best,
        sharp_odds=sharp,
        book_counts=counts,
    )


def parse_iso_datetime(raw: str) -> datetime | None:
    if not raw:
        return None
    text = str(raw).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def api_key_for(env_names: Iterable[str], explicit: str | None = None) -> str:
    """Read a key from the argument or the environment, or explain how to set it."""
    if explicit:
        return explicit
    names = list(env_names)
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    raise ProviderError(
        "no API key supplied. Pass --api-key, or set one of: " + ", ".join(names)
    )


def request_json(url: str, params: dict[str, Any], timeout: int = 25,
                 headers: dict[str, str] | None = None,
                 capture_headers: dict[str, str] | None = None) -> Any:
    """GET some JSON, turning the usual failures into readable messages.

    ``capture_headers``, when given, is filled in with the response headers, so
    a caller can read quota counters without this function having to know what
    any particular provider calls them.
    """
    import requests

    try:
        response = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
    except Exception as exc:
        raise ProviderError(f"could not reach {url}: {exc}") from exc

    if capture_headers is not None:
        # Before the status checks, so a 429 still hands back the counters that
        # explain it.
        capture_headers.update({k.lower(): v for k, v in response.headers.items()})

    if response.status_code == 401:
        raise ProviderError("the provider rejected your API key (401).")
    if response.status_code == 403:
        raise ProviderError(
            "the provider refused the request (403) — usually an inactive key or a "
            "market not included in your plan."
        )
    if response.status_code == 429:
        raise ProviderError(
            "rate limit or monthly quota exhausted (429). Free tiers are small; "
            "results are cached, so re-running costs nothing until the cache expires."
        )
    if response.status_code >= 400:
        raise ProviderError(f"provider returned HTTP {response.status_code}: "
                            f"{response.text[:200]}")
    try:
        return response.json()
    except ValueError as exc:
        raise ProviderError(f"provider returned something that is not JSON: "
                            f"{response.text[:200]}") from exc
