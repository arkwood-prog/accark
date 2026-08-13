"""Build the market's honest opinion of a fixture.

For each complete market we take the sharpest available prices, strip the
margin, and record how much margin there was and how many books quoted it.
Those last two are the market-quality signals: a two-book market with a 12%
overround deserves far less trust than a ten-book market at 3%.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import MarketConfig
from ..data.schema import Fixture
from ..models.markets import PARTITIONS, ou_partition
from .devig import devig, overround


@dataclass
class MarketQuote:
    """What the market says about one selection, and where to back it."""

    selection_id: str
    best_odds: float
    best_source: str          # "best available" or "sharp"
    sharp_odds: float | None
    fair_probability: float   # margin removed
    fair_odds: float
    overround: float
    book_count: int

    @property
    def trusted(self) -> bool:
        return self.book_count >= 3 and self.overround <= 0.10

    def to_dict(self) -> dict:
        return {
            "selection": self.selection_id,
            "best_odds": round(self.best_odds, 3),
            "sharp_odds": round(self.sharp_odds, 3) if self.sharp_odds else None,
            "fair_probability": round(self.fair_probability, 4),
            "fair_odds": round(self.fair_odds, 3),
            "overround": round(self.overround, 4),
            "book_count": self.book_count,
        }


def _candidate_partitions(fixture: Fixture, ou_lines: tuple[float, ...]) -> list[tuple[str, ...]]:
    groups = [PARTITIONS["1X2"], PARTITIONS["BTTS"]]
    groups += [ou_partition(line) for line in ou_lines]
    available = []
    for group in groups:
        if all(fixture.sharp_or_best(sel) for sel in group):
            available.append(group)
    return available


def market_view(
    fixture: Fixture,
    config: MarketConfig | None = None,
    ou_lines: tuple[float, ...] = (1.5, 2.5, 3.5),
) -> dict[str, MarketQuote]:
    """Fair probabilities and best prices for every market this fixture has."""
    config = config or MarketConfig()
    quotes: dict[str, MarketQuote] = {}

    for group in _candidate_partitions(fixture, ou_lines):
        # Fair probabilities come from the sharp line where we have one.
        sharp_prices = [fixture.sharp_odds.get(sel) for sel in group]
        use_sharp = all(price for price in sharp_prices)
        if use_sharp:
            pricing_odds = [float(p) for p in sharp_prices]
        else:
            # Fall back per selection: a CSV may carry a sharp column for one
            # outcome and only a best price for another.
            fallback = [fixture.sharp_or_best(sel) for sel in group]
            if not all(fallback):
                continue
            pricing_odds = [float(p) for p in fallback]

        margin = overround(pricing_odds)
        method = config.devig_three_way if len(group) >= 3 else config.devig_two_way
        try:
            fair = devig(pricing_odds, method)
        except ValueError:
            continue

        for selection_id, probability, price in zip(group, fair, pricing_odds):
            best = float(fixture.odds.get(selection_id) or price)
            # Never claim a best price worse than the sharp one we already saw.
            best = max(best, price)
            quotes[selection_id] = MarketQuote(
                selection_id=selection_id,
                best_odds=best,
                best_source="best available" if best > price else "sharp",
                sharp_odds=float(fixture.sharp_odds[selection_id]) if use_sharp else None,
                fair_probability=probability,
                fair_odds=1.0 / probability if probability > 0 else float("inf"),
                overround=margin,
                book_count=int(fixture.book_counts.get(selection_id, 1)),
            )

    # Double chance is rarely quoted directly; derive it from the 1X2 view so
    # the whole price set stays consistent.
    if all(sel in quotes for sel in PARTITIONS["1X2"]):
        home, draw, away = (quotes[sel] for sel in PARTITIONS["1X2"])
        derived = {
            "DC:1X": (home.fair_probability + draw.fair_probability,
                      _parlay_free_price(home.best_odds, draw.best_odds)),
            "DC:12": (home.fair_probability + away.fair_probability,
                      _parlay_free_price(home.best_odds, away.best_odds)),
            "DC:X2": (draw.fair_probability + away.fair_probability,
                      _parlay_free_price(draw.best_odds, away.best_odds)),
        }
        for selection_id, (probability, price) in derived.items():
            if price is None or probability <= 0:
                continue
            quotes[selection_id] = MarketQuote(
                selection_id=selection_id,
                best_odds=price,
                best_source="derived from 1X2",
                sharp_odds=None,
                fair_probability=probability,
                fair_odds=1.0 / probability,
                overround=home.overround,
                book_count=home.book_count,
            )
    return quotes


# Haircut applied to synthetic double-chance prices. The construction below is
# genuinely executable, but it needs two stakes placed at two different books
# at prices that may move between them, so the theoretical price is shaded.
DERIVED_PRICE_HAIRCUT = 0.98


def _parlay_free_price(odds_a: float, odds_b: float) -> float | None:
    """Synthetic double-chance price built from two 1X2 outcomes.

    Splitting a stake across both outcomes in proportion to their prices, so
    that either result returns the same amount, pays ``1 / (1/a + 1/b)``. That
    is a real, placeable bet — it is just two tickets rather than one — and it
    usually beats the double chance a book will quote you directly.

    It is shaded by :data:`DERIVED_PRICE_HAIRCUT` for execution risk and
    flagged as derived so the analysis can tell you to check the real quote.
    """
    if odds_a <= 1.0 or odds_b <= 1.0:
        return None
    return DERIVED_PRICE_HAIRCUT / (1.0 / odds_a + 1.0 / odds_b)
