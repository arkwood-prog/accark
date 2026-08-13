"""Finding value: blending the model with the market, then measuring edge.

The central judgement call in this whole project is *how much to trust the
model over the market*. Closing prices in liquid football markets are close to
efficient. A model that ignores them will find "edges" that are really just
model error.

So probabilities are blended in **logit space**, which is the right space for
combining opinions about probabilities (it is symmetric, respects the
boundaries, and averages odds ratios rather than raw probabilities), and then
renormalised within each market so the set still sums to one.

Edge is then plain expected value per unit staked:

    EV = p * (odds - 1) - (1 - p) = p * odds - 1
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from ..config import Config
from ..data.schema import Fixture, Selection
from ..market.consensus import MarketQuote
from ..models.markets import PARTITIONS, ou_partition

_EPS = 1e-9


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def expected_value(probability: float, odds: float) -> float:
    """Expected profit per unit staked."""
    return probability * odds - 1.0


def kelly_fraction(probability: float, odds: float) -> float:
    """Full-Kelly stake as a fraction of bankroll. Zero when there is no edge."""
    if odds <= 1.0:
        return 0.0
    edge = expected_value(probability, odds)
    return max(0.0, edge / (odds - 1.0))


def blend(model_probability: float, market_probability: float, weight: float) -> float:
    """Weighted average of two probabilities in logit space."""
    return _sigmoid(weight * _logit(model_probability) + (1.0 - weight) * _logit(market_probability))


def blend_market_set(
    model_probs: dict[str, float],
    quotes: dict[str, MarketQuote],
    weight: float,
    ou_lines: tuple[float, ...] = (1.5, 2.5, 3.5),
) -> dict[str, float]:
    """Blend model and market across every market where both have an opinion.

    Blending is done partition by partition and renormalised, so the result is
    still a valid probability distribution over each market's outcomes.
    Selections the market does not quote keep the pure model probability.
    """
    groups = [PARTITIONS["1X2"], PARTITIONS["BTTS"]] + [ou_partition(line) for line in ou_lines]
    blended = dict(model_probs)

    for group in groups:
        if not all(sel in model_probs for sel in group):
            continue
        if not all(sel in quotes for sel in group):
            continue  # no market view: keep the model's own numbers
        raw = [blend(model_probs[sel], quotes[sel].fair_probability, weight) for sel in group]
        total = sum(raw)
        if total <= 0:
            continue
        for sel, value in zip(group, raw):
            blended[sel] = value / total

    # Double chance is the sum of its 1X2 parts — derive it rather than
    # blending it separately, or the numbers stop agreeing with each other.
    if all(sel in blended for sel in PARTITIONS["1X2"]):
        home, draw, away = (blended[sel] for sel in PARTITIONS["1X2"])
        blended["DC:1X"] = home + draw
        blended["DC:12"] = home + away
        blended["DC:X2"] = draw + away
    return blended


@dataclass
class Candidate:
    """A single bet that has been priced and measured against the market."""

    fixture_key: str
    date: date
    league: str
    home: str
    away: str
    selection_id: str
    label: str
    short_label: str
    odds: float
    model_probability: float
    market_probability: float
    probability: float          # blended — this is what we bet on
    edge: float                 # expected value per unit staked
    kelly: float                # full-Kelly fraction
    fair_odds: float
    market_fair_odds: float
    overround: float
    book_count: int
    data_confidence: float
    confidence: float           # 0-100
    tier: str
    price_source: str
    derived: bool

    @property
    def selection(self) -> Selection:
        return Selection.parse(self.selection_id)

    @property
    def match(self) -> str:
        return f"{self.home} v {self.away}"

    def to_dict(self) -> dict:
        return {
            "fixture_key": self.fixture_key,
            "date": self.date.isoformat(),
            "league": self.league,
            "home": self.home,
            "away": self.away,
            "match": self.match,
            "selection": self.selection_id,
            "label": self.label,
            "short_label": self.short_label,
            "odds": round(self.odds, 3),
            "model_probability": round(self.model_probability, 4),
            "market_probability": round(self.market_probability, 4),
            "probability": round(self.probability, 4),
            "edge": round(self.edge, 4),
            "edge_pct": round(self.edge * 100, 2),
            "kelly": round(self.kelly, 4),
            "fair_odds": round(self.fair_odds, 3),
            "market_fair_odds": round(self.market_fair_odds, 3),
            "overround": round(self.overround, 4),
            "book_count": self.book_count,
            "confidence": round(self.confidence, 1),
            "tier": self.tier,
            "price_source": self.price_source,
            "derived": self.derived,
        }


def _confidence_score(
    edge: float,
    probability: float,
    quote: MarketQuote,
    data_confidence: float,
    model_probability: float,
    market_probability: float,
) -> float:
    """0-100 score for how much weight to put behind a bet.

    Deliberately not just "bigger edge = better". An implausibly large edge
    usually means the model is wrong or the price is stale, so the edge
    component saturates and then gets penalised.
    """
    edge_component = min(1.0, max(0.0, edge) / 0.08)
    if edge > 0.25:
        edge_component *= 0.55        # too good to be true, treat it as such
    elif edge > 0.15:
        edge_component *= 0.80

    liquidity = min(1.0, quote.book_count / 8.0)
    margin_quality = min(1.0, max(0.0, (0.12 - quote.overround) / 0.09))
    market_component = 0.5 * liquidity + 0.5 * margin_quality

    # Disagreement between model and market: some is required for an edge to
    # exist at all, but a wild gap is a red flag rather than an opportunity.
    gap = abs(_logit(model_probability) - _logit(market_probability))
    stability = math.exp(-gap / 0.55)

    # Extreme probabilities are where both model and market are least reliable.
    balance = 1.0 - abs(probability - 0.5) / 0.5
    balance = 0.55 + 0.45 * balance

    score = (0.30 * data_confidence
             + 0.20 * market_component
             + 0.30 * edge_component
             + 0.20 * stability) * balance
    return round(100.0 * min(1.0, max(0.0, score)), 1)


def tier_for(confidence: float) -> str:
    if confidence >= 62:
        return "High"
    if confidence >= 48:
        return "Medium"
    if confidence >= 34:
        return "Speculative"
    return "Low"


def build_candidates(
    fixture: Fixture,
    blended_probs: dict[str, float],
    model_probs: dict[str, float],
    quotes: dict[str, MarketQuote],
    config: Config,
    data_confidence: float,
    min_confidence: float | None = None,
) -> list[Candidate]:
    """Every selection on this fixture that clears the selection filters.

    ``min_confidence`` overrides the configured floor, which the pipeline uses
    to raise the bar automatically when the model is working from a thin
    sample.
    """
    sel_cfg = config.selection
    floor = sel_cfg.min_confidence if min_confidence is None else min_confidence
    candidates: list[Candidate] = []

    for selection_id, quote in quotes.items():
        selection = Selection.parse(selection_id)
        if selection.market not in sel_cfg.markets:
            continue
        if selection.market == "OU" and selection.line not in sel_cfg.ou_lines:
            continue
        probability = blended_probs.get(selection_id)
        if probability is None:
            continue

        odds = quote.best_odds
        if not (sel_cfg.min_odds <= odds <= sel_cfg.max_odds):
            continue
        if not (sel_cfg.min_probability <= probability <= sel_cfg.max_probability):
            continue

        edge = expected_value(probability, odds)
        if edge < sel_cfg.min_edge:
            continue

        confidence = _confidence_score(
            edge=edge,
            probability=probability,
            quote=quote,
            data_confidence=data_confidence,
            model_probability=model_probs.get(selection_id, probability),
            market_probability=quote.fair_probability,
        )
        if confidence < floor:
            continue
        candidates.append(
            Candidate(
                fixture_key=fixture.key,
                date=fixture.date,
                league=fixture.league,
                home=fixture.home,
                away=fixture.away,
                selection_id=selection_id,
                label=selection.describe(fixture.home, fixture.away),
                short_label=selection.short(),
                odds=odds,
                model_probability=model_probs.get(selection_id, probability),
                market_probability=quote.fair_probability,
                probability=probability,
                edge=edge,
                kelly=kelly_fraction(probability, odds),
                fair_odds=1.0 / probability if probability > 0 else float("inf"),
                market_fair_odds=quote.fair_odds,
                overround=quote.overround,
                book_count=quote.book_count,
                data_confidence=data_confidence,
                confidence=confidence,
                tier=tier_for(confidence),
                price_source=quote.best_source,
                derived=quote.best_source.startswith("derived"),
            )
        )

    candidates.sort(key=lambda c: (-c.confidence, -c.edge))
    return candidates
