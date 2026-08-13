"""Building doubles, trebles and accumulators — properly.

Two things get combined bets wrong more than anything else:

1. **Margin compounds.** A book with a 5% margin per leg has roughly a 16%
   margin on a treble. That is why a multi built from break-even legs is a
   losing bet: every leg you add multiplies the hold against you. Legs must
   therefore clear a *higher* individual bar than a single would.

2. **Correlation.** "Home win" and "Over 2.5 goals" in the same match are not
   independent, so multiplying their probabilities is simply wrong. Legs from
   different matches are combined by multiplication; legs from the same match
   are priced from the joint score distribution instead.

Ranking is by **expected log growth** rather than raw expected value. Two
slips can have the same EV while one is a 60% shot and the other a 4% shot;
log growth is what the Kelly criterion actually maximises and it correctly
prefers the first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

from ..config import Config
from ..models.markets import joint_probability
from .value import Candidate, expected_value, kelly_fraction

SIZE_NAMES = {1: "Single", 2: "Double", 3: "Treble", 4: "4-fold", 5: "5-fold",
              6: "6-fold", 7: "7-fold", 8: "8-fold"}


def size_name(n: int) -> str:
    return SIZE_NAMES.get(n, f"{n}-fold")


def expected_log_growth(probability: float, odds: float, stake_fraction: float) -> float:
    """Expected log bankroll growth from staking `stake_fraction` at these terms."""
    if stake_fraction <= 0:
        return 0.0
    stake_fraction = min(stake_fraction, 0.999)
    win = math.log1p(stake_fraction * (odds - 1.0))
    lose = math.log1p(-stake_fraction)
    return probability * win + (1.0 - probability) * lose


@dataclass
class Slip:
    """A bet slip: one leg for a single, several for a multi."""

    kind: str                     # "Single", "Double", "Treble", "4-fold", ...
    legs: list[Candidate]
    probability: float
    odds: float
    edge: float
    kelly: float
    log_growth: float
    correlated: bool = False
    profile: str = ""
    note: str = ""
    stake: float = 0.0
    analysis: str = ""
    leg_analysis: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.legs)

    @property
    def potential_return(self) -> float:
        return self.stake * self.odds

    @property
    def confidence(self) -> float:
        """A multi is only as trustworthy as its weakest leg."""
        return min(leg.confidence for leg in self.legs) if self.legs else 0.0

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "size": self.size,
            "profile": self.profile,
            "note": self.note,
            "correlated": self.correlated,
            "probability": round(self.probability, 4),
            "probability_pct": round(self.probability * 100, 2),
            "odds": round(self.odds, 3),
            "fair_odds": round(1.0 / self.probability, 3) if self.probability > 0 else None,
            "edge": round(self.edge, 4),
            "edge_pct": round(self.edge * 100, 2),
            "kelly": round(self.kelly, 4),
            "log_growth": round(self.log_growth, 6),
            "confidence": round(self.confidence, 1),
            "stake": round(self.stake, 2),
            "potential_return": round(self.potential_return, 2),
            "potential_profit": round(self.potential_return - self.stake, 2),
            "analysis": self.analysis,
            "leg_analysis": self.leg_analysis,
            "legs": [leg.to_dict() for leg in self.legs],
        }


def _score(probability: float, odds: float, kelly_cap: float) -> tuple[float, float, float]:
    edge = expected_value(probability, odds)
    kelly = kelly_fraction(probability, odds)
    growth = expected_log_growth(probability, odds, min(kelly * kelly_cap, kelly_cap))
    return edge, kelly, growth


def _best_per_fixture(candidates: Iterable[Candidate]) -> list[Candidate]:
    """One leg per match — two selections from the same match are correlated."""
    best: dict[str, Candidate] = {}
    for candidate in candidates:
        current = best.get(candidate.fixture_key)
        if current is None or candidate.confidence > current.confidence or (
            candidate.confidence == current.confidence and candidate.edge > current.edge
        ):
            best[candidate.fixture_key] = candidate
    return list(best.values())


def build_multis(candidates: Sequence[Candidate], config: Config) -> list[Slip]:
    """Cross-match doubles through accumulators, ranked by expected log growth."""
    cfg = config.parlay
    kelly_cap = config.staking.kelly_fraction

    eligible = [
        c for c in candidates
        if c.edge >= cfg.min_leg_edge and c.probability >= cfg.min_leg_probability
    ]
    pool = _best_per_fixture(eligible)
    pool.sort(key=lambda c: (-(c.confidence / 100.0) * (1 + c.edge), -c.edge))
    pool = pool[: cfg.leg_pool_size]

    slips: list[Slip] = []
    max_legs = min(cfg.max_legs, len(pool))

    for size in range(2, max_legs + 1):
        sized: list[Slip] = []
        for legs in combinations(pool, size):
            probability = 1.0
            odds = 1.0
            for leg in legs:
                probability *= leg.probability
                odds *= leg.odds
            if probability < cfg.min_combined_probability:
                continue
            if odds > cfg.max_combined_odds:
                continue
            edge, kelly, growth = _score(probability, odds, kelly_cap)
            if edge < cfg.min_combined_edge:
                continue
            sized.append(
                Slip(
                    kind=size_name(size),
                    legs=list(legs),
                    probability=probability,
                    odds=odds,
                    edge=edge,
                    kelly=kelly,
                    log_growth=growth,
                )
            )
        sized.sort(key=lambda s: -s.log_growth)
        chosen = _diversify(sized, cfg.max_per_size)
        _tag_profiles(chosen)
        slips.extend(chosen)
    return slips


def _diversify(slips: list[Slip], limit: int) -> list[Slip]:
    """Avoid surfacing near-identical slips that share most of their legs."""
    picked: list[Slip] = []
    for slip in slips:
        keys = {leg.fixture_key for leg in slip.legs}
        if any(len(keys & {leg.fixture_key for leg in other.legs}) >= slip.size - 1
               for other in picked):
            continue
        picked.append(slip)
        if len(picked) >= limit:
            break
    return picked


def _tag_profiles(slips: list[Slip]) -> None:
    if not slips:
        return
    by_growth = max(slips, key=lambda s: s.log_growth)
    by_edge = max(slips, key=lambda s: s.edge)
    by_probability = max(slips, key=lambda s: s.probability)
    by_growth.profile = "Balanced"
    by_growth.note = "Best expected bankroll growth for this leg count"
    if by_edge is not by_growth:
        by_edge.profile = "Value"
        by_edge.note = "Largest expected value, higher variance"
    if by_probability is not by_growth and by_probability is not by_edge:
        by_probability.profile = "Banker"
        by_probability.note = "Highest strike rate of the qualifying combinations"
    for slip in slips:
        if not slip.profile:
            slip.profile = "Alternative"


def build_same_game_slips(
    candidates: Sequence[Candidate],
    matrices: dict[str, np.ndarray],
    config: Config,
    limit: int = 4,
) -> list[Slip]:
    """Same-match combinations priced from the joint score distribution.

    A book that prices a same-game multi by multiplying its legs together is
    quoting the *independent* probability. Where the true joint probability is
    higher than that — positively correlated legs — the multiplied price is too
    generous. That gap is the edge, and it is only visible if you model the
    joint distribution rather than the margins.
    """
    cfg = config.parlay
    if not cfg.allow_same_game:
        return []
    kelly_cap = config.staking.kelly_fraction

    by_fixture: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_fixture.setdefault(candidate.fixture_key, []).append(candidate)

    slips: list[Slip] = []
    for fixture_key, legs in by_fixture.items():
        matrix = matrices.get(fixture_key)
        if matrix is None or len(legs) < 2:
            continue
        # Only combine different markets; two legs of the same market are
        # either identical or mutually exclusive.
        for pair in combinations(sorted(legs, key=lambda c: -c.confidence)[:5], 2):
            markets = {leg.selection.market for leg in pair}
            if len(markets) < 2:
                continue
            probability = joint_probability(matrix, [leg.selection_id for leg in pair])
            if probability < cfg.min_combined_probability:
                continue
            independent = pair[0].probability * pair[1].probability
            if independent <= 0:
                continue
            odds = pair[0].odds * pair[1].odds
            edge, kelly, growth = _score(probability, odds, kelly_cap)
            if edge < cfg.min_combined_edge:
                continue
            correlation = probability / independent
            slips.append(
                Slip(
                    kind="Same-game double",
                    legs=list(pair),
                    probability=probability,
                    odds=odds,
                    edge=edge,
                    kelly=kelly,
                    log_growth=growth,
                    correlated=True,
                    profile="Correlated",
                    note=(
                        f"Legs are {correlation:.2f}x more likely together than if they "
                        "were independent. Only worth taking at a book that prices "
                        "same-game multis by multiplying the legs."
                    ),
                )
            )
    slips.sort(key=lambda s: -s.log_growth)
    return slips[:limit]
