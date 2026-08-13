"""Bankroll management.

Kelly gives the growth-optimal stake *if your probability is exact*. It never
is, so three corrections are applied, all of them standard practice:

1. **Fractional Kelly** (default a quarter). Halving the fraction roughly
   quarters the variance while only costing a quarter of the growth rate, and
   it protects against the probabilities being a little wrong.
2. **Per-bet caps**, tighter for multis than singles, because a multi's
   probability is a product of estimates and its error compounds too.
3. **Simultaneous-bet scaling.** Kelly assumes bets settle sequentially. A
   Saturday slate does not, so if the whole book of bets exceeds the exposure
   ceiling every stake is scaled back proportionally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..config import StakingConfig
from .parlays import Slip


@dataclass
class Portfolio:
    """Summary of everything staked on a slate."""

    bankroll: float
    total_staked: float
    expected_profit: float
    exposure_pct: float
    bet_count: int
    scaled_by: float

    def to_dict(self) -> dict:
        return {
            "bankroll": round(self.bankroll, 2),
            "total_staked": round(self.total_staked, 2),
            "expected_profit": round(self.expected_profit, 2),
            "expected_roi_pct": round(
                100 * self.expected_profit / self.total_staked, 2
            ) if self.total_staked > 0 else 0.0,
            "exposure_pct": round(self.exposure_pct, 2),
            "bet_count": self.bet_count,
            "kelly_scaled_by": round(self.scaled_by, 3),
        }


def _round_stake(amount: float, step: float) -> float:
    if step <= 0:
        return round(amount, 2)
    return round(round(amount / step) * step, 2)


def assign_stakes(slips: Sequence[Slip], config: StakingConfig) -> Portfolio:
    """Set `slip.stake` on every slip and return the portfolio summary.

    Mutates the slips in place — they are the objects the report renders.
    """
    if not slips:
        return Portfolio(config.bankroll, 0.0, 0.0, 0.0, 0, 1.0)

    fractions: list[float] = []
    for slip in slips:
        cap = config.max_stake_pct if slip.size == 1 else config.max_parlay_stake_pct
        fractions.append(min(slip.kelly * config.kelly_fraction, cap))

    total_fraction = sum(fractions)
    scale = 1.0
    if total_fraction > config.max_total_exposure > 0:
        scale = config.max_total_exposure / total_fraction
        fractions = [f * scale for f in fractions]

    total_staked = 0.0
    expected_profit = 0.0
    bet_count = 0
    for slip, fraction in zip(slips, fractions):
        raw = config.bankroll * fraction
        stake = _round_stake(raw, config.round_to)
        if raw <= 0:
            stake = 0.0
        elif stake < config.min_stake:
            # Too small to be worth placing at the rounding granularity, but the
            # edge is real — round it up to the minimum rather than drop it.
            stake = config.min_stake
        slip.stake = stake
        if stake > 0:
            total_staked += stake
            expected_profit += stake * slip.edge
            bet_count += 1

    return Portfolio(
        bankroll=config.bankroll,
        total_staked=total_staked,
        expected_profit=expected_profit,
        exposure_pct=100.0 * total_staked / config.bankroll if config.bankroll else 0.0,
        bet_count=bet_count,
        scaled_by=scale,
    )
