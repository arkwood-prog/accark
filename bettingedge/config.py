"""Tunable parameters for the whole pipeline.

Every number a trader would want to argue about lives here rather than being
buried in the modelling code. Defaults are deliberately conservative: the
betting market is close to efficient, so the model is used as a *tilt* on the
market price, not as a replacement for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Dixon-Coles fitting options."""

    # Matches lose half their weight in the likelihood after this many days.
    # ~180d keeps roughly a season and a half of signal with recent form
    # dominating. Shorter reacts faster but overfits streaks.
    half_life_days: float = 180.0

    # Score matrix truncation. 10 covers >99.99% of realistic football scores.
    max_goals: int = 10

    # L2 shrinkage of attack/defence ratings toward the league mean. Keeps
    # small-sample teams (newly promoted, early season) from being extreme.
    ridge: float = 0.02

    # Teams with fewer matches than this are flagged as low-confidence.
    min_matches_per_team: int = 6

    # Dixon-Coles low-score dependence parameter is bounded for stability.
    rho_bounds: tuple[float, float] = (-0.35, 0.35)

    # Ignore matches older than this when fitting (0 = no limit).
    max_history_days: int = 900


@dataclass(frozen=True)
class MarketConfig:
    """How bookmaker prices are converted to probabilities."""

    # 'shin' handles favourite-longshot bias best on 3-way markets;
    # 'power' is a good default on 2-way; 'multiplicative' is the naive one.
    devig_three_way: str = "shin"
    devig_two_way: str = "power"

    # Weight on the model when blending with the market's fair price, in
    # logit space. 0.0 = pure market, 1.0 = pure model.
    # 0.35 reflects the reality that closing prices are hard to beat.
    model_weight: float = 0.35

    # Books whose margin exceeds this are treated as untrustworthy for
    # fair-price estimation (still usable as a place to take a price).
    max_trusted_overround: float = 0.12


@dataclass(frozen=True)
class SelectionConfig:
    """Which single bets are allowed to become recommendations."""

    # Minimum expected value per unit staked, e.g. 0.03 = +3%.
    min_edge: float = 0.03

    # Guard rails on price. Very short prices tie up bankroll for little
    # return; very long prices are where model error is largest.
    min_probability: float = 0.10
    max_probability: float = 0.92
    min_odds: float = 1.20
    max_odds: float = 8.00

    # Cap on how many singles are surfaced per matchday.
    max_singles: int = 25

    # Refuse to price a fixture when the model has never seen one of the teams.
    # Without this an unrecognised name is silently treated as a league-average
    # side and the fixture still produces confident-looking bets — the single
    # most dangerous failure mode when fixtures come from a different source
    # than the results the model was fitted on.
    require_known_teams: bool = True

    # Markets the engine is allowed to quote.
    markets: tuple[str, ...] = ("1X2", "DC", "OU", "BTTS")

    # Over/Under lines to price.
    ou_lines: tuple[float, ...] = (1.5, 2.5, 3.5)


@dataclass(frozen=True)
class ParlayConfig:
    """Multi-leg (double / treble / accumulator) construction rules."""

    max_legs: int = 5
    max_per_size: int = 3            # how many combos to surface per leg-count
    min_leg_edge: float = 0.03       # each leg must clear this on its own
    min_leg_probability: float = 0.40
    min_combined_probability: float = 0.03
    min_combined_edge: float = 0.05  # multis must clear a higher bar than singles
    max_combined_odds: float = 60.0

    # Legs from the same fixture are correlated. When True the engine prices
    # them from the joint score distribution instead of multiplying, and only
    # then only for explicitly requested same-game slips.
    allow_same_game: bool = True

    # Candidate legs considered when searching combinations (keeps the
    # combinatorics tractable and the quality high).
    leg_pool_size: int = 14


@dataclass(frozen=True)
class StakingConfig:
    """Bankroll management."""

    bankroll: float = 1000.0

    # Fractional Kelly. Full Kelly is theoretically growth-optimal but assumes
    # your probabilities are exact; they are not. Quarter Kelly is the standard
    # professional compromise.
    kelly_fraction: float = 0.25

    # Hard cap on any single stake, as a fraction of bankroll.
    max_stake_pct: float = 0.02
    max_parlay_stake_pct: float = 0.005

    # Total amount at risk across the whole slate, as a fraction of bankroll.
    # Kelly assumes bets settle one at a time; a full matchday does not, so
    # simultaneous exposure is scaled back to this ceiling.
    max_total_exposure: float = 0.15

    min_stake: float = 1.0
    round_to: float = 0.5


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    parlay: ParlayConfig = field(default_factory=ParlayConfig)
    staking: StakingConfig = field(default_factory=StakingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Build a Config from a (possibly partial) nested dict."""
        sections = {
            "model": ModelConfig,
            "market": MarketConfig,
            "selection": SelectionConfig,
            "parlay": ParlayConfig,
            "staking": StakingConfig,
        }
        kwargs: dict[str, Any] = {}
        for name, klass in sections.items():
            payload = dict(data.get(name) or {})
            valid = {f for f in klass.__dataclass_fields__}
            filtered = {k: v for k, v in payload.items() if k in valid}
            # tuple fields arrive from JSON as lists
            for key, value in list(filtered.items()):
                declared = klass.__dataclass_fields__[key].type
                if isinstance(value, list) and "tuple" in str(declared):
                    filtered[key] = tuple(value)
            kwargs[name] = klass(**filtered)
        return cls(**kwargs)


DEFAULT_CONFIG = Config()
