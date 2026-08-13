"""An expected-goals proxy built from shot counts.

Goals are a noisy measure of how a team played. A side that creates twelve
chances and scores once played well and got unlucky, but the scoreline records
only the luck. Ratings fitted on goals therefore chase variance.

Shots on target are far less noisy — there are roughly ten times as many of
them per match as goals — so a rate model over shots gives a steadier estimate
of underlying strength:

    expected goals = alpha * (shots on target) + beta * (shots off target)

Alpha and beta are fitted per league from its own history rather than assumed,
because conversion rates differ by division: lower leagues take worse shots.
Off-target shots carry a small weight because shot volume still says something
about territory even when the shot itself was never going in.

This is not the shot-location xG a commercial provider sells; without shot
coordinates it cannot be. It is the best available proxy from data that comes
free with the results, and it is a clear improvement on goals alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..data.schema import Match

# Sensible fallbacks if a league has too little shot data to fit its own.
DEFAULT_ALPHA = 0.30    # a shot on target is worth about a third of a goal
DEFAULT_BETA = 0.02


@dataclass
class ShotConversion:
    """Fitted conversion rates for one league."""

    alpha: float           # goals per shot on target
    beta: float            # goals per off-target shot
    matches_used: int
    fitted: bool

    def expected_goals(self, shots_on_target: int | None,
                       shots: int | None) -> float | None:
        if shots_on_target is None:
            return None
        off_target = max((shots or shots_on_target) - shots_on_target, 0)
        return self.alpha * shots_on_target + self.beta * off_target

    def to_dict(self) -> dict:
        return {
            "goals_per_shot_on_target": round(self.alpha, 4),
            "goals_per_off_target_shot": round(self.beta, 4),
            "matches_used": self.matches_used,
            "fitted": self.fitted,
        }


def fit_conversion(matches: Sequence[Match], min_matches: int = 60) -> ShotConversion:
    """Least-squares fit of goals on shot counts, pooled over both teams."""
    on_target: list[float] = []
    off_target: list[float] = []
    goals: list[float] = []

    for match in matches:
        if not match.has_shot_data:
            continue
        for sot, shots, scored in (
            (match.home_shots_on_target, match.home_shots, match.home_goals),
            (match.away_shots_on_target, match.away_shots, match.away_goals),
        ):
            if sot is None:
                continue
            on_target.append(float(sot))
            off_target.append(float(max((shots or sot) - sot, 0)))
            goals.append(float(scored))

    if len(goals) < min_matches:
        return ShotConversion(DEFAULT_ALPHA, DEFAULT_BETA, len(goals), fitted=False)

    design = np.column_stack([on_target, off_target])
    # No intercept: a team with no shots is expected to score no goals.
    solution, *_ = np.linalg.lstsq(design, np.array(goals), rcond=None)
    alpha, beta = (float(solution[0]), float(solution[1]))

    # Guard against a degenerate fit on thin or strange data.
    if not (0.05 <= alpha <= 1.0):
        return ShotConversion(DEFAULT_ALPHA, DEFAULT_BETA, len(goals), fitted=False)
    beta = min(max(beta, 0.0), 0.2)
    return ShotConversion(alpha, beta, len(goals), fitted=True)


def blended_targets(
    matches: Sequence[Match],
    conversion: ShotConversion,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-match attacking targets: goals, expected goals, or a blend.

    ``weight`` is how far to move from goals toward expected goals. Matches
    with no shot data keep their actual goals, so a partial feed degrades
    gracefully instead of dropping matches.

    The result is continuous, which the Dixon-Coles likelihood accepts as a
    quasi-Poisson target: the score equations stay unbiased for non-integer
    values, and the factorial term is constant with respect to the parameters.
    """
    home = np.empty(len(matches))
    away = np.empty(len(matches))

    for index, match in enumerate(matches):
        home_goals = float(match.home_goals)
        away_goals = float(match.away_goals)
        if weight <= 0 or not match.has_shot_data:
            home[index], away[index] = home_goals, away_goals
            continue
        home_xg = conversion.expected_goals(match.home_shots_on_target, match.home_shots)
        away_xg = conversion.expected_goals(match.away_shots_on_target, match.away_shots)
        if home_xg is None or away_xg is None:
            home[index], away[index] = home_goals, away_goals
            continue
        home[index] = (1 - weight) * home_goals + weight * home_xg
        away[index] = (1 - weight) * away_goals + weight * away_xg
    return home, away
