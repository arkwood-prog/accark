"""Dixon-Coles bivariate Poisson goal model.

Reference: Dixon & Coles (1997), "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market".

Goals are modelled as

    home goals ~ Poisson(lambda),  lambda = exp(attack_home - defence_away + gamma)
    away goals ~ Poisson(mu),      mu     = exp(attack_away - defence_home)

with a correction factor tau(x, y) applied to the four low-score cells
(0-0, 0-1, 1-0, 1-1) where independent Poissons are known to misprice, and an
exponential time-decay weight so that recent matches count for more.

Attack and defence ratings are mean-centred inside the likelihood (they are
otherwise unidentifiable) and shrunk toward zero with an L2 penalty, which
keeps newly-promoted and early-season teams from taking extreme values.

The analytic gradient matters: a walk-forward backtest refits the model
hundreds of times, and finite differences would make that unusable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from ..config import ModelConfig
from ..data.schema import Match
from .expected_goals import ShotConversion, blended_targets, fit_conversion


@dataclass
class TeamRating:
    team: str
    attack: float
    defence: float
    matches: int

    @property
    def net(self) -> float:
        return self.attack + self.defence


@dataclass
class FittedModel:
    """A fitted model, able to price any fixture between known teams."""

    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    home_advantage: float
    rho: float
    match_counts: dict[str, int]
    config: ModelConfig
    n_matches: int
    effective_sample: float
    log_likelihood: float
    converged: bool
    conversion: ShotConversion | None = None
    xg_weight: float = 0.0
    fitted_through: date | None = None
    _index: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {team: i for i, team in enumerate(self.teams)}

    # -- team access ----------------------------------------------------
    def knows(self, team: str) -> bool:
        return team in self._index

    def rating(self, team: str) -> TeamRating:
        if team in self._index:
            i = self._index[team]
            return TeamRating(team, float(self.attack[i]), float(self.defence[i]),
                              self.match_counts.get(team, 0))
        # Unknown team: assume a league-average side, and say so via matches=0.
        return TeamRating(team, 0.0, 0.0, 0)

    def ratings(self) -> list[TeamRating]:
        return sorted((self.rating(t) for t in self.teams), key=lambda r: -r.net)

    @property
    def effective_matches_per_team(self) -> float:
        """Time-weighted matches behind the average team's rating.

        The headline match count flatters an early-season fit: with a 180-day
        half-life most of that window is the off-season, so the ratings rest on
        far less than the raw number suggests. Below roughly 20 the ratings are
        stale last-season values and the model will disagree with the market
        loudly and wrongly.
        """
        n_teams = max(len(self.teams), 1)
        return 2.0 * self.effective_sample / n_teams

    @property
    def thin_sample(self) -> bool:
        return self.effective_matches_per_team < 20.0

    def data_confidence(self, home: str, away: str) -> float:
        """0-1 score for how much match data underpins this fixture."""
        need = max(1, self.config.min_matches_per_team)
        seen = [self.match_counts.get(home, 0), self.match_counts.get(away, 0)]
        return float(min(1.0, min(seen) / (need * 2.0)))

    # -- pricing --------------------------------------------------------
    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        home_r, away_r = self.rating(home), self.rating(away)
        lam = math.exp(home_r.attack - away_r.defence + self.home_advantage)
        mu = math.exp(away_r.attack - home_r.defence)
        return lam, mu

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        """P(home goals = i, away goals = j) as a (n+1, n+1) matrix."""
        lam, mu = self.expected_goals(home, away)
        return score_matrix_from_rates(lam, mu, self.rho, self.config.max_goals)

    def to_dict(self) -> dict:
        return {
            "home_advantage": round(self.home_advantage, 4),
            "rho": round(self.rho, 4),
            "n_matches": self.n_matches,
            "effective_sample": round(self.effective_sample, 1),
            "effective_matches_per_team": round(self.effective_matches_per_team, 1),
            "thin_sample": self.thin_sample,
            "log_likelihood": round(self.log_likelihood, 2),
            "converged": self.converged,
            "half_life_days": self.config.half_life_days,
            "xg_weight": self.xg_weight,
            "shot_conversion": self.conversion.to_dict() if self.conversion else None,
            "fitted_through": self.fitted_through.isoformat() if self.fitted_through else None,
            "teams": [
                {
                    "team": r.team,
                    "attack": round(r.attack, 3),
                    "defence": round(r.defence, 3),
                    "net": round(r.net, 3),
                    "matches": r.matches,
                }
                for r in self.ratings()
            ],
        }


def score_matrix_from_rates(lam: float, mu: float, rho: float, max_goals: int = 10) -> np.ndarray:
    """Dixon-Coles joint score distribution, truncated and renormalised."""
    goals = np.arange(max_goals + 1)
    log_fact = gammaln(goals + 1.0)
    home_pmf = np.exp(-lam + goals * np.log(max(lam, 1e-12)) - log_fact)
    away_pmf = np.exp(-mu + goals * np.log(max(mu, 1e-12)) - log_fact)
    matrix = np.outer(home_pmf, away_pmf)

    matrix[0, 0] *= max(1.0 - lam * mu * rho, 1e-9)
    matrix[0, 1] *= max(1.0 + lam * rho, 1e-9)
    matrix[1, 0] *= max(1.0 + mu * rho, 1e-9)
    matrix[1, 1] *= max(1.0 - rho, 1e-9)

    total = matrix.sum()
    return matrix / total if total > 0 else matrix


def _time_weights(match_dates: np.ndarray, as_of: date, half_life_days: float) -> np.ndarray:
    if half_life_days <= 0:
        return np.ones(match_dates.shape[0])
    days_ago = np.array([(as_of - d).days for d in match_dates], dtype=float)
    days_ago = np.maximum(days_ago, 0.0)
    return np.power(0.5, days_ago / half_life_days)


class DixonColesModel:
    """Fits :class:`FittedModel` from a list of completed matches."""

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()

    def fit(self, matches: Sequence[Match], as_of: date | None = None) -> FittedModel:
        cfg = self.config
        if not matches:
            raise ValueError("cannot fit a model with no matches")

        as_of = as_of or max(m.date for m in matches)
        usable = [m for m in matches if m.date <= as_of]
        if cfg.max_history_days > 0:
            cutoff_days = cfg.max_history_days
            usable = [m for m in usable if (as_of - m.date).days <= cutoff_days]
        if not usable:
            raise ValueError("no matches remain after applying the history window")

        teams = sorted({m.home for m in usable} | {m.away for m in usable})
        index = {team: i for i, team in enumerate(teams)}
        n_teams = len(teams)

        home_idx = np.array([index[m.home] for m in usable])
        away_idx = np.array([index[m.away] for m in usable])
        home_goals = np.array([m.home_goals for m in usable], dtype=float)
        away_goals = np.array([m.away_goals for m in usable], dtype=float)

        # Attacking target: goals, a shots-based expected-goals proxy, or a
        # blend. The tau correction below still keys off the *actual* scoreline,
        # because it describes the dependence of real low scores.
        xg_weight = float(min(max(cfg.xg_weight, 0.0), 1.0))
        conversion = fit_conversion(usable) if xg_weight > 0 else None
        if conversion is not None:
            home_target, away_target = blended_targets(usable, conversion, xg_weight)
        else:
            home_target, away_target = home_goals, away_goals
        weights = _time_weights(np.array([m.date for m in usable]), as_of, cfg.half_life_days)
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            raise ValueError("all matches have zero weight; widen half_life_days")

        counts: dict[str, int] = {team: 0 for team in teams}
        for m in usable:
            counts[m.home] += 1
            counts[m.away] += 1

        # Cells needing the tau correction.
        cell_00 = (home_goals == 0) & (away_goals == 0)
        cell_01 = (home_goals == 0) & (away_goals == 1)
        cell_10 = (home_goals == 1) & (away_goals == 0)
        cell_11 = (home_goals == 1) & (away_goals == 1)

        ridge = cfg.ridge
        rho_lo, rho_hi = cfg.rho_bounds

        def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
            raw_attack = theta[:n_teams]
            raw_defence = theta[n_teams:2 * n_teams]
            # Centring makes the parameters identifiable.
            return (raw_attack - raw_attack.mean(),
                    raw_defence - raw_defence.mean(),
                    float(theta[-2]),
                    float(theta[-1]))

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            attack, defence, gamma, rho = unpack(theta)

            log_lam = attack[home_idx] - defence[away_idx] + gamma
            log_mu = attack[away_idx] - defence[home_idx]
            lam = np.exp(np.clip(log_lam, -8, 4))
            mu = np.exp(np.clip(log_mu, -8, 4))

            tau = np.ones_like(lam)
            tau[cell_00] = 1.0 - lam[cell_00] * mu[cell_00] * rho
            tau[cell_01] = 1.0 + lam[cell_01] * rho
            tau[cell_10] = 1.0 + mu[cell_10] * rho
            tau[cell_11] = 1.0 - rho
            valid = tau > 1e-9
            tau_safe = np.where(valid, tau, 1e-9)

            log_lik_terms = (-lam + home_target * log_lam - mu + away_target * log_mu
                             + np.log(tau_safe))
            mean_log_lik = float((weights * log_lik_terms).sum() / weight_sum)
            penalty = ridge * float((attack**2).sum() + (defence**2).sum()) / n_teams

            # --- gradient ---
            d_tau_d_lam = np.zeros_like(lam)
            d_tau_d_mu = np.zeros_like(mu)
            d_tau_d_rho = np.zeros_like(lam)
            d_tau_d_lam[cell_00] = -mu[cell_00] * rho
            d_tau_d_lam[cell_01] = rho
            d_tau_d_mu[cell_00] = -lam[cell_00] * rho
            d_tau_d_mu[cell_10] = rho
            d_tau_d_rho[cell_00] = -lam[cell_00] * mu[cell_00]
            d_tau_d_rho[cell_01] = lam[cell_01]
            d_tau_d_rho[cell_10] = mu[cell_10]
            d_tau_d_rho[cell_11] = -1.0
            inv_tau = np.where(valid, 1.0 / tau_safe, 0.0)

            w = weights / weight_sum
            # d(loglik)/d(log lambda) = lambda * d/d lambda
            d_log_lam = w * (-lam + home_target + lam * inv_tau * d_tau_d_lam)
            d_log_mu = w * (-mu + away_target + mu * inv_tau * d_tau_d_mu)
            d_rho = float((w * inv_tau * d_tau_d_rho).sum())

            grad_attack = (np.bincount(home_idx, weights=d_log_lam, minlength=n_teams)
                           + np.bincount(away_idx, weights=d_log_mu, minlength=n_teams))
            grad_defence = (-np.bincount(away_idx, weights=d_log_lam, minlength=n_teams)
                            - np.bincount(home_idx, weights=d_log_mu, minlength=n_teams))
            grad_gamma = float(d_log_lam.sum())

            # Penalty gradient (wrt the centred values).
            grad_attack -= 2.0 * ridge * attack / n_teams
            grad_defence -= 2.0 * ridge * defence / n_teams

            # Chain through the mean-centring: dc_i/dr_j = delta_ij - 1/n.
            grad_attack -= grad_attack.mean()
            grad_defence -= grad_defence.mean()

            grad = np.concatenate([grad_attack, grad_defence, [grad_gamma], [d_rho]])
            # We minimise the negative.
            return -(mean_log_lik - penalty), -grad

        # Sensible starting point: everyone average, typical home edge, mild
        # negative low-score dependence.
        theta0 = np.concatenate([
            np.zeros(n_teams), np.zeros(n_teams), [0.25], [-0.05],
        ])
        bounds = ([(-3.0, 3.0)] * n_teams + [(-3.0, 3.0)] * n_teams
                  + [(-1.5, 1.5), (rho_lo, rho_hi)])

        result = minimize(objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7})

        attack, defence, gamma, rho = unpack(result.x)
        # Report the true log-likelihood (with factorial terms) for diagnostics.
        log_lik = -result.fun * weight_sum - float(
            (weights * (gammaln(home_target + 1) + gammaln(away_target + 1))).sum()
        )

        return FittedModel(
            teams=teams,
            attack=attack,
            defence=defence,
            home_advantage=gamma,
            rho=rho,
            match_counts=counts,
            config=cfg,
            n_matches=len(usable),
            effective_sample=weight_sum,
            log_likelihood=log_lik,
            converged=bool(result.success),
            conversion=conversion,
            xg_weight=xg_weight,
            fitted_through=as_of,
        )
