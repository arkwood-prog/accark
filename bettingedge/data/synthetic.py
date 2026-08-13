"""A self-contained synthetic league, for offline demos and for tests.

THIS IS NOT REAL DATA. It is generated from a known Dixon-Coles process so
that (a) the app runs with zero network access and (b) tests can check the
estimator recovers parameters it was never told.

Simulated bookmakers price off the *true* probabilities with a small amount
of noise plus a realistic margin. That means the demo backtest measures
whether the machinery works, not whether the strategy prints money — a model
cannot reliably beat a book that knows the truth. Real edges come from real
data, where nobody knows the truth.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np

from .schema import Fixture, Match

TEAM_NAMES = [
    "Ashford United", "Brackenfield", "Carrow Rovers", "Dunmore City",
    "Eastvale Athletic", "Fenwick Town", "Granby Wanderers", "Harlow Albion",
    "Ilkeston FC", "Jarrow Park", "Kingsmere", "Lowfield Rangers",
    "Marlow Sporting", "Netherton", "Oakhurst FC", "Pendle County",
    "Quarryhill", "Redmoor Athletic", "Southgate Vale", "Thornbury",
]

TRUE_HOME_ADV = 0.26
TRUE_RHO = -0.06
LEAGUE_CODE = "SYN"


def _tau(home_goals: int, away_goals: int, lam: float, mu: float, rho: float) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lam * mu * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lam * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + mu * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def true_score_matrix(lam: float, mu: float, rho: float, max_goals: int = 10) -> np.ndarray:
    goals = np.arange(max_goals + 1)
    home_pmf = np.exp(-lam) * lam**goals / np.array([math.factorial(g) for g in goals])
    away_pmf = np.exp(-mu) * mu**goals / np.array([math.factorial(g) for g in goals])
    matrix = np.outer(home_pmf, away_pmf)
    for i in (0, 1):
        for j in (0, 1):
            matrix[i, j] *= _tau(i, j, lam, mu, rho)
    return matrix / matrix.sum()


def _round_robin(n_teams: int, rng: np.random.Generator) -> list[list[tuple[int, int]]]:
    """Berger-style double round robin: 2*(n-1) rounds of n/2 matches."""
    teams = list(range(n_teams))
    rounds: list[list[tuple[int, int]]] = []
    for r in range(n_teams - 1):
        pairs = []
        for i in range(n_teams // 2):
            home, away = teams[i], teams[n_teams - 1 - i]
            if (r + i) % 2 == 0:
                pairs.append((home, away))
            else:
                pairs.append((away, home))
        rounds.append(pairs)
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    # Reverse fixtures for the second half of the season.
    rounds += [[(a, h) for h, a in rnd] for rnd in rounds]
    rng.shuffle(rounds)
    return rounds


def _simulated_book_prices(
    true_probs: dict[str, float],
    groups: list[tuple[str, ...]],
    rng: np.random.Generator,
    n_books: int = 8,
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    """Turn true probabilities into a realistic set of bookmaker prices.

    Each book perturbs the truth in logit space and applies a margin. The
    sharp book has the least noise and the thinnest margin; `best` is the
    highest price any book offers, which is what a real bettor would take.
    """
    best: dict[str, float] = {}
    sharp: dict[str, float] = {}
    counts: dict[str, int] = {}

    for group in groups:
        probs = np.array([true_probs[sel] for sel in group])
        prices_by_book = []
        for book in range(n_books):
            is_sharp = book == 0
            noise_sd = 0.035 if is_sharp else float(rng.uniform(0.06, 0.14))
            margin = 0.022 if is_sharp else float(rng.uniform(0.04, 0.085))
            logits = np.log(probs / (1 - probs)) + rng.normal(0, noise_sd, size=probs.size)
            noisy = 1 / (1 + np.exp(-logits))
            noisy = noisy / noisy.sum()
            prices_by_book.append(1.0 / (noisy * (1 + margin)))
        stacked = np.array(prices_by_book)
        for idx, sel in enumerate(group):
            best[sel] = round(float(stacked[:, idx].max()), 3)
            sharp[sel] = round(float(stacked[0, idx]), 3)
            counts[sel] = n_books
    return best, sharp, counts


def _market_probs(matrix: np.ndarray) -> dict[str, float]:
    n = matrix.shape[0]
    idx_h = np.arange(n)[:, None]
    idx_a = np.arange(n)[None, :]
    total = idx_h + idx_a
    return {
        "1X2:H": float(matrix[idx_h > idx_a].sum()),
        "1X2:D": float(np.trace(matrix)),
        "1X2:A": float(matrix[idx_h < idx_a].sum()),
        "OU2.5:O": float(matrix[total > 2.5].sum()),
        "OU2.5:U": float(matrix[total < 2.5].sum()),
    }


def generate(
    seasons: int = 4,
    seed: int = 7,
    n_teams: int = 20,
    end_date: date | None = None,
    upcoming_matchdays: int = 1,
) -> tuple[list[Match], list[Fixture]]:
    """Generate `seasons` of completed matches plus a slate of fixtures."""
    rng = np.random.default_rng(seed)
    teams = TEAM_NAMES[:n_teams]

    attack = rng.normal(0.0, 0.30, size=n_teams)
    defence = rng.normal(0.0, 0.24, size=n_teams)
    attack -= attack.mean()
    defence -= defence.mean()

    end_date = end_date or date.today()
    rounds_per_season = 2 * (n_teams - 1)
    total_rounds = seasons * rounds_per_season + upcoming_matchdays
    # One matchday every 4 days, ending with the upcoming slate a week out.
    start = end_date - timedelta(days=4 * total_rounds - 7)

    matches: list[Match] = []
    fixtures: list[Fixture] = []
    round_index = 0

    for season in range(seasons + 1):
        schedule = _round_robin(n_teams, rng)
        for pairs in schedule:
            if round_index >= total_rounds:
                break
            match_date = start + timedelta(days=4 * round_index)
            is_upcoming = round_index >= total_rounds - upcoming_matchdays

            # Ratings drift slowly: form, transfers, managers.
            attack += rng.normal(0, 0.018, size=n_teams)
            defence += rng.normal(0, 0.018, size=n_teams)
            attack -= attack.mean()
            defence -= defence.mean()

            for home_idx, away_idx in pairs:
                lam = math.exp(attack[home_idx] - defence[away_idx] + TRUE_HOME_ADV)
                mu = math.exp(attack[away_idx] - defence[home_idx])
                matrix = true_score_matrix(lam, mu, TRUE_RHO)
                probs = _market_probs(matrix)
                best, sharp, counts = _simulated_book_prices(
                    probs, [("1X2:H", "1X2:D", "1X2:A"), ("OU2.5:O", "OU2.5:U")], rng
                )

                if is_upcoming:
                    fixtures.append(
                        Fixture(
                            date=match_date,
                            league=LEAGUE_CODE,
                            home=teams[home_idx],
                            away=teams[away_idx],
                            kickoff="15:00",
                            odds=best,
                            sharp_odds=sharp,
                            book_counts=counts,
                        )
                    )
                else:
                    flat = matrix.ravel()
                    draw = rng.choice(flat.size, p=flat / flat.sum())
                    home_goals, away_goals = divmod(int(draw), matrix.shape[1])
                    matches.append(
                        Match(
                            date=match_date,
                            league=LEAGUE_CODE,
                            home=teams[home_idx],
                            away=teams[away_idx],
                            home_goals=home_goals,
                            away_goals=away_goals,
                            closing_odds=sharp,
                            best_odds=best,
                        )
                    )
            round_index += 1
        if round_index >= total_rounds:
            break

    matches.sort(key=lambda m: m.date)
    fixtures.sort(key=lambda f: (f.date, f.home))
    return matches, fixtures
