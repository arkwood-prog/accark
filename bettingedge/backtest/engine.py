"""Walk-forward backtesting.

The rules that make a backtest honest, all enforced here:

* The model is refitted using **only matches that had already been played**
  at the time of each simulated bet. No future data touches a past decision.
* Bets are settled at the **best price that was actually on offer** for that
  match, taken from the same historical file as the result.
* Performance is reported as flat-stake yield *and* Kelly bankroll growth,
  because the two answer different questions.
* The model's probability forecasts are scored against the market's own
  margin-free forecasts. If the model cannot beat the closing line on log
  loss, no amount of positive backtest ROI should be believed — it is noise.
* Closing-line value is tracked separately, because beating the closing price
  is the single most reliable predictor of long-run profitability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import Sequence

import numpy as np

from ..betting.parlays import Slip, expected_log_growth
from ..betting.staking import assign_stakes
from ..betting.value import blend_market_set, build_candidates, kelly_fraction
from ..config import Config
from ..data.schema import Fixture, Match
from ..market.consensus import market_view
from ..market.devig import devig
from ..models.dixon_coles import DixonColesModel
from ..models.markets import PARTITIONS, price_markets, selection_mask


def settles(selection_id: str, home_goals: int, away_goals: int) -> bool:
    """Did this selection win, given the final score?"""
    size = max(home_goals, away_goals) + 2
    return bool(selection_mask(selection_id, size)[home_goals, away_goals])


def _fixture_from_match(match: Match) -> Fixture:
    return Fixture(
        date=match.date,
        league=match.league,
        home=match.home,
        away=match.away,
        odds=dict(match.best_odds),
        sharp_odds=dict(match.closing_odds),
        book_counts={sel: 6 for sel in match.best_odds},
    )


@dataclass
class PlacedBet:
    date: date
    match: str
    selection: str
    label: str
    odds: float
    probability: float
    market_probability: float
    edge: float
    stake: float
    won: bool
    profit: float
    closing_value: float | None

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "match": self.match,
            "selection": self.selection,
            "label": self.label,
            "odds": round(self.odds, 3),
            "probability": round(self.probability, 4),
            "edge_pct": round(self.edge * 100, 2),
            "stake": round(self.stake, 2),
            "won": self.won,
            "profit": round(self.profit, 2),
            "clv_pct": round(self.closing_value * 100, 2) if self.closing_value is not None else None,
        }


@dataclass
class BacktestResult:
    bets: list[PlacedBet]
    bankroll_curve: list[tuple[date, float]]
    starting_bankroll: float
    final_bankroll: float
    max_drawdown: float
    model_log_loss: float
    market_log_loss: float
    model_brier: float
    market_brier: float
    forecasts: int
    calibration: list[dict] = field(default_factory=list)
    refits: int = 0

    @property
    def turnover(self) -> float:
        return sum(bet.stake for bet in self.bets)

    @property
    def profit(self) -> float:
        return sum(bet.profit for bet in self.bets)

    @property
    def roi(self) -> float:
        """Return on money staked (Kelly-sized)."""
        return self.profit / self.turnover if self.turnover else 0.0

    @property
    def flat_yield(self) -> float:
        """Profit per unit staked if every bet were the same size."""
        if not self.bets:
            return 0.0
        units = sum((bet.odds - 1.0) if bet.won else -1.0 for bet in self.bets)
        return units / len(self.bets)

    @property
    def hit_rate(self) -> float:
        return sum(1 for bet in self.bets if bet.won) / len(self.bets) if self.bets else 0.0

    @property
    def average_odds(self) -> float:
        return sum(bet.odds for bet in self.bets) / len(self.bets) if self.bets else 0.0

    @property
    def average_clv(self) -> float:
        values = [bet.closing_value for bet in self.bets if bet.closing_value is not None]
        return sum(values) / len(values) if values else 0.0

    @property
    def beat_closing_rate(self) -> float:
        values = [bet.closing_value for bet in self.bets if bet.closing_value is not None]
        return sum(1 for v in values if v > 0) / len(values) if values else 0.0

    def to_dict(self) -> dict:
        return {
            "bets": len(self.bets),
            "turnover": round(self.turnover, 2),
            "profit": round(self.profit, 2),
            "roi_pct": round(self.roi * 100, 2),
            "flat_yield_pct": round(self.flat_yield * 100, 2),
            "hit_rate_pct": round(self.hit_rate * 100, 2),
            "average_odds": round(self.average_odds, 2),
            "starting_bankroll": round(self.starting_bankroll, 2),
            "final_bankroll": round(self.final_bankroll, 2),
            "growth_pct": round(
                100 * (self.final_bankroll / self.starting_bankroll - 1), 2
            ) if self.starting_bankroll else 0.0,
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "average_clv_pct": round(self.average_clv * 100, 2),
            "beat_closing_rate_pct": round(self.beat_closing_rate * 100, 2),
            "forecasts": self.forecasts,
            "model_log_loss": round(self.model_log_loss, 4),
            "market_log_loss": round(self.market_log_loss, 4),
            "log_loss_edge": round(self.market_log_loss - self.model_log_loss, 4),
            "model_brier": round(self.model_brier, 4),
            "market_brier": round(self.market_brier, 4),
            "refits": self.refits,
            "calibration": self.calibration,
            "bankroll_curve": [[d.isoformat(), round(v, 2)] for d, v in self.bankroll_curve],
            "sample_bets": [bet.to_dict() for bet in self.bets[-40:]],
        }

    def summary(self) -> str:
        lines = [
            f"Bets placed         : {len(self.bets)}",
            f"Turnover            : {self.turnover:,.2f}",
            f"Profit              : {self.profit:,.2f}",
            f"ROI (staked)        : {self.roi * 100:+.2f}%",
            f"Flat-stake yield    : {self.flat_yield * 100:+.2f}%",
            f"Hit rate            : {self.hit_rate * 100:.1f}% at average odds {self.average_odds:.2f}",
            f"Bankroll            : {self.starting_bankroll:,.2f} -> {self.final_bankroll:,.2f}",
            f"Max drawdown        : {self.max_drawdown * 100:.1f}%",
            f"Closing-line value  : {self.average_clv * 100:+.2f}% (beat close on "
            f"{self.beat_closing_rate * 100:.1f}% of bets)",
            "",
            f"Forecast quality over {self.forecasts} matches (1X2):",
            f"  model log loss    : {self.model_log_loss:.4f}",
            f"  market log loss   : {self.market_log_loss:.4f}",
            f"  model advantage   : {self.market_log_loss - self.model_log_loss:+.4f} "
            f"({'model adds information' if self.model_log_loss < self.market_log_loss else 'market is sharper'})",
            f"  model Brier       : {self.model_brier:.4f}   market Brier: {self.market_brier:.4f}",
        ]
        return "\n".join(lines)


def _calibration_table(records: list[tuple[float, bool]], bins: int = 10) -> list[dict]:
    """Predicted vs realised frequency, the acid test of a probability model."""
    table: list[dict] = []
    if not records:
        return table
    edges = np.linspace(0.0, 1.0, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        bucket = [(p, hit) for p, hit in records if low <= p < high]
        if not bucket:
            continue
        table.append({
            "bucket": f"{low:.0%}-{high:.0%}",
            "n": len(bucket),
            "predicted": round(sum(p for p, _ in bucket) / len(bucket), 4),
            "actual": round(sum(1 for _, hit in bucket if hit) / len(bucket), 4),
        })
    return table


def run_backtest(
    matches: Sequence[Match],
    config: Config | None = None,
    train_days: int = 400,
    refit_every_days: int = 7,
    start: date | None = None,
    end: date | None = None,
    progress: bool = False,
) -> BacktestResult:
    """Simulate the strategy forward through history, one matchday at a time."""
    config = config or Config()
    ordered = sorted(matches, key=lambda m: m.date)
    if not ordered:
        raise ValueError("no matches to backtest")

    first_date = ordered[0].date
    start = start or (first_date + timedelta(days=train_days))
    end = end or ordered[-1].date

    matchdays = sorted({m.date for m in ordered if start <= m.date <= end})
    if not matchdays:
        raise ValueError(
            "no matchdays in the backtest window — try a longer history or a smaller "
            "train_days"
        )

    bankroll = config.staking.bankroll
    peak = bankroll
    max_drawdown = 0.0
    curve: list[tuple[date, float]] = [(matchdays[0], bankroll)]

    bets: list[PlacedBet] = []
    model_losses: list[float] = []
    market_losses: list[float] = []
    model_briers: list[float] = []
    market_briers: list[float] = []
    calibration_records: list[tuple[float, bool]] = []

    fitter = DixonColesModel(config.model)
    model = None
    last_fit: date | None = None
    refits = 0

    for day in matchdays:
        history = [m for m in ordered if m.date < day]
        if len(history) < 50:
            continue
        if model is None or last_fit is None or (day - last_fit).days >= refit_every_days:
            model = fitter.fit(history, as_of=day)
            last_fit = day
            refits += 1
            if progress:
                print(f"  fitted through {day} ({len(history)} matches)")

        todays = [m for m in ordered if m.date == day]
        # Stakes compound: size today's bets off the bankroll as it stands now,
        # not the amount we started the backtest with.
        day_staking = replace(config.staking, bankroll=bankroll)

        day_slips = []
        day_meta = []
        for match in todays:
            if not match.best_odds:
                continue
            fixture = _fixture_from_match(match)
            matrix = model.score_matrix(match.home, match.away)
            model_probs = price_markets(matrix, config.selection.markets,
                                        config.selection.ou_lines)
            quotes = market_view(fixture, config.market, config.selection.ou_lines)
            if not quotes:
                continue
            blended = blend_market_set(model_probs, quotes, config.market.model_weight,
                                       config.selection.ou_lines)

            _score_forecast(match, model_probs, quotes, model_losses, market_losses,
                            model_briers, market_briers)

            candidates = build_candidates(
                fixture=fixture,
                blended_probs=blended,
                model_probs=model_probs,
                quotes=quotes,
                config=config,
                data_confidence=model.data_confidence(match.home, match.away),
            )
            for candidate in candidates:
                kelly = kelly_fraction(candidate.probability, candidate.odds)
                day_slips.append(
                    Slip(
                        kind="Single",
                        legs=[candidate],
                        probability=candidate.probability,
                        odds=candidate.odds,
                        edge=candidate.edge,
                        kelly=kelly,
                        log_growth=expected_log_growth(
                            candidate.probability, candidate.odds,
                            min(kelly * config.staking.kelly_fraction,
                                config.staking.max_stake_pct),
                        ),
                    )
                )
                day_meta.append((match, quotes))

        if not day_slips:
            curve.append((day, bankroll))
            continue

        assign_stakes(day_slips, day_staking)

        day_profit = 0.0
        for slip, (match, quotes) in zip(day_slips, day_meta):
            if slip.stake <= 0:
                continue
            candidate = slip.legs[0]
            won = settles(candidate.selection_id, match.home_goals, match.away_goals)
            profit = slip.stake * (candidate.odds - 1.0) if won else -slip.stake
            day_profit += profit
            bets.append(
                PlacedBet(
                    date=day,
                    match=f"{match.home} v {match.away}",
                    selection=candidate.selection_id,
                    label=candidate.label,
                    odds=candidate.odds,
                    probability=candidate.probability,
                    market_probability=candidate.market_probability,
                    edge=candidate.edge,
                    stake=slip.stake,
                    won=won,
                    profit=profit,
                    closing_value=_closing_line_value(candidate.odds, candidate.selection_id,
                                                      match, config),
                )
            )
            calibration_records.append((candidate.probability, won))

        bankroll += day_profit
        peak = max(peak, bankroll)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - bankroll) / peak)
        curve.append((day, bankroll))
        if bankroll <= 0:
            break

    return BacktestResult(
        bets=bets,
        bankroll_curve=curve,
        starting_bankroll=config.staking.bankroll,
        final_bankroll=bankroll,
        max_drawdown=max_drawdown,
        model_log_loss=float(np.mean(model_losses)) if model_losses else float("nan"),
        market_log_loss=float(np.mean(market_losses)) if market_losses else float("nan"),
        model_brier=float(np.mean(model_briers)) if model_briers else float("nan"),
        market_brier=float(np.mean(market_briers)) if market_briers else float("nan"),
        forecasts=len(model_losses),
        calibration=_calibration_table(calibration_records),
        refits=refits,
    )


def _score_forecast(
    match: Match,
    model_probs: dict[str, float],
    quotes: dict,
    model_losses: list[float],
    market_losses: list[float],
    model_briers: list[float],
    market_briers: list[float],
) -> None:
    """Score the 1X2 forecast against the result, model versus market."""
    group = PARTITIONS["1X2"]
    if not all(sel in quotes for sel in group):
        return
    if not all(sel in model_probs for sel in group):
        return
    outcome = {"H": "1X2:H", "D": "1X2:D", "A": "1X2:A"}[match.result]

    model_vector = np.array([model_probs[sel] for sel in group], dtype=float)
    market_vector = np.array([quotes[sel].fair_probability for sel in group], dtype=float)
    model_vector /= model_vector.sum()
    market_vector /= market_vector.sum()
    actual = np.array([1.0 if sel == outcome else 0.0 for sel in group])

    model_losses.append(-math.log(max(float(model_vector[actual == 1][0]), 1e-12)))
    market_losses.append(-math.log(max(float(market_vector[actual == 1][0]), 1e-12)))
    model_briers.append(float(((model_vector - actual) ** 2).sum()))
    market_briers.append(float(((market_vector - actual) ** 2).sum()))


def _closing_line_value(
    taken_odds: float,
    selection_id: str,
    match: Match,
    config: Config,
) -> float | None:
    """How the price taken compares with the margin-free closing price.

    Positive CLV means you got a better price than the market's final,
    sharpest opinion. Over a large sample this predicts profitability far
    more reliably than the ROI of the same sample does.
    """
    groups = [PARTITIONS["1X2"], PARTITIONS["BTTS"], ("OU2.5:O", "OU2.5:U")]
    for group in groups:
        if selection_id not in group:
            continue
        prices = [match.closing_odds.get(sel) for sel in group]
        if not all(prices):
            return None
        method = (config.market.devig_three_way if len(group) >= 3
                  else config.market.devig_two_way)
        try:
            fair = devig([float(p) for p in prices], method)
        except ValueError:
            return None
        probability = fair[group.index(selection_id)]
        return taken_odds * probability - 1.0
    return None
