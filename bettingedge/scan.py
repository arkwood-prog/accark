"""Compare leagues.

The Premier League is the most heavily traded football market on earth. If a
model is going to find anything, it is far more likely to be somewhere the
books price with less care and lower limits. This runs the same walk-forward
test across several divisions and puts the results side by side, so the choice
of where to play is made from numbers rather than familiarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .backtest.engine import BacktestResult, run_backtest
from .config import Config
from .data.footballdata import LEAGUES, FootballDataUK, recent_seasons
from .verify import sharp_only

DEFAULT_LEAGUES = ("E0", "E1", "E2", "E3", "EC", "SC0", "D1", "I1", "SP1", "F1", "N1")


@dataclass
class LeagueResult:
    league: str
    name: str
    matches: int
    error: str | None = None
    optimistic: BacktestResult | None = None
    pessimistic: BacktestResult | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.optimistic is not None

    def to_dict(self) -> dict:
        if not self.ok:
            return {"league": self.league, "name": self.name, "error": self.error}
        best, sharp = self.optimistic, self.pessimistic
        return {
            "league": self.league,
            "name": self.name,
            "matches": self.matches,
            "model_log_loss": round(best.model_log_loss, 4),
            "market_log_loss": round(best.market_log_loss, 4),
            "log_loss_gap": round(best.market_log_loss - best.model_log_loss, 4),
            "bets": len(best.bets),
            "flat_yield_pct": round(best.flat_yield * 100, 2),
            "sharp_yield_pct": round(sharp.flat_yield * 100, 2) if sharp else None,
            "clv_pct": round(best.average_clv * 100, 2),
        }


@dataclass
class ScanReport:
    results: list[LeagueResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"leagues": [r.to_dict() for r in self.results]}

    def render(self) -> str:
        header = (f"{'code':<5}{'league':<26}{'matches':>8}{'model':>8}{'market':>8}"
                  f"{'gap':>8}{'bets':>7}{'yield':>9}{'sharp':>9}{'CLV':>8}")
        lines = [header, "-" * len(header)]

        usable = [r for r in self.results if r.ok]
        for result in self.results:
            if not result.ok:
                lines.append(f"{result.league:<5}{result.name[:25]:<26}"
                             f"{'— ' + (result.error or 'failed'):>48}")
                continue
            row = result.to_dict()
            sharp = f"{row['sharp_yield_pct']:+.2f}%" if row["sharp_yield_pct"] is not None else "—"
            lines.append(
                f"{row['league']:<5}{row['name'][:25]:<26}{row['matches']:>8}"
                f"{row['model_log_loss']:>8.4f}{row['market_log_loss']:>8.4f}"
                f"{row['log_loss_gap']:>+8.4f}{row['bets']:>7}"
                f"{row['flat_yield_pct']:>+8.2f}%{sharp:>9}{row['clv_pct']:>+7.2f}%"
            )

        lines += ["", "gap    = market log loss minus model log loss. Positive means the "
                      "model forecast better", "         than the closing line in that "
                      "division — the thing worth hunting for.",
                  "yield  = flat-stake return at the best price across books.",
                  "sharp  = the same with no price shopping at all.",
                  "CLV    = average closing-line value of the bets placed."]

        if usable:
            best = max(usable, key=lambda r: r.optimistic.market_log_loss
                       - r.optimistic.model_log_loss)
            gap = best.optimistic.market_log_loss - best.optimistic.model_log_loss
            lines += ["", f"Best forecast gap: {best.league} ({best.name}) at {gap:+.4f}."]
            lines.append(
                "The model beat the closing line there. Treat it as a lead to test "
                "further, not a result." if gap > 0 else
                "The closing line beat the model in every division scanned."
            )
        return "\n".join(lines)


def scan_leagues(
    leagues: Sequence[str] = DEFAULT_LEAGUES,
    seasons: int = 6,
    config: Config | None = None,
    train_days: int = 400,
    refit_every: int = 21,
    include_pessimistic: bool = True,
    offline: bool = False,
    progress: Callable[[str], None] | None = None,
) -> ScanReport:
    config = config or Config()
    say = progress or (lambda _: None)
    source = FootballDataUK(offline=offline)
    codes = recent_seasons(seasons)
    report = ScanReport()

    for league in leagues:
        name = LEAGUES.get(league, league)
        say(f"{league:<5} {name} ...")
        try:
            matches = source.results(league, codes)
        except Exception as exc:
            report.results.append(LeagueResult(league, name, 0, error=str(exc)[:60]))
            continue
        if len(matches) < 300:
            report.results.append(
                LeagueResult(league, name, len(matches),
                             error=f"only {len(matches)} matches"))
            continue
        try:
            optimistic = run_backtest(matches, config=config, train_days=train_days,
                                      refit_every_days=refit_every)
            pessimistic = None
            if include_pessimistic:
                stripped = sharp_only(matches)
                if len(stripped) > 300:
                    pessimistic = run_backtest(stripped, config=config,
                                               train_days=train_days,
                                               refit_every_days=refit_every)
        except Exception as exc:
            report.results.append(LeagueResult(league, name, len(matches),
                                               error=str(exc)[:60]))
            continue
        report.results.append(LeagueResult(league, name, len(matches),
                                           optimistic=optimistic,
                                           pessimistic=pessimistic))
    return report
