"""A verification ladder for running the engine against real data.

Five stages, cheapest and most diagnostic first. Each check compares a
computed value against a range that real football produces, and reports
PASS / WARN / FAIL with a plain reason.

The design principle: **a check that can only pass is worthless**. Several
of these fire on results that look *too good*, because on this problem an
implausibly strong number is far more likely to be a bug or a data leak than
an edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

from .backtest.engine import BacktestResult, run_backtest
from .config import Config
from .data.schema import Match
from .models.dixon_coles import DixonColesModel

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_RANK = {PASS: 0, WARN: 1, FAIL: 2}


@dataclass
class Check:
    name: str
    value: str
    expected: str
    status: str
    note: str = ""


@dataclass
class Stage:
    title: str
    checks: list[Check] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.checks:
            return PASS
        return max((c.status for c in self.checks), key=lambda s: _RANK[s])


@dataclass
class VerificationReport:
    label: str
    stages: list[Stage] = field(default_factory=list)
    verdict: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.stages:
            return PASS
        return max((s.status for s in self.stages), key=lambda s: _RANK[s])

    @property
    def failures(self) -> list[Check]:
        return [c for s in self.stages for c in s.checks if c.status == FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [c for s in self.stages for c in s.checks if c.status == WARN]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "status": self.status,
            "stages": [
                {
                    "title": stage.title,
                    "status": stage.status,
                    "checks": [vars(c) for c in stage.checks],
                    "lines": stage.lines,
                }
                for stage in self.stages
            ],
            "verdict": self.verdict,
        }

    def render(self) -> str:
        width = 76
        out = ["=" * width, f"  BETTINGEDGE VERIFICATION — {self.label}", "=" * width, ""]
        for index, stage in enumerate(self.stages, start=1):
            out.append(f"STAGE {index}  {stage.title.upper()}   [{stage.status}]")
            for check in stage.checks:
                out.append(
                    f"  [{check.status}] {check.name:<30}{check.value:<22}{check.expected}"
                )
                if check.note:
                    for line in _wrap(check.note, width - 8):
                        out.append(f"         {line}")
            for line in stage.lines:
                out.append(f"  {line}")
            out.append("")
        out += ["-" * width, "VERDICT", "-" * width]
        out += [f"  {line}" for line in self.verdict]
        out.append("")
        return "\n".join(out)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def _band(value: float, good: tuple[float, float], tolerable: tuple[float, float]) -> str:
    if good[0] <= value <= good[1]:
        return PASS
    if tolerable[0] <= value <= tolerable[1]:
        return WARN
    return FAIL


# --------------------------------------------------------------------------
# Pessimistic pricing — works for any source, not just football-data.co.uk
# --------------------------------------------------------------------------
def sharp_only(matches: Sequence[Match]) -> list[Match]:
    """Rewrite history so every bet settles at the sharp closing price.

    This removes the entire price-shopping component of any edge: you are no
    longer taking the best of ten books, only the sharpest single line. What
    survives is what the model itself contributes.
    """
    rewritten = []
    for match in matches:
        if not match.closing_odds:
            continue
        rewritten.append(replace(match, best_odds=dict(match.closing_odds)))
    return rewritten


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
def stage_data_integrity(matches: Sequence[Match]) -> Stage:
    stage = Stage("data integrity")
    n = len(matches)
    stage.checks.append(Check(
        "matches loaded", f"{n}", "expect > 500",
        PASS if n > 500 else (WARN if n > 200 else FAIL),
        "" if n > 500 else "Too little history to fit stable ratings or backtest. "
                           "Increase --seasons.",
    ))
    if not matches:
        return stage

    goals = sum(m.total_goals for m in matches) / n
    stage.checks.append(Check(
        "goals per game", f"{goals:.2f}", "expect 2.2 - 3.6",
        _band(goals, (2.2, 3.6), (1.8, 4.2)),
        "" if 2.2 <= goals <= 3.6 else "Outside anything real football produces — "
                                       "suspect a parsing error in the score columns.",
    ))

    home_rate = sum(1 for m in matches if m.result == "H") / n
    stage.checks.append(Check(
        "home win rate", f"{home_rate:.1%}", "expect 35% - 55%",
        _band(home_rate, (0.35, 0.55), (0.30, 0.60)),
        "" if 0.35 <= home_rate <= 0.55 else "A home/away column swap is the usual cause.",
    ))

    with_1x2 = sum(1 for m in matches if "1X2:H" in m.best_odds) / n
    stage.checks.append(Check(
        "1X2 price coverage", f"{with_1x2:.1%}", "expect > 90%",
        PASS if with_1x2 > 0.90 else (WARN if with_1x2 > 0.5 else FAIL),
        "" if with_1x2 > 0.90 else "Matches without prices cannot be bet or backtested.",
    ))

    with_ou = sum(1 for m in matches if "OU2.5:O" in m.best_odds) / n
    stage.checks.append(Check(
        "over/under coverage", f"{with_ou:.1%}", "expect > 50%",
        PASS if with_ou > 0.50 else WARN,
        "" if with_ou > 0.50 else "Older seasons often lack totals pricing; not fatal.",
    ))

    with_sharp = sum(1 for m in matches if "1X2:H" in m.closing_odds) / n
    stage.checks.append(Check(
        "sharp reference coverage", f"{with_sharp:.1%}", "expect > 80%",
        PASS if with_sharp > 0.80 else WARN,
        "" if with_sharp > 0.80 else "Without a sharp line the closing-line-value and "
                                     "pessimistic tests lose their reference point.",
    ))

    # Is the "best price" a single book's quote, or a maximum taken across many?
    # The answer changes what a plausible overround looks like, so infer it from
    # the data rather than assuming.
    comparable = [m for m in matches if m.closing_odds and m.best_odds]
    shopped_share = 0.0
    if comparable:
        shopped = sum(
            1 for m in comparable
            if any(m.best_odds.get(s, 0) > m.closing_odds.get(s, 0) * 1.005
                   for s in m.closing_odds)
        )
        shopped_share = shopped / len(comparable)
    cross_book = shopped_share > 0.5

    overrounds = [
        sum(1 / m.best_odds[s] for s in ("1X2:H", "1X2:D", "1X2:A")) - 1
        for m in matches if all(s in m.best_odds for s in ("1X2:H", "1X2:D", "1X2:A"))
    ]
    if overrounds:
        median = sorted(overrounds)[len(overrounds) // 2]
        if cross_book:
            # Taking the highest price on all three outcomes routinely produces a
            # book that sums to under 100%. That is expected here, not a bug — but
            # it is exactly why the sharp-only re-run exists.
            status = PASS if -0.04 <= median <= 0.06 else WARN
            note = (
                f"Prices are a maximum across books on {shopped_share:.0%} of matches, "
                "so a sub-100% book is normal. It also means no single account could "
                "have taken every one of these prices — stage 4 re-runs without any "
                "shopping for that reason."
            )
        else:
            status = _band(median, (0.01, 0.09), (-0.01, 0.15))
            note = "" if 0.01 <= median <= 0.09 else (
                "A single-book price set should carry a positive margin. A negative one "
                "means the columns being read are not one book's quote."
            )
        stage.checks.append(Check(
            "median 1X2 overround", f"{median:.2%}",
            "expect -4% to 6% (cross-book)" if cross_book else "expect 1% - 9%",
            status, note,
        ))

    ordered = sorted(matches, key=lambda m: m.date)
    stage.lines.append(f"date range: {ordered[0].date} to {ordered[-1].date}")
    last = ordered[-1]
    price = last.best_odds.get("1X2:H")
    stage.lines.append(
        f"spot-check the most recent row against a result you remember:\n"
        f"    {last.date}  {last.home} {last.home_goals}-{last.away_goals} {last.away}"
        + (f"  (home price {price:.2f})" if price else "")
    )
    return stage


def stage_model_sanity(matches: Sequence[Match], config: Config) -> tuple[Stage, object]:
    stage = Stage("model sanity")
    model = DixonColesModel(config.model).fit(matches)

    stage.checks.append(Check(
        "optimiser converged", "yes" if model.converged else "no", "expect yes",
        PASS if model.converged else FAIL,
        "" if model.converged else "Try more seasons or a longer --half-life.",
    ))

    gamma = model.home_advantage
    stage.checks.append(Check(
        "home advantage", f"{gamma:+.3f}", "expect +0.10 to +0.45 log-goals",
        _band(gamma, (0.10, 0.45), (0.0, 0.60)),
        "" if 0.10 <= gamma <= 0.45 else
        "Real leagues run about +0.20 to +0.30. A negative value means home and away "
        "are swapped somewhere in the pipeline.",
    ))

    stage.checks.append(Check(
        "low-score rho", f"{model.rho:+.3f}", "expect -0.25 to +0.10",
        _band(model.rho, (-0.25, 0.10), (-0.35, 0.20)),
        "" if -0.25 <= model.rho <= 0.10 else
        "Rho pinned at a bound suggests the fit is unstable.",
    ))

    ratings = model.ratings()
    spread = ratings[0].net - ratings[-1].net
    stage.checks.append(Check(
        "rating spread", f"{spread:.2f}", "expect 0.5 - 2.5 log-goals",
        _band(spread, (0.5, 2.5), (0.2, 3.5)),
        "" if 0.5 <= spread <= 2.5 else
        "A near-zero spread means the model is not separating teams; a huge one means "
        "shrinkage is not doing its job.",
    ))

    stage.lines.append("strongest and weakest by net rating — check these against what "
                       "you know about the league:")
    for rating in ratings[:3]:
        stage.lines.append(f"    {rating.net:+.3f}  {rating.team}  ({rating.matches} played)")
    stage.lines.append("    ...")
    for rating in ratings[-3:]:
        stage.lines.append(f"    {rating.net:+.3f}  {rating.team}  ({rating.matches} played)")
    stage.lines.append("If the top and bottom of that list look wrong to you, stop here — "
                       "no amount of")
    stage.lines.append("downstream maths fixes a model that disagrees with the table.")
    return stage, model


def stage_forecast_quality(result: BacktestResult) -> Stage:
    stage = Stage("forecast quality")
    if not result.forecasts:
        stage.checks.append(Check("forecasts scored", "0", "expect > 200", FAIL,
                                  "No matches were scored — widen the backtest window."))
        return stage

    stage.checks.append(Check(
        "forecasts scored", f"{result.forecasts}", "expect > 200",
        PASS if result.forecasts > 200 else WARN,
    ))

    model_ll, market_ll = result.model_log_loss, result.market_log_loss
    stage.checks.append(Check(
        "model 1X2 log loss", f"{model_ll:.4f}", "expect 0.85 - 1.15",
        _band(model_ll, (0.85, 1.15), (0.75, 1.30)),
        "" if 0.85 <= model_ll <= 1.15 else
        "Below 0.85 on a 3-way football market is not achievable — suspect a data leak.",
    ))
    stage.checks.append(Check(
        "market 1X2 log loss", f"{market_ll:.4f}", "expect 0.85 - 1.10",
        _band(market_ll, (0.85, 1.10), (0.75, 1.25)),
    ))

    gap = market_ll - model_ll   # positive = model is better
    if gap > 0.02:
        status, note = WARN, (
            "The model beats the closing line by more than 0.02 nats. That is a larger "
            "information advantage than published models achieve against Pinnacle. "
            "Check for lookahead before believing it."
        )
    elif gap > 0:
        status, note = PASS, (
            "The model adds information over the closing line. This is the necessary "
            "condition for a real edge — necessary, not sufficient."
        )
    elif gap > -0.05:
        status, note = PASS, (
            "The market is sharper, which is the normal and expected result for a "
            "goals-only model. Edges then have to come from price shopping or from "
            "markets the closing line prices less carefully."
        )
    else:
        status, note = WARN, (
            "The model is a long way behind the market. Any positive ROI is variance. "
            "Try more seasons or a different --half-life before betting anything."
        )
    stage.checks.append(Check("model vs market", f"{gap:+.4f}", "positive = model wins",
                              status, note))

    if result.calibration:
        worst = max(result.calibration, key=lambda r: abs(r["actual"] - r["predicted"]))
        deviation = abs(worst["actual"] - worst["predicted"])
        stage.checks.append(Check(
            "worst calibration band", f"{worst['bucket']} off by {deviation:.1%}",
            "expect < 10 points",
            PASS if deviation < 0.10 else (WARN if deviation < 0.20 else FAIL),
            "" if deviation < 0.10 else
            f"Only {worst['n']} bets in that band, so some of this is noise; if it "
            "persists across runs the probabilities are miscalibrated there.",
        ))
    return stage


def stage_price_realism(optimistic: BacktestResult, pessimistic: BacktestResult | None,
                        price_mode_note: str) -> Stage:
    stage = Stage("price realism")
    stage.lines.append(price_mode_note)

    stage.checks.append(Check(
        "bets placed", f"{len(optimistic.bets)}", "expect > 300",
        PASS if len(optimistic.bets) > 300 else WARN,
        "" if len(optimistic.bets) > 300 else
        "Too few bets for any performance number to mean much.",
    ))

    optimistic_yield = optimistic.flat_yield
    stage.checks.append(Check(
        "flat yield (best price)", f"{optimistic_yield:+.2%}", "expect -5% to +8%",
        _band(optimistic_yield, (-0.05, 0.08), (-0.15, 0.15)),
        "" if optimistic_yield <= 0.08 else
        "A yield above 8% on football does not survive contact with a real account. "
        "It usually means the price used was never actually available.",
    ))

    if optimistic.bets:
        implied = sum(1 / b.odds for b in optimistic.bets) / len(optimistic.bets)
        drift = optimistic.hit_rate - implied
        stage.checks.append(Check(
            "hit rate vs implied", f"{optimistic.hit_rate:.1%} vs {implied:.1%}",
            "expect within 10 points",
            PASS if abs(drift) < 0.10 else FAIL,
            "" if abs(drift) < 0.10 else
            "A large gap means bets are being settled against the wrong outcome.",
        ))

    if pessimistic is not None:
        pessimistic_yield = pessimistic.flat_yield
        if pessimistic_yield > 0.08:
            status, note = WARN, (
                "Positive without any price shopping, which is the right sign — but a "
                "yield this large against a sharp closing line is not credible on real "
                "data. Check for lookahead before trusting it."
            )
        elif pessimistic_yield > 0:
            status, note = PASS, (
                "The edge survives without price shopping, so it is coming from the "
                "model rather than from taking the best of ten books."
            )
        else:
            status, note = WARN, (
                "The edge exists only when taking the best price across every book. "
                "That is still real money, but it decays as books converge and it is "
                "what gets accounts limited. Do not present it as model skill."
            )
        stage.checks.append(Check(
            "flat yield (sharp only)", f"{pessimistic_yield:+.2%}",
            "the number that matters", status, note,
        ))
        stage.lines.append(
            f"best price: {len(optimistic.bets)} bets at {optimistic_yield:+.2%}   |   "
            f"sharp only: {len(pessimistic.bets)} bets at {pessimistic_yield:+.2%}"
        )
    return stage


def stage_sample_size(result: BacktestResult) -> Stage:
    """Is the sample large enough for the headline number to mean anything?"""
    stage = Stage("statistical power")
    bets = result.bets
    if len(bets) < 2:
        stage.checks.append(Check("sample", f"{len(bets)}", "expect > 1000", FAIL,
                                  "No usable sample."))
        return stage

    outcomes = [(b.odds - 1.0) if b.won else -1.0 for b in bets]
    n = len(outcomes)
    mean = sum(outcomes) / n
    variance = sum((o - mean) ** 2 for o in outcomes) / (n - 1)
    sd = math.sqrt(variance)
    standard_error = sd / math.sqrt(n)

    # A thin sample limits what you can conclude; it does not mean anything is
    # broken, so it never escalates past a warning.
    stage.checks.append(Check(
        "bets in sample", f"{n}", "expect > 1000",
        PASS if n > 1000 else WARN,
        "" if n > 1000 else "Enough to check the machinery, not enough to judge "
                            "profitability. Add seasons or leagues.",
    ))
    stage.checks.append(Check(
        "yield ± 2 standard errors",
        f"{mean:+.2%} ± {2 * standard_error:.2%}", "should exclude zero",
        PASS if abs(mean) > 2 * standard_error else WARN,
        "" if abs(mean) > 2 * standard_error else
        "The observed yield is statistically indistinguishable from zero.",
    ))

    # Bets needed for a 2% edge to clear two standard errors.
    needed = int((2 * sd / 0.02) ** 2)
    stage.lines.append(
        f"per-bet standard deviation is {sd:.2f}, so proving a 2% edge at two standard "
        f"errors\nneeds roughly {needed:,} bets. A single league season supplies a few "
        f"hundred."
    )
    stage.lines.append(
        "This is why closing-line value and log loss are the evidence, and ROI is the "
        "smoke test."
    )
    return stage


# --------------------------------------------------------------------------
def verify(
    matches: Sequence[Match],
    config: Config | None = None,
    label: str = "dataset",
    train_days: int = 400,
    refit_every: int = 14,
    price_mode_note: str = "",
    run_pessimistic: bool = True,
    progress: Callable[[str], None] | None = None,
) -> VerificationReport:
    """Run the whole ladder and return a report."""
    config = config or Config()
    say = progress or (lambda _: None)
    report = VerificationReport(label=label)

    say("stage 1/5  checking data integrity ...")
    data_stage = stage_data_integrity(matches)
    report.stages.append(data_stage)
    if data_stage.status == FAIL and len(matches) < 200:
        report.verdict = ["Data did not load well enough to go further. Fix stage 1 first."]
        return report

    say("stage 2/5  fitting the model ...")
    model_stage, _ = stage_model_sanity(matches, config)
    report.stages.append(model_stage)

    say("stage 3/5  walk-forward backtest at the loaded prices ...")
    optimistic = run_backtest(matches, config=config, train_days=train_days,
                              refit_every_days=refit_every)

    report.stages.append(stage_forecast_quality(optimistic))

    pessimistic = None
    if run_pessimistic:
        say("stage 4/5  re-running at sharp prices only ...")
        stripped = sharp_only(matches)
        if len(stripped) > 200:
            pessimistic = run_backtest(stripped, config=config, train_days=train_days,
                                       refit_every_days=refit_every)
    report.stages.append(stage_price_realism(optimistic, pessimistic, price_mode_note))

    say("stage 5/5  assessing statistical power ...")
    report.stages.append(stage_sample_size(optimistic))

    report.verdict = _verdict(report, optimistic, pessimistic)
    return report


def _verdict(report: VerificationReport, optimistic: BacktestResult,
             pessimistic: BacktestResult | None) -> list[str]:
    lines: list[str] = []
    failures, warnings = report.failures, report.warnings

    # Stages 1-2 test whether the pipeline is sound; 3-5 test what the results
    # are worth. Only the first kind means "broken".
    integrity_stages = {stage.title for stage in report.stages[:2]}
    integrity_failures = [
        check for stage in report.stages if stage.title in integrity_stages
        for check in stage.checks if check.status == FAIL
    ]
    other_failures = [c for c in failures if c not in integrity_failures]

    if integrity_failures:
        lines.append("Data or model checks FAILED. Treat the pipeline as broken until "
                     "these are resolved:")
        lines += [f"  - {c.name}: {c.value} ({c.expected})" for c in integrity_failures]
        lines.append("")
    if other_failures:
        lines.append("The pipeline is sound, but these results are out of range:")
        lines += [f"  - {c.name}: {c.value} ({c.expected})" for c in other_failures]
        lines.append("")

    beats_market = optimistic.model_log_loss < optimistic.market_log_loss
    lines.append(
        "The model adds information over the closing line."
        if beats_market else
        "The market is sharper than the model on log loss — the normal result."
    )

    if pessimistic is not None:
        if pessimistic.flat_yield > 0 and optimistic.flat_yield > 0:
            lines.append(
                "The edge survives at sharp prices only, so it is not purely price "
                "shopping. This is the most encouraging combination available here."
            )
        elif optimistic.flat_yield > 0:
            lines.append(
                "Profitable only at the best price across every book. Real, but it is "
                "shopping rather than modelling, and it is what gets accounts limited."
            )
        else:
            lines.append("Not profitable in this window under either pricing assumption.")

    if warnings:
        lines.append(f"{len(warnings)} warning(s) — read the notes above before acting.")

    lines.append("")
    lines.append("Nothing here proves future profit. The only real test is forward "
                 "paper-trading:")
    lines.append("  bettingedge recommend --league <code> --json slates/$(date +%F).json")
    lines.append("Log the closing price yourself and track closing-line value; it "
                 "converges about")
    lines.append("ten times faster than ROI.")
    return lines
