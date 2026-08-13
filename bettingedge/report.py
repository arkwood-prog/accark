"""Terminal and Markdown rendering of a slate."""

from __future__ import annotations

from .analysis.narrative import match_preview
from .betting.parlays import Slip
from .pipeline import Slate

RULE = "=" * 78
THIN = "-" * 78

DISCLAIMER = (
    "This is a statistical modelling tool, not a tipping service and not financial "
    "advice. A positive expected value is an estimate, not a promise. Never stake "
    "money you cannot afford to lose. If gambling stops being fun, stop — "
    "BeGambleAware.org / 0808 8020 133 (UK), 1-800-GAMBLER (US)."
)


def _slip_block(slip: Slip, index: int) -> list[str]:
    lines = [
        f"{index}. {slip.kind}"
        + (f"  [{slip.profile}]" if slip.profile else "")
        + f"  |  odds {slip.odds:.2f}  |  {slip.probability * 100:.1f}% to land"
        f"  |  edge {slip.edge * 100:+.2f}%  |  confidence {slip.confidence:.0f}/100",
    ]
    for leg in slip.legs:
        lines.append(
            f"     - {leg.match} ({leg.date:%d %b}): {leg.label} @ {leg.odds:.2f}"
            f"   [model {leg.model_probability * 100:.1f}% | market "
            f"{leg.market_probability * 100:.1f}% | blend "
            f"{leg.probability * 100:.1f}% -> fair {leg.fair_odds:.2f}]"
        )
    if slip.stake > 0:
        lines.append(
            f"     Stake {slip.stake:.2f} -> returns {slip.potential_return:.2f} "
            f"(profit {slip.potential_return - slip.stake:+.2f})"
        )
    if slip.note:
        lines.append(f"     Note: {slip.note}")
    return lines


def render_slate(slate: Slate, detail: bool = True, previews: bool = False) -> str:
    out: list[str] = [RULE, "  BETTINGEDGE - DATA-DRIVEN FOOTBALL BET RECOMMENDATIONS", RULE, ""]

    model = slate.model
    out += [
        f"Generated      : {slate.generated_at:%Y-%m-%d %H:%M}",
    ]
    if len(slate.models) > 1:
        out.append(f"Models         : {len(slate.models)} leagues, fitted separately")
        for code, fit in slate.models.items():
            flag = "  << thin sample" if fit.thin_sample else ""
            out.append(f"   {code:<5} {fit.n_matches:>5} matches, "
                       f"{fit.effective_matches_per_team:>4.0f} eff/team, "
                       f"home {fit.home_advantage:+.3f}{flag}")
    else:
        out += [
            f"Model          : Dixon-Coles, {model.n_matches} matches "
            f"(effective sample {model.effective_sample:.0f}, "
            f"{model.effective_matches_per_team:.0f} per team, "
            f"half-life {model.config.half_life_days:.0f}d)",
            f"Home advantage : {model.home_advantage:+.3f} log-goals   "
            f"low-score rho: {model.rho:+.3f}",
        ]
    out += [
        f"Blend          : {slate.config.market.model_weight:.0%} model / "
        f"{1 - slate.config.market.model_weight:.0%} market fair price",
        f"Fixtures       : {len(slate.contexts)} priced"
        + (f", {len(slate.skipped)} skipped" if slate.skipped else ""),
        "",
    ]

    if any(m.thin_sample for m in (slate.models.values() or [model])):
        out += [
            "!! THIN SAMPLE WARNING",
            "   " + ", ".join(
                f"{code} {fit.effective_matches_per_team:.0f}"
                for code, fit in (slate.models or {"model": model}).items()
                if fit.thin_sample
            ) + " time-weighted matches per team.",
            "   Early in a season the decay window is mostly off-season, so ratings are "
            "stale",
            "   last-season values. Expect the model to disagree with the market loudly "
            "and to be",
            "   wrong when it does. The confidence floor has been raised "
            f"automatically to {slate.config.selection.thin_sample_min_confidence:.0f}",
            "   for the affected leagues, so fewer bets survive than usual.",
            "",
        ]

    portfolio = slate.portfolio
    out += [
        THIN,
        "PORTFOLIO",
        THIN,
        f"Bankroll {portfolio.bankroll:,.2f}   staked {portfolio.total_staked:,.2f} "
        f"({portfolio.exposure_pct:.1f}% exposure) across {portfolio.bet_count} bets",
        f"Expected profit {portfolio.expected_profit:+,.2f} "
        f"({100 * portfolio.expected_profit / portfolio.total_staked:+.2f}% of turnover)"
        if portfolio.total_staked else "No qualifying bets.",
        f"Kelly stakes scaled by {portfolio.scaled_by:.2f} to respect the exposure cap"
        if portfolio.scaled_by < 1 else "",
        "",
    ]

    sections = [
        ("SINGLES", slate.singles),
        ("MULTIPLES (doubles, trebles, accumulators)", slate.multis),
        ("SAME-GAME COMBINATIONS (correlation-priced)", slate.same_game),
    ]
    for title, slips in sections:
        out += [THIN, title, THIN]
        if not slips:
            out += ["No qualifying bets in this category.", ""]
            continue
        for index, slip in enumerate(slips, start=1):
            out += _slip_block(slip, index)
            out.append("")

    if detail:
        out += [RULE, "  ANALYSIS", RULE, ""]
        for title, slips in sections:
            if not slips:
                continue
            out += [f"### {title}", ""]
            for index, slip in enumerate(slips, start=1):
                out += [f"--- {index}. {slip.kind} ---", "", slip.analysis, ""]

    if previews:
        out += [RULE, "  FIXTURE PREVIEWS", RULE, ""]
        for context in sorted(slate.contexts.values(),
                              key=lambda c: (c.fixture.date, c.fixture.home)):
            out += [
                f"{context.fixture.date:%a %d %b}  {context.fixture.home} v "
                f"{context.fixture.away}",
                match_preview(context),
                "",
            ]

    out += [RULE, DISCLAIMER, RULE]
    return "\n".join(line for line in out if line is not None)


def render_markdown(slate: Slate) -> str:
    """Markdown version, for saving or pasting into a notes app."""
    lines = [
        "# Football bet recommendations",
        "",
        f"*Generated {slate.generated_at:%Y-%m-%d %H:%M}. "
        f"Dixon-Coles model on {slate.model.n_matches} matches, blended "
        f"{slate.config.market.model_weight:.0%} model / "
        f"{1 - slate.config.market.model_weight:.0%} market.*",
        "",
        "## Portfolio",
        "",
        f"- Bankroll: **{slate.portfolio.bankroll:,.2f}**",
        f"- Staked: **{slate.portfolio.total_staked:,.2f}** "
        f"({slate.portfolio.exposure_pct:.1f}% exposure, "
        f"{slate.portfolio.bet_count} bets)",
        f"- Expected profit: **{slate.portfolio.expected_profit:+,.2f}**",
        "",
    ]

    for title, slips in [("Singles", slate.singles),
                         ("Multiples", slate.multis),
                         ("Same-game combinations", slate.same_game)]:
        lines += [f"## {title}", ""]
        if not slips:
            lines += ["_No qualifying bets._", ""]
            continue
        lines += ["| # | Bet | Odds | Model % | Edge | Conf | Stake |",
                  "|---|-----|------|---------|------|------|-------|"]
        for index, slip in enumerate(slips, start=1):
            description = " + ".join(f"{leg.match}: {leg.label}" for leg in slip.legs)
            lines.append(
                f"| {index} | {description} | {slip.odds:.2f} | "
                f"{slip.probability * 100:.1f}% | {slip.edge * 100:+.2f}% | "
                f"{slip.confidence:.0f} | {slip.stake:.2f} |"
            )
        lines.append("")
        for index, slip in enumerate(slips, start=1):
            lines += [f"### {title[:-1] if title.endswith('s') else title} {index}: "
                      f"{slip.kind}", "", slip.analysis, ""]

    lines += ["---", "", f"_{DISCLAIMER}_", ""]
    return "\n".join(lines)
