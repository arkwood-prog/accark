"""Written analysis for each recommendation.

Every sentence here is generated from a number the engine actually computed.
The aim is the read a trader would give a colleague: what the bet is, what
the model thinks, what the market thinks, where the disagreement comes from,
and what would make it wrong.
"""

from __future__ import annotations

from ..betting.parlays import Slip
from ..betting.value import Candidate
from ..models.form import TeamForm
from .context import FixtureContext


def _pct(value: float, places: int = 1) -> str:
    return f"{value * 100:.{places}f}%"


def _signed_pct(value: float, places: int = 1) -> str:
    return f"{value * 100:+.{places}f}%"


def _form_phrase(form: TeamForm, team: str) -> str:
    if form.matches == 0:
        return f"{team} have no matches in the sample window."
    parts = [
        f"{team} are {form.form_string} in their last {form.matches} "
        f"({form.points_per_game:.2f} points per game)",
        f"scoring {form.goals_for_per_game:.2f} and conceding "
        f"{form.goals_against_per_game:.2f} per game",
    ]
    if form.unbeaten_run >= 4:
        parts.append(f"unbeaten in {form.unbeaten_run}")
    elif form.winless_run >= 4:
        parts.append(f"without a win in {form.winless_run}")
    return ", ".join(parts) + "."


def _strength_phrase(context: FixtureContext) -> str:
    home, away = context.fixture.home, context.fixture.away
    hr, ar = context.home_rating, context.away_rating
    attack_gap = hr.attack - ar.attack
    defence_gap = hr.defence - ar.defence

    if abs(attack_gap) < 0.08:
        attack_line = "The two attacks rate within a whisker of each other"
    else:
        stronger, weaker = (home, away) if attack_gap > 0 else (away, home)
        attack_line = (f"{stronger}'s attack rates {abs(attack_gap):.2f} log-goals above "
                       f"{weaker}'s")
    if abs(defence_gap) < 0.08:
        defence_line = "and the defences are similarly matched"
    else:
        stronger, weaker = (home, away) if defence_gap > 0 else (away, home)
        defence_line = (f"while {stronger} defend {abs(defence_gap):.2f} better "
                        f"than {weaker}")
    return f"{attack_line}, {defence_line}."


def match_preview(context: FixtureContext) -> str:
    """A short model read of the fixture itself, independent of any bet."""
    home_xg, away_xg = context.expected_goals
    scores = ", ".join(f"{score} ({_pct(p)})" for score, p in context.likely_scores[:3])
    home, away = context.fixture.home, context.fixture.away

    lines = [
        f"Model line: {home} {home_xg:.2f} - {away_xg:.2f} {away} "
        f"({home_xg + away_xg:.2f} total goals). Most likely scorelines: {scores}.",
        _strength_phrase(context),
        _form_phrase(context.home_form, home),
        _form_phrase(context.away_form, away),
    ]

    home_venue, away_venue = context.home_venue_form, context.away_venue_form
    if home_venue.matches >= 3 and away_venue.matches >= 3:
        lines.append(
            f"At home {home} average {home_venue.goals_for_per_game:.2f} scored and "
            f"{home_venue.goals_against_per_game:.2f} conceded; on the road {away} "
            f"average {away_venue.goals_for_per_game:.2f} and "
            f"{away_venue.goals_against_per_game:.2f}."
        )

    over_rate = (context.home_form.over_2_5_rate + context.away_form.over_2_5_rate) / 2
    btts_rate = (context.home_form.btts_rate + context.away_form.btts_rate) / 2
    if context.home_form.matches and context.away_form.matches:
        lines.append(
            f"Recent games involving these sides have gone over 2.5 goals "
            f"{_pct(over_rate, 0)} of the time and seen both teams score "
            f"{_pct(btts_rate, 0)} of the time."
        )

    if context.head_to_head:
        meetings = ", ".join(
            f"{m.home_goals}-{m.away_goals} ({m.home[:3].upper()} h, {m.date.year})"
            for m in context.head_to_head[:3]
        )
        lines.append(f"Recent meetings: {meetings}.")
    return " ".join(lines)


def _edge_source(candidate: Candidate, context: FixtureContext) -> str:
    """Explain *why* there is an edge — model disagreement, or a shopped price."""
    quote = context.quotes.get(candidate.selection_id)
    model_vs_market = candidate.model_probability - candidate.market_probability
    sentences = []

    if abs(model_vs_market) >= 0.02:
        direction = "higher" if model_vs_market > 0 else "lower"
        sentences.append(
            f"The model rates this {_pct(abs(model_vs_market))} {direction} than the "
            f"market's margin-free price ({_pct(candidate.model_probability)} against "
            f"{_pct(candidate.market_probability)}), and the blended number used for "
            f"staking is {_pct(candidate.probability)}."
        )
    else:
        sentences.append(
            f"Model and market broadly agree on the probability "
            f"({_pct(candidate.model_probability)} against "
            f"{_pct(candidate.market_probability)})."
        )

    if quote and quote.sharp_odds and candidate.odds > quote.sharp_odds * 1.005:
        gain = candidate.odds / quote.sharp_odds - 1
        sentences.append(
            f"The best available price of {candidate.odds:.2f} is {_signed_pct(gain)} "
            f"better than the sharp line of {quote.sharp_odds:.2f} — a meaningful part "
            f"of this edge is price shopping, so it disappears if you take a worse price."
        )
    if quote:
        sentences.append(
            f"That market is quoted by {quote.book_count} book(s) at a "
            f"{_pct(quote.overround)} overround."
        )
    return " ".join(sentences)


def _risk_notes(candidate: Candidate, context: FixtureContext) -> list[str]:
    risks: list[str] = []
    quote = context.quotes.get(candidate.selection_id)

    if context.data_confidence < 0.6:
        risks.append(
            "Thin match history for one or both teams, so the ratings are shrunk "
            "toward league average and carry real uncertainty."
        )
    if candidate.edge > 0.15:
        risks.append(
            f"An edge of {_pct(candidate.edge)} is larger than genuine football edges "
            "usually are. Check the price is live and that no team news has moved the "
            "market since these odds were captured."
        )
    if quote and quote.overround > 0.08:
        risks.append(
            f"The {_pct(quote.overround)} margin on this market makes the fair-price "
            "estimate noisier than usual."
        )
    if quote and quote.book_count <= 2:
        risks.append("Few books quoting this market, so the consensus is weakly supported.")
    if candidate.derived:
        risks.append(
            "This price is not quoted directly — it is replicated by splitting the "
            "stake across two 1X2 outcomes in proportion to their prices, which means "
            "two tickets and the risk that one leg moves before you place the other. "
            "Compare it against the double chance the book quotes you."
        )
    if candidate.probability > 0.80:
        risks.append(
            "Short price: one goal against the run of play wipes out a long sequence "
            "of these."
        )
    if candidate.odds > 5.0:
        risks.append(
            "Long price: expect long losing runs even when the edge is real, and size "
            "the stake accordingly."
        )
    if not risks:
        risks.append(
            "No specific red flags — the main risk is ordinary model error and late "
            "team news."
        )
    return risks


def analyse_candidate(candidate: Candidate, context: FixtureContext) -> str:
    """Full written analysis for a single bet."""
    home_xg, away_xg = context.expected_goals
    value_gap = candidate.odds / candidate.fair_odds - 1

    header = (
        f"{candidate.label} at {candidate.odds:.2f} "
        f"({candidate.match}, {candidate.date:%a %d %b})."
    )
    verdict = (
        f"Fair price {candidate.fair_odds:.2f}, available at {candidate.odds:.2f} — "
        f"{_signed_pct(value_gap)} of value, an expected return of "
        f"{_signed_pct(candidate.edge)} per unit staked. "
        f"Confidence {candidate.confidence:.0f}/100 ({candidate.tier.lower()})."
    )
    model_read = (
        f"The goal model projects {context.fixture.home} {home_xg:.2f} - "
        f"{away_xg:.2f} {context.fixture.away}. {_strength_phrase(context)}"
    )
    market_read = _edge_source(candidate, context)
    form_read = " ".join([
        _form_phrase(context.home_form, context.fixture.home),
        _form_phrase(context.away_form, context.fixture.away),
    ])
    staking = (
        f"Full Kelly on these numbers is {_pct(candidate.kelly)} of bankroll; the "
        f"recommended stake applies a fraction of that with per-bet and slate-wide caps."
    )
    risks = "\n".join(f"- {risk}" for risk in _risk_notes(candidate, context))

    return "\n\n".join([
        f"**{header}**",
        f"**Verdict.** {verdict}",
        f"**Model read.** {model_read}",
        f"**Market read.** {market_read}",
        f"**Form.** {form_read}",
        f"**Staking.** {staking}",
        f"**Risks.** {risks}",
    ])


def analyse_slip(slip: Slip, contexts: dict[str, FixtureContext]) -> str:
    """Written analysis for a multi, including the honest maths on margin."""
    if slip.size == 1:
        candidate = slip.legs[0]
        return analyse_candidate(candidate, contexts[candidate.fixture_key])

    naive_price = 1.0
    for leg in slip.legs:
        naive_price *= leg.odds
    fair_price = 1.0 / slip.probability if slip.probability > 0 else float("inf")

    header = (
        f"{slip.kind} at {slip.odds:.2f} — {slip.size} legs, "
        f"{_pct(slip.probability)} to land."
    )
    legs_line = " + ".join(
        f"{leg.label} ({leg.odds:.2f}, {_pct(leg.probability)})" for leg in slip.legs
    )

    if slip.correlated:
        independent = 1.0
        for leg in slip.legs:
            independent *= leg.probability
        correlation = slip.probability / independent if independent > 0 else 1.0
        structure = (
            f"Both legs come from the same match, so they are not independent. Priced "
            f"from the joint score distribution the combination lands "
            f"{_pct(slip.probability)} of the time, against {_pct(independent)} if you "
            f"wrongly multiplied the legs — a correlation factor of {correlation:.2f}x. "
            f"The bet is only available at the multiplied price of {slip.odds:.2f} at "
            f"books that do not model the correlation."
        )
    else:
        structure = (
            f"The legs are in different matches and treated as independent, so the "
            f"combined probability is the product of the legs: {_pct(slip.probability)}. "
            f"Note that the bookmaker's margin compounds across legs — that is why each "
            f"leg here has to clear a higher individual edge than a single would."
        )

    economics = (
        f"Fair price on the model is {fair_price:.2f} against {naive_price:.2f} on offer, "
        f"an expected return of {_signed_pct(slip.edge)} per unit. Expected log growth is "
        f"{slip.log_growth:.5f}, which is the number the ranking uses — it prefers a "
        f"combination that actually lands over a lottery ticket with the same headline EV."
    )
    reality = (
        f"Expect this to lose {_pct(1 - slip.probability, 0)} of the time. A "
        f"{slip.size}-leg bet is a variance product: even with a real edge, the sensible "
        f"stake is a fraction of what you would put on the same edge as a single, which "
        f"is what the staking model does."
    )
    weakest = min(slip.legs, key=lambda leg: leg.confidence)
    weak_link = (
        f"Weakest leg is {weakest.label} at confidence {weakest.confidence:.0f}/100 — the "
        f"slip is only as good as that one."
    )

    return "\n\n".join([
        f"**{header}**",
        f"**Legs.** {legs_line}",
        f"**Structure.** {structure}",
        f"**Economics.** {economics}",
        f"**Reality check.** {reality}",
        f"**Weak link.** {weak_link}",
    ])
