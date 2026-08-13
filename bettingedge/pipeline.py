"""The engine: history in, ranked bet slips with written analysis out.

    matches + fixtures
        -> fit Dixon-Coles
        -> score matrix per fixture
        -> model probabilities for every market
        -> devig the bookmaker prices into fair probabilities
        -> blend the two in logit space
        -> push the blend back into the score matrix (so correct scores and
           same-game combinations stay consistent with it)
        -> keep the selections with a real edge
        -> build singles, doubles, trebles, accumulators
        -> size every bet with fractional Kelly
        -> write the analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Sequence

import numpy as np

from .analysis.context import FixtureContext
from .analysis.narrative import analyse_candidate, analyse_slip, match_preview
from .betting.parlays import Slip, build_multis, build_same_game_slips, expected_log_growth
from .betting.staking import Portfolio, assign_stakes
from .betting.value import Candidate, blend_market_set, build_candidates, kelly_fraction
from .config import Config
from .data.schema import Fixture, Match
from .market.consensus import market_view
from .models.dixon_coles import DixonColesModel, FittedModel
from .models.markets import calibrate_matrix, price_markets
from .models.form import head_to_head, team_form


@dataclass
class Slate:
    """A complete set of recommendations for one matchday."""

    generated_at: datetime
    model: FittedModel                      # primary model (first league)
    contexts: dict[str, FixtureContext]
    singles: list[Slip]
    multis: list[Slip]
    same_game: list[Slip]
    portfolio: Portfolio
    config: Config
    skipped: list[dict] = field(default_factory=list)
    # One fitted model per league when the card spans several divisions.
    # Ratings are only comparable within a division, so each gets its own fit.
    models: dict[str, FittedModel] = field(default_factory=dict)

    @property
    def all_slips(self) -> list[Slip]:
        return [*self.singles, *self.multis, *self.same_game]

    def to_dict(self, include_contexts: bool = True) -> dict:
        payload = {
            "generated_at": self.generated_at.isoformat(timespec="seconds"),
            "model": self.model.to_dict(),
            "models": {code: m.to_dict() for code, m in self.models.items()},
            "portfolio": self.portfolio.to_dict(),
            "singles": [slip.to_dict() for slip in self.singles],
            "multis": [slip.to_dict() for slip in self.multis],
            "same_game": [slip.to_dict() for slip in self.same_game],
            "fixtures_analysed": len(self.contexts),
            "skipped": self.skipped,
            "config": self.config.to_dict(),
        }
        if include_contexts:
            payload["previews"] = [
                {**context.to_dict(), "preview": match_preview(context)}
                for context in sorted(self.contexts.values(),
                                      key=lambda c: (c.fixture.date, c.fixture.home))
            ]
        return payload


class Engine:
    """Ties the modelling, market and staking layers together."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

    # -- modelling ------------------------------------------------------
    def fit(self, matches: Sequence[Match], as_of: date | None = None) -> FittedModel:
        return DixonColesModel(self.config.model).fit(matches, as_of=as_of)

    def context_for(
        self,
        fixture: Fixture,
        model: FittedModel,
        history: Sequence[Match],
    ) -> FixtureContext:
        """Price one fixture and gather everything needed to explain it."""
        cfg = self.config
        matrix = model.score_matrix(fixture.home, fixture.away)
        model_probs = price_markets(matrix, cfg.selection.markets, cfg.selection.ou_lines)
        quotes = market_view(fixture, cfg.market, cfg.selection.ou_lines)
        blended = blend_market_set(model_probs, quotes, cfg.market.model_weight,
                                   cfg.selection.ou_lines)
        # Fold the blended view back into the joint distribution so that
        # correct scores and same-game combinations agree with the prices we
        # are actually betting on.
        calibrated = calibrate_matrix(matrix, blended)

        return FixtureContext(
            fixture=fixture,
            matrix=calibrated,
            model_probs=model_probs,
            blended_probs=blended,
            quotes=quotes,
            home_rating=model.rating(fixture.home),
            away_rating=model.rating(fixture.away),
            home_form=team_form(history, fixture.home, before=fixture.date),
            away_form=team_form(history, fixture.away, before=fixture.date),
            home_venue_form=team_form(history, fixture.home, before=fixture.date, venue="home"),
            away_venue_form=team_form(history, fixture.away, before=fixture.date, venue="away"),
            head_to_head=head_to_head(history, fixture.home, fixture.away, before=fixture.date),
            data_confidence=model.data_confidence(fixture.home, fixture.away),
        )

    # -- full slate -----------------------------------------------------
    def build_slate(
        self,
        matches: Sequence[Match],
        fixtures: Sequence[Fixture],
        as_of: date | None = None,
        model: FittedModel | None = None,
    ) -> Slate:
        """Price one league's card."""
        return self.build_multi_slate(
            [(matches, fixtures)], as_of=as_of,
            models=[model] if model is not None else None,
        )

    def build_multi_slate(
        self,
        groups: Sequence[tuple[Sequence[Match], Sequence[Fixture]]],
        as_of: date | None = None,
        models: Sequence[FittedModel] | None = None,
    ) -> Slate:
        """Price several leagues into one card.

        Each division gets its own fit — attack and defence ratings are only
        meaningful relative to the league they were estimated in, and a
        Championship +0.3 is not a Premier League +0.3. Combination bets and
        staking then run across the whole card, since legs in different
        matches are independent regardless of which league they are in.
        """
        cfg = self.config

        contexts: dict[str, FixtureContext] = {}
        candidates: list[Candidate] = []
        matrices: dict[str, np.ndarray] = {}
        skipped: list[dict] = []
        fitted: dict[str, FittedModel] = {}
        primary: FittedModel | None = None

        for index, (matches, fixtures) in enumerate(groups):
            if not matches:
                continue
            model = (models[index] if models and index < len(models) and models[index]
                     else self.fit(matches, as_of=as_of))
            league = fixtures[0].league if fixtures else (matches[0].league or f"g{index}")
            fitted[league] = model
            if primary is None:
                primary = model

            # A thin sample is exactly when the model produces its largest and
            # least reliable disagreements, so the bar rises automatically.
            floor = (max(cfg.selection.min_confidence,
                         cfg.selection.thin_sample_min_confidence)
                     if model.thin_sample else cfg.selection.min_confidence)

            for fixture in fixtures:
                if not fixture.odds:
                    skipped.append({"fixture": f"{fixture.home} v {fixture.away}",
                                    "reason": "no odds available"})
                    continue
                if cfg.selection.require_known_teams:
                    unknown = [team for team in (fixture.home, fixture.away)
                               if not model.knows(team)]
                    if unknown:
                        # Pricing this would use league-average ratings and look
                        # exactly as confident as a real read. Usually a team-name
                        # mismatch between the odds source and the results source.
                        skipped.append({
                            "fixture": f"{fixture.home} v {fixture.away}",
                            "reason": f"model has never seen {', '.join(unknown)} — "
                                      "check team name matching between your odds and "
                                      "results sources",
                        })
                        continue
                context = self.context_for(fixture, model, matches)
                if not context.quotes:
                    skipped.append({"fixture": f"{fixture.home} v {fixture.away}",
                                    "reason": "no complete market could be devigged"})
                    continue
                contexts[fixture.key] = context
                matrices[fixture.key] = context.matrix
                candidates.extend(
                    build_candidates(
                        fixture=fixture,
                        blended_probs=context.blended_probs,
                        model_probs=context.model_probs,
                        quotes=context.quotes,
                        config=cfg,
                        data_confidence=context.data_confidence,
                        min_confidence=floor,
                    )
                )

        if primary is None:
            raise ValueError("no league supplied any match history to fit on")
        model = primary

        candidates.sort(key=lambda c: (-c.confidence, -c.edge))
        shortlist = candidates[: cfg.selection.max_singles]

        singles = [
            Slip(
                kind="Single",
                legs=[candidate],
                probability=candidate.probability,
                odds=candidate.odds,
                edge=candidate.edge,
                kelly=kelly_fraction(candidate.probability, candidate.odds),
                log_growth=0.0,
                profile=candidate.tier,
            )
            for candidate in shortlist
        ]
        for slip in singles:
            slip.log_growth = expected_log_growth(
                slip.probability, slip.odds,
                min(slip.kelly * cfg.staking.kelly_fraction, cfg.staking.max_stake_pct),
            )

        multis = build_multis(shortlist, cfg)
        same_game = build_same_game_slips(shortlist, matrices, cfg)

        all_slips = [*singles, *multis, *same_game]
        portfolio = assign_stakes(all_slips, cfg.staking)

        for slip in all_slips:
            slip.analysis = analyse_slip(slip, contexts)
            if slip.size > 1:
                slip.leg_analysis = [
                    analyse_candidate(leg, contexts[leg.fixture_key]) for leg in slip.legs
                ]

        return Slate(
            generated_at=datetime.now(),
            model=model,
            models=fitted,
            contexts=contexts,
            singles=singles,
            multis=multis,
            same_game=same_game,
            portfolio=portfolio,
            config=cfg,
            skipped=skipped,
        )
