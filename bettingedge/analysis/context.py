"""Everything known about one fixture, gathered in a single object."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.schema import Fixture, Match
from ..market.consensus import MarketQuote
from ..models.dixon_coles import TeamRating
from ..models.form import TeamForm
from ..models.markets import expected_goals_from_matrix, top_scorelines


@dataclass
class FixtureContext:
    """The model's full view of a fixture, ready to be explained or bet on."""

    fixture: Fixture
    matrix: np.ndarray                       # calibrated joint score distribution
    model_probs: dict[str, float]            # pure model
    blended_probs: dict[str, float]          # model blended with the market
    quotes: dict[str, MarketQuote]
    home_rating: TeamRating
    away_rating: TeamRating
    home_form: TeamForm
    away_form: TeamForm
    home_venue_form: TeamForm
    away_venue_form: TeamForm
    head_to_head: list[Match] = field(default_factory=list)
    data_confidence: float = 0.0

    @property
    def expected_goals(self) -> tuple[float, float]:
        return expected_goals_from_matrix(self.matrix)

    @property
    def total_expected_goals(self) -> float:
        home, away = self.expected_goals
        return home + away

    @property
    def likely_scores(self) -> list[tuple[str, float]]:
        return top_scorelines(self.matrix, 4)

    def to_dict(self) -> dict:
        home_xg, away_xg = self.expected_goals
        return {
            "date": self.fixture.date.isoformat(),
            "kickoff": self.fixture.kickoff,
            "league": self.fixture.league,
            "home": self.fixture.home,
            "away": self.fixture.away,
            "expected_goals": {"home": round(home_xg, 2), "away": round(away_xg, 2),
                               "total": round(home_xg + away_xg, 2)},
            "likely_scores": [{"score": s, "probability": round(p, 4)}
                              for s, p in self.likely_scores],
            "model_probabilities": {k: round(v, 4) for k, v in sorted(self.model_probs.items())},
            "blended_probabilities": {k: round(v, 4)
                                      for k, v in sorted(self.blended_probs.items())},
            "market": {k: q.to_dict() for k, q in sorted(self.quotes.items())},
            "ratings": {
                "home": {"attack": round(self.home_rating.attack, 3),
                         "defence": round(self.home_rating.defence, 3),
                         "matches": self.home_rating.matches},
                "away": {"attack": round(self.away_rating.attack, 3),
                         "defence": round(self.away_rating.defence, 3),
                         "matches": self.away_rating.matches},
            },
            "form": {
                "home": self.home_form.to_dict(),
                "away": self.away_form.to_dict(),
                "home_at_home": self.home_venue_form.to_dict(),
                "away_on_road": self.away_venue_form.to_dict(),
            },
            "head_to_head": [
                {"date": m.date.isoformat(), "home": m.home, "away": m.away,
                 "score": f"{m.home_goals}-{m.away_goals}"}
                for m in self.head_to_head
            ],
            "data_confidence": round(self.data_confidence, 3),
        }
