"""Goal modelling and market derivation."""

from .dixon_coles import DixonColesModel, FittedModel, TeamRating, score_matrix_from_rates
from .form import TeamForm, head_to_head, team_form
from .markets import (
    calibrate_matrix,
    expected_goals_from_matrix,
    joint_probability,
    price_markets,
    probability,
    selection_mask,
    top_scorelines,
)

__all__ = [
    "DixonColesModel",
    "FittedModel",
    "TeamRating",
    "score_matrix_from_rates",
    "TeamForm",
    "team_form",
    "head_to_head",
    "calibrate_matrix",
    "expected_goals_from_matrix",
    "joint_probability",
    "price_markets",
    "probability",
    "selection_mask",
    "top_scorelines",
]
