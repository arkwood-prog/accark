"""Selection parsing, market masks and settlement."""

import numpy as np
import pytest

from bettingedge.backtest.engine import settles
from bettingedge.data.schema import Selection
from bettingedge.models.markets import (
    calibrate_matrix,
    joint_probability,
    price_markets,
    probability,
    selection_mask,
)
from bettingedge.models.dixon_coles import score_matrix_from_rates

ALL_IDS = ["1X2:H", "1X2:D", "1X2:A", "DC:1X", "DC:12", "DC:X2",
           "OU0.5:O", "OU2.5:O", "OU2.5:U", "OU3.5:U", "BTTS:Y", "BTTS:N", "CS:2-1"]


@pytest.mark.parametrize("selection_id", ALL_IDS)
def test_selection_id_round_trips(selection_id):
    assert Selection.parse(selection_id).id == selection_id


def test_1x2_is_not_parsed_as_market_1x_with_line_2():
    parsed = Selection.parse("1X2:H")
    assert parsed.market == "1X2"
    assert parsed.line is None


def test_over_under_keeps_its_line():
    parsed = Selection.parse("OU2.5:O")
    assert (parsed.market, parsed.pick, parsed.line) == ("OU", "O", 2.5)


@pytest.mark.parametrize("bad", ["", "nonsense", "ZZ:H", "1X2"])
def test_malformed_ids_are_rejected(bad):
    with pytest.raises(ValueError):
        Selection.parse(bad)


def test_1x2_partitions_the_whole_space():
    size = 8
    masks = [selection_mask(sel, size) for sel in ("1X2:H", "1X2:D", "1X2:A")]
    stacked = np.stack(masks).sum(axis=0)
    assert (stacked == 1).all(), "every scoreline belongs to exactly one 1X2 outcome"


def test_over_under_partitions_the_whole_space():
    size = 8
    over = selection_mask("OU2.5:O", size)
    under = selection_mask("OU2.5:U", size)
    assert not (over & under).any()
    assert (over | under).all()


def test_double_chance_is_the_union_of_its_parts():
    size = 8
    home = selection_mask("1X2:H", size)
    draw = selection_mask("1X2:D", size)
    assert (selection_mask("DC:1X", size) == (home | draw)).all()


@pytest.mark.parametrize(
    "selection_id,home_goals,away_goals,expected",
    [
        ("1X2:H", 2, 1, True), ("1X2:H", 1, 1, False), ("1X2:H", 0, 3, False),
        ("1X2:D", 2, 2, True), ("1X2:D", 2, 1, False),
        ("1X2:A", 0, 1, True), ("1X2:A", 1, 0, False),
        ("DC:1X", 1, 1, True), ("DC:1X", 0, 1, False),
        ("DC:12", 1, 1, False), ("DC:12", 3, 0, True),
        ("DC:X2", 0, 0, True), ("DC:X2", 4, 1, False),
        ("OU2.5:O", 2, 1, True), ("OU2.5:O", 1, 1, False),
        ("OU2.5:U", 1, 1, True), ("OU2.5:U", 3, 0, False),
        ("OU3.5:O", 2, 2, True), ("OU3.5:O", 2, 1, False),
        ("BTTS:Y", 1, 1, True), ("BTTS:Y", 3, 0, False),
        ("BTTS:N", 3, 0, True), ("BTTS:N", 1, 2, False),
        ("CS:2-1", 2, 1, True), ("CS:2-1", 1, 2, False),
    ],
)
def test_settlement_truth_table(selection_id, home_goals, away_goals, expected):
    assert settles(selection_id, home_goals, away_goals) is expected


def test_score_matrix_is_a_distribution():
    matrix = score_matrix_from_rates(1.6, 1.1, -0.05, max_goals=10)
    assert matrix.sum() == pytest.approx(1.0, abs=1e-12)
    assert (matrix >= 0).all()


def test_market_probabilities_are_internally_consistent():
    matrix = score_matrix_from_rates(1.5, 1.2, -0.06)
    probs = price_markets(matrix)
    assert probs["1X2:H"] + probs["1X2:D"] + probs["1X2:A"] == pytest.approx(1.0)
    assert probs["OU2.5:O"] + probs["OU2.5:U"] == pytest.approx(1.0)
    assert probs["BTTS:Y"] + probs["BTTS:N"] == pytest.approx(1.0)
    # Double chance must equal the sum of the 1X2 outcomes it covers.
    assert probs["DC:1X"] == pytest.approx(probs["1X2:H"] + probs["1X2:D"])
    assert probs["DC:X2"] == pytest.approx(probs["1X2:D"] + probs["1X2:A"])
    assert probs["DC:12"] == pytest.approx(probs["1X2:H"] + probs["1X2:A"])


def test_same_game_legs_are_correlated_not_independent():
    """A home win and over 2.5 in one match must not be priced by multiplying."""
    matrix = score_matrix_from_rates(1.9, 0.9, -0.05)
    joint = joint_probability(matrix, ["1X2:H", "OU2.5:O"])
    independent = probability(matrix, "1X2:H") * probability(matrix, "OU2.5:O")
    assert joint != pytest.approx(independent, abs=1e-4)
    # A joint probability can never exceed either of its parts.
    assert joint <= probability(matrix, "1X2:H") + 1e-12
    assert joint <= probability(matrix, "OU2.5:O") + 1e-12


def test_mutually_exclusive_legs_have_zero_joint_probability():
    matrix = score_matrix_from_rates(1.4, 1.4, -0.05)
    assert joint_probability(matrix, ["1X2:H", "1X2:A"]) == pytest.approx(0.0)


def test_calibration_reproduces_the_target_margins():
    matrix = score_matrix_from_rates(1.5, 1.2, -0.06)
    targets = {"1X2:H": 0.50, "1X2:D": 0.25, "1X2:A": 0.25,
               "OU2.5:O": 0.60, "OU2.5:U": 0.40}
    adjusted = calibrate_matrix(matrix, targets)

    assert adjusted.sum() == pytest.approx(1.0, abs=1e-9)
    for selection_id, target in targets.items():
        assert probability(adjusted, selection_id) == pytest.approx(target, abs=1e-3)


def test_calibration_leaves_an_already_matching_matrix_alone():
    matrix = score_matrix_from_rates(1.5, 1.2, -0.06)
    targets = {sel: probability(matrix, sel) for sel in ("1X2:H", "1X2:D", "1X2:A")}
    adjusted = calibrate_matrix(matrix, targets)
    assert np.allclose(adjusted, matrix, atol=1e-9)
