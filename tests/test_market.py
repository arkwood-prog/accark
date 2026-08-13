"""Margin removal, fair pricing and the model/market blend."""

import pytest

from bettingedge.betting.value import blend, blend_market_set, expected_value, kelly_fraction
from bettingedge.config import MarketConfig
from bettingedge.data.schema import Fixture
from bettingedge.market.consensus import market_view
from bettingedge.market.devig import (
    devig,
    devig_multiplicative,
    devig_power,
    devig_shin,
    overround,
)
from datetime import date

METHODS = ["multiplicative", "power", "shin"]
THREE_WAY = [2.10, 3.40, 3.70]
TWO_WAY = [1.85, 2.05]


def test_overround_is_measured_correctly():
    assert overround([2.0, 2.0]) == pytest.approx(0.0)
    assert overround([1.90, 1.90]) == pytest.approx(0.05263, abs=1e-4)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("odds", [THREE_WAY, TWO_WAY, [1.2, 6.0, 15.0]])
def test_every_method_returns_a_valid_distribution(method, odds):
    probs = devig(odds, method)
    assert sum(probs) == pytest.approx(1.0, abs=1e-9)
    assert all(0.0 < p < 1.0 for p in probs)


@pytest.mark.parametrize("method", METHODS)
def test_devigging_lowers_every_implied_probability(method):
    """Removing margin can only reduce probabilities on an over-round book."""
    raw = [1 / o for o in THREE_WAY]
    probs = devig(THREE_WAY, method)
    assert all(p < q for p, q in zip(probs, raw))


@pytest.mark.parametrize("method", METHODS)
def test_devigging_preserves_the_favourite_ordering(method):
    probs = devig([1.5, 4.5, 7.0], method)
    assert probs[0] > probs[1] > probs[2]


def test_shin_and_power_shade_longshots_more_than_multiplicative():
    """The point of Shin/power: the margin is not spread evenly."""
    odds = [1.35, 5.00, 11.0]
    naive = devig_multiplicative(odds)
    shin = devig_shin(odds)
    power = devig_power(odds)
    # The longshot loses more probability than under naive normalisation.
    assert shin[-1] < naive[-1]
    assert power[-1] < naive[-1]
    # And the favourite keeps more.
    assert shin[0] > naive[0]


def test_a_fair_book_is_left_alone():
    fair = [3.0, 3.0, 3.0]
    for method in METHODS:
        assert devig(fair, method) == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-9)


@pytest.mark.parametrize("bad", [[2.0], [0.5, 3.0], [1.0, 2.0]])
def test_invalid_price_sets_are_rejected(bad):
    with pytest.raises(ValueError):
        devig(bad, "shin")


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        devig(THREE_WAY, "wishful-thinking")


# ---------------------------------------------------------------- blending
def test_blend_weight_zero_is_the_market():
    assert blend(0.20, 0.55, 0.0) == pytest.approx(0.55)


def test_blend_weight_one_is_the_model():
    assert blend(0.20, 0.55, 1.0) == pytest.approx(0.20)


def test_blend_sits_between_the_two_opinions():
    result = blend(0.30, 0.60, 0.35)
    assert 0.30 < result < 0.60


def test_blended_market_still_sums_to_one():
    fixture = Fixture(
        date=date(2026, 1, 1), league="X", home="A", away="B",
        odds={"1X2:H": 2.30, "1X2:D": 3.40, "1X2:A": 3.20},
        sharp_odds={"1X2:H": 2.25, "1X2:D": 3.35, "1X2:A": 3.10},
        book_counts={"1X2:H": 5, "1X2:D": 5, "1X2:A": 5},
    )
    quotes = market_view(fixture, MarketConfig())
    model_probs = {"1X2:H": 0.50, "1X2:D": 0.25, "1X2:A": 0.25}
    blended = blend_market_set(model_probs, quotes, 0.35)
    total = blended["1X2:H"] + blended["1X2:D"] + blended["1X2:A"]
    assert total == pytest.approx(1.0, abs=1e-9)
    # Double chance is derived, so it must agree with the 1X2 numbers.
    assert blended["DC:1X"] == pytest.approx(blended["1X2:H"] + blended["1X2:D"])


def test_markets_without_a_quote_keep_the_pure_model_number():
    fixture = Fixture(date=date(2026, 1, 1), league="X", home="A", away="B",
                      odds={"1X2:H": 2.0, "1X2:D": 3.5, "1X2:A": 4.0})
    quotes = market_view(fixture, MarketConfig())
    model_probs = {"1X2:H": 0.5, "1X2:D": 0.25, "1X2:A": 0.25, "BTTS:Y": 0.61, "BTTS:N": 0.39}
    blended = blend_market_set(model_probs, quotes, 0.35)
    assert blended["BTTS:Y"] == pytest.approx(0.61)


# ---------------------------------------------------------------- market view
def test_market_view_produces_fair_probabilities_below_implied():
    fixture = Fixture(
        date=date(2026, 1, 1), league="X", home="A", away="B",
        odds={"1X2:H": 2.10, "1X2:D": 3.40, "1X2:A": 3.70},
        sharp_odds={"1X2:H": 2.05, "1X2:D": 3.35, "1X2:A": 3.60},
        book_counts={"1X2:H": 8, "1X2:D": 8, "1X2:A": 8},
    )
    quotes = market_view(fixture, MarketConfig())
    total = sum(quotes[sel].fair_probability for sel in ("1X2:H", "1X2:D", "1X2:A"))
    assert total == pytest.approx(1.0, abs=1e-9)
    assert quotes["1X2:H"].overround > 0
    # Best price is never reported as worse than the sharp price we saw.
    assert quotes["1X2:H"].best_odds >= 2.05


def test_derived_double_chance_is_shaded_below_the_theoretical_price():
    fixture = Fixture(date=date(2026, 1, 1), league="X", home="A", away="B",
                      odds={"1X2:H": 2.00, "1X2:D": 4.00, "1X2:A": 4.00})
    quotes = market_view(fixture, MarketConfig())
    theoretical = 1 / (1 / 2.00 + 1 / 4.00)
    assert quotes["DC:1X"].best_odds < theoretical
    assert quotes["DC:1X"].best_source.startswith("derived")


def test_incomplete_markets_are_skipped():
    """A market missing one outcome cannot be devigged and must be dropped."""
    fixture = Fixture(date=date(2026, 1, 1), league="X", home="A", away="B",
                      odds={"1X2:H": 2.0, "1X2:D": 3.5})
    assert market_view(fixture, MarketConfig()) == {}


# ---------------------------------------------------------------- bet maths
def test_expected_value_is_zero_at_a_fair_price():
    assert expected_value(0.5, 2.0) == pytest.approx(0.0)
    assert expected_value(0.25, 4.0) == pytest.approx(0.0)


def test_expected_value_signs():
    assert expected_value(0.55, 2.0) > 0
    assert expected_value(0.45, 2.0) < 0


def test_kelly_is_zero_without_an_edge():
    assert kelly_fraction(0.5, 2.0) == pytest.approx(0.0)
    assert kelly_fraction(0.40, 2.0) == 0.0


def test_kelly_matches_the_textbook_formula():
    # p=0.6 at even money: f = (bp - q)/b = (0.6 - 0.4)/1 = 0.2
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)
    # p=0.5 at 3.0: f = (2*0.5 - 0.5)/2 = 0.25
    assert kelly_fraction(0.5, 3.0) == pytest.approx(0.25)
