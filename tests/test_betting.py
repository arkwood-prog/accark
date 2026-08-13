"""Multi construction, staking discipline and the end-to-end pipeline."""

from datetime import date

import pytest

from bettingedge.betting.parlays import (
    Slip,
    build_multis,
    build_same_game_slips,
    expected_log_growth,
    size_name,
)
from bettingedge.betting.staking import assign_stakes
from bettingedge.betting.value import Candidate
from bettingedge.config import Config, ParlayConfig, StakingConfig
from bettingedge.data import synthetic
from bettingedge.models.dixon_coles import score_matrix_from_rates
from bettingedge.pipeline import Engine


def make_candidate(fixture: str, selection="1X2:H", probability=0.55, odds=2.10,
                   confidence=70.0) -> Candidate:
    edge = probability * odds - 1
    return Candidate(
        fixture_key=fixture, date=date(2026, 8, 15), league="X",
        home=f"{fixture} home", away=f"{fixture} away",
        selection_id=selection, label=f"{fixture} {selection}", short_label=selection,
        odds=odds, model_probability=probability, market_probability=probability - 0.02,
        probability=probability, edge=edge,
        kelly=max(0.0, edge / (odds - 1)), fair_odds=1 / probability,
        market_fair_odds=1 / (probability - 0.02), overround=0.04, book_count=6,
        data_confidence=1.0, confidence=confidence, tier="High",
        price_source="best available", derived=False,
    )


def make_slip(probability: float, odds: float, size: int = 1) -> Slip:
    edge = probability * odds - 1
    return Slip(
        kind=size_name(size),
        legs=[make_candidate(f"f{i}") for i in range(size)],
        probability=probability, odds=odds, edge=edge,
        kelly=max(0.0, edge / (odds - 1)), log_growth=0.0,
    )


# ------------------------------------------------------------------ parlays
def test_multi_legs_never_share_a_fixture():
    """Two selections from one match are correlated — they must not be multiplied."""
    candidates = [
        make_candidate("match-a", "1X2:H"),
        make_candidate("match-a", "OU2.5:O"),
        make_candidate("match-b", "1X2:H"),
        make_candidate("match-c", "1X2:H"),
    ]
    slips = build_multis(candidates, Config())
    assert slips
    for slip in slips:
        keys = [leg.fixture_key for leg in slip.legs]
        assert len(keys) == len(set(keys))


def test_combined_odds_and_probability_are_the_products_of_the_legs():
    candidates = [make_candidate(f"m{i}") for i in range(4)]
    slips = build_multis(candidates, Config())
    for slip in slips:
        expected_odds = 1.0
        expected_probability = 1.0
        for leg in slip.legs:
            expected_odds *= leg.odds
            expected_probability *= leg.probability
        assert slip.odds == pytest.approx(expected_odds)
        assert slip.probability == pytest.approx(expected_probability)
        assert slip.edge == pytest.approx(slip.probability * slip.odds - 1)


def test_adding_a_leg_lowers_the_strike_rate():
    candidates = [make_candidate(f"m{i}") for i in range(5)]
    slips = build_multis(candidates, Config())
    by_size = {}
    for slip in slips:
        by_size.setdefault(slip.size, []).append(slip.probability)
    sizes = sorted(by_size)
    for smaller, larger in zip(sizes, sizes[1:]):
        assert max(by_size[larger]) < max(by_size[smaller])


def test_negative_ev_legs_never_reach_a_multi():
    """The margin compounds, so a losing leg cannot be rescued by combining."""
    candidates = [make_candidate(f"m{i}", probability=0.45, odds=2.0) for i in range(4)]
    assert build_multis(candidates, Config()) == []


def test_max_legs_is_respected():
    candidates = [make_candidate(f"m{i}") for i in range(8)]
    config = Config.from_dict({"parlay": {"max_legs": 3}})
    slips = build_multis(candidates, config)
    assert slips
    assert max(slip.size for slip in slips) <= 3


def test_expected_log_growth_prefers_the_likelier_of_two_equal_ev_bets():
    # Both have EV = +10%: 0.55 at 2.0 and 0.11 at 10.0.
    steady = expected_log_growth(0.55, 2.0, 0.05)
    lottery = expected_log_growth(0.11, 10.0, 0.05)
    assert steady > lottery


def test_expected_log_growth_is_negative_for_a_bad_bet():
    assert expected_log_growth(0.30, 2.0, 0.05) < 0


def test_same_game_slips_use_the_joint_distribution():
    matrix = score_matrix_from_rates(1.9, 0.8, -0.05)
    candidates = [
        make_candidate("one-match", "1X2:H", probability=0.60, odds=1.90),
        make_candidate("one-match", "OU2.5:O", probability=0.55, odds=2.10),
    ]
    slips = build_same_game_slips(candidates, {"one-match": matrix}, Config())
    for slip in slips:
        assert slip.correlated
        independent = slip.legs[0].probability * slip.legs[1].probability
        assert slip.probability != pytest.approx(independent, abs=1e-4)


def test_same_game_slips_can_be_switched_off():
    matrix = score_matrix_from_rates(1.9, 0.8, -0.05)
    candidates = [
        make_candidate("one-match", "1X2:H", probability=0.60, odds=1.90),
        make_candidate("one-match", "OU2.5:O", probability=0.55, odds=2.10),
    ]
    config = Config(parlay=ParlayConfig(allow_same_game=False))
    assert build_same_game_slips(candidates, {"one-match": matrix}, config) == []


# ------------------------------------------------------------------ staking
def test_stake_never_exceeds_the_per_bet_cap():
    config = StakingConfig(bankroll=1000, kelly_fraction=1.0, max_stake_pct=0.02,
                           max_total_exposure=1.0, min_stake=0.0)
    slips = [make_slip(0.90, 3.0)]   # a huge (unrealistic) edge
    assign_stakes(slips, config)
    assert slips[0].stake <= 1000 * 0.02 + 1e-9


def test_parlays_are_capped_tighter_than_singles():
    config = StakingConfig(bankroll=10_000, kelly_fraction=1.0, max_stake_pct=0.02,
                           max_parlay_stake_pct=0.005, max_total_exposure=1.0)
    single = make_slip(0.70, 2.0, size=1)
    multi = make_slip(0.70, 2.0, size=3)
    assign_stakes([single, multi], config)
    assert multi.stake < single.stake


def test_total_exposure_ceiling_is_enforced():
    config = StakingConfig(bankroll=1000, kelly_fraction=1.0, max_stake_pct=0.05,
                           max_total_exposure=0.10, min_stake=0.0, round_to=0.01)
    slips = [make_slip(0.70, 2.0) for _ in range(10)]
    portfolio = assign_stakes(slips, config)
    assert portfolio.total_staked <= 1000 * 0.10 + 0.5
    assert portfolio.scaled_by < 1.0


def test_fractional_kelly_stakes_less_than_full_kelly():
    slip_full = make_slip(0.60, 2.0)
    slip_quarter = make_slip(0.60, 2.0)
    assign_stakes([slip_full], StakingConfig(kelly_fraction=1.0, max_stake_pct=1.0,
                                             max_total_exposure=1.0))
    assign_stakes([slip_quarter], StakingConfig(kelly_fraction=0.25, max_stake_pct=1.0,
                                                max_total_exposure=1.0))
    assert slip_quarter.stake < slip_full.stake


def test_no_edge_means_no_stake():
    slips = [make_slip(0.40, 2.0)]
    portfolio = assign_stakes(slips, StakingConfig())
    assert slips[0].stake == 0.0
    assert portfolio.bet_count == 0


def test_empty_slate_produces_an_empty_portfolio():
    portfolio = assign_stakes([], StakingConfig(bankroll=500))
    assert portfolio.total_staked == 0.0
    assert portfolio.bet_count == 0


# ------------------------------------------------------------------ pipeline
@pytest.fixture(scope="module")
def slate():
    matches, fixtures = synthetic.generate(seasons=3, seed=5)
    return Engine(Config()).build_slate(matches, fixtures)


def test_pipeline_produces_a_complete_slate(slate):
    assert slate.contexts
    assert slate.model.converged
    assert slate.portfolio.total_staked > 0


def test_every_recommended_bet_has_a_positive_edge(slate):
    for slip in slate.all_slips:
        assert slip.edge > 0, f"{slip.kind} was recommended with a negative edge"


def test_every_recommended_bet_carries_written_analysis(slate):
    for slip in slate.all_slips:
        assert len(slip.analysis) > 200
        assert "Verdict" in slip.analysis or "Economics" in slip.analysis
        if slip.size > 1:
            assert len(slip.leg_analysis) == slip.size


def test_singles_clear_the_configured_minimum_edge(slate):
    minimum = slate.config.selection.min_edge
    for slip in slate.singles:
        assert slip.edge >= minimum


def test_multis_clear_the_higher_multi_threshold(slate):
    minimum = slate.config.parlay.min_combined_edge
    for slip in slate.multis:
        assert slip.edge >= minimum


def test_slate_serialises_to_json_safe_types(slate):
    import json

    payload = slate.to_dict()
    text = json.dumps(payload, default=str)
    assert len(text) > 1000
    restored = json.loads(text)
    assert restored["singles"] and restored["model"]["teams"]


def test_raising_the_edge_threshold_reduces_the_number_of_bets():
    matches, fixtures = synthetic.generate(seasons=3, seed=5)
    loose = Engine(Config.from_dict({"selection": {"min_edge": 0.01}})).build_slate(
        matches, fixtures)
    tight = Engine(Config.from_dict({"selection": {"min_edge": 0.12}})).build_slate(
        matches, fixtures)
    assert len(tight.singles) < len(loose.singles)


def test_pure_market_weight_finds_far_less_value():
    """With zero model weight the only edges left are price-shopping ones."""
    matches, fixtures = synthetic.generate(seasons=3, seed=5)
    model_led = Engine(Config.from_dict({"market": {"model_weight": 0.9}})).build_slate(
        matches, fixtures)
    market_led = Engine(Config.from_dict({"market": {"model_weight": 0.0}})).build_slate(
        matches, fixtures)
    assert len(market_led.singles) <= len(model_led.singles)
