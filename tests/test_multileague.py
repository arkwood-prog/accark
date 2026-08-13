"""Confidence floor, automatic thin-sample tightening, and multi-league cards."""

from dataclasses import replace
from datetime import date, timedelta

import pytest

from bettingedge.config import Config, ModelConfig, SelectionConfig
from bettingedge.data import synthetic
from bettingedge.data.schema import Match
from bettingedge.pipeline import Engine


@pytest.fixture(scope="module")
def league_a():
    return synthetic.generate(seasons=3, seed=11)


@pytest.fixture(scope="module")
def league_b():
    """A second division with its own teams and its own price history."""
    matches, fixtures = synthetic.generate(seasons=3, seed=77)
    renamed_matches = [
        replace(m, league="SYN2", home=f"B {m.home}", away=f"B {m.away}") for m in matches
    ]
    renamed_fixtures = [
        replace(f, league="SYN2", home=f"B {f.home}", away=f"B {f.away}") for f in fixtures
    ]
    return renamed_matches, renamed_fixtures


# ---------------------------------------------------------- confidence floor
def test_confidence_floor_removes_weak_bets(league_a):
    matches, fixtures = league_a
    loose = Engine(Config()).build_slate(matches, fixtures)
    strict = Engine(Config(selection=SelectionConfig(min_confidence=80.0))).build_slate(
        matches, fixtures)

    assert len(strict.singles) < len(loose.singles)
    for slip in strict.singles:
        assert slip.legs[0].confidence >= 80.0


def test_a_floor_above_everything_empties_the_card(league_a):
    matches, fixtures = league_a
    slate = Engine(Config(selection=SelectionConfig(min_confidence=101.0))).build_slate(
        matches, fixtures)
    assert slate.singles == []
    assert slate.portfolio.total_staked == 0.0


def test_zero_floor_is_the_default_and_changes_nothing(league_a):
    matches, fixtures = league_a
    assert Config().selection.min_confidence == 0.0
    a = Engine(Config()).build_slate(matches, fixtures)
    b = Engine(Config(selection=SelectionConfig(min_confidence=0.0))).build_slate(
        matches, fixtures)
    assert len(a.singles) == len(b.singles)


# ------------------------------------------------- automatic thin-sample bar
def _thin_history(n_teams: int = 12) -> list[Match]:
    """A finished season followed by a long gap — the August situation."""
    start = date(2024, 1, 1)
    return [
        Match(start + timedelta(days=i * 3), "X", f"T{i % n_teams}",
              f"T{(i + 5) % n_teams}", 2, 1,
              closing_odds={"1X2:H": 2.0, "1X2:D": 3.4, "1X2:A": 3.8},
              best_odds={"1X2:H": 2.1, "1X2:D": 3.5, "1X2:A": 4.0})
        for i in range(200)
    ]


def test_thin_sample_raises_the_floor_on_its_own(league_a):
    """The floor must tighten without the caller having to remember."""
    matches, fixtures = league_a
    config = Config(
        model=ModelConfig(half_life_days=180, max_history_days=0),
        selection=SelectionConfig(min_confidence=0.0, thin_sample_min_confidence=99.0),
    )
    # Age the fixtures far past the history so the decay window is empty.
    stale = [replace(f, date=f.date + timedelta(days=1500)) for f in fixtures]
    aged = Engine(config)
    model = aged.fit(matches, as_of=stale[0].date)
    if not model.thin_sample:
        pytest.skip("fixture did not produce a thin sample")
    slate = aged.build_multi_slate([(matches, stale)], as_of=stale[0].date)
    assert slate.singles == [], "thin sample should have raised the bar to 99"


def test_a_healthy_sample_does_not_trigger_the_tighter_floor(league_a):
    matches, fixtures = league_a
    config = Config(selection=SelectionConfig(thin_sample_min_confidence=99.0))
    slate = Engine(config).build_slate(matches, fixtures)
    assert not slate.model.thin_sample
    assert slate.singles, "a dense sample must not be tightened"


# ---------------------------------------------------------------- multi-league
def test_each_league_gets_its_own_fit(league_a, league_b):
    slate = Engine(Config()).build_multi_slate([league_a, league_b])
    assert len(slate.models) == 2
    assert set(slate.models) == {"SYN", "SYN2"}
    # Separate fits, so separate team vocabularies.
    a_teams = set(slate.models["SYN"].teams)
    b_teams = set(slate.models["SYN2"].teams)
    assert not (a_teams & b_teams)


def test_a_combined_card_covers_fixtures_from_both_leagues(league_a, league_b):
    slate = Engine(Config()).build_multi_slate([league_a, league_b])
    leagues = {leg.league for slip in slate.singles for leg in slip.legs}
    assert leagues == {"SYN", "SYN2"}


def test_combined_card_holds_more_than_either_league_alone(league_a, league_b):
    engine = Engine(Config())
    only_a = engine.build_slate(*league_a)
    combined = engine.build_multi_slate([league_a, league_b])
    assert len(combined.contexts) > len(only_a.contexts)


def test_multis_may_span_leagues(league_a, league_b):
    """Legs in different matches are independent whichever division they are in."""
    slate = Engine(Config()).build_multi_slate([league_a, league_b])
    if not slate.multis:
        pytest.skip("no qualifying multis in this fixture")
    for slip in slate.multis:
        keys = [leg.fixture_key for leg in slip.legs]
        assert len(keys) == len(set(keys))


def test_staking_is_shared_across_the_whole_card(league_a, league_b):
    """The exposure ceiling applies to the portfolio, not per league."""
    config = Config()
    slate = Engine(config).build_multi_slate([league_a, league_b])
    cap = config.staking.max_total_exposure * config.staking.bankroll
    assert slate.portfolio.total_staked <= cap + config.staking.min_stake * 5


def test_build_slate_still_works_and_matches_the_multi_path(league_a):
    engine = Engine(Config())
    single = engine.build_slate(*league_a)
    via_multi = engine.build_multi_slate([league_a])
    assert len(single.singles) == len(via_multi.singles)
    assert single.model.n_matches == via_multi.model.n_matches


def test_an_empty_group_list_is_rejected():
    with pytest.raises(ValueError, match="no league"):
        Engine(Config()).build_multi_slate([])


def test_multi_slate_serialises_every_model(league_a, league_b):
    import json

    slate = Engine(Config()).build_multi_slate([league_a, league_b])
    payload = json.loads(json.dumps(slate.to_dict(), default=str))
    assert set(payload["models"]) == {"SYN", "SYN2"}
    assert payload["model"]["teams"]
