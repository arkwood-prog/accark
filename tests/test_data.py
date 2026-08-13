"""Data loading: football-data.co.uk parsing, CSV import, backtest plumbing."""

from datetime import date

import pytest

from bettingedge.backtest.engine import run_backtest
from bettingedge.config import Config
from bettingedge.data.csvsource import load_fixtures_csv, load_results_csv
from bettingedge.data.footballdata import (
    parse_fixtures_csv,
    parse_results_csv,
    recent_seasons,
    season_code,
)
from bettingedge.data import synthetic

RESULTS_CSV = """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,MaxH,MaxD,MaxA,AvgH,AvgD,AvgA,PSCH,PSCD,PSCA,MaxCH,MaxCD,MaxCA,B365>2.5,B365<2.5,Max>2.5,Max<2.5,Avg>2.5,Avg<2.5
E0,16/08/2024,20:00,Man United,Fulham,1,0,H,1.80,3.75,4.50,1.88,3.90,4.75,1.79,3.70,4.40,1.83,3.80,4.60,1.90,3.95,4.80,1.90,1.95,1.98,2.00,1.88,1.92
E0,17/08/2024,12:30,Ipswich,Liverpool,0,2,A,6.50,4.50,1.50,6.80,4.70,1.55,6.40,4.45,1.49,6.60,4.60,1.52,6.90,4.75,1.57,1.75,2.10,1.80,2.15,1.74,2.08
E0,17/08/2024,15:00,Arsenal,Wolves,2,0,H,1.28,6.00,10.0,1.32,6.20,11.0,1.27,5.90,9.80,1.30,6.10,10.5,1.34,6.30,11.5,1.60,2.35,1.65,2.40,1.59,2.33
"""

FIXTURES_CSV = """Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A,MaxH,MaxD,MaxA,AvgH,AvgD,AvgA,B365>2.5,B365<2.5,Max>2.5,Max<2.5,Avg>2.5,Avg<2.5
E0,24/08/2024,15:00,Chelsea,Everton,1.55,4.20,6.00,1.60,4.40,6.30,1.54,4.15,5.90,1.70,2.15,1.75,2.20,1.69,2.12
D1,24/08/2024,14:30,Bayern Munich,Freiburg,1.20,7.50,13.0,1.24,7.80,14.0,1.19,7.40,12.5,1.40,3.00,1.45,3.10,1.39,2.95
"""


# ---------------------------------------------------------- football-data.co.uk
def test_results_are_parsed_with_scores_and_prices():
    matches = parse_results_csv(RESULTS_CSV, "E0")
    assert len(matches) == 3

    first = matches[0]
    assert (first.home, first.away) == ("Man United", "Fulham")
    assert first.date == date(2024, 8, 16)
    assert (first.home_goals, first.away_goals) == (1, 0)
    assert first.result == "H"
    assert first.total_goals == 1


def test_best_price_is_preferred_over_the_sharp_price():
    """MaxC* is the best closing price; PSC* is Pinnacle's closing line."""
    match = parse_results_csv(RESULTS_CSV, "E0")[0]
    assert match.best_odds["1X2:H"] == 1.90      # MaxCH
    assert match.closing_odds["1X2:H"] == 1.83   # PSCH
    assert match.best_odds["1X2:H"] > match.closing_odds["1X2:H"]


def test_over_under_prices_are_picked_up():
    match = parse_results_csv(RESULTS_CSV, "E0")[0]
    assert "OU2.5:O" in match.best_odds
    assert "OU2.5:U" in match.best_odds


def test_matches_come_back_in_date_order():
    matches = parse_results_csv(RESULTS_CSV, "E0")
    assert [m.date for m in matches] == sorted(m.date for m in matches)


def test_rows_without_a_result_are_skipped():
    broken = RESULTS_CSV + "E0,18/08/2024,15:00,Spurs,Leicester,,,,,,,,,,,,,,,,,,,,,,,\n"
    assert len(parse_results_csv(broken, "E0")) == 3


def test_fixtures_can_be_filtered_by_league():
    assert len(parse_fixtures_csv(FIXTURES_CSV)) == 2
    only_english = parse_fixtures_csv(FIXTURES_CSV, ["E0"])
    assert len(only_english) == 1
    assert only_english[0].home == "Chelsea"


def test_fixture_carries_best_and_sharp_prices():
    fixture = parse_fixtures_csv(FIXTURES_CSV, ["E0"])[0]
    assert fixture.odds["1X2:H"] == 1.60          # MaxH
    assert fixture.sharp_odds["1X2:H"] == 1.54    # AvgH
    assert fixture.sharp_or_best("1X2:H") == 1.54
    assert fixture.book_counts["1X2:H"] >= 1


def test_season_codes():
    assert season_code(2024) == "2425"
    assert season_code(1999) == "9900"
    assert recent_seasons(3, today=date(2025, 3, 1)) == ["2223", "2324", "2425"]
    # Seasons roll over in July, so August is already the new season.
    assert recent_seasons(1, today=date(2025, 8, 1)) == ["2526"]


# ---------------------------------------------------------------- own CSV
def test_own_fixtures_csv_with_friendly_column_names(tmp_path):
    path = tmp_path / "fixtures.csv"
    path.write_text(
        "date,home,away,H,D,A,O2.5,U2.5,BTTS_Y,BTTS_N,sharp_H,sharp_D,sharp_A\n"
        "2026-08-15,Arsenal,Chelsea,2.10,3.50,3.60,1.85,1.95,1.70,2.10,2.05,3.40,3.55\n",
        encoding="utf-8",
    )
    fixtures = load_fixtures_csv(path)
    assert len(fixtures) == 1
    fixture = fixtures[0]
    assert fixture.date == date(2026, 8, 15)
    assert fixture.odds["1X2:H"] == 2.10
    assert fixture.odds["OU2.5:O"] == 1.85
    assert fixture.odds["BTTS:Y"] == 1.70
    assert fixture.sharp_odds["1X2:H"] == 2.05


def test_own_fixtures_csv_with_canonical_ids(tmp_path):
    path = tmp_path / "fixtures.csv"
    path.write_text(
        "date,home,away,1X2:H,1X2:D,1X2:A\n2026-08-15,A,B,2.0,3.5,4.0\n",
        encoding="utf-8",
    )
    fixture = load_fixtures_csv(path)[0]
    assert fixture.odds == {"1X2:H": 2.0, "1X2:D": 3.5, "1X2:A": 4.0}


def test_own_results_csv(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text(
        "date,home,away,home_goals,away_goals,H,D,A\n"
        "2026-01-05,A,B,2,1,2.0,3.5,4.0\n"
        "2026-01-12,B,A,0,0,3.0,3.3,2.4\n",
        encoding="utf-8",
    )
    matches = load_results_csv(path)
    assert len(matches) == 2
    assert matches[0].result == "H"
    assert matches[1].result == "D"
    assert matches[0].best_odds["1X2:H"] == 2.0


def test_unparseable_dates_are_reported_clearly(tmp_path):
    path = tmp_path / "fixtures.csv"
    path.write_text("date,home,away,H,D,A\nAugust 5th,A,B,2.0,3.5,4.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unrecognised date"):
        load_fixtures_csv(path)


# ---------------------------------------------------------------- backtest
@pytest.fixture(scope="module")
def backtest_result():
    matches, _ = synthetic.generate(seasons=3, seed=9)
    return run_backtest(matches, Config(), train_days=420, refit_every_days=28)


def test_backtest_places_bets_and_tracks_a_bankroll(backtest_result):
    assert backtest_result.bets
    assert len(backtest_result.bankroll_curve) > 5
    assert backtest_result.forecasts > 50


def test_backtest_accounting_adds_up(backtest_result):
    expected = sum(
        bet.stake * (bet.odds - 1) if bet.won else -bet.stake
        for bet in backtest_result.bets
    )
    assert backtest_result.profit == pytest.approx(expected, abs=1e-6)
    assert backtest_result.final_bankroll == pytest.approx(
        backtest_result.starting_bankroll + backtest_result.profit, abs=1e-6
    )


def test_backtest_settles_every_bet_correctly(backtest_result):
    from bettingedge.backtest.engine import settles

    for bet in backtest_result.bets[:200]:
        # A won bet must return more than it staked, a lost bet exactly the stake.
        if bet.won:
            assert bet.profit == pytest.approx(bet.stake * (bet.odds - 1))
        else:
            assert bet.profit == pytest.approx(-bet.stake)


def test_backtest_hit_rate_is_plausible_for_the_prices_taken(backtest_result):
    """A wildly off hit rate would mean settlement or pricing is broken."""
    implied = sum(1 / bet.odds for bet in backtest_result.bets) / len(backtest_result.bets)
    assert abs(backtest_result.hit_rate - implied) < 0.10


def test_backtest_scores_both_model_and_market(backtest_result):
    assert backtest_result.model_log_loss > 0
    assert backtest_result.market_log_loss > 0
    # A 3-way market cannot be predicted better than this in football.
    assert 0.6 < backtest_result.model_log_loss < 1.6


def test_backtest_refuses_an_impossible_window():
    matches, _ = synthetic.generate(seasons=2, seed=1)
    with pytest.raises(ValueError):
        run_backtest(matches, Config(), train_days=100_000)


def test_backtest_needs_matches():
    with pytest.raises(ValueError):
        run_backtest([], Config())


# ---------------------------------------------------------------- synthetic
def test_synthetic_generator_is_deterministic():
    first, _ = synthetic.generate(seasons=2, seed=42)
    second, _ = synthetic.generate(seasons=2, seed=42)
    assert [(m.home, m.away, m.home_goals, m.away_goals) for m in first] == \
           [(m.home, m.away, m.home_goals, m.away_goals) for m in second]


def test_synthetic_fixtures_have_no_result_and_carry_prices():
    matches, fixtures = synthetic.generate(seasons=2)
    assert fixtures
    for fixture in fixtures:
        assert fixture.odds and fixture.sharp_odds
        assert fixture.date > max(m.date for m in matches)


def test_synthetic_home_advantage_shows_up_in_the_results():
    matches, _ = synthetic.generate(seasons=4, seed=3)
    home_wins = sum(1 for m in matches if m.result == "H")
    away_wins = sum(1 for m in matches if m.result == "A")
    assert home_wins > away_wins
