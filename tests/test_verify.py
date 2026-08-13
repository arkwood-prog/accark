"""Price modes, the pessimistic transform and the verification ladder."""

import pytest

from bettingedge.config import Config
from bettingedge.data import synthetic
from bettingedge.data.footballdata import (
    PRICE_MODES,
    parse_results_csv,
    resolve_price_mode,
)
from bettingedge.verify import FAIL, PASS, WARN, sharp_only, verify

# One row carrying every snapshot: pre-closing (B365/PS/Max/Avg) and closing
# (B365C/PSC/MaxC/AvgC), for both 1X2 and over/under.
FULL_CSV = """Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A,PSH,PSD,PSA,MaxH,MaxD,MaxA,AvgH,AvgD,AvgA,PSCH,PSCD,PSCA,MaxCH,MaxCD,MaxCA,AvgCH,AvgCD,AvgCA,P>2.5,P<2.5,Max>2.5,Max<2.5,Avg>2.5,Avg<2.5,PC>2.5,PC<2.5,MaxC>2.5,MaxC<2.5
E0,16/08/2024,Home Team,Away Team,1,0,1.80,3.75,4.50,1.82,3.78,4.55,1.88,3.90,4.75,1.79,3.70,4.40,1.83,3.80,4.60,1.90,3.95,4.80,1.81,3.76,4.52,1.90,1.95,1.98,2.00,1.88,1.92,1.92,1.93,1.99,2.01
"""


def test_every_mode_is_resolvable():
    for key in PRICE_MODES:
        assert resolve_price_mode(key).key == key
    assert resolve_price_mode(None).key == "best-closing"


def test_unknown_mode_names_the_valid_ones():
    with pytest.raises(ValueError, match="unknown price mode"):
        resolve_price_mode("whatever")


def test_best_closing_mode_reads_the_closing_maximum():
    match = parse_results_csv(FULL_CSV, "E0", "best-closing")[0]
    assert match.best_odds["1X2:H"] == 1.90      # MaxCH
    assert match.closing_odds["1X2:H"] == 1.83   # PSCH
    assert match.best_odds["OU2.5:O"] == 1.99    # MaxC>2.5
    assert match.closing_odds["OU2.5:O"] == 1.92  # PC>2.5


def test_early_mode_bets_pre_close_and_scores_against_the_close():
    """This is the pairing that makes closing-line value mean anything."""
    match = parse_results_csv(FULL_CSV, "E0", "early")[0]
    assert match.best_odds["1X2:H"] == 1.88      # MaxH, taken before the close
    assert match.closing_odds["1X2:H"] == 1.83   # PSCH, the line to beat
    assert match.best_odds["OU2.5:O"] == 1.98    # Max>2.5
    assert match.closing_odds["OU2.5:O"] == 1.92  # PC>2.5


def test_sharp_only_mode_gives_you_no_shopping_at_all():
    match = parse_results_csv(FULL_CSV, "E0", "sharp-only")[0]
    assert match.best_odds["1X2:H"] == 1.83
    assert match.best_odds["1X2:H"] == match.closing_odds["1X2:H"]
    assert match.best_odds["OU2.5:O"] == match.closing_odds["OU2.5:O"]


def test_modes_are_ordered_best_to_worst_for_the_bettor():
    prices = {
        mode: parse_results_csv(FULL_CSV, "E0", mode)[0].best_odds["1X2:H"]
        for mode in PRICE_MODES
    }
    assert prices["best-closing"] > prices["early"] > prices["sharp-only"]


def test_modes_fall_back_when_closing_columns_are_missing():
    """Seasons before ~2019 have no closing columns; parsing must still work."""
    early_only = """Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A,MaxH,MaxD,MaxA,AvgH,AvgD,AvgA
E0,16/08/2014,Home Team,Away Team,2,1,1.80,3.75,4.50,1.88,3.90,4.75,1.79,3.70,4.40
"""
    match = parse_results_csv(early_only, "E0", "best-closing")[0]
    assert match.best_odds["1X2:H"] == 1.88      # falls through to MaxH
    assert match.closing_odds["1X2:H"] == 1.79   # falls through to AvgH


def test_early_mode_has_no_sharp_reference_without_closing_columns():
    """It must degrade quietly rather than silently faking a closing line."""
    early_only = """Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,MaxH,MaxD,MaxA,AvgH,AvgD,AvgA
E0,16/08/2014,Home Team,Away Team,2,1,1.88,3.90,4.75,1.79,3.70,4.40
"""
    match = parse_results_csv(early_only, "E0", "early")[0]
    assert match.best_odds["1X2:H"] == 1.88
    assert "1X2:H" not in match.closing_odds


# ------------------------------------------------------------ sharp_only()
def test_sharp_only_transform_removes_the_shopping_edge():
    matches, _ = synthetic.generate(seasons=2, seed=8)
    stripped = sharp_only(matches)
    assert len(stripped) == len(matches)
    for original, rewritten in zip(matches, stripped):
        assert rewritten.best_odds == original.closing_odds
        # The result itself must be untouched.
        assert (rewritten.home_goals, rewritten.away_goals) == \
               (original.home_goals, original.away_goals)


def test_sharp_only_prices_are_never_better_than_the_originals():
    matches, _ = synthetic.generate(seasons=2, seed=8)
    for original, rewritten in zip(matches, sharp_only(matches)):
        for selection, price in rewritten.best_odds.items():
            assert price <= original.best_odds[selection] + 1e-9


def test_sharp_only_drops_matches_with_no_sharp_reference():
    from dataclasses import replace

    matches, _ = synthetic.generate(seasons=2, seed=8)
    blinded = [replace(m, closing_odds={}) for m in matches[:10]] + list(matches[10:])
    assert len(sharp_only(blinded)) == len(matches) - 10


# ------------------------------------------------------------ the ladder
@pytest.fixture(scope="module")
def report():
    matches, _ = synthetic.generate(seasons=4, seed=6)
    return verify(matches, Config(), label="test", train_days=420, refit_every=28)


def test_report_covers_all_five_stages(report):
    assert len(report.stages) == 5
    titles = [stage.title for stage in report.stages]
    assert titles == ["data integrity", "model sanity", "forecast quality",
                      "price realism", "statistical power"]


def test_every_check_has_a_valid_status(report):
    checks = [c for stage in report.stages for c in stage.checks]
    assert checks
    assert all(c.status in (PASS, WARN, FAIL) for c in checks)
    assert all(c.name and c.value and c.expected for c in checks)


def test_clean_synthetic_data_passes_the_integrity_stages(report):
    """Stages 1-2 test the pipeline, and the pipeline is sound here."""
    assert report.stages[0].status != FAIL, [vars(c) for c in report.stages[0].checks]
    assert report.stages[1].status != FAIL, [vars(c) for c in report.stages[1].checks]


def test_report_renders_and_serialises(report):
    text = report.render()
    assert "BETTINGEDGE VERIFICATION" in text
    assert "VERDICT" in text
    for stage in report.stages:
        assert stage.title.upper() in text

    payload = report.to_dict()
    assert payload["status"] in (PASS, WARN, FAIL)
    assert len(payload["stages"]) == 5


def test_verdict_always_points_at_forward_testing(report):
    assert any("paper-trading" in line for line in report.verdict)


def test_a_thin_sample_warns_rather_than_failing(report):
    """Low statistical power is a limit on conclusions, not a broken pipeline."""
    power = report.stages[4]
    sample_check = next(c for c in power.checks if c.name == "bets in sample")
    assert sample_check.status != FAIL


def test_cross_book_prices_do_not_fail_the_overround_check(report):
    """Best-of-N pricing legitimately sums to under 100% — that is not a bug."""
    overround = next(c for c in report.stages[0].checks
                     if c.name == "median 1X2 overround")
    assert overround.status != FAIL
    assert "cross-book" in overround.expected


def test_swapped_home_and_away_is_caught():
    """The check that would catch the most damaging silent data bug."""
    from dataclasses import replace

    matches, _ = synthetic.generate(seasons=3, seed=6)
    flipped = [
        replace(m, home=m.away, away=m.home,
                home_goals=m.away_goals, away_goals=m.home_goals)
        for m in matches
    ]
    stage = verify(flipped, Config(), label="flipped", train_days=420,
                   refit_every=60, run_pessimistic=False).stages[0]
    home_rate = next(c for c in stage.checks if c.name == "home win rate")
    assert home_rate.status in (WARN, FAIL)


def test_broken_scores_are_caught():
    from dataclasses import replace

    matches, _ = synthetic.generate(seasons=3, seed=6)
    absurd = [replace(m, home_goals=9, away_goals=9) for m in matches]
    stage = verify(absurd, Config(), label="broken", train_days=420,
                   refit_every=60, run_pessimistic=False).stages[0]
    goals = next(c for c in stage.checks if c.name == "goals per game")
    assert goals.status == FAIL


def test_verify_stops_early_when_there_is_almost_no_data():
    matches, _ = synthetic.generate(seasons=2, seed=6)
    report = verify(matches[:40], Config(), label="tiny")
    assert len(report.stages) == 1
    assert report.verdict
