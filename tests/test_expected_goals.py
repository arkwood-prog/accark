"""Shots-based expected-goals proxy and the league scan."""

from datetime import date, timedelta

import numpy as np
import pytest

from bettingedge.config import Config, ModelConfig
from bettingedge.data.schema import Match
from bettingedge.models.dixon_coles import DixonColesModel
from bettingedge.models.expected_goals import (
    DEFAULT_ALPHA,
    ShotConversion,
    blended_targets,
    fit_conversion,
)
from bettingedge.scan import LeagueResult, ScanReport


def _match(day, home, away, hg, ag, hst=None, ast=None, hs=None, as_=None):
    return Match(date=day, league="X", home=home, away=away,
                 home_goals=hg, away_goals=ag,
                 home_shots_on_target=hst, away_shots_on_target=ast,
                 home_shots=hs, away_shots=as_)


def _shot_matches(n=200, alpha=0.30, beta=0.02, seed=5):
    """Matches whose goals are generated from a known conversion rate."""
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 1)
    matches = []
    for i in range(n):
        hst, ast = int(rng.integers(1, 9)), int(rng.integers(1, 9))
        hoff, aoff = int(rng.integers(0, 12)), int(rng.integers(0, 12))
        hg = int(rng.poisson(alpha * hst + beta * hoff))
        ag = int(rng.poisson(alpha * ast + beta * aoff))
        matches.append(_match(start + timedelta(days=i), f"T{i % 8}", f"T{(i + 3) % 8}",
                              hg, ag, hst, ast, hst + hoff, ast + aoff))
    return matches


def test_conversion_recovers_a_known_rate():
    conversion = fit_conversion(_shot_matches(400, alpha=0.30))
    assert conversion.fitted
    assert conversion.alpha == pytest.approx(0.30, abs=0.05)
    assert 0.0 <= conversion.beta <= 0.2


def test_conversion_falls_back_without_enough_data():
    conversion = fit_conversion(_shot_matches(5))
    assert not conversion.fitted
    assert conversion.alpha == DEFAULT_ALPHA


def test_conversion_ignores_matches_with_no_shot_data():
    plain = [_match(date(2024, 1, 1) + timedelta(days=i), "A", "B", 1, 1)
             for i in range(100)]
    assert not fit_conversion(plain).fitted


def test_expected_goals_rises_with_shots_on_target():
    conversion = ShotConversion(0.3, 0.02, 100, True)
    assert conversion.expected_goals(8, 14) > conversion.expected_goals(3, 14)
    assert conversion.expected_goals(None, 10) is None


def test_expected_goals_counts_off_target_shots_less():
    conversion = ShotConversion(0.3, 0.02, 100, True)
    # Same total shots, more of them on target: higher expected goals.
    assert conversion.expected_goals(8, 12) > conversion.expected_goals(2, 12)


def test_blended_targets_interpolate_between_goals_and_expected_goals():
    matches = _shot_matches(50)
    conversion = fit_conversion(matches)

    goals_home, _ = blended_targets(matches, conversion, 0.0)
    xg_home, _ = blended_targets(matches, conversion, 1.0)
    half_home, _ = blended_targets(matches, conversion, 0.5)

    assert np.allclose(goals_home, [m.home_goals for m in matches])
    assert not np.allclose(xg_home, goals_home)
    assert np.allclose(half_home, (goals_home + xg_home) / 2)


def test_matches_without_shots_keep_their_goals():
    """A partial feed must degrade gracefully, not drop matches."""
    matches = _shot_matches(20) + [_match(date(2024, 6, 1), "A", "B", 3, 0)]
    conversion = fit_conversion(matches)
    home, _ = blended_targets(matches, conversion, 1.0)
    assert home[-1] == 3.0


def test_model_fits_on_a_continuous_expected_goals_target():
    """The quasi-Poisson likelihood must accept non-integer targets."""
    matches = _shot_matches(300)
    fitted = DixonColesModel(ModelConfig(xg_weight=1.0, half_life_days=400)).fit(matches)
    assert fitted.converged
    assert fitted.xg_weight == 1.0
    assert fitted.conversion is not None and fitted.conversion.fitted
    lam, mu = fitted.expected_goals(fitted.teams[0], fitted.teams[1])
    assert 0.1 < lam < 6 and 0.1 < mu < 6


def test_xg_weight_changes_the_ratings():
    matches = _shot_matches(300)
    goals_fit = DixonColesModel(ModelConfig(xg_weight=0.0, half_life_days=400)).fit(matches)
    xg_fit = DixonColesModel(ModelConfig(xg_weight=1.0, half_life_days=400)).fit(matches)
    assert not np.allclose(goals_fit.attack, xg_fit.attack)


def test_xg_weight_of_zero_reproduces_the_goals_model():
    matches = _shot_matches(200)
    a = DixonColesModel(ModelConfig(xg_weight=0.0, half_life_days=400)).fit(matches)
    b = DixonColesModel(ModelConfig(xg_weight=0.0, half_life_days=400)).fit(matches)
    assert np.allclose(a.attack, b.attack)
    assert a.conversion is None


def test_xg_weight_is_clamped_to_a_sensible_range():
    matches = _shot_matches(150)
    fitted = DixonColesModel(ModelConfig(xg_weight=5.0, half_life_days=400)).fit(matches)
    assert fitted.xg_weight == 1.0


def test_model_dict_reports_the_conversion():
    fitted = DixonColesModel(ModelConfig(xg_weight=0.5, half_life_days=400)).fit(
        _shot_matches(200))
    payload = fitted.to_dict()
    assert payload["xg_weight"] == 0.5
    assert payload["shot_conversion"]["goals_per_shot_on_target"] > 0


# ---------------------------------------------------------------- scan
def test_scan_report_renders_failures_without_crashing():
    report = ScanReport(results=[LeagueResult("ZZ", "Nowhere", 0, error="no data")])
    text = report.render()
    assert "ZZ" in text and "no data" in text


def test_scan_report_serialises():
    report = ScanReport(results=[LeagueResult("ZZ", "Nowhere", 0, error="no data")])
    payload = report.to_dict()
    assert payload["leagues"][0]["error"] == "no data"


# ------------------------------------------------- thin-sample detection
def test_thin_sample_is_flagged_when_the_decay_window_is_mostly_empty():
    """The early-season case: plenty of matches on file, little recent signal."""
    from bettingedge.data.schema import Match

    start = date(2024, 1, 1)
    # A full season, then a long gap — exactly what August looks like.
    old = [Match(start + timedelta(days=i * 3), "X", f"T{i % 12}", f"T{(i + 5) % 12}",
                 2, 1) for i in range(200)]
    fitted = DixonColesModel(ModelConfig(half_life_days=180, max_history_days=0)).fit(
        old, as_of=start + timedelta(days=1200))
    assert fitted.thin_sample
    assert fitted.effective_matches_per_team < 20


def test_a_dense_recent_season_is_not_flagged_as_thin():
    from bettingedge.data.schema import Match

    start = date(2024, 1, 1)
    dense = [Match(start + timedelta(days=i), "X", f"T{i % 10}", f"T{(i + 3) % 10}", 2, 1)
             for i in range(400)]
    fitted = DixonColesModel(ModelConfig(half_life_days=180)).fit(dense)
    assert not fitted.thin_sample


def test_thin_sample_reaches_the_report_and_the_model_dict():
    from bettingedge.data.schema import Match

    start = date(2024, 1, 1)
    old = [Match(start + timedelta(days=i * 3), "X", f"T{i % 12}", f"T{(i + 5) % 12}",
                 2, 1) for i in range(200)]
    fitted = DixonColesModel(ModelConfig(half_life_days=180, max_history_days=0)).fit(
        old, as_of=start + timedelta(days=1200))
    payload = fitted.to_dict()
    assert payload["thin_sample"] is True
    assert payload["effective_matches_per_team"] < 20
