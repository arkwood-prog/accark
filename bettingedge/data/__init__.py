"""Data ingestion: schemas, the football-data.co.uk source, and offline fixtures."""

from .schema import Fixture, Match, Selection, teams_in
from .footballdata import LEAGUES, FootballDataUK, recent_seasons, season_code
from .csvsource import load_fixtures_csv, load_results_csv
from . import synthetic

__all__ = [
    "Fixture",
    "Match",
    "Selection",
    "teams_in",
    "load_fixtures_csv",
    "load_results_csv",
    "LEAGUES",
    "FootballDataUK",
    "recent_seasons",
    "season_code",
    "synthetic",
]
