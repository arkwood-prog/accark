"""Descriptive form and situational splits.

These numbers do not feed the Dixon-Coles fit — the model already learns team
strength from results. They exist so a recommendation can be *explained* in
the language a trader actually uses: run of form, goals per game, home/away
split, over rate, both-teams-to-score rate.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from typing import Sequence

from ..data.schema import Match


@dataclass
class TeamForm:
    team: str
    matches: int
    points_per_game: float
    goals_for_per_game: float
    goals_against_per_game: float
    over_2_5_rate: float
    btts_rate: float
    clean_sheet_rate: float
    failed_to_score_rate: float
    form_string: str          # most recent first, e.g. "WWDLW"
    unbeaten_run: int
    winless_run: int
    venue: str                # "all", "home" or "away"

    def to_dict(self) -> dict:
        return asdict(self)


def _empty(team: str, venue: str) -> TeamForm:
    return TeamForm(team, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "", 0, 0, venue)


def team_form(
    matches: Sequence[Match],
    team: str,
    before: date | None = None,
    last_n: int = 8,
    venue: str = "all",
) -> TeamForm:
    """Form over the team's most recent `last_n` matches before `before`."""
    relevant = []
    for match in matches:
        if before is not None and match.date >= before:
            continue
        if match.home == team and venue in ("all", "home"):
            relevant.append((match, True))
        elif match.away == team and venue in ("all", "away"):
            relevant.append((match, False))
    if not relevant:
        return _empty(team, venue)

    relevant.sort(key=lambda pair: pair[0].date)
    window = relevant[-last_n:]

    points = goals_for = goals_against = 0
    overs = btts = clean_sheets = blanks = 0
    letters: list[str] = []

    for match, is_home in window:
        scored = match.home_goals if is_home else match.away_goals
        conceded = match.away_goals if is_home else match.home_goals
        goals_for += scored
        goals_against += conceded
        if scored > conceded:
            points += 3
            letters.append("W")
        elif scored == conceded:
            points += 1
            letters.append("D")
        else:
            letters.append("L")
        if match.total_goals > 2.5:
            overs += 1
        if match.home_goals > 0 and match.away_goals > 0:
            btts += 1
        if conceded == 0:
            clean_sheets += 1
        if scored == 0:
            blanks += 1

    n = len(window)
    recent_first = letters[::-1]

    unbeaten = 0
    for letter in recent_first:
        if letter == "L":
            break
        unbeaten += 1
    winless = 0
    for letter in recent_first:
        if letter == "W":
            break
        winless += 1

    return TeamForm(
        team=team,
        matches=n,
        points_per_game=points / n,
        goals_for_per_game=goals_for / n,
        goals_against_per_game=goals_against / n,
        over_2_5_rate=overs / n,
        btts_rate=btts / n,
        clean_sheet_rate=clean_sheets / n,
        failed_to_score_rate=blanks / n,
        form_string="".join(recent_first),
        unbeaten_run=unbeaten,
        winless_run=winless,
        venue=venue,
    )


def head_to_head(
    matches: Sequence[Match],
    home: str,
    away: str,
    before: date | None = None,
    limit: int = 6,
) -> list[Match]:
    """Previous meetings between the two teams, most recent first."""
    meetings = [
        m for m in matches
        if {m.home, m.away} == {home, away} and (before is None or m.date < before)
    ]
    meetings.sort(key=lambda m: m.date, reverse=True)
    return meetings[:limit]
