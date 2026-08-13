"""Core value objects.

A *selection* is identified by a canonical string id so that odds, model
probabilities and bet slips can all be keyed the same way:

    1X2:H      home win            DC:1X    home win or draw
    1X2:D      draw                DC:12    either team wins
    1X2:A      away win            DC:X2    draw or away win
    OU2.5:O    over 2.5 goals      BTTS:Y   both teams to score
    OU2.5:U    under 2.5 goals     BTTS:N   not both teams to score
    CS:2-1     correct score 2-1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable


@dataclass(frozen=True)
class Selection:
    """One outcome that can be backed."""

    market: str           # "1X2", "DC", "OU", "BTTS", "CS"
    pick: str             # "H", "1X", "O", "Y", "2-1", ...
    line: float | None = None   # only meaningful for OU

    @property
    def id(self) -> str:
        if self.line is not None:
            return f"{self.market}{self.line:g}:{self.pick}"
        return f"{self.market}:{self.pick}"

    @staticmethod
    def parse(selection_id: str) -> "Selection":
        head, _, pick = selection_id.partition(":")
        if not pick:
            raise ValueError(f"malformed selection id: {selection_id!r}")
        # Longest first, so "1X2" is never read as market "1X" with line 2.
        for market in ("BTTS", "1X2", "DC", "OU", "CS"):
            if head == market:
                return Selection(market, pick)
            if head.startswith(market):
                suffix = head[len(market):]
                try:
                    return Selection(market, pick, float(suffix))
                except ValueError:
                    break
        raise ValueError(f"malformed selection id: {selection_id!r}")

    def describe(self, home: str, away: str) -> str:
        """Human-readable name of the bet, e.g. 'Arsenal to win'."""
        if self.market == "1X2":
            return {
                "H": f"{home} to win",
                "D": "Draw",
                "A": f"{away} to win",
            }[self.pick]
        if self.market == "DC":
            return {
                "1X": f"{home} or Draw (double chance)",
                "12": "Either team to win (no draw)",
                "X2": f"{away} or Draw (double chance)",
            }[self.pick]
        if self.market == "OU":
            side = "Over" if self.pick == "O" else "Under"
            return f"{side} {self.line:g} goals"
        if self.market == "BTTS":
            return "Both teams to score" if self.pick == "Y" else "Both teams to score - No"
        if self.market == "CS":
            return f"Correct score {self.pick}"
        return f"{self.market} {self.pick}"

    def short(self) -> str:
        if self.market == "OU":
            return f"{'Over' if self.pick == 'O' else 'Under'} {self.line:g}"
        if self.market == "BTTS":
            return "BTTS Yes" if self.pick == "Y" else "BTTS No"
        return f"{self.market} {self.pick}"


@dataclass(frozen=True)
class Match:
    """A played match with its result and, where available, closing prices."""

    date: date
    league: str
    home: str
    away: str
    home_goals: int
    away_goals: int
    # Sharpest available closing price per selection id (used as the market's
    # opinion when backtesting).
    closing_odds: dict[str, float] = field(default_factory=dict)
    # Best price across all books (what you could actually have taken).
    best_odds: dict[str, float] = field(default_factory=dict)

    @property
    def total_goals(self) -> int:
        return self.home_goals + self.away_goals

    @property
    def result(self) -> str:
        if self.home_goals > self.away_goals:
            return "H"
        if self.home_goals < self.away_goals:
            return "A"
        return "D"


@dataclass(frozen=True)
class Fixture:
    """An upcoming match to be priced."""

    date: date
    league: str
    home: str
    away: str
    kickoff: str | None = None
    # Best available price per selection id.
    odds: dict[str, float] = field(default_factory=dict)
    # Sharp reference price per selection id (Pinnacle-style, or market
    # average). Falls back to `odds` when absent.
    sharp_odds: dict[str, float] = field(default_factory=dict)
    # How many books contributed a price, per selection id.
    book_counts: dict[str, int] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.date.isoformat()}|{self.home}|{self.away}"

    def sharp_or_best(self, selection_id: str) -> float | None:
        return self.sharp_odds.get(selection_id) or self.odds.get(selection_id)


def teams_in(matches: Iterable[Match]) -> list[str]:
    names: set[str] = set()
    for match in matches:
        names.add(match.home)
        names.add(match.away)
    return sorted(names)
