"""Matching team names across data sources.

This is the least glamorous file in the project and the one most likely to
save you money.

Every odds provider spells teams differently. football-data.co.uk says
"Man United", The Odds API says "Manchester United", API-Football says
"Manchester United" too but "Nott'm Forest" becomes "Nottingham Forest". The
model is fitted on one set of names; the fixtures arrive with another.

If a name fails to match, the model does not error — it quietly treats the
team as league-average and prices the game anyway. You get a full card of
confident-looking recommendations built on nothing. So matching is done
explicitly, it reports what it could not resolve, and it refuses to guess
when two candidates are close.

That last rule matters more than the matching itself: "Man United" and
"Man City" are similar strings. A resolver that returns its best guess
without checking the runner-up will eventually hand you the wrong team.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Sequence

# Words that carry no identifying information and differ between sources.
_NOISE = {
    "fc", "afc", "cf", "sc", "ac", "as", "ss", "us", "ssc", "sv", "sk", "if",
    "club", "cd", "ud", "rc", "cp", "fk", "bk", "nk", "hk", "vfl", "vfb", "tsg",
    "calcio", "de", "futbol", "football",
}

# Curated aliases for names that fuzzy matching should never be trusted with.
# Keys are normalised; values are the football-data.co.uk spelling the model
# is fitted on.
ALIASES: dict[str, str] = {}


def _register(canonical: str, *aliases: str) -> None:
    for alias in (canonical, *aliases):
        ALIASES[_normalise(alias)] = canonical


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))


def _normalise(name: str) -> str:
    """Lower-case, de-accent, drop punctuation and non-identifying words.

    Deliberately does NOT drop "united", "city", "rovers" and friends — those
    are the only thing separating Manchester United from Manchester City.
    """
    text = _strip_accents(str(name)).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t and t not in _NOISE]
    return " ".join(tokens)


@dataclass(frozen=True)
class Resolution:
    """The outcome of trying to match one incoming name."""

    source_name: str
    matched: str | None
    score: float
    method: str            # exact | alias | fuzzy | ambiguous | unmatched
    runner_up: str | None = None

    @property
    def ok(self) -> bool:
        return self.matched is not None

    def describe(self) -> str:
        if self.method == "exact":
            return f"{self.source_name!r} matched exactly"
        if self.method == "alias":
            return f"{self.source_name!r} -> {self.matched!r} (known alias)"
        if self.method == "fuzzy":
            return f"{self.source_name!r} -> {self.matched!r} ({self.score:.0%} similar)"
        if self.method == "ambiguous":
            return (f"{self.source_name!r} is ambiguous: {self.matched!r} and "
                    f"{self.runner_up!r} score too close to call")
        return f"{self.source_name!r} matched nothing in the model"


class TeamResolver:
    """Maps provider team names onto the names the model was fitted on."""

    def __init__(self, known: Iterable[str], threshold: float = 0.72,
                 margin: float = 0.08, extra_aliases: dict[str, str] | None = None):
        self.known = sorted(set(known))
        self.threshold = threshold
        # A match must beat the runner-up by this much to be trusted.
        self.margin = margin
        self._by_normal = {}
        for team in self.known:
            self._by_normal.setdefault(_normalise(team), team)
        self._aliases = dict(ALIASES)
        for alias, canonical in (extra_aliases or {}).items():
            self._aliases[_normalise(alias)] = canonical
        self._cache: dict[str, Resolution] = {}

    def resolve(self, name: str) -> Resolution:
        if name in self._cache:
            return self._cache[name]
        result = self._resolve(name)
        self._cache[name] = result
        return result

    def _resolve(self, name: str) -> Resolution:
        if name in self._by_normal.values() and name in self.known:
            return Resolution(name, name, 1.0, "exact")

        normal = _normalise(name)
        if not normal:
            return Resolution(name, None, 0.0, "unmatched")

        direct = self._by_normal.get(normal)
        if direct:
            return Resolution(name, direct, 1.0, "exact")

        # Curated alias, but only if the model actually knows that team.
        alias_target = self._aliases.get(normal)
        if alias_target:
            resolved = self._by_normal.get(_normalise(alias_target))
            if resolved:
                return Resolution(name, resolved, 1.0, "alias")

        scored = sorted(
            ((SequenceMatcher(None, normal, candidate).ratio(), team)
             for candidate, team in self._by_normal.items()),
            reverse=True,
        )
        if not scored:
            return Resolution(name, None, 0.0, "unmatched")

        best_score, best = scored[0]
        second_score, second = scored[1] if len(scored) > 1 else (0.0, None)

        if best_score < self.threshold:
            return Resolution(name, None, best_score, "unmatched", second)
        if best_score - second_score < self.margin:
            # Too close to call. Refusing here is the whole point: guessing
            # between two similar names silently prices the wrong team.
            return Resolution(name, None, best_score, "ambiguous", second)
        return Resolution(name, best, best_score, "fuzzy", second)


@dataclass
class ReconciliationReport:
    """What happened when a set of fixtures met the model's vocabulary."""

    resolutions: dict[str, Resolution]
    matched_fixtures: int
    dropped_fixtures: int
    dropped_detail: list[str]

    @property
    def unresolved(self) -> list[Resolution]:
        return [r for r in self.resolutions.values() if not r.ok]

    @property
    def fuzzy(self) -> list[Resolution]:
        return [r for r in self.resolutions.values() if r.method == "fuzzy"]

    def render(self) -> str:
        lines = [
            f"Team name matching: {self.matched_fixtures} fixture(s) resolved, "
            f"{self.dropped_fixtures} dropped."
        ]
        if self.fuzzy:
            lines.append("  Matched by similarity — check these are right:")
            lines += [f"    {r.describe()}" for r in sorted(self.fuzzy, key=lambda r: r.score)]
        if self.unresolved:
            lines.append("  Could not resolve:")
            lines += [f"    {r.describe()}" for r in self.unresolved]
            lines.append("  Fix with --team-alias \"Provider Name=Model Name\" "
                         "(repeatable), or check the league code matches.")
        for detail in self.dropped_detail:
            lines.append(f"    dropped: {detail}")
        return "\n".join(lines)


def reconcile_fixtures(fixtures: Sequence, known_teams: Iterable[str],
                       extra_aliases: dict[str, str] | None = None,
                       drop_unresolved: bool = True) -> tuple[list, ReconciliationReport]:
    """Rewrite fixture team names into the model's vocabulary.

    Fixtures whose teams cannot be resolved are dropped by default — pricing a
    match against a team the model has never seen produces a confident number
    with nothing behind it.
    """
    from dataclasses import replace as _replace

    resolver = TeamResolver(known_teams, extra_aliases=extra_aliases)
    resolutions: dict[str, Resolution] = {}
    kept, dropped_detail = [], []

    for fixture in fixtures:
        home = resolver.resolve(fixture.home)
        away = resolver.resolve(fixture.away)
        resolutions[fixture.home] = home
        resolutions[fixture.away] = away

        if home.ok and away.ok:
            kept.append(_replace(fixture, home=home.matched, away=away.matched))
        elif drop_unresolved:
            unknown = [r.source_name for r in (home, away) if not r.ok]
            dropped_detail.append(
                f"{fixture.home} v {fixture.away} (unresolved: {', '.join(unknown)})"
            )
        else:
            kept.append(fixture)

    return kept, ReconciliationReport(
        resolutions=resolutions,
        matched_fixtures=len(kept),
        dropped_fixtures=len(fixtures) - len(kept),
        dropped_detail=dropped_detail,
    )


def parse_alias_arguments(values: Iterable[str] | None) -> dict[str, str]:
    """Turn ``--team-alias "Manchester United=Man United"`` into a mapping."""
    aliases: dict[str, str] = {}
    for value in values or []:
        provider, separator, model = value.partition("=")
        if not separator or not provider.strip() or not model.strip():
            raise ValueError(
                f"bad --team-alias {value!r}; expected \"Provider Name=Model Name\""
            )
        aliases[provider.strip()] = model.strip()
    return aliases


# --------------------------------------------------------------------------
# Curated aliases. Canonical spelling is football-data.co.uk's, because that
# is what the model is normally fitted on.
# --------------------------------------------------------------------------
# England
_register("Man United", "Manchester United", "Manchester Utd", "Man Utd")
_register("Man City", "Manchester City", "Manchester Cty")
_register("Tottenham", "Tottenham Hotspur", "Spurs")
_register("Newcastle", "Newcastle United", "Newcastle Utd")
_register("Wolves", "Wolverhampton Wanderers", "Wolverhampton")
_register("Nott'm Forest", "Nottingham Forest", "Notts Forest", "Nottingham")
_register("Sheffield United", "Sheffield Utd", "Sheff United", "Sheff Utd")
_register("Sheffield Weds", "Sheffield Wednesday", "Sheff Wed", "Sheffield Wed")
_register("West Brom", "West Bromwich Albion", "West Bromwich")
_register("West Ham", "West Ham United")
_register("Leicester", "Leicester City")
_register("Brighton", "Brighton and Hove Albion", "Brighton & Hove Albion")
_register("Bournemouth", "AFC Bournemouth")
_register("Leeds", "Leeds United")
_register("Norwich", "Norwich City")
_register("Ipswich", "Ipswich Town")
_register("Luton", "Luton Town")
_register("Hull", "Hull City")
_register("Stoke", "Stoke City")
_register("Swansea", "Swansea City")
_register("Cardiff", "Cardiff City")
_register("Birmingham", "Birmingham City")
_register("Coventry", "Coventry City")
_register("Derby", "Derby County")
_register("Preston", "Preston North End")
_register("QPR", "Queens Park Rangers")
_register("Blackburn", "Blackburn Rovers")
_register("Bolton", "Bolton Wanderers")
_register("Bristol City", "Bristol City FC")
_register("Huddersfield", "Huddersfield Town")
_register("Plymouth", "Plymouth Argyle")
_register("Rotherham", "Rotherham United")
_register("Oxford", "Oxford United")
_register("Wycombe", "Wycombe Wanderers")
_register("Peterboro", "Peterborough United", "Peterborough")
_register("Milton Keynes Dons", "MK Dons")

# Spain
_register("Ath Madrid", "Atletico Madrid", "Atlético Madrid", "Atletico de Madrid")
_register("Ath Bilbao", "Athletic Bilbao", "Athletic Club")
_register("Real Madrid", "Real Madrid CF")
_register("Barcelona", "FC Barcelona", "Barca")
_register("Sociedad", "Real Sociedad")
_register("Betis", "Real Betis")
_register("Celta", "Celta Vigo", "Celta de Vigo")
_register("Espanol", "Espanyol", "RCD Espanyol")
_register("Vallecano", "Rayo Vallecano")
_register("Alaves", "Deportivo Alaves", "Deportivo Alavés")
_register("Sevilla", "Sevilla FC")
_register("Valencia", "Valencia CF")
_register("Villarreal", "Villarreal CF")
_register("Mallorca", "RCD Mallorca")
_register("Vigo", "Celta Vigo")

# Germany
_register("Bayern Munich", "FC Bayern Munich", "Bayern Munchen", "Bayern München")
_register("Dortmund", "Borussia Dortmund", "BVB")
_register("M'gladbach", "Borussia Monchengladbach", "Borussia Mönchengladbach",
          "Monchengladbach", "Gladbach")
_register("Leverkusen", "Bayer Leverkusen", "Bayer 04 Leverkusen")
_register("Ein Frankfurt", "Eintracht Frankfurt", "Frankfurt")
_register("RB Leipzig", "RasenBallsport Leipzig", "Leipzig")
_register("Hoffenheim", "TSG Hoffenheim", "1899 Hoffenheim")
_register("Stuttgart", "VfB Stuttgart")
_register("Werder Bremen", "Bremen", "SV Werder Bremen")
_register("Wolfsburg", "VfL Wolfsburg")
_register("Mainz", "Mainz 05", "FSV Mainz 05")
_register("Union Berlin", "1. FC Union Berlin")
_register("FC Koln", "FC Cologne", "1. FC Koln", "1. FC Köln", "Cologne")
_register("Schalke 04", "Schalke", "FC Schalke 04")
_register("Hertha", "Hertha Berlin", "Hertha BSC")
_register("Augsburg", "FC Augsburg")
_register("Freiburg", "SC Freiburg")

# Italy
_register("Inter", "Inter Milan", "Internazionale", "FC Internazionale Milano")
_register("Milan", "AC Milan")
_register("Juventus", "Juventus FC", "Juve")
_register("Napoli", "SSC Napoli")
_register("Roma", "AS Roma")
_register("Lazio", "SS Lazio")
_register("Atalanta", "Atalanta BC")
_register("Fiorentina", "ACF Fiorentina")
_register("Verona", "Hellas Verona")
_register("Torino", "Torino FC")
_register("Bologna", "Bologna FC")
_register("Udinese", "Udinese Calcio")

# France
_register("Paris SG", "Paris Saint Germain", "Paris Saint-Germain", "PSG")
_register("Marseille", "Olympique Marseille", "Olympique de Marseille")
_register("Lyon", "Olympique Lyonnais", "Olympique Lyon")
_register("St Etienne", "Saint Etienne", "AS Saint-Etienne", "Saint-Étienne")
_register("Nice", "OGC Nice")
_register("Lille", "LOSC Lille")
_register("Rennes", "Stade Rennais", "Stade Rennes")
_register("Monaco", "AS Monaco")
_register("Nantes", "FC Nantes")
_register("Lens", "RC Lens")

# Netherlands / Portugal / Belgium
_register("Ajax", "AFC Ajax", "Ajax Amsterdam")
_register("PSV Eindhoven", "PSV")
_register("Feyenoord", "Feyenoord Rotterdam")
_register("AZ Alkmaar", "AZ")
_register("Sp Lisbon", "Sporting CP", "Sporting Lisbon", "Sporting Clube de Portugal")
_register("Porto", "FC Porto")
_register("Benfica", "SL Benfica")
_register("Braga", "SC Braga", "Sporting Braga")
_register("Club Brugge", "Club Bruges", "Club Brugge KV")
_register("Anderlecht", "RSC Anderlecht")

# Scotland
_register("Celtic", "Celtic FC")
_register("Rangers", "Rangers FC")
_register("Hearts", "Heart of Midlothian")
_register("Hibernian", "Hibs")
_register("Aberdeen", "Aberdeen FC")
