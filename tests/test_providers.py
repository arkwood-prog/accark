"""Live odds providers, team-name matching and the unknown-team guard."""

from datetime import date, datetime, timedelta, timezone

import pytest

from bettingedge.config import Config, SelectionConfig
from bettingedge.data import synthetic
from bettingedge.data.providers import PROVIDER_INFO, available_providers, get_provider
from bettingedge.data.providers.base import (
    BookQuote,
    ProviderError,
    assemble_fixture,
    parse_iso_datetime,
    pick_sharp_book,
)
from bettingedge.data.providers.apifootball import APIFootball
from bettingedge.data.providers.theoddsapi import TheOddsAPI
from bettingedge.data.teams import (
    TeamResolver,
    parse_alias_arguments,
    reconcile_fixtures,
)
from bettingedge.data.schema import Fixture
from bettingedge.pipeline import Engine

E0_TEAMS = ["Arsenal", "Aston Villa", "Bournemouth", "Brighton", "Chelsea", "Everton",
            "Fulham", "Ipswich", "Leicester", "Liverpool", "Man City", "Man United",
            "Newcastle", "Nott'm Forest", "Southampton", "Tottenham", "West Ham",
            "Wolves", "Brentford", "Crystal Palace"]


# ---------------------------------------------------------------- resolver
def test_exact_names_resolve_to_themselves():
    resolver = TeamResolver(E0_TEAMS)
    for team in E0_TEAMS:
        assert resolver.resolve(team).matched == team


@pytest.mark.parametrize("provider_name,expected", [
    ("Manchester United", "Man United"),
    ("Manchester City", "Man City"),
    ("Tottenham Hotspur", "Tottenham"),
    ("Wolverhampton Wanderers", "Wolves"),
    ("Nottingham Forest", "Nott'm Forest"),
    ("AFC Bournemouth", "Bournemouth"),
    ("Brighton and Hove Albion", "Brighton"),
    ("West Ham United", "West Ham"),
    ("Newcastle United", "Newcastle"),
    ("Leicester City", "Leicester"),
    ("Ipswich Town", "Ipswich"),
])
def test_common_provider_spellings_resolve(provider_name, expected):
    assert TeamResolver(E0_TEAMS).resolve(provider_name).matched == expected


def test_manchester_clubs_are_never_confused():
    """The failure that would quietly price the wrong team."""
    resolver = TeamResolver(E0_TEAMS)
    assert resolver.resolve("Manchester United").matched == "Man United"
    assert resolver.resolve("Manchester City").matched == "Man City"


def test_ambiguous_names_are_refused_rather_than_guessed():
    """Two equally plausible candidates must produce no match at all."""
    resolver = TeamResolver(["Real Sociedad", "Real Sociedad B"])
    result = resolver.resolve("Real Sociedad C")
    assert result.method in ("ambiguous", "fuzzy")
    if result.method == "ambiguous":
        assert result.matched is None
        assert "ambiguous" in result.describe()


def test_completely_unknown_names_do_not_match():
    result = TeamResolver(E0_TEAMS).resolve("Shakhtar Donetsk")
    assert not result.ok
    assert result.method in ("unmatched", "ambiguous")
    assert "matched nothing" in result.describe() or "ambiguous" in result.describe()


def test_accents_and_punctuation_are_ignored():
    resolver = TeamResolver(["Atletico Madrid", "Malaga"])
    assert resolver.resolve("Atlético Madrid").matched == "Atletico Madrid"
    assert resolver.resolve("Málaga").matched == "Malaga"


def test_suffixes_are_ignored_but_identity_words_are_not():
    resolver = TeamResolver(["Sevilla", "Man United", "Man City"])
    assert resolver.resolve("Sevilla FC").matched == "Sevilla"
    # "United" and "City" carry the identity and must not be stripped.
    assert resolver.resolve("Man United").matched == "Man United"
    assert resolver.resolve("Man City").matched == "Man City"


def test_an_alias_for_a_team_not_in_the_league_does_not_match():
    """Knowing 'Bayern Munich' must not conjure it into the Premier League."""
    assert not TeamResolver(E0_TEAMS).resolve("FC Bayern Munich").ok


def test_custom_aliases_win():
    resolver = TeamResolver(E0_TEAMS, extra_aliases={"The Gunners": "Arsenal"})
    assert resolver.resolve("The Gunners").matched == "Arsenal"


def test_alias_arguments_parse():
    assert parse_alias_arguments(["Manchester United=Man United", "A=B"]) == {
        "Manchester United": "Man United", "A": "B"}
    assert parse_alias_arguments(None) == {}


@pytest.mark.parametrize("bad", ["no-equals-sign", "=Man United", "Manchester United="])
def test_bad_alias_arguments_are_rejected(bad):
    with pytest.raises(ValueError, match="team-alias"):
        parse_alias_arguments([bad])


# ------------------------------------------------------------ reconciliation
def _fixture(home: str, away: str) -> Fixture:
    return Fixture(date=date(2026, 8, 15), league="E0", home=home, away=away,
                   odds={"1X2:H": 2.0, "1X2:D": 3.5, "1X2:A": 4.0})


def test_reconciliation_rewrites_names_into_the_models_vocabulary():
    fixtures = [_fixture("Manchester United", "Tottenham Hotspur")]
    resolved, report = reconcile_fixtures(fixtures, E0_TEAMS)
    assert len(resolved) == 1
    assert (resolved[0].home, resolved[0].away) == ("Man United", "Tottenham")
    assert report.dropped_fixtures == 0
    # The prices must survive the rewrite untouched.
    assert resolved[0].odds == fixtures[0].odds


def test_unresolvable_fixtures_are_dropped_and_reported():
    fixtures = [_fixture("Manchester United", "Some Invented FC")]
    resolved, report = reconcile_fixtures(fixtures, E0_TEAMS)
    assert resolved == []
    assert report.dropped_fixtures == 1
    assert "Some Invented FC" in report.render()
    assert "--team-alias" in report.render()


def test_reconciliation_can_be_told_not_to_drop():
    fixtures = [_fixture("Manchester United", "Some Invented FC")]
    resolved, _ = reconcile_fixtures(fixtures, E0_TEAMS, drop_unresolved=False)
    assert len(resolved) == 1


# -------------------------------------------------------- unknown-team guard
def test_engine_refuses_to_price_a_team_it_has_never_seen():
    """Without this the fixture prices off league average and looks confident."""
    matches, fixtures = synthetic.generate(seasons=2, seed=3)
    from dataclasses import replace

    mystery = replace(fixtures[0], home="Completely Unknown FC")
    slate = Engine(Config()).build_slate(matches, [mystery])

    assert slate.singles == []
    assert slate.skipped
    assert "never seen" in slate.skipped[0]["reason"]
    assert "Completely Unknown FC" in slate.skipped[0]["reason"]


def test_the_guard_can_be_switched_off_deliberately():
    matches, fixtures = synthetic.generate(seasons=2, seed=3)
    from dataclasses import replace

    mystery = replace(fixtures[0], home="Completely Unknown FC")
    config = Config(selection=SelectionConfig(require_known_teams=False))
    slate = Engine(config).build_slate(matches, [mystery])
    assert not slate.skipped


# ------------------------------------------------------------ base helpers
def test_best_price_is_the_maximum_across_books():
    quotes = [
        BookQuote("bet365", {"1X2:H": 2.00, "1X2:D": 3.40, "1X2:A": 4.00}),
        BookQuote("williamhill", {"1X2:H": 2.10, "1X2:D": 3.30, "1X2:A": 3.90}),
        BookQuote("pinnacle", {"1X2:H": 2.05, "1X2:D": 3.45, "1X2:A": 4.10}),
    ]
    fixture = assemble_fixture(date(2026, 8, 15), "E0", "Arsenal", "Chelsea", quotes)
    assert fixture.odds["1X2:H"] == 2.10
    assert fixture.odds["1X2:D"] == 3.45
    assert fixture.odds["1X2:A"] == 4.10
    assert fixture.book_counts["1X2:H"] == 3


def test_pinnacle_is_used_as_the_sharp_reference():
    quotes = [
        BookQuote("bet365", {"1X2:H": 2.00, "1X2:D": 3.40, "1X2:A": 4.00}),
        BookQuote("pinnacle", {"1X2:H": 2.05, "1X2:D": 3.45, "1X2:A": 4.10}),
    ]
    fixture = assemble_fixture(date(2026, 8, 15), "E0", "Arsenal", "Chelsea", quotes)
    assert fixture.sharp_odds["1X2:H"] == 2.05
    assert pick_sharp_book(quotes).book == "pinnacle"


def test_without_a_known_sharp_book_the_thinnest_margin_wins():
    fat = BookQuote("fatbook", {"1X2:H": 1.80, "1X2:D": 3.20, "1X2:A": 3.60})
    thin = BookQuote("thinbook", {"1X2:H": 2.05, "1X2:D": 3.45, "1X2:A": 4.05})
    assert pick_sharp_book([fat, thin]).book == "thinbook"


def test_a_fixture_with_no_usable_prices_is_dropped():
    assert assemble_fixture(date(2026, 8, 15), "E0", "A", "B", []) is None
    assert assemble_fixture(date(2026, 8, 15), "E0", "A", "B",
                            [BookQuote("x", {"1X2:H": 0.5})]) is None


def test_iso_datetimes_parse_with_and_without_zulu():
    assert parse_iso_datetime("2026-08-15T14:00:00Z").tzinfo is not None
    assert parse_iso_datetime("2026-08-15T14:00:00+00:00") is not None
    assert parse_iso_datetime("") is None
    assert parse_iso_datetime("not a date") is None


# ------------------------------------------------------------ The Odds API
def _odds_api_event(commence: str) -> dict:
    return {
        "id": "abc123",
        "sport_key": "soccer_epl",
        "commence_time": commence,
        "home_team": "Manchester United",
        "away_team": "Fulham",
        "bookmakers": [
            {
                "key": "pinnacle", "title": "Pinnacle",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Manchester United", "price": 1.83},
                        {"name": "Fulham", "price": 4.60},
                        {"name": "Draw", "price": 3.80},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "price": 1.90, "point": 2.5},
                        {"name": "Under", "price": 1.95, "point": 2.5},
                        {"name": "Over", "price": 3.10, "point": 3.5},
                        {"name": "Under", "price": 1.38, "point": 3.5},
                    ]},
                ],
            },
            {
                "key": "bet365", "title": "Bet365",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Manchester United", "price": 1.90},
                        {"name": "Fulham", "price": 4.20},
                        {"name": "Draw", "price": 3.75},
                    ]},
                    {"key": "btts", "outcomes": [
                        {"name": "Yes", "price": 1.72},
                        {"name": "No", "price": 2.05},
                    ]},
                ],
            },
        ],
    }


@pytest.fixture
def odds_api(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    return TheOddsAPI()


def test_odds_api_builds_a_fixture_from_an_event(odds_api):
    soon = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    cutoff = datetime.now(timezone.utc) + timedelta(days=7)
    fixture = odds_api._event_to_fixture(_odds_api_event(soon), "E0", cutoff)

    assert fixture is not None
    assert (fixture.home, fixture.away) == ("Manchester United", "Fulham")
    # Best price across the two books.
    assert fixture.odds["1X2:H"] == 1.90
    assert fixture.odds["1X2:A"] == 4.60
    # Pinnacle supplies the sharp reference.
    assert fixture.sharp_odds["1X2:H"] == 1.83
    assert fixture.book_counts["1X2:H"] == 2


def test_odds_api_reads_every_supported_totals_line(odds_api):
    soon = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    cutoff = datetime.now(timezone.utc) + timedelta(days=7)
    fixture = odds_api._event_to_fixture(_odds_api_event(soon), "E0", cutoff)
    assert fixture.odds["OU2.5:O"] == 1.90
    assert fixture.odds["OU2.5:U"] == 1.95
    assert fixture.odds["OU3.5:O"] == 3.10


def test_odds_api_reads_btts_when_a_book_offers_it(odds_api):
    soon = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    cutoff = datetime.now(timezone.utc) + timedelta(days=7)
    fixture = odds_api._event_to_fixture(_odds_api_event(soon), "E0", cutoff)
    assert fixture.odds["BTTS:Y"] == 1.72
    assert fixture.odds["BTTS:N"] == 2.05


def test_odds_api_matches_outcomes_by_team_not_by_position(odds_api):
    """Outcome order is not guaranteed, so ordering must never be assumed."""
    soon = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    event = _odds_api_event(soon)
    event["bookmakers"][0]["markets"][0]["outcomes"].reverse()
    cutoff = datetime.now(timezone.utc) + timedelta(days=7)
    fixture = odds_api._event_to_fixture(event, "E0", cutoff)
    assert fixture.sharp_odds["1X2:H"] == 1.83
    assert fixture.sharp_odds["1X2:A"] == 4.60


def test_odds_api_skips_events_beyond_the_window(odds_api):
    far = (datetime.now(timezone.utc) + timedelta(days=40)).isoformat()
    cutoff = datetime.now(timezone.utc) + timedelta(days=7)
    assert odds_api._event_to_fixture(_odds_api_event(far), "E0", cutoff) is None


@pytest.mark.parametrize("broken", [
    {}, {"home_team": "A"}, {"home_team": "A", "away_team": "B"},
    {"home_team": "A", "away_team": "B", "commence_time": "nonsense"},
    "not even a dict",
])
def test_odds_api_survives_malformed_events(odds_api, broken):
    cutoff = datetime.now(timezone.utc) + timedelta(days=7)
    assert odds_api._event_to_fixture(broken, "E0", cutoff) is None


def test_odds_api_explains_an_unmapped_league(odds_api):
    with pytest.raises(ProviderError, match="no The Odds API sport key"):
        odds_api._sport_key("ZZ")


def test_odds_api_demands_a_key_only_when_it_makes_a_request(monkeypatch):
    """Constructing must work without a key so captured payloads can be replayed;
    the key is required the moment a real request needs one."""
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)

    client = TheOddsAPI()          # no key: fine, nothing has been requested
    with pytest.raises(ProviderError, match="no API key"):
        _ = client.api_key


# ------------------------------------------------------------ API-Football
@pytest.fixture
def api_football(monkeypatch):
    monkeypatch.setenv("API_FOOTBALL_KEY", "test-key")
    return APIFootball()


def test_api_football_parses_a_bookmakers_markets(api_football):
    bookmaker = {
        "name": "Pinnacle",
        "bets": [
            {"name": "Match Winner", "values": [
                {"value": "Home", "odd": "1.83"},
                {"value": "Draw", "odd": "3.80"},
                {"value": "Away", "odd": "4.60"},
            ]},
            {"name": "Goals Over/Under", "values": [
                {"value": "Over 2.5", "odd": "1.90"},
                {"value": "Under 2.5", "odd": "1.95"},
                {"value": "Over 7.5", "odd": "21.0"},
            ]},
            {"name": "Both Teams Score", "values": [
                {"value": "Yes", "odd": "1.72"},
                {"value": "No", "odd": "2.05"},
            ]},
            {"name": "Some Market We Do Not Model", "values": [
                {"value": "Whatever", "odd": "2.00"},
            ]},
        ],
    }
    prices = api_football._book_prices(bookmaker)
    assert prices["1X2:H"] == 1.83
    assert prices["1X2:A"] == 4.60
    assert prices["OU2.5:O"] == 1.90
    assert prices["BTTS:Y"] == 1.72
    # Lines the engine does not price are ignored, not guessed at.
    assert "OU7.5:O" not in prices


def test_api_football_indexes_odds_by_fixture(api_football):
    pages = [{
        "fixture": {"id": 12345},
        "bookmakers": [{"name": "Bet365", "bets": [
            {"name": "Match Winner", "values": [
                {"value": "Home", "odd": "2.00"},
                {"value": "Draw", "odd": "3.40"},
                {"value": "Away", "odd": "4.00"},
            ]},
        ]}],
    }]
    indexed = api_football._index_odds(pages)
    assert 12345 in indexed
    assert indexed[12345][0].prices["1X2:H"] == 2.00


def test_api_football_ignores_junk_odds(api_football):
    prices = api_football._book_prices({"bets": [
        {"name": "Match Winner", "values": [
            {"value": "Home", "odd": "not-a-number"},
            {"value": "Draw", "odd": "0.5"},
        ]},
    ]})
    assert prices == {}


def test_api_football_explains_an_unmapped_league(api_football):
    with pytest.raises(ProviderError, match="no API-Football league id"):
        api_football._league_id("ZZ")


# ------------------------------------------------------------ registry
def test_registry_lists_every_provider():
    assert set(available_providers()) == {"footballdata", "theoddsapi", "apifootball"}
    for key in available_providers():
        assert key in PROVIDER_INFO


def test_unknown_provider_is_rejected_by_name():
    with pytest.raises(ProviderError, match="unknown odds provider"):
        get_provider("bookies-r-us")


def test_footballdata_is_not_a_live_provider():
    """It supplies history; it is not constructed through the live registry."""
    with pytest.raises(ProviderError):
        get_provider("footballdata")


# ------------------------------------------------------------ capture/replay
def _write_capture(tmp_path, payload):
    import json
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _odds_api_payload():
    soon = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    return [_odds_api_event(soon)]


def test_replay_needs_no_api_key(tmp_path, monkeypatch):
    """The whole point: it must work on a machine with no key and no network."""
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)
    from bettingedge.data.providers import ReplayProvider

    path = _write_capture(tmp_path, _odds_api_payload())
    fixtures = ReplayProvider(path).fixtures("E0")
    assert len(fixtures) == 1
    assert fixtures[0].home == "Manchester United"


def test_replay_reproduces_the_live_parse(tmp_path, odds_api):
    """A replayed capture must give exactly what the live client would."""
    from bettingedge.data.providers import ReplayProvider

    payload = _odds_api_payload()
    cutoff = datetime.now(timezone.utc) + timedelta(days=7)
    live = odds_api._event_to_fixture(payload[0], "E0", cutoff)
    replayed = ReplayProvider(_write_capture(tmp_path, payload)).fixtures("E0")[0]

    assert replayed.odds == live.odds
    assert replayed.sharp_odds == live.sharp_odds
    assert (replayed.home, replayed.away) == (live.home, live.away)


def test_replay_keeps_events_that_have_since_gone_stale(tmp_path):
    """A capture is always in the past by the time it is replayed."""
    from bettingedge.data.providers import ReplayProvider

    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    path = _write_capture(tmp_path, [_odds_api_event(old)])
    assert len(ReplayProvider(path).fixtures("E0")) == 1


def test_replay_detects_api_football_captures(tmp_path):
    from bettingedge.data.providers import ReplayProvider, detect_shape

    payload = {
        "fixtures": [{
            "fixture": {"id": 1, "date": "2026-08-15T14:00:00+00:00"},
            "teams": {"home": {"name": "Manchester United"},
                      "away": {"name": "Fulham"}},
        }],
        "odds": [{
            "fixture": {"id": 1},
            "bookmakers": [{"name": "Pinnacle", "bets": [
                {"name": "Match Winner", "values": [
                    {"value": "Home", "odd": "1.83"},
                    {"value": "Draw", "odd": "3.80"},
                    {"value": "Away", "odd": "4.60"},
                ]},
            ]}],
        }],
    }
    assert detect_shape(payload) == "apifootball"
    fixtures = ReplayProvider(_write_capture(tmp_path, payload)).fixtures("E0")
    assert len(fixtures) == 1
    assert fixtures[0].odds["1X2:H"] == 1.83


def test_replay_explains_an_unrecognisable_payload(tmp_path):
    from bettingedge.data.providers import ReplayProvider

    with pytest.raises(ProviderError, match="could not tell which provider"):
        ReplayProvider(_write_capture(tmp_path, {"something": "else"}))


def test_replay_surfaces_a_captured_api_error(tmp_path):
    from bettingedge.data.providers import ReplayProvider

    with pytest.raises(ProviderError, match="API error"):
        ReplayProvider(_write_capture(tmp_path, {"errors": {"token": "invalid"}}))


def test_replay_reports_a_missing_file():
    from bettingedge.data.providers import ReplayProvider

    with pytest.raises(ProviderError, match="no captured payload"):
        ReplayProvider("/no/such/capture.json")


def test_capture_summary_names_the_markets_and_teams(tmp_path):
    from bettingedge.data.providers import ReplayProvider, describe_capture

    fixtures = ReplayProvider(_write_capture(tmp_path, _odds_api_payload())).fixtures("E0")
    summary = describe_capture(fixtures)
    assert "Manchester United" in summary       # provider's exact spelling
    assert "1X2:H" in summary and "OU2.5:O" in summary
    assert "book(s) per selection" in summary


def test_capture_summary_explains_an_empty_result():
    from bettingedge.data.providers import describe_capture

    assert "No fixtures parsed" in describe_capture([])
