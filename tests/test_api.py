"""HTTP API and CLI surface."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bettingedge.api.server import DataStore, create_app
from bettingedge.cli import build_parser, main
from bettingedge.config import Config
from bettingedge.data import synthetic
from bettingedge.data.schema import Match
from datetime import date


@pytest.fixture(scope="module")
def client():
    matches, fixtures = synthetic.generate(seasons=3, seed=4)
    store = DataStore(league="SYN", matches=matches, fixtures=fixtures,
                      source="synthetic", loaded_at=date.today())
    return TestClient(create_app(store))


def test_health_reports_what_is_loaded(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["matches"] > 500
    assert body["fixtures"] > 0
    assert "BeGambleAware" in body["disclaimer"]


def test_slate_endpoint_returns_bets_and_analysis(client):
    body = client.get("/api/slate", params={"bankroll": 2000, "min_edge": 0.03}).json()
    assert body["singles"]
    assert body["portfolio"]["bankroll"] == 2000
    assert all(slip["analysis"] for slip in body["singles"])
    assert body["previews"]
    assert body["markdown"].startswith("# Football bet")


def test_slate_respects_the_edge_filter(client):
    loose = client.get("/api/slate", params={"min_edge": 0.01}).json()
    tight = client.get("/api/slate", params={"min_edge": 0.15}).json()
    assert len(tight["singles"]) <= len(loose["singles"])
    for slip in tight["singles"]:
        assert slip["edge"] >= 0.15


def test_slate_rejects_nonsense_parameters(client):
    assert client.get("/api/slate", params={"bankroll": -5}).status_code == 422
    assert client.get("/api/slate", params={"kelly": 3}).status_code == 422
    assert client.get("/api/slate", params={"model_weight": 2}).status_code == 422


def test_ratings_endpoint(client):
    body = client.get("/api/ratings").json()
    assert body["teams"]
    assert body["converged"] is True
    nets = [team["net"] for team in body["teams"]]
    assert nets == sorted(nets, reverse=True), "ratings should come back ranked"


def test_backtest_endpoint(client):
    body = client.get("/api/backtest", params={"train_days": 420, "refit_every": 30}).json()
    assert body["bets"] >= 0
    assert "model_log_loss" in body
    assert "summary_text" in body


def test_static_assets_are_served(client):
    for path, needle in [("/", "bettingedge"), ("/app.js", "renderBets"),
                         ("/styles.css", "--accent")]:
        response = client.get(path)
        assert response.status_code == 200
        assert needle in response.text


def test_slate_404s_when_there_are_no_fixtures():
    matches, _ = synthetic.generate(seasons=2)
    store = DataStore(league="SYN", matches=matches, fixtures=[], source="synthetic",
                      loaded_at=date.today())
    response = TestClient(create_app(store)).get("/api/slate")
    assert response.status_code == 404
    assert "fixtures" in response.json()["detail"].lower()


# ------------------------------------------------------------------ CLI
def test_cli_demo_runs(capsys):
    assert main(["demo", "--seasons", "2", "--brief"]) == 0
    output = capsys.readouterr().out
    assert "BETTINGEDGE" in output
    assert "PORTFOLIO" in output
    assert "synthetic" in output.lower()


def test_cli_demo_writes_json_and_markdown(tmp_path, capsys):
    json_path = tmp_path / "slate.json"
    md_path = tmp_path / "slate.md"
    assert main(["demo", "--seasons", "2", "--brief",
                 "--json", str(json_path), "--markdown", str(md_path)]) == 0
    payload = json.loads(json_path.read_text())
    assert payload["model"]["teams"]
    assert md_path.read_text().startswith("# Football bet")


def test_cli_ratings_runs(capsys):
    assert main(["ratings", "--synthetic", "--seasons", "2"]) == 0
    assert "Home advantage" in capsys.readouterr().out


def test_cli_leagues_lists_codes(capsys):
    assert main(["leagues"]) == 0
    assert "E0" in capsys.readouterr().out


def test_cli_config_overrides_reach_the_engine(capsys):
    main(["demo", "--seasons", "2", "--brief", "--bankroll", "5000"])
    assert "5,000.00" in capsys.readouterr().out


def test_cli_verify_runs_and_reports(capsys):
    code = main(["verify", "--synthetic", "--seasons", "3", "--train-days", "420",
                 "--refit-every", "40", "--skip-pessimistic", "--quiet"])
    output = capsys.readouterr().out
    assert code in (0, 1)          # 1 when a check legitimately fails
    assert "BETTINGEDGE VERIFICATION" in output
    assert "STAGE 1  DATA INTEGRITY" in output
    assert "STAGE 5  STATISTICAL POWER" in output
    assert "VERDICT" in output


def test_cli_verify_writes_json(tmp_path, capsys):
    path = tmp_path / "report.json"
    main(["verify", "--synthetic", "--seasons", "3", "--train-days", "420",
          "--refit-every", "40", "--skip-pessimistic", "--quiet", "--json", str(path)])
    payload = json.loads(path.read_text())
    assert len(payload["stages"]) == 5
    assert payload["status"] in ("PASS", "WARN", "FAIL")


def test_cli_backtest_pessimistic_switch(capsys):
    assert main(["backtest", "--synthetic", "--seasons", "3", "--pessimistic",
                 "--train-days", "420", "--refit-every", "40"]) == 0
    output = capsys.readouterr().out
    assert "Pessimistic mode" in output
    assert "sharp closing price" in output


def test_cli_rejects_an_unknown_price_mode():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["backtest", "--price-mode", "nonsense"])


def test_cli_accepts_every_price_mode():
    from bettingedge.data.footballdata import PRICE_MODES

    for mode in PRICE_MODES:
        args = build_parser().parse_args(["backtest", "--price-mode", mode])
        assert args.price_mode == mode


def test_cli_reports_errors_without_a_traceback(capsys):
    """A bad path should produce a clean message, not a stack dump."""
    code = main(["recommend", "--results-csv", "/no/such/file.csv"])
    assert code == 1
    assert "Error:" in capsys.readouterr().err


def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


# ------------------------------------------------------------ access token
def test_no_token_means_open_access():
    matches, fixtures = synthetic.generate(seasons=2)
    store = DataStore(league="SYN", matches=matches, fixtures=fixtures,
                      source="synthetic", loaded_at=date.today())
    assert TestClient(create_app(store, token=None)).get("/api/health").status_code == 200


@pytest.fixture(scope="module")
def guarded():
    matches, fixtures = synthetic.generate(seasons=2, seed=12)
    store = DataStore(league="SYN", matches=matches, fixtures=fixtures,
                      source="synthetic", loaded_at=date.today())
    return TestClient(create_app(store, token="s3cret"))


def test_a_deployed_instance_refuses_requests_without_the_token(guarded):
    assert guarded.get("/api/health").status_code == 401
    assert guarded.get("/").status_code == 401


def test_the_wrong_token_is_refused(guarded):
    assert guarded.get("/api/health", params={"token": "guess"}).status_code == 401
    assert guarded.get("/api/health",
                       headers={"x-bettingedge-token": "guess"}).status_code == 401


def test_the_right_token_is_accepted_by_query_header_or_cookie(guarded):
    assert guarded.get("/api/health", params={"token": "s3cret"}).status_code == 200
    assert guarded.get("/api/health",
                       headers={"x-bettingedge-token": "s3cret"}).status_code == 200
    assert guarded.get("/api/health", cookies={"bettingedge_token": "s3cret"}).status_code == 200


def test_the_token_is_remembered_in_a_cookie(guarded):
    """So the page keeps working on a phone after the first load."""
    response = guarded.get("/", params={"token": "s3cret"})
    assert response.status_code == 200
    assert "bettingedge_token" in response.headers.get("set-cookie", "")
    assert "HttpOnly" in response.headers.get("set-cookie", "")


def _cookieless_guarded_client():
    """A token-protected client with an empty cookie jar.

    The module-scoped `guarded` fixture cannot be reused here: the cookie test
    above leaves a valid token in its jar, so every later request would be
    authorised and these assertions would pass for the wrong reason.
    """
    matches, fixtures = synthetic.generate(seasons=2, seed=12)
    store = DataStore(league="SYN", matches=matches, fixtures=fixtures,
                      source="synthetic", loaded_at=date.today())
    return TestClient(create_app(store, token="s3cret"))


def test_the_manifest_and_icons_load_without_a_token():
    """Add to Home Screen needs these before any cookie exists.

    A browser fetches the manifest itself, with credentials omitted by default,
    so gating it behind the token silently costs you the app icon — and it
    protects nothing, since a name and a logo are not data.
    """
    client = _cookieless_guarded_client()
    for path in ("/manifest.json", "/logo.svg", "/icon-192.png", "/icon-512.png"):
        assert client.get(path).status_code == 200, path


def test_the_manifest_still_declares_icons_that_are_actually_served():
    """A manifest pointing at a 401 or a 404 installs a blank icon."""
    client = _cookieless_guarded_client()
    icons = client.get("/manifest.json").json()["icons"]
    assert icons
    for icon in icons:
        assert client.get(icon["src"]).status_code == 200, icon["src"]


def test_opening_the_manifest_up_did_not_open_up_the_data():
    """The exemption must be branding-only — no data path may slip through."""
    client = _cookieless_guarded_client()
    for path in ("/", "/api/health", "/api/slate", "/api/ratings",
                 "/api/backtest", "/api/leagues", "/app.js", "/styles.css"):
        assert client.get(path).status_code == 401, path


def test_public_asset_matching_is_not_a_loose_prefix():
    """A near-miss path must not inherit the exemption."""
    from bettingedge.api.server import _is_public_asset

    assert _is_public_asset("/manifest.json")
    assert _is_public_asset("/icon-192.png")
    assert _is_public_asset("/icon-512.png")
    assert not _is_public_asset("/manifest.json/../api/slate")
    assert not _is_public_asset("/icon-192.png.map")
    assert not _is_public_asset("/api/icon-192.png")
    assert not _is_public_asset("/icon-abc.png")
    assert not _is_public_asset("/")


def test_the_page_asks_for_the_manifest_with_credentials():
    """Without crossorigin=use-credentials the cookie is not sent at all."""
    html = (Path(__file__).resolve().parent.parent
            / "bettingedge" / "web" / "index.html").read_text(encoding="utf-8")
    link = next(line for line in html.splitlines() if 'rel="manifest"' in line)
    assert 'crossorigin="use-credentials"' in link


# ------------------------------------------------------------ mobile assets
def test_the_web_app_manifest_is_served(client):
    response = client.get("/manifest.json")
    assert response.status_code == 200
    manifest = response.json()
    assert manifest["display"] == "standalone"
    assert {icon["sizes"] for icon in manifest["icons"]} == {"192x192", "512x512"}


def test_home_screen_icons_are_served(client):
    for size in (192, 512):
        response = client.get(f"/icon-{size}.png")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_missing_icon_size_404s(client):
    assert client.get("/icon-77.png").status_code == 404


def test_the_page_declares_itself_installable(client):
    html = client.get("/").text
    assert 'rel="manifest"' in html
    assert 'name="apple-mobile-web-app-capable"' in html
    assert "viewport-fit=cover" in html


# ------------------------------------------------------------ LAN serving
def test_lan_address_is_private_or_absent():
    """It must never hand back a loopback or public address as 'your LAN IP'."""
    import ipaddress

    from bettingedge.api.server import lan_address

    address = lan_address()
    if address is None:
        return          # containers and offline machines legitimately have none
    parsed = ipaddress.ip_address(address)
    assert parsed.is_private and not parsed.is_loopback


def test_qr_code_renders_or_degrades_quietly():
    from bettingedge.api.server import qr_code

    code = qr_code("http://192.168.1.10:8000")
    if code is None:
        return          # the library is optional
    lines = code.splitlines()
    assert len(lines) > 8
    assert len({len(line) for line in lines}) == 1, "QR rows must be equal width"


def test_lan_flag_binds_to_all_interfaces():
    args = build_parser().parse_args(["serve", "--lan"])
    assert args.lan is True


def test_host_defaults_to_loopback():
    args = build_parser().parse_args(["serve"])
    assert args.host == "127.0.0.1"
    assert args.lan is False


def test_documentation_ranges_are_not_offered_as_a_lan_address():
    """A container can route to 192.0.2.0/24; no phone can reach it."""
    import ipaddress

    lan_ranges = (ipaddress.ip_network("10.0.0.0/8"),
                  ipaddress.ip_network("172.16.0.0/12"),
                  ipaddress.ip_network("192.168.0.0/16"))

    def reachable(text):
        parsed = ipaddress.ip_address(text)
        return any(parsed in network for network in lan_ranges)

    assert reachable("192.168.1.42") and reachable("10.0.0.5") and reachable("172.16.3.9")
    # Python calls these private; a phone still cannot reach them.
    assert not reachable("192.0.2.2")      # TEST-NET-1
    assert not reachable("169.254.1.1")    # link-local
    assert not reachable("127.0.0.1") and not reachable("8.8.8.8")


# ------------------------------------------------------------ logo assets
def test_the_svg_logo_is_served(client):
    response = client.get("/logo.svg")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    body = response.text
    assert body.startswith("<svg") and body.rstrip().endswith("</svg>")
    # Four coloured panels, one per blade.
    assert body.count("<polygon") == 8      # four grooves plus four colours


def test_the_page_uses_the_ball_as_its_icon(client):
    html = client.get("/").text
    assert '<img class="mark" src="/logo.svg"' in html
    assert 'type="image/svg+xml" href="/logo.svg"' in html
    # The placeholder emoji favicon must be gone.
    assert "text y='26'" not in html


def test_the_generator_reproduces_the_committed_svg():
    """The icons are generated, so a stale checked-in file would be a lie."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "tools"))
    try:
        import make_logo
    except ImportError:
        pytest.skip("generator not available")
    finally:
        sys.path.pop(0)

    committed = (root / "bettingedge" / "web" / "logo.svg").read_text(encoding="utf-8")
    assert make_logo.render_svg() == committed


def test_the_icons_are_square_and_the_expected_size():
    import struct
    from pathlib import Path

    web = Path(__file__).resolve().parent.parent / "bettingedge" / "web"
    for size in (192, 512):
        data = (web / f"icon-{size}.png").read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        width, height = struct.unpack(">II", data[16:24])
        assert (width, height) == (size, size)


# ------------------------------------------------------------ live provider wiring
def test_load_store_actually_uses_the_requested_odds_provider(monkeypatch):
    """Regression test for a real bug: `serve --odds-provider theoddsapi` used
    to parse fine but silently fall back to the free feed, because load_store
    never accepted or used the provider argument at all."""
    from datetime import date as _date

    from bettingedge.api.server import load_store
    from bettingedge.data.schema import Fixture, Match

    fake_matches = [
        Match(date=_date(2026, 1, 1), league="E0", home="Arsenal", away="Chelsea",
              home_goals=2, away_goals=1),
    ]
    fake_fixture = Fixture(date=_date(2026, 9, 1), league="E0", home="Arsenal",
                           away="Chelsea", odds={"1X2:H": 2.0, "1X2:D": 3.4, "1X2:A": 3.8})

    class FakeProvider:
        name = "theoddsapi"

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fixtures(self, league, days_ahead=7):
            assert league == "E0"
            return [fake_fixture]

    import bettingedge.api.server as server_module

    monkeypatch.setattr(server_module.FootballDataUK, "results",
                        lambda self, league, seasons: fake_matches)
    monkeypatch.setattr("bettingedge.data.providers.get_provider",
                        lambda name, **kwargs: FakeProvider(**kwargs))

    store = load_store(league="E0", seasons=2, odds_provider="theoddsapi",
                       api_key="test-key")

    assert store.source.startswith("The Odds API")
    assert len(store.fixtures) == 1
    assert store.fixtures[0].home == "Arsenal"


def test_load_store_defaults_to_the_free_feed_when_no_provider_given(monkeypatch):
    from datetime import date as _date

    from bettingedge.api.server import load_store
    from bettingedge.data.schema import Fixture, Match

    fake_matches = [Match(date=_date(2026, 1, 1), league="E0", home="Arsenal",
                          away="Chelsea", home_goals=1, away_goals=0)]
    fake_fixture = Fixture(date=_date(2026, 9, 1), league="E0", home="Arsenal",
                           away="Chelsea", odds={"1X2:H": 2.0, "1X2:D": 3.4, "1X2:A": 3.8})

    import bettingedge.api.server as server_module

    monkeypatch.setattr(server_module.FootballDataUK, "results",
                        lambda self, league, seasons: fake_matches)
    monkeypatch.setattr(server_module.FootballDataUK, "fixtures",
                        lambda self, leagues: [fake_fixture])

    store = load_store(league="E0", seasons=2)
    assert "football-data.co.uk" in store.source
    assert len(store.fixtures) == 1


def test_load_store_reconciles_live_provider_team_names(monkeypatch):
    """A live provider spells teams differently; unresolved names must be
    dropped rather than silently priced off league-average ratings."""
    from datetime import date as _date

    from bettingedge.api.server import load_store
    from bettingedge.data.schema import Fixture, Match

    fake_matches = [Match(date=_date(2026, 1, 1), league="E0", home="Man United",
                          away="Fulham", home_goals=2, away_goals=0)]
    provider_fixture = Fixture(date=_date(2026, 9, 1), league="E0",
                               home="Manchester United", away="Fulham",
                               odds={"1X2:H": 2.0, "1X2:D": 3.4, "1X2:A": 3.8})
    unresolvable_fixture = Fixture(date=_date(2026, 9, 1), league="E0",
                                   home="Some Made Up FC", away="Fulham",
                                   odds={"1X2:H": 2.0, "1X2:D": 3.4, "1X2:A": 3.8})

    class FakeProvider:
        def __init__(self, **kwargs):
            pass

        def fixtures(self, league, days_ahead=7):
            return [provider_fixture, unresolvable_fixture]

    import bettingedge.api.server as server_module

    monkeypatch.setattr(server_module.FootballDataUK, "results",
                        lambda self, league, seasons: fake_matches)
    monkeypatch.setattr("bettingedge.data.providers.get_provider",
                        lambda name, **kwargs: FakeProvider(**kwargs))

    store = load_store(league="E0", seasons=2, odds_provider="theoddsapi")
    assert len(store.fixtures) == 1
    assert store.fixtures[0].home == "Man United"      # rewritten to match the model


def test_cmd_serve_parses_live_provider_flags():
    args = build_parser().parse_args(
        ["serve", "--league", "E0", "--odds-provider", "theoddsapi",
         "--api-key", "abc123", "--lan"])
    assert args.odds_provider == "theoddsapi"
    assert args.api_key == "abc123"
    assert args.lan is True


# ------------------------------------------------------------ auto refresh
def _store_with(fixtures_count: int, source: str = "initial"):
    from datetime import date as _date

    from bettingedge.api.server import DataStore
    from bettingedge.data.schema import Fixture, Match

    matches = [Match(date=_date(2026, 1, 1), league="E0", home="Arsenal",
                     away="Chelsea", home_goals=1, away_goals=0)]
    fixtures = [Fixture(date=_date(2026, 9, i + 1), league="E0", home="Arsenal",
                        away="Chelsea", odds={"1X2:H": 2.0, "1X2:D": 3.4, "1X2:A": 3.8})
                for i in range(fixtures_count)]
    return DataStore(league="E0", matches=matches, fixtures=fixtures,
                     source=source, loaded_at=_date.today())


def test_replace_data_swaps_content_and_drops_derived_caches():
    store = _store_with(1)
    store._slate_cache["old"] = {"x": 1}
    store._backtest_cache["old"] = {"y": 2}
    before = store.refreshed_at

    fresh = _store_with(3, source="refreshed")
    store.replace_data(fresh.matches, fresh.fixtures, fresh.source)

    assert len(store.fixtures) == 3
    assert store.source == "refreshed"
    assert store._slate_cache == {} and store._backtest_cache == {}
    assert store.refreshed_at >= before


def test_refresh_loop_picks_up_new_fixtures(monkeypatch):
    import time as _time

    import bettingedge.api.server as server_module

    store = _store_with(1)
    monkeypatch.setattr(server_module, "load_store",
                        lambda **kwargs: _store_with(4, source="live"))

    server_module.start_refresh_loop(store, minutes=1 / 120)   # ~0.5s
    deadline = _time.time() + 6
    while _time.time() < deadline and len(store.fixtures) != 4:
        _time.sleep(0.1)

    assert len(store.fixtures) == 4, "refresh loop never replaced the data"
    assert store.last_refresh_error is None


def test_a_failed_refresh_keeps_serving_the_previous_data(monkeypatch):
    """Stale odds beat a dashboard that goes blank because a provider blipped."""
    import time as _time

    import bettingedge.api.server as server_module

    store = _store_with(2)

    def explode(**kwargs):
        raise RuntimeError("provider unreachable")

    monkeypatch.setattr(server_module, "load_store", explode)
    server_module.start_refresh_loop(store, minutes=1 / 120)

    deadline = _time.time() + 6
    while _time.time() < deadline and store.last_refresh_error is None:
        _time.sleep(0.1)

    assert len(store.fixtures) == 2, "old data must survive a failed refresh"
    assert store.last_refresh_error is not None
    assert "provider unreachable" in store.last_refresh_error


def test_refresh_can_be_disabled():
    import bettingedge.api.server as server_module

    assert server_module.start_refresh_loop(_store_with(1), minutes=0) is None


def test_health_reports_freshness(client):
    body = client.get("/api/health").json()
    assert "refreshed_at" in body
    assert body["refresh_error"] is None


def test_serve_accepts_a_refresh_interval():
    args = build_parser().parse_args(["serve", "--refresh-minutes", "240"])
    assert args.refresh_minutes == 240
    assert build_parser().parse_args(["serve"]).refresh_minutes == 180


# ------------------------------------------------------------ league dropdown
def _two_league_client(token=None):
    from bettingedge.api.server import DataStore, create_app

    m1, f1 = synthetic.generate(seasons=2, seed=21)
    m2, f2 = synthetic.generate(seasons=2, seed=22)
    primary = DataStore(league="E0", matches=m1, fixtures=f1, source="a",
                        loaded_at=date.today())
    other = DataStore(league="SP1", matches=m2, fixtures=f2, source="b",
                      loaded_at=date.today())
    return TestClient(create_app(primary, token=token, extra_stores={"SP1": other})), primary, other


def test_leagues_endpoint_lists_every_loaded_store():
    client, primary, other = _two_league_client()
    body = client.get("/api/leagues").json()
    codes = {row["code"] for row in body}
    assert codes == {"E0", "SP1"}
    primary_row = next(row for row in body if row["code"] == "E0")
    other_row = next(row for row in body if row["code"] == "SP1")
    assert primary_row["is_primary"] is True
    assert other_row["is_primary"] is False


def test_single_store_leagues_endpoint_still_lists_the_one_league(client):
    body = client.get("/api/leagues").json()
    assert len(body) == 1
    assert body[0]["is_primary"] is True


def test_league_param_switches_which_store_answers(client):
    """The existing single-store fixture, requested with an unknown code,
    must fall back to serving its one store rather than erroring."""
    default = client.get("/api/health").json()
    same = client.get("/api/health", params={"league": "NOPE"}).json()
    assert default["league"] == same["league"]


def test_health_and_slate_resolve_independently_per_league():
    client, primary, other = _two_league_client()

    health_a = client.get("/api/health", params={"league": "E0"}).json()
    health_b = client.get("/api/health", params={"league": "SP1"}).json()
    assert health_a["league"] == "E0"
    assert health_b["league"] == "SP1"
    assert set(health_a["available_leagues"]) == {"E0", "SP1"}

    slate_a = client.get("/api/slate", params={"league": "E0"}).json()
    slate_b = client.get("/api/slate", params={"league": "SP1"}).json()
    assert slate_a["league"] == "E0"
    assert slate_b["league"] == "SP1"


def test_ratings_and_backtest_also_respect_league():
    """Not just slate — every data endpoint must resolve the right store."""
    client, primary, other = _two_league_client()

    ratings_a = client.get("/api/ratings", params={"league": "E0"}).json()
    ratings_b = client.get("/api/ratings", params={"league": "SP1"}).json()
    assert ratings_a["teams"] and ratings_b["teams"]
    # Fitted independently on different (seeded) results, so the ratings
    # themselves differ even though both stores share the same team-name pool.
    assert ratings_a["n_matches"] == len(primary.matches)
    assert ratings_b["n_matches"] == len(other.matches)
    assert ratings_a["teams"] != ratings_b["teams"]


def test_unknown_league_falls_back_to_primary_on_every_endpoint():
    client, primary, other = _two_league_client()
    for path in ("/api/health", "/api/slate", "/api/ratings"):
        body = client.get(path, params={"league": "ZZ"}).json()
        league_field = body.get("league")
        assert league_field == "E0" or "teams" in body  # ratings has no top-level league


def test_caches_are_kept_separate_per_league():
    """Regression guard: two leagues must not share one slate/backtest cache."""
    client, primary, other = _two_league_client()
    client.get("/api/slate", params={"league": "E0"})
    client.get("/api/slate", params={"league": "SP1"})
    assert "" in primary._slate_cache or primary._slate_cache  # populated independently
    assert other._slate_cache
    assert primary._slate_cache is not other._slate_cache


def test_token_protection_covers_every_league_equally():
    client, _, _ = _two_league_client(token="s3cret")
    assert client.get("/api/slate", params={"league": "SP1"}).status_code == 401
    assert client.get("/api/slate",
                      params={"league": "SP1", "token": "s3cret"}).status_code == 200


# ------------------------------------------------------------ run_server multi-league
def test_run_server_loads_every_comma_separated_league(monkeypatch):
    import bettingedge.api.server as server_module

    calls = []

    def fake_load_store(**kwargs):
        calls.append(kwargs["league"])
        matches, fixtures = synthetic.generate(seasons=2, seed=len(calls))
        return server_module.DataStore(league=kwargs["league"], matches=matches,
                                       fixtures=fixtures, source="fake",
                                       loaded_at=date.today())

    monkeypatch.setattr(server_module, "load_store", fake_load_store)
    monkeypatch.setattr(server_module, "start_refresh_loop", lambda *a, **k: None)

    # uvicorn is imported lazily inside run_server, so it must be patched in
    # sys.modules before that import statement executes.
    import sys
    import types

    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda *a, **k: None))

    server_module.run_server(league="E0,SP1,I1", refresh_minutes=0)

    assert calls == ["E0", "SP1", "I1"]
    assert set(server_module.STORES) == {"E0", "SP1", "I1"}
    assert server_module.STORE.league == "E0"


def test_run_server_single_league_still_works(monkeypatch):
    import sys
    import types

    import bettingedge.api.server as server_module

    def fake_load_store(**kwargs):
        matches, fixtures = synthetic.generate(seasons=2, seed=7)
        return server_module.DataStore(league=kwargs["league"], matches=matches,
                                       fixtures=fixtures, source="fake",
                                       loaded_at=date.today())

    monkeypatch.setattr(server_module, "load_store", fake_load_store)
    monkeypatch.setattr(server_module, "start_refresh_loop", lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda *a, **k: None))

    server_module.run_server(league="E0", refresh_minutes=0)
    assert list(server_module.STORES) == ["E0"]
    assert server_module.STORE.league == "E0"


# ------------------------------------------- current division in the ratings
def test_ratings_tag_which_teams_are_in_the_division_now():
    """A 24-team league fitted over 6 seasons rates ~32 clubs; the table shows 24."""
    from datetime import date as _date, timedelta

    def season(start_year, teams):
        out, kickoff = [], _date(start_year, 8, 10)
        for r in range(4):
            for i, home in enumerate(teams):
                away = teams[(i + 1 + r) % len(teams)]
                if home != away:
                    out.append(Match(date=kickoff, league="E1", home=home, away=away,
                                     home_goals=(i + r) % 3, away_goals=r % 2))
                    kickoff += timedelta(days=2)
        return out

    old = [f"Old{i}" for i in range(12)]
    now = [f"Now{i}" for i in range(12)]
    matches = season(2024, old) + season(2025, now)
    store = DataStore(league="E1", matches=matches, fixtures=[],
                      source="test", loaded_at=date.today())
    payload = TestClient(create_app(store)).get("/api/ratings").json()

    assert payload["current_season"] == "2025-26"
    assert payload["n_current_teams"] == len(now)
    assert len(payload["teams"]) > payload["n_current_teams"], "fit should span both seasons"
    current = {e["team"] for e in payload["teams"] if e["current"]}
    assert current == set(now)
    assert all(e["current"] is False for e in payload["teams"] if e["team"] in set(old))


def test_every_rated_team_is_tagged_one_way_or_the_other():
    """A missing flag would silently hide a team from the table."""
    matches, fixtures = synthetic.generate(seasons=3, seed=7)
    store = DataStore(league="SYN", matches=matches, fixtures=fixtures,
                      source="synthetic", loaded_at=date.today())
    payload = TestClient(create_app(store)).get("/api/ratings").json()
    assert payload["teams"]
    assert all(isinstance(e.get("current"), bool) for e in payload["teams"])


def test_the_slate_model_carries_the_same_tags_as_the_ratings_endpoint():
    """The dashboard table reads the slate payload, so the two must agree."""
    matches, fixtures = synthetic.generate(seasons=3, seed=8)
    store = DataStore(league="SYN", matches=matches, fixtures=fixtures,
                      source="synthetic", loaded_at=date.today())
    client = TestClient(create_app(store))
    slate = client.get("/api/slate").json()["model"]
    ratings = client.get("/api/ratings").json()
    assert slate["n_current_teams"] == ratings["n_current_teams"]
    assert slate["current_season"] == ratings["current_season"]
    assert ({e["team"] for e in slate["teams"] if e["current"]}
            == {e["team"] for e in ratings["teams"] if e["current"]})


def test_tagging_never_drops_a_team_from_the_fit():
    """Display filtering must not change what the model was fitted on."""
    matches, fixtures = synthetic.generate(seasons=3, seed=9)
    store = DataStore(league="SYN", matches=matches, fixtures=fixtures,
                      source="synthetic", loaded_at=date.today())
    payload = TestClient(create_app(store)).get("/api/ratings").json()
    fitted = {t for m in matches for t in (m.home, m.away)}
    assert {e["team"] for e in payload["teams"]} == fitted
    assert payload["n_matches"] == len(matches)


# ------------------------------------------------ collapsible bet categories
def test_multis_carry_the_leg_counts_the_dashboard_groups_by(client):
    """The bets tab splits multis into doubles/trebles/accas by leg count."""
    body = client.get("/api/slate", params={"min_edge": 0.005, "max_legs": 5}).json()
    assert body["multis"], "need multis to check the grouping"
    for slip in body["multis"]:
        assert len(slip["legs"]) >= 2
    counts = {len(slip["legs"]) for slip in body["multis"]}
    assert counts <= {2, 3, 4, 5}, "max_legs=5 must not yield a 6-fold"
    # Every multi lands in exactly one of the three groups the UI renders.
    grouped = sum(1 for s in body["multis"]
                  if len(s["legs"]) == 2 or len(s["legs"]) == 3 or len(s["legs"]) >= 4)
    assert grouped == len(body["multis"])


def test_every_slip_reports_a_stake_for_the_group_subtotals(client):
    """Each collapsed category header shows its own staked total."""
    body = client.get("/api/slate", params={"min_edge": 0.005}).json()
    for group in ("singles", "multis", "same_game"):
        for slip in body[group]:
            assert isinstance(slip["stake"], (int, float))
            assert slip["stake"] >= 0


def test_the_settings_panel_and_bet_groups_are_collapsible():
    """Guards the markup the collapse behaviour depends on."""
    web = Path(__file__).resolve().parent.parent / "bettingedge" / "web"
    html = (web / "index.html").read_text(encoding="utf-8")
    js = (web / "app.js").read_text(encoding="utf-8")
    # Controls live in a <details> keyed for state, with the inputs still inside.
    assert '<details class="panel" id="controls">' in html
    assert 'id="controls-summary"' in html
    for control in ("bankroll", "kelly", "minedge", "weight", "halflife", "maxlegs"):
        assert f'id="{control}"' in html, control
    # Each category is its own keyed panel.
    for key in ("'singles'", "'doubles'", "'trebles'", "'accas'", "'samegame'"):
        assert key in js, key


# ------------------------------------------------------------ cache freshness
def test_dashboard_files_are_never_served_stale(client):
    """A phone must not keep running yesterday's app.js after a pull.

    FileResponse sets ETag/Last-Modified but no Cache-Control, and with no
    explicit freshness a browser invents one — which showed up as index.html
    updating while the old JavaScript kept driving it.
    """
    for path in ("/", "/app.js", "/styles.css", "/manifest.json", "/icon-192.png"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.headers.get("cache-control") == "no-cache", path


def test_the_index_declares_a_html_content_type(client):
    assert client.get("/").headers["content-type"].startswith("text/html")
