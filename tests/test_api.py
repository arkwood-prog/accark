"""HTTP API and CLI surface."""

import json

import pytest
from fastapi.testclient import TestClient

from bettingedge.api.server import DataStore, create_app
from bettingedge.cli import build_parser, main
from bettingedge.config import Config
from bettingedge.data import synthetic
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
