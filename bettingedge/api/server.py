"""FastAPI backend for the dashboard.

Data is loaded once at start-up and held in memory; the model is refitted
whenever the tuning parameters change, which is fast enough to feel live.
"""

from __future__ import annotations

import os
import secrets
import threading
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from ..backtest.engine import run_backtest
from ..config import Config
from ..data import synthetic
from ..data.footballdata import (
    DEFAULT_PRICE_MODE,
    LEAGUES,
    FootballDataUK,
    recent_seasons,
)
from ..data.schema import Fixture, Match
from ..pipeline import Engine
from ..report import DISCLAIMER, render_markdown

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@dataclass
class DataStore:
    """Loaded history and fixtures, plus a cache of derived results."""

    league: str
    matches: list[Match]
    fixtures: list[Fixture]
    source: str
    loaded_at: date
    # default_factory, not a shared instance — every store gets its own lock
    # and its own caches.
    lock: threading.Lock = field(default_factory=threading.Lock)
    _slate_cache: dict[str, Any] = field(default_factory=dict)
    _backtest_cache: dict[str, Any] = field(default_factory=dict)


STORE: DataStore | None = None


def _config_from_query(
    bankroll: float,
    kelly: float,
    min_edge: float,
    model_weight: float,
    half_life: float,
    max_legs: int,
    min_leg_edge: float,
) -> Config:
    return Config.from_dict({
        "model": {"half_life_days": half_life},
        "market": {"model_weight": model_weight},
        "selection": {"min_edge": min_edge},
        "parlay": {"max_legs": max_legs, "min_leg_edge": min_leg_edge},
        "staking": {"bankroll": bankroll, "kelly_fraction": kelly},
    })


def lan_address() -> str | None:
    """This machine's address on the local network.

    Opens a UDP socket toward a public address and asks the OS which local
    interface it would use. Nothing is actually sent, and it works offline —
    it is just the reliable way to learn which of several interfaces is the
    one a phone on the same wifi can reach.
    """
    import ipaddress
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No packet is sent; this only asks the routing table which local
        # interface would be used to reach the internet.
        probe.connect(("8.8.8.8", 80))
        address = probe.getsockname()[0]
    except Exception:
        return None
    finally:
        probe.close()
    if not address:
        return None
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return None
    # Only the actual home-network ranges. Python's is_private is broader than
    # this — it also covers documentation and test ranges like 192.0.2.0/24,
    # which a container can hand back and which no phone could ever reach.
    lan_ranges = (
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
    )
    if not any(parsed in network for network in lan_ranges):
        return None
    return address


def qr_code(text: str) -> str | None:
    """A scannable terminal QR for the URL, when the optional library is there."""
    try:
        import qrcode
    except ImportError:
        return None
    try:
        code = qrcode.QRCode(border=1)
        code.add_data(text)
        code.make(fit=True)
        matrix = code.get_matrix()
    except Exception:
        return None

    # Two rows per line using half-block characters, so it stays square.
    lines = []
    for top in range(0, len(matrix), 2):
        row = ""
        for column in range(len(matrix[top])):
            upper = matrix[top][column]
            lower = matrix[top + 1][column] if top + 1 < len(matrix) else False
            row += {(True, True): "\u2588", (True, False): "\u2580",
                    (False, True): "\u2584", (False, False): " "}[(upper, lower)]
        lines.append(row)
    return "\n".join(lines)


ACCESS_TOKEN_ENV = "BETTINGEDGE_TOKEN"
_COOKIE = "bettingedge_token"


def create_app(store: DataStore, token: str | None = None) -> FastAPI:
    app = FastAPI(title="bettingedge", version="1.0.0",
                  description="Data-driven football betting analysis")

    token = token if token is not None else os.environ.get(ACCESS_TOKEN_ENV)

    if token:
        # A deployed instance is reachable by anyone who finds the URL. One
        # shared token, remembered in a cookie so the page keeps working after
        # the first load: open https://host/?token=... once on the phone.
        @app.middleware("http")
        async def require_token(request: Request, call_next):
            supplied = (request.query_params.get("token")
                        or request.cookies.get(_COOKIE)
                        or request.headers.get("x-bettingedge-token"))
            if not supplied or not secrets.compare_digest(supplied, token):
                return PlainTextResponse(
                    "Not authorised. Append ?token=... to the URL.", status_code=401)
            response = await call_next(request)
            if request.query_params.get("token"):
                response.set_cookie(_COOKIE, token, httponly=True, samesite="lax",
                                    max_age=60 * 60 * 24 * 365, secure=
                                    request.url.scheme == "https")
            return response

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "league": store.league,
            "league_name": LEAGUES.get(store.league, store.league),
            "source": store.source,
            "matches": len(store.matches),
            "fixtures": len(store.fixtures),
            "history_from": store.matches[0].date.isoformat() if store.matches else None,
            "history_to": store.matches[-1].date.isoformat() if store.matches else None,
            "disclaimer": DISCLAIMER,
        }

    @app.get("/api/slate")
    def slate(
        bankroll: float = Query(1000.0, gt=0),
        kelly: float = Query(0.25, gt=0, le=1.0),
        min_edge: float = Query(0.03, ge=0.0, le=0.5),
        model_weight: float = Query(0.35, ge=0.0, le=1.0),
        half_life: float = Query(180.0, gt=1.0),
        max_legs: int = Query(5, ge=2, le=8),
        min_leg_edge: float = Query(0.03, ge=0.0, le=0.5),
        previews: bool = Query(True),
    ) -> JSONResponse:
        if not store.fixtures:
            raise HTTPException(
                status_code=404,
                detail="No upcoming fixtures with prices are loaded. The free feed is "
                       "empty between seasons — restart with --synthetic for a demo, or "
                       "supply your own fixtures CSV.",
            )
        key = f"{bankroll}|{kelly}|{min_edge}|{model_weight}|{half_life}|{max_legs}|{min_leg_edge}|{previews}"
        with store.lock:
            if key in store._slate_cache:
                return JSONResponse(store._slate_cache[key])
        config = _config_from_query(bankroll, kelly, min_edge, model_weight, half_life,
                                    max_legs, min_leg_edge)
        built = Engine(config).build_slate(store.matches, store.fixtures)
        payload = built.to_dict(include_contexts=previews)
        payload["source"] = store.source
        payload["league"] = store.league
        payload["league_name"] = LEAGUES.get(store.league, store.league)
        payload["disclaimer"] = DISCLAIMER
        payload["markdown"] = render_markdown(built)
        with store.lock:
            store._slate_cache[key] = payload
        return JSONResponse(payload)

    @app.get("/api/ratings")
    def ratings(half_life: float = Query(180.0, gt=1.0)) -> dict:
        config = Config.from_dict({"model": {"half_life_days": half_life}})
        model = Engine(config).fit(store.matches)
        return model.to_dict()

    @app.get("/api/backtest")
    def backtest(
        kelly: float = Query(0.25, gt=0, le=1.0),
        min_edge: float = Query(0.03, ge=0.0, le=0.5),
        model_weight: float = Query(0.35, ge=0.0, le=1.0),
        half_life: float = Query(180.0, gt=1.0),
        bankroll: float = Query(1000.0, gt=0),
        train_days: int = Query(400, ge=120),
        refit_every: int = Query(7, ge=1, le=60),
    ) -> JSONResponse:
        if len(store.matches) < 200:
            raise HTTPException(status_code=400,
                                detail="Not enough history loaded to backtest.")
        key = f"{kelly}|{min_edge}|{model_weight}|{half_life}|{bankroll}|{train_days}|{refit_every}"
        with store.lock:
            if key in store._backtest_cache:
                return JSONResponse(store._backtest_cache[key])
        config = _config_from_query(bankroll, kelly, min_edge, model_weight, half_life, 5,
                                    min_edge)
        result = run_backtest(store.matches, config=config, train_days=train_days,
                              refit_every_days=refit_every)
        payload = result.to_dict()
        payload["summary_text"] = result.summary()
        with store.lock:
            store._backtest_cache[key] = payload
        return JSONResponse(payload)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/app.js")
    def script() -> FileResponse:
        return FileResponse(WEB_DIR / "app.js", media_type="application/javascript")

    @app.get("/styles.css")
    def styles() -> FileResponse:
        return FileResponse(WEB_DIR / "styles.css", media_type="text/css")

    @app.get("/logo.svg")
    def logo() -> FileResponse:
        return FileResponse(WEB_DIR / "logo.svg", media_type="image/svg+xml")

    @app.get("/manifest.json")
    def manifest() -> FileResponse:
        return FileResponse(WEB_DIR / "manifest.json", media_type="application/manifest+json")

    @app.get("/icon-{size}.png")
    def icon(size: int) -> FileResponse:
        path = WEB_DIR / f"icon-{size}.png"
        if not path.exists():
            raise HTTPException(status_code=404, detail="no icon at that size")
        return FileResponse(path, media_type="image/png")

    return app


def load_store(league: str = "E0", seasons: int = 4, offline: bool = False,
               use_synthetic: bool = False,
               price_mode: str = DEFAULT_PRICE_MODE) -> DataStore:
    if use_synthetic:
        matches, fixtures = synthetic.generate(seasons=max(2, seasons))
        return DataStore(league="SYN", matches=matches, fixtures=fixtures,
                         source="synthetic (offline demo data — not real matches)",
                         loaded_at=date.today())

    source = FootballDataUK(offline=offline, price_mode=price_mode)
    codes = recent_seasons(seasons)
    print(f"Loading {league} history for seasons {', '.join(codes)} ...")
    matches = source.results(league, codes)
    print(f"  {len(matches)} matches")
    try:
        fixtures = source.fixtures([league])
        print(f"  {len(fixtures)} upcoming fixtures with prices")
    except Exception as exc:
        print(f"  ! could not load fixtures: {exc}")
        fixtures = []
    return DataStore(league=league, matches=matches, fixtures=fixtures,
                     source=f"football-data.co.uk ({price_mode} prices)",
                     loaded_at=date.today())


def run_server(host: str = "127.0.0.1", port: int = 8000, league: str = "E0",
               seasons: int = 4, offline: bool = False, use_synthetic: bool = False,
               config: Config | None = None,
               price_mode: str = DEFAULT_PRICE_MODE, token: str | None = None) -> None:
    import uvicorn

    global STORE
    STORE = load_store(league=league, seasons=seasons, offline=offline,
                       use_synthetic=use_synthetic, price_mode=price_mode)
    if not STORE.matches:
        raise SystemExit(
            "No match history could be loaded. Check the league code, or start with "
            "--synthetic to explore the app offline."
        )
    app = create_app(STORE, token=token)
    active_token = token if token is not None else os.environ.get(ACCESS_TOKEN_ENV)
    suffix = f"/?token={active_token}" if active_token else ""

    print()
    if host == "0.0.0.0":
        address = lan_address()
        if address:
            url = f"http://{address}:{port}{suffix}"
            print("=" * 62)
            print("  OPEN THIS ON YOUR PHONE (same wifi):")
            print(f"  {url}")
            print("=" * 62)
            code = qr_code(url)
            if code:
                print()
                print(code)
            else:
                print("  (pip install qrcode for a scannable code here)")
            print()
            print(f"  On this machine: http://127.0.0.1:{port}{suffix}")
            print("  Phone can't reach it? Your firewall is probably blocking the port,")
            print("  and both devices must be on the same network — not one on mobile data.")
        else:
            print(f"Dashboard: http://127.0.0.1:{port}{suffix}")
            print("Could not work out this machine's network address; find it with "
                  "`ipconfig` or `ifconfig`.")
        if not active_token:
            print()
            print(f"  Note: no access token set, so anyone on this network can open it.")
            print(f"  Fine at home. On shared or public wifi, set {ACCESS_TOKEN_ENV}.")
    else:
        print(f"Dashboard: http://{host}:{port}{suffix}")
        if active_token:
            print("(the token is remembered in a cookie after the first load)")
    print()
    uvicorn.run(app, host=host, port=port, log_level="warning")
