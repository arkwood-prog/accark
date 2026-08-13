"""FastAPI backend for the dashboard.

Data is loaded once at start-up and held in memory; the model is refitted
whenever the tuning parameters change, which is fast enough to feel live.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

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


def create_app(store: DataStore) -> FastAPI:
    app = FastAPI(title="bettingedge", version="1.0.0",
                  description="Data-driven football betting analysis")

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
               price_mode: str = DEFAULT_PRICE_MODE) -> None:
    import uvicorn

    global STORE
    STORE = load_store(league=league, seasons=seasons, offline=offline,
                       use_synthetic=use_synthetic, price_mode=price_mode)
    if not STORE.matches:
        raise SystemExit(
            "No match history could be loaded. Check the league code, or start with "
            "--synthetic to explore the app offline."
        )
    app = create_app(STORE)
    print(f"\nDashboard: http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
