"""FastAPI backend for the dashboard.

Data is loaded once at start-up and held in memory; the model is refitted
whenever the tuning parameters change, which is fast enough to feel live.
"""

from __future__ import annotations

import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from ..backtest.engine import run_backtest
from ..config import Config
from ..data import synthetic
from ..data.teams import current_squad
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
    refreshed_at: datetime = field(default_factory=datetime.now)
    last_refresh_error: str | None = None

    def replace_data(self, matches: list[Match], fixtures: list[Fixture],
                     source: str) -> None:
        """Swap in newly fetched data and drop everything derived from the old.

        Holding the lock for the whole swap means a request in flight either
        sees entirely the old data or entirely the new — never a slate priced
        from fresh fixtures against a stale model.
        """
        with self.lock:
            self.matches = matches
            self.fixtures = fixtures
            self.source = source
            self.loaded_at = date.today()
            self.refreshed_at = datetime.now()
            self.last_refresh_error = None
            self._slate_cache.clear()
            self._backtest_cache.clear()


STORE: DataStore | None = None
STORES: dict[str, DataStore] = {}


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

# The web app manifest and its icons are fetched by the browser itself, not by
# our page's JavaScript, and per spec a manifest request carries no credentials
# unless the <link> opts in with crossorigin="use-credentials" — manifest-driven
# icon fetches are inconsistent across browsers even then. Gating these behind
# the token breaks "Add to Home Screen" (you get a generic bookmark showing a
# page screenshot instead of the app icon) while protecting nothing at all:
# they are a name and a logo, carrying no fixtures, prices or ratings.
_PUBLIC_PATHS = frozenset({"/manifest.json", "/logo.svg"})
_PUBLIC_ICON = re.compile(r"/icon-\d+\.png\Z")


def _is_public_asset(path: str) -> bool:
    """True for branding assets that must load before a token is presented."""
    return path in _PUBLIC_PATHS or _PUBLIC_ICON.fullmatch(path) is not None


def _web_file(name: str, media_type: str) -> FileResponse:
    """Serve a dashboard file that must never be served stale.

    FileResponse sets ETag and Last-Modified but no Cache-Control, and with no
    explicit freshness a browser is free to guess one — mobile browsers guess
    generously, and a phone will happily run yesterday's app.js for hours
    without ever asking whether it changed. After a `git pull` that shows up as
    a partial upgrade: the new index.html renders while the old JavaScript still
    drives it, which looks like a broken feature rather than a stale cache.

    ``no-cache`` does not mean "do not store" — it means "revalidate before
    using". FileResponse sends an ETag but does not itself answer conditional
    requests with a 304, so revalidation refetches the file: about 50KB for the
    whole dashboard, which is nothing over a LAN and buys correctness that
    matters every time you pull.
    """
    return FileResponse(WEB_DIR / name, media_type=media_type,
                        headers={"Cache-Control": "no-cache"})


def _tag_current_squad(model_payload: dict, matches: list[Match]) -> dict:
    """Mark which rated teams are in the division now.

    The fit spans several seasons on purpose, so it rates every club that passed
    through — 32 for a 24-team Championship. Narrowing the fit to fix that would
    throw away real matches; tagging the rows instead lets the table show the
    league as it stands while the model keeps all its evidence.

    Applied to both /api/slate and /api/ratings, since the dashboard's ratings
    table is fed by the slate payload and the two must not disagree.
    """
    teams = model_payload.get("teams")
    if not teams:
        return model_payload
    squad, season = current_squad(matches)
    for entry in teams:
        # No usable roster (synthetic data, a single part-season) means every
        # team is shown rather than none.
        entry["current"] = entry["team"] in squad if squad else True
    model_payload["current_season"] = (f"{season}-{str(season + 1)[2:]}"
                                       if season is not None else None)
    model_payload["n_current_teams"] = sum(1 for e in teams if e["current"])
    return model_payload


def create_app(store: DataStore, token: str | None = None,
               extra_stores: dict[str, DataStore] | None = None) -> FastAPI:
    """Build the app around a primary store, optionally with more leagues loaded.

    Every additional league in `extra_stores` becomes selectable from any
    endpoint via `?league=CODE`, and appears in the dashboard's league
    dropdown. A single-store deployment (the common case, and every existing
    test) behaves exactly as before — `extra_stores` defaults to nothing.
    """
    app = FastAPI(title="bettingedge", version="1.0.0",
                  description="Data-driven football betting analysis")

    all_stores: dict[str, DataStore] = {store.league: store, **(extra_stores or {})}

    def resolve(league: str | None) -> DataStore:
        if league and league.upper() in all_stores:
            return all_stores[league.upper()]
        return store

    token = token if token is not None else os.environ.get(ACCESS_TOKEN_ENV)

    if token:
        # A deployed instance is reachable by anyone who finds the URL. One
        # shared token, remembered in a cookie so the page keeps working after
        # the first load: open https://host/?token=... once on the phone.
        @app.middleware("http")
        async def require_token(request: Request, call_next):
            if _is_public_asset(request.url.path):
                return await call_next(request)
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

    @app.get("/api/leagues")
    def leagues_endpoint() -> list[dict]:
        """Every league currently loaded in memory — what the dropdown offers.

        Not every league this project *can* model, just the ones this
        particular server was started with. Adding more is a restart with a
        longer --league list, not a runtime action, because each one adds to
        a live provider's monthly quota.
        """
        return [
            {
                "code": code,
                "name": LEAGUES.get(code, code),
                "matches": len(s.matches),
                "fixtures": len(s.fixtures),
                "is_primary": code == store.league,
            }
            for code, s in all_stores.items()
        ]

    @app.get("/api/health")
    def health(league: str | None = Query(None)) -> dict:
        s = resolve(league)
        return {
            "status": "ok",
            "league": s.league,
            "league_name": LEAGUES.get(s.league, s.league),
            "source": s.source,
            "matches": len(s.matches),
            "fixtures": len(s.fixtures),
            "history_from": s.matches[0].date.isoformat() if s.matches else None,
            "history_to": s.matches[-1].date.isoformat() if s.matches else None,
            "refreshed_at": s.refreshed_at.isoformat(timespec="seconds"),
            "refresh_error": s.last_refresh_error,
            "available_leagues": sorted(all_stores),
            "disclaimer": DISCLAIMER,
        }

    @app.get("/api/slate")
    def slate(
        league: str | None = Query(None),
        bankroll: float = Query(1000.0, gt=0),
        kelly: float = Query(0.25, gt=0, le=1.0),
        min_edge: float = Query(0.03, ge=0.0, le=0.5),
        model_weight: float = Query(0.35, ge=0.0, le=1.0),
        half_life: float = Query(180.0, gt=1.0),
        max_legs: int = Query(5, ge=2, le=8),
        min_leg_edge: float = Query(0.03, ge=0.0, le=0.5),
        previews: bool = Query(True),
    ) -> JSONResponse:
        s = resolve(league)
        if not s.fixtures:
            raise HTTPException(
                status_code=404,
                detail="No upcoming fixtures with prices are loaded. The free feed is "
                       "empty between seasons — restart with --synthetic for a demo, or "
                       "supply your own fixtures CSV.",
            )
        return JSONResponse(build_slate_for(s, bankroll, kelly, min_edge, model_weight,
                                            half_life, max_legs, min_leg_edge, previews))

    def build_slate_for(s: DataStore, bankroll: float, kelly: float, min_edge: float,
                        model_weight: float, half_life: float, max_legs: int,
                        min_leg_edge: float, previews: bool) -> dict:
        """Build (or reuse) one league's priced card.

        Shared with /api/best, which needs every league's card at once — going
        through the same per-store cache means a cross-league shortlist costs
        nothing extra once the tabs have been looked at.
        """
        key = f"{bankroll}|{kelly}|{min_edge}|{model_weight}|{half_life}|{max_legs}|{min_leg_edge}|{previews}"
        with s.lock:
            if key in s._slate_cache:
                return s._slate_cache[key]
        config = _config_from_query(bankroll, kelly, min_edge, model_weight, half_life,
                                    max_legs, min_leg_edge)
        built = Engine(config).build_slate(s.matches, s.fixtures)
        payload = built.to_dict(include_contexts=previews)
        if isinstance(payload.get("model"), dict):
            _tag_current_squad(payload["model"], s.matches)
        payload["source"] = s.source
        payload["league"] = s.league
        payload["league_name"] = LEAGUES.get(s.league, s.league)
        payload["disclaimer"] = DISCLAIMER
        payload["markdown"] = render_markdown(built)
        with s.lock:
            s._slate_cache[key] = payload
        return payload

    @app.get("/api/best")
    def best(
        days: int = Query(7, ge=1, le=60),
        limit: int = Query(5, ge=1, le=25),
        scope: str = Query("all"),
        include: str = Query("singles"),
        league: str | None = Query(None),
        bankroll: float = Query(1000.0, gt=0),
        kelly: float = Query(0.25, gt=0, le=1.0),
        min_edge: float = Query(0.03, ge=0.0, le=0.5),
        model_weight: float = Query(0.35, ge=0.0, le=1.0),
        half_life: float = Query(180.0, gt=1.0),
        max_legs: int = Query(5, ge=2, le=8),
        min_leg_edge: float = Query(0.03, ge=0.0, le=0.5),
    ) -> JSONResponse:
        """The strongest bets kicking off within the next `days`.

        Ranked by expected log growth rather than raw edge. Edge alone rewards
        a longshot whose price the model happens to disagree with most, which
        is exactly where model error is largest; log growth is what fractional
        Kelly is trying to maximise, so it prefers a bet that will actually
        compound a bankroll over one with a fat headline number.

        Spans every loaded league by default — the reason to load seven is to
        pick from all of them, and a shortlist confined to whichever league the
        dropdown happens to be showing would defeat that.

        Singles only unless ``include=all``. Ranking every slip type together
        puts same-game doubles at the top of every shortlist, because those are
        priced off the joint distribution and disagree with the book most; they
        are also the least actionable, being worth taking only where a book
        multiplies the legs. A list headed "best bets" should be bets you can
        place, so multis are opt-in.
        """
        groups = (("singles", "multis", "same_game") if include == "all" else ("singles",))
        stores = ([resolve(league)] if scope == "league"
                  else [all_stores[code] for code in sorted(all_stores)])
        horizon = date.today() + timedelta(days=days)

        picks, leagues_seen, errors = [], [], []
        for s in stores:
            if not s.fixtures:
                continue
            leagues_seen.append(s.league)
            try:
                card = build_slate_for(s, bankroll, kelly, min_edge, model_weight,
                                       half_life, max_legs, min_leg_edge, False)
            except Exception as exc:                       # one bad league must not
                errors.append(f"{s.league}: {exc}")        # sink the whole shortlist
                continue
            for group in groups:
                for slip in card.get(group, []):
                    legs = slip.get("legs") or []
                    dates = [leg.get("date") for leg in legs if leg.get("date")]
                    if len(dates) != len(legs) or not dates:
                        continue
                    # Every leg must land inside the window: a multi that runs
                    # past it is not a bet for this week.
                    if max(dates) > horizon.isoformat():
                        continue
                    picks.append({**slip, "league": card["league"],
                                  "league_name": card["league_name"],
                                  "group": group, "kicks_off": min(dates)})

        picks.sort(key=lambda p: (p.get("log_growth") or 0.0, p.get("edge") or 0.0),
                   reverse=True)
        return JSONResponse({
            "days": days,
            "limit": limit,
            "scope": scope,
            "include": include,
            "through": horizon.isoformat(),
            "leagues_considered": leagues_seen,
            "n_candidates": len(picks),
            "bets": picks[:limit],
            "errors": errors,
            "disclaimer": DISCLAIMER,
        })

    @app.get("/api/ratings")
    def ratings(league: str | None = Query(None),
               half_life: float = Query(180.0, gt=1.0)) -> dict:
        s = resolve(league)
        config = Config.from_dict({"model": {"half_life_days": half_life}})
        model = Engine(config).fit(s.matches)
        return _tag_current_squad(model.to_dict(), s.matches)

    @app.get("/api/backtest")
    def backtest(
        league: str | None = Query(None),
        kelly: float = Query(0.25, gt=0, le=1.0),
        min_edge: float = Query(0.03, ge=0.0, le=0.5),
        model_weight: float = Query(0.35, ge=0.0, le=1.0),
        half_life: float = Query(180.0, gt=1.0),
        bankroll: float = Query(1000.0, gt=0),
        train_days: int = Query(400, ge=120),
        refit_every: int = Query(7, ge=1, le=60),
    ) -> JSONResponse:
        s = resolve(league)
        if len(s.matches) < 200:
            raise HTTPException(status_code=400,
                                detail="Not enough history loaded to backtest.")
        key = f"{kelly}|{min_edge}|{model_weight}|{half_life}|{bankroll}|{train_days}|{refit_every}"
        with s.lock:
            if key in s._backtest_cache:
                return JSONResponse(s._backtest_cache[key])
        config = _config_from_query(bankroll, kelly, min_edge, model_weight, half_life, 5,
                                    min_edge)
        result = run_backtest(s.matches, config=config, train_days=train_days,
                              refit_every_days=refit_every)
        payload = result.to_dict()
        payload["summary_text"] = result.summary()
        with s.lock:
            s._backtest_cache[key] = payload
        return JSONResponse(payload)

    @app.get("/")
    def index() -> FileResponse:
        return _web_file("index.html", "text/html")

    @app.get("/app.js")
    def script() -> FileResponse:
        return _web_file("app.js", "application/javascript")

    @app.get("/styles.css")
    def styles() -> FileResponse:
        return _web_file("styles.css", "text/css")

    @app.get("/logo.svg")
    def logo() -> FileResponse:
        return _web_file("logo.svg", "image/svg+xml")

    @app.get("/manifest.json")
    def manifest() -> FileResponse:
        return _web_file("manifest.json", "application/manifest+json")

    @app.get("/icon-{size}.png")
    def icon(size: int) -> FileResponse:
        if not (WEB_DIR / f"icon-{size}.png").exists():
            raise HTTPException(status_code=404, detail="no icon at that size")
        return _web_file(f"icon-{size}.png", "image/png")

    return app


def load_store(league: str = "E0", seasons: int = 4, offline: bool = False,
               use_synthetic: bool = False,
               price_mode: str = DEFAULT_PRICE_MODE,
               odds_provider: str = "footballdata",
               api_key: str | None = None,
               sport_key: str | None = None,
               team_aliases: dict[str, str] | None = None) -> DataStore:
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

    if odds_provider == "footballdata":
        try:
            fixtures = source.fixtures([league])
            print(f"  {len(fixtures)} upcoming fixtures with prices")
        except Exception as exc:
            print(f"  ! could not load fixtures: {exc}")
            fixtures = []
        source_label = f"football-data.co.uk ({price_mode} prices)"
    else:
        # Local import avoids a hard dependency on the providers package for
        # everyone who never asks for a live source.
        from ..data.providers import PROVIDER_INFO, get_provider
        from ..data.teams import reconcile_fixtures

        info = PROVIDER_INFO[odds_provider]
        print(f"Loading live prices from {info.title} ...")
        extra = {"sport_key": sport_key} if odds_provider == "theoddsapi" and sport_key else {}
        try:
            provider = get_provider(odds_provider, api_key=api_key, **extra)
            fixtures = provider.fixtures(league)
            print(f"  {len(fixtures)} upcoming fixtures with prices")
        except Exception as exc:
            print(f"  ! could not load live fixtures: {exc}")
            fixtures = []

        if fixtures:
            # A live provider spells teams differently than the results this
            # model is fitted on. Skipping this reconciliation is exactly the
            # bug that shipped first: fixtures loaded, matched nothing, and
            # 'serve' silently fell back to looking empty.
            known = {m.home for m in matches} | {m.away for m in matches}
            fixtures, report = reconcile_fixtures(fixtures, known,
                                                   extra_aliases=team_aliases)
            if report.dropped_fixtures or report.fuzzy or report.unresolved:
                print(report.render())
        source_label = f"{info.title} (live)"

    return DataStore(league=league, matches=matches, fixtures=fixtures,
                     source=source_label, loaded_at=date.today())


def start_refresh_loop(store: DataStore, minutes: int, **load_kwargs) -> threading.Thread | None:
    """Re-fetch fixtures and prices every `minutes`, in the background.

    The point of this is that the dashboard is worth opening at any hour
    without anyone having restarted anything. It is a daemon thread, so it
    never keeps the process alive on its own.

    A failed refresh is deliberately non-fatal: the previous data stays
    served and the error is surfaced on /api/health. Odds going stale is a
    far better outcome than the dashboard going dark because a provider
    had a bad minute.
    """
    if minutes <= 0:
        return None

    def loop() -> None:
        while True:
            time.sleep(minutes * 60)
            try:
                fresh = load_store(**load_kwargs)
                if fresh.matches:
                    store.replace_data(fresh.matches, fresh.fixtures, fresh.source)
                    print(f"[refresh] {datetime.now():%H:%M} — {len(fresh.fixtures)} "
                          f"fixtures, {len(fresh.matches)} matches")
                else:
                    raise RuntimeError("refresh returned no match history")
            except Exception as exc:
                store.last_refresh_error = f"{datetime.now():%Y-%m-%d %H:%M} — {exc}"
                print(f"[refresh] failed, keeping previous data: {exc}")

    thread = threading.Thread(target=loop, daemon=True, name="bettingedge-refresh")
    thread.start()
    return thread


def run_server(host: str = "127.0.0.1", port: int = 8000, league: str = "E0",
               seasons: int = 4, offline: bool = False, use_synthetic: bool = False,
               config: Config | None = None,
               price_mode: str = DEFAULT_PRICE_MODE, token: str | None = None,
               odds_provider: str = "footballdata", api_key: str | None = None,
               sport_key: str | None = None,
               team_aliases: dict[str, str] | None = None,
               refresh_minutes: int = 180) -> None:
    import uvicorn

    global STORE, STORES
    # Comma-separated leagues each get their own fit and their own fixture
    # feed — ratings from one division are not comparable to another — and
    # each becomes a choice in the dashboard's league dropdown rather than
    # being merged into one card, the way `recommend` merges them.
    codes = [code.strip().upper() for code in league.split(",") if code.strip()] or ["E0"]
    if use_synthetic:
        codes = codes[:1]      # the synthetic league ignores the code entirely

    STORES = {}
    live_leagues_refreshing = 0
    for code in codes:
        load_kwargs = dict(league=code, seasons=seasons, offline=offline,
                           use_synthetic=use_synthetic, price_mode=price_mode,
                           odds_provider=odds_provider, api_key=api_key,
                           sport_key=sport_key, team_aliases=team_aliases)
        built_store = load_store(**load_kwargs)
        STORES[built_store.league] = built_store
        if refresh_minutes > 0:
            start_refresh_loop(built_store, refresh_minutes, **load_kwargs)
            if odds_provider != "footballdata":
                live_leagues_refreshing += 1

    if not any(s.matches for s in STORES.values()):
        raise SystemExit(
            "No match history could be loaded for any requested league. Check the "
            "league code(s), or start with --synthetic to explore the app offline."
        )

    if refresh_minutes > 0:
        per_month = (24 * 60 / refresh_minutes) * 30 * max(live_leagues_refreshing, 1)
        note = f"refreshing every {refresh_minutes} min"
        if live_leagues_refreshing:
            # The Odds API free tier is 500 requests/month, shared across every
            # league loaded — burning through it silently at 3am is exactly the
            # kind of thing worth saying out loud, and it scales with league count.
            note += (f" — {live_leagues_refreshing} live league(s), about "
                     f"{per_month:.0f} provider calls/month combined")
            if per_month > 450:
                note += "  ** likely to exhaust a 500/month free tier **"
        print(note)

    primary_code = codes[0] if codes[0] in STORES else next(iter(STORES))
    STORE = STORES[primary_code]
    extra_stores = {code: s for code, s in STORES.items() if code != primary_code}
    if extra_stores:
        print(f"Leagues loaded: {', '.join(STORES)} — switch between them in the "
              "dashboard's league dropdown.")

    app = create_app(STORE, token=token, extra_stores=extra_stores)
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
