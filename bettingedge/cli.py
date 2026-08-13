"""Command line interface.

    bettingedge demo                       run on the bundled offline dataset
    bettingedge recommend --league E0      today's card from live prices
    bettingedge backtest  --league E0      walk-forward test on real history
    bettingedge verify    --league E0      full verification ladder on real data
    bettingedge scan                       compare leagues side by side
    bettingedge ratings   --league E0      current team strength table
    bettingedge serve                      web dashboard on localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from .backtest.engine import run_backtest
from .config import Config
from .data import synthetic
from .data.csvsource import load_fixtures_csv, load_results_csv
from .data.providers import (
    PROVIDER_INFO,
    ProviderError,
    ReplayProvider,
    available_providers,
    describe_capture,
    get_provider,
)
from .data.teams import parse_alias_arguments, reconcile_fixtures
from .data.footballdata import (
    DEFAULT_PRICE_MODE,
    LEAGUES,
    PRICE_MODES,
    FootballDataUK,
    recent_seasons,
    resolve_price_mode,
)
from .pipeline import Engine
from .report import DISCLAIMER, render_markdown, render_slate
from .scan import DEFAULT_LEAGUES, scan_leagues
from .verify import sharp_only, verify


# --------------------------------------------------------------------------
# shared argument handling
# --------------------------------------------------------------------------
def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--league", default="E0",
                        help="football-data.co.uk league code, or several separated by "
                             "commas (e.g. E0,E1,EC). Each division is fitted separately; "
                             "the card is combined. See `bettingedge leagues`.")
    parser.add_argument("--seasons", type=int, default=4,
                        help="how many recent seasons of history to fit on (default: 4)")
    parser.add_argument("--results-csv", help="use your own results CSV instead of downloading")
    parser.add_argument("--fixtures-csv", help="use your own fixtures/odds CSV")
    parser.add_argument("--cache", help="directory for downloaded files")
    parser.add_argument("--offline", action="store_true",
                        help="never hit the network; use cached files only")
    parser.add_argument("--synthetic", action="store_true",
                        help="use the generated offline dataset (no network needed)")
    parser.add_argument("--odds-provider", default="footballdata",
                        choices=available_providers(),
                        help="where live prices come from (default: footballdata). "
                             "Live providers need an API key — see `bettingedge providers`.")
    parser.add_argument("--api-key", help="API key for the odds provider "
                                          "(otherwise read from the environment)")
    parser.add_argument("--days-ahead", type=int, default=7,
                        help="how far ahead to pull fixtures from a live provider")
    parser.add_argument("--team-alias", action="append", metavar="PROVIDER=MODEL",
                        help="force a team name mapping, e.g. "
                             "--team-alias \"Manchester United=Man United\". Repeatable.")
    parser.add_argument("--sport-key", help="override the provider's league identifier "
                                            "(see `bettingedge providers --list-sports`)")
    parser.add_argument("--dump-raw", help="write the provider's raw payload here for "
                                           "debugging")
    parser.add_argument("--replay-raw", metavar="PATH",
                        help="read prices from a payload captured earlier by "
                             "`bettingedge capture`, making no network calls at all")
    parser.add_argument("--price-mode", default=DEFAULT_PRICE_MODE, choices=list(PRICE_MODES),
                        help="which price snapshot to read: 'best-closing' (default, best "
                             "price across books at the close), 'early' (pre-closing price "
                             "scored against the closing line — the honest CLV test), or "
                             "'sharp-only' (assume you only get the sharp closing line)")


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bankroll", type=float, help="bankroll for staking (default: 1000)")
    parser.add_argument("--kelly", type=float, help="Kelly fraction, e.g. 0.25 (default: 0.25)")
    parser.add_argument("--min-edge", type=float,
                        help="minimum expected value per unit staked, e.g. 0.03")
    parser.add_argument("--model-weight", type=float,
                        help="weight on the model when blending with the market (0-1)")
    parser.add_argument("--half-life", type=float,
                        help="time-decay half life in days (default: 180)")
    parser.add_argument("--max-legs", type=int, help="largest accumulator to build")
    parser.add_argument("--min-confidence", type=float,
                        help="drop bets scoring below this out of 100. Raised "
                             "automatically when the model is on a thin sample.")
    parser.add_argument("--xg-weight", type=float,
                        help="how far to move team ratings from goals toward a "
                             "shots-based expected-goals proxy (0-1, default 0.5)")


def _config_from(args: argparse.Namespace) -> Config:
    overrides: dict[str, dict] = {"model": {}, "market": {}, "selection": {},
                                  "parlay": {}, "staking": {}}
    if getattr(args, "bankroll", None) is not None:
        overrides["staking"]["bankroll"] = args.bankroll
    if getattr(args, "kelly", None) is not None:
        overrides["staking"]["kelly_fraction"] = args.kelly
    if getattr(args, "min_edge", None) is not None:
        overrides["selection"]["min_edge"] = args.min_edge
    if getattr(args, "model_weight", None) is not None:
        overrides["market"]["model_weight"] = args.model_weight
    if getattr(args, "half_life", None) is not None:
        overrides["model"]["half_life_days"] = args.half_life
    if getattr(args, "max_legs", None) is not None:
        overrides["parlay"]["max_legs"] = args.max_legs
    if getattr(args, "xg_weight", None) is not None:
        overrides["model"]["xg_weight"] = args.xg_weight
    if getattr(args, "min_confidence", None) is not None:
        overrides["selection"]["min_confidence"] = args.min_confidence
    return Config.from_dict(overrides)


def _leagues(args: argparse.Namespace) -> list[str]:
    raw = getattr(args, "league", "E0") or "E0"
    return [code.strip().upper() for code in raw.split(",") if code.strip()]


def _source(args: argparse.Namespace) -> FootballDataUK:
    kwargs = {
        "offline": bool(getattr(args, "offline", False)),
        "price_mode": getattr(args, "price_mode", DEFAULT_PRICE_MODE),
    }
    if getattr(args, "cache", None):
        kwargs["cache_dir"] = args.cache
    return FootballDataUK(**kwargs)


def _load_history(args: argparse.Namespace):
    """Completed matches, from whichever source was requested."""
    if getattr(args, "synthetic", False):
        matches, _ = synthetic.generate(seasons=max(2, args.seasons))
        return matches
    if getattr(args, "results_csv", None):
        return load_results_csv(args.results_csv)
    seasons = recent_seasons(args.seasons)
    mode = resolve_price_mode(getattr(args, "price_mode", DEFAULT_PRICE_MODE))
    print(f"Loading {args.league} ({LEAGUES.get(args.league, 'unknown league')}), "
          f"seasons {', '.join(seasons)} ...")
    print(f"  price mode: {mode.key} — {mode.description}")
    matches = _source(args).results(args.league, seasons)
    print(f"  {len(matches)} matches loaded")
    return matches


def _load_fixtures(args: argparse.Namespace):
    if getattr(args, "fixtures_csv", None):
        return load_fixtures_csv(args.fixtures_csv, default_league=args.league)
    if getattr(args, "synthetic", False):
        _, fixtures = synthetic.generate(seasons=max(2, args.seasons))
        return fixtures

    if getattr(args, "replay_raw", None):
        print(f"Replaying captured prices from {args.replay_raw} (no network) ...")
        fixtures = ReplayProvider(args.replay_raw).fixtures(
            args.league, days_ahead=getattr(args, "days_ahead", 7))
        print(f"  {len(fixtures)} fixtures with prices")
        print("  NOTE: a capture is a snapshot. Use it to check the pipeline, "
              "not to place bets.")
        return fixtures

    provider_name = getattr(args, "odds_provider", "footballdata")
    if provider_name == "footballdata":
        print("Loading upcoming fixtures and prices ...")
        fixtures = _source(args).fixtures([args.league])
    else:
        info = PROVIDER_INFO[provider_name]
        print(f"Loading live prices from {info.title} ...")
        extra = {}
        if provider_name == "theoddsapi" and getattr(args, "sport_key", None):
            extra["sport_key"] = args.sport_key
        provider = get_provider(
            provider_name,
            api_key=getattr(args, "api_key", None),
            dump_raw=getattr(args, "dump_raw", None),
            **extra,
        )
        fixtures = provider.fixtures(args.league,
                                     days_ahead=getattr(args, "days_ahead", 7))
    print(f"  {len(fixtures)} fixtures with prices")
    return fixtures


def _reconcile(fixtures, matches, args: argparse.Namespace):
    """Map provider team names onto the names the model was fitted on.

    Always run, even for the default source: a mismatch here silently prices
    every fixture off league-average ratings.
    """
    if not fixtures or not matches:
        return fixtures
    known = {m.home for m in matches} | {m.away for m in matches}
    try:
        aliases = parse_alias_arguments(getattr(args, "team_alias", None))
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}")

    resolved, report = reconcile_fixtures(fixtures, known, extra_aliases=aliases)
    if report.dropped_fixtures or report.fuzzy or report.unresolved:
        print()
        print(report.render())
    return resolved


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_demo(args: argparse.Namespace) -> int:
    print("Generating the offline demo league (synthetic data, no network) ...")
    matches, fixtures = synthetic.generate(seasons=args.seasons)
    print(f"  {len(matches)} historical matches, {len(fixtures)} fixtures to price\n")
    slate = Engine(_config_from(args)).build_slate(matches, fixtures)
    print(render_slate(slate, detail=not args.brief, previews=args.previews))
    _maybe_write(args, slate)
    print("\nNOTE: this demo runs on generated data so it works with no network "
          "access.\nThe numbers exercise the machinery; they are not real betting "
          "opportunities.\nRun `bettingedge recommend --league E0` for real prices.")
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    codes = _leagues(args)
    groups = []
    for code in codes:
        per_league = argparse.Namespace(**{**vars(args), "league": code})
        matches = _load_history(per_league)
        if not matches:
            print(f"  {code}: no history, skipping", file=sys.stderr)
            continue
        fixtures = _load_fixtures(per_league)
        if not fixtures:
            print(f"  {code}: no upcoming fixtures with prices, skipping")
            continue
        fixtures = _reconcile(fixtures, matches, args)
        if not fixtures:
            print(f"  {code}: every fixture dropped in team-name matching")
            continue
        if args.max_fixtures:
            fixtures = fixtures[: args.max_fixtures]
        groups.append((matches, fixtures))

    if not groups:
        print("\nNothing to price.\n"
              "The free fixtures feed only covers the next few days and is empty "
              "between seasons.\nSupply your own with --fixtures-csv, or try "
              "`bettingedge demo`.", file=sys.stderr)
        return 1

    slate = Engine(_config_from(args)).build_multi_slate(groups)
    print()
    print(render_slate(slate, detail=not args.brief, previews=args.previews))
    _maybe_write(args, slate)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    matches = _load_history(args)
    if len(matches) < 200:
        print(f"Only {len(matches)} matches — a backtest needs a lot more history. "
              "Increase --seasons.", file=sys.stderr)
        return 1
    config = _config_from(args)

    if args.pessimistic:
        stripped = sharp_only(matches)
        if len(stripped) < 200:
            print("\nNot enough matches carry a sharp closing price for the pessimistic "
                  "run.\nDrop --pessimistic, or load seasons that include the closing "
                  "columns.", file=sys.stderr)
            return 1
        print(f"\nPessimistic mode: settling every bet at the sharp closing price "
              f"({len(stripped)} of {len(matches)} matches have one).")
        print("Price shopping contributes nothing here — what survives is model edge.")
        matches = stripped

    print(f"\nWalk-forward backtest over {len(matches)} matches "
          f"({matches[0].date} to {matches[-1].date})")
    print(f"Training window opens after {args.train_days} days; refitting every "
          f"{args.refit_every} days.\n")
    result = run_backtest(
        matches,
        config=config,
        train_days=args.train_days,
        refit_every_days=args.refit_every,
        progress=args.verbose,
    )
    print()
    print("=" * 62)
    print("  BACKTEST RESULT")
    print("=" * 62)
    print(result.summary())
    if result.calibration:
        print("\nCalibration (predicted vs actual strike rate on bets placed):")
        print(f"  {'bucket':>12}  {'n':>5}  {'predicted':>9}  {'actual':>7}")
        for row in result.calibration:
            print(f"  {row['bucket']:>12}  {row['n']:>5}  {row['predicted']:>9.1%}  "
                  f"{row['actual']:>7.1%}")
    print()
    print(_backtest_verdict(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    print(f"\n{DISCLAIMER}")
    return 0


def _backtest_verdict(result) -> str:
    """Say plainly whether the numbers support betting this."""
    lines = ["VERDICT"]
    if not result.bets:
        return ("VERDICT\n  No bets qualified. Either the filters are too tight or the "
                "model found\n  no disagreement with the market worth backing.")

    beats_market = result.model_log_loss < result.market_log_loss
    if beats_market:
        lines.append("  The model's probabilities scored better than the market's on log "
                     "loss.\n  That is the necessary condition for an edge to be real.")
    else:
        lines.append("  The market's probabilities scored BETTER than the model's on log "
                     "loss.\n  Any positive ROI above is very likely variance, not skill. "
                     "Do not bet\n  this configuration with real money.")

    if result.average_clv > 0.005:
        lines.append(f"  Average closing-line value of {result.average_clv * 100:+.2f}% is "
                     "the strongest\n  available evidence that the prices taken were good.")
    elif result.average_clv < -0.005:
        lines.append(f"  Negative closing-line value ({result.average_clv * 100:+.2f}%) "
                     "means the market moved\n  against these bets on average — a bad sign "
                     "regardless of the ROI.")

    if len(result.bets) < 300:
        lines.append(f"  Only {len(result.bets)} bets. At football-sized edges you need "
                     "well over a\n  thousand before ROI means anything at all.")
    return "\n".join(lines)


def cmd_verify(args: argparse.Namespace) -> int:
    """Run the whole verification ladder against whichever data is loaded."""
    matches = _load_history(args)
    if not matches:
        print("No history loaded — nothing to verify.", file=sys.stderr)
        return 1

    if getattr(args, "synthetic", False):
        label = "synthetic offline dataset"
        mode_note = ("Synthetic data: simulated books price off the true probabilities, "
                     "so treat every performance number here as a machinery check only.")
    else:
        mode = resolve_price_mode(args.price_mode)
        label = f"{args.league} ({LEAGUES.get(args.league, 'unknown league')})"
        mode_note = f"Price mode '{mode.key}': {mode.measures}"

    report = verify(
        matches,
        config=_config_from(args),
        label=label,
        train_days=args.train_days,
        refit_every=args.refit_every,
        price_mode_note=mode_note,
        run_pessimistic=not args.skip_pessimistic,
        progress=(lambda line: print(f"  {line}")) if not args.quiet else None,
    )
    print()
    print(report.render())
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")
    print(DISCLAIMER)
    # Non-zero exit on a failed check, so this can gate a script.
    return 1 if report.status == "FAIL" else 0


def cmd_ratings(args: argparse.Namespace) -> int:
    matches = _load_history(args)
    if not matches:
        print("No history loaded.", file=sys.stderr)
        return 1
    model = Engine(_config_from(args)).fit(matches)
    print(f"\nDixon-Coles ratings — {len(matches)} matches through "
          f"{model.fitted_through}")
    print(f"Home advantage {model.home_advantage:+.3f} log-goals, rho {model.rho:+.3f}\n")
    print(f"{'#':>3}  {'team':<26}{'attack':>8}{'defence':>9}{'net':>8}{'played':>8}")
    print("-" * 62)
    for position, rating in enumerate(model.ratings(), start=1):
        print(f"{position:>3}  {rating.team:<26}{rating.attack:>+8.3f}"
              f"{rating.defence:>+9.3f}{rating.net:>+8.3f}{rating.matches:>8}")
    print("\nAttack and defence are in log-goals relative to an average team in this "
          "league.\nHigher is better for both. Net is their sum.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .api.server import run_server

    run_server(host=args.host, port=args.port, league=args.league, seasons=args.seasons,
               offline=args.offline, use_synthetic=args.synthetic,
               config=_config_from(args), price_mode=args.price_mode,
               token=getattr(args, "token", None))
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """Run the same walk-forward test across several divisions."""
    leagues = ([code.strip().upper() for code in args.leagues.split(",")]
               if args.leagues else list(DEFAULT_LEAGUES))
    print(f"\nScanning {len(leagues)} league(s) over {args.seasons} seasons.")
    print("Each one is loaded, fitted walk-forward and re-run without price "
          "shopping, so this takes a few minutes.\n")
    report = scan_leagues(
        leagues=leagues,
        seasons=args.seasons,
        config=_config_from(args),
        train_days=args.train_days,
        refit_every=args.refit_every,
        include_pessimistic=not args.skip_pessimistic,
        offline=args.offline,
        progress=(lambda line: print(f"  {line}")) if not args.quiet else None,
    )
    print()
    print(report.render())
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    print(f"\n{DISCLAIMER}")
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    """Save a provider's raw response so it can be replayed without network."""
    out = Path(args.out)
    if args.odds_provider == "footballdata":
        print("Capture is for live API providers. football-data.co.uk is already "
              "cached on disk — use --offline to work from that cache.",
              file=sys.stderr)
        return 1

    info = PROVIDER_INFO[args.odds_provider]
    print(f"Capturing {args.league} prices from {info.title} ...")
    extra = {}
    if args.odds_provider == "theoddsapi" and getattr(args, "sport_key", None):
        extra["sport_key"] = args.sport_key
    provider = get_provider(args.odds_provider, api_key=args.api_key, dump_raw=out, **extra)
    fixtures = provider.fixtures(args.league, days_ahead=args.days_ahead)

    print(f"\nWrote {out} ({out.stat().st_size / 1024:.0f} KB)\n")
    print(describe_capture(fixtures))
    print(f"""
The file holds the provider's untouched response. It contains no API key —
keys travel in the request, not the reply — so it is safe to commit or hand to
someone else for debugging. Have a look before sharing anyway.

Replay it anywhere, with no network:

    bettingedge recommend --league {args.league} --replay-raw {out}

A capture is a snapshot: use it to check parsing, team matching and pricing,
not to place bets. Prices go stale within minutes.""")
    return 0


def cmd_providers(args: argparse.Namespace) -> int:
    if args.list_sports:
        provider = get_provider("theoddsapi", api_key=args.api_key)
        print("\nSport keys your Odds API key can access:\n")
        for sport in provider.sports():
            if str(sport.get("key", "")).startswith("soccer"):
                print(f"  {sport.get('key'):<40} {sport.get('title')}")
        return 0

    print("\nOdds sources\n" + "=" * 74)
    for info in PROVIDER_INFO.values():
        print(f"\n  {info.title}   [--odds-provider {info.key}]")
        print(f"    {info.url}")
        print(f"    free tier : {info.free_tier}")
        print(f"    markets   : {info.markets}")
        if info.env_vars:
            print(f"    api key   : set {' or '.join(info.env_vars)}, or pass --api-key")
        for line in _wrap_text(info.notes, 66):
            print(f"    {line}")

    print("""
Getting started with live prices
--------------------------------
  1. Sign up for a free key at https://the-odds-api.com
  2. export ODDS_API_KEY=your-key-here
  3. bettingedge recommend --league E0 --odds-provider theoddsapi

History still comes from football-data.co.uk, because it is the only free
source that carries results and closing prices together — which is what makes
backtesting honest. Live providers supply the upcoming card only.

Because the two sources spell teams differently, names are reconciled
automatically and anything unresolved is reported rather than guessed at. Fix
stragglers with --team-alias "Provider Name=Model Name".
""")
    return 0


def _wrap_text(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def cmd_leagues(args: argparse.Namespace) -> int:
    print("\nLeague codes (football-data.co.uk):\n")
    for code, name in LEAGUES.items():
        print(f"  {code:<5} {name}")
    print()
    return 0


def _maybe_write(args: argparse.Namespace, slate) -> None:
    if getattr(args, "json", None):
        Path(args.json).write_text(json.dumps(slate.to_dict(), indent=2, default=str),
                                   encoding="utf-8")
        print(f"\nWrote {args.json}")
    if getattr(args, "markdown", None):
        Path(args.markdown).write_text(render_markdown(slate), encoding="utf-8")
        print(f"Wrote {args.markdown}")


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bettingedge",
        description="Data-driven football betting analysis: model the goals, price "
                    "the markets, find the value, size the stake.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=DISCLAIMER,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run on the bundled offline dataset")
    demo.add_argument("--seasons", type=int, default=4)
    demo.add_argument("--brief", action="store_true", help="skip the written analysis")
    demo.add_argument("--previews", action="store_true", help="include fixture previews")
    demo.add_argument("--json", help="also write the slate as JSON")
    demo.add_argument("--markdown", help="also write the slate as Markdown")
    _add_config_arguments(demo)
    demo.set_defaults(func=cmd_demo)

    recommend = subparsers.add_parser("recommend", help="recommend bets for upcoming fixtures")
    _add_data_arguments(recommend)
    _add_config_arguments(recommend)
    recommend.add_argument("--brief", action="store_true", help="skip the written analysis")
    recommend.add_argument("--previews", action="store_true", help="include fixture previews")
    recommend.add_argument("--max-fixtures", type=int, help="only price the first N fixtures")
    recommend.add_argument("--json", help="also write the slate as JSON")
    recommend.add_argument("--markdown", help="also write the slate as Markdown")
    recommend.set_defaults(func=cmd_recommend)

    backtest = subparsers.add_parser("backtest", help="walk-forward test on real history")
    _add_data_arguments(backtest)
    _add_config_arguments(backtest)
    backtest.add_argument("--train-days", type=int, default=400,
                          help="days of history before the first simulated bet")
    backtest.add_argument("--refit-every", type=int, default=7,
                          help="refit the model every N days (default: 7)")
    backtest.add_argument("--pessimistic", action="store_true",
                          help="settle every bet at the sharp closing price instead of the "
                               "best price across books — removes all price-shopping edge")
    backtest.add_argument("--json", help="write full results as JSON")
    backtest.add_argument("--verbose", action="store_true")
    backtest.set_defaults(func=cmd_backtest)

    verify_cmd = subparsers.add_parser(
        "verify", help="run the full verification ladder on real data")
    _add_data_arguments(verify_cmd)
    _add_config_arguments(verify_cmd)
    verify_cmd.add_argument("--train-days", type=int, default=400,
                            help="days of history before the first simulated bet")
    verify_cmd.add_argument("--refit-every", type=int, default=14,
                            help="refit the model every N days (default: 14)")
    verify_cmd.add_argument("--skip-pessimistic", action="store_true",
                            help="skip the sharp-price-only re-run (roughly halves runtime)")
    verify_cmd.add_argument("--json", help="write the report as JSON")
    verify_cmd.add_argument("--quiet", action="store_true", help="no progress output")
    verify_cmd.set_defaults(func=cmd_verify)

    ratings = subparsers.add_parser("ratings", help="show the team strength table")
    _add_data_arguments(ratings)
    _add_config_arguments(ratings)
    ratings.set_defaults(func=cmd_ratings)

    serve = subparsers.add_parser("serve", help="run the web dashboard")
    _add_data_arguments(serve)
    _add_config_arguments(serve)
    serve.add_argument("--host", default="127.0.0.1",
                       help="use 0.0.0.0 to reach it from your phone on the same network")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--token", help="require this token to access the dashboard "
                                       "(or set BETTINGEDGE_TOKEN)")
    serve.set_defaults(func=cmd_serve)

    leagues = subparsers.add_parser("leagues", help="list supported league codes")
    leagues.set_defaults(func=cmd_leagues)

    scan = subparsers.add_parser(
        "scan", help="compare model performance across several leagues")
    scan.add_argument("--leagues", help="comma-separated league codes "
                                        "(default: a broad sweep)")
    scan.add_argument("--seasons", type=int, default=6)
    scan.add_argument("--train-days", type=int, default=400)
    scan.add_argument("--refit-every", type=int, default=21)
    scan.add_argument("--skip-pessimistic", action="store_true")
    scan.add_argument("--offline", action="store_true")
    scan.add_argument("--json", help="write the comparison as JSON")
    scan.add_argument("--quiet", action="store_true")
    _add_config_arguments(scan)
    scan.set_defaults(func=cmd_scan)

    capture = subparsers.add_parser(
        "capture",
        help="save a provider's raw response so it can be replayed with no network")
    _add_data_arguments(capture)
    capture.add_argument("--out", default="odds-capture.json",
                         help="where to write the payload (default: odds-capture.json)")
    capture.set_defaults(func=cmd_capture)

    providers = subparsers.add_parser(
        "providers", help="list live odds sources and how to set them up")
    providers.add_argument("--list-sports", action="store_true",
                           help="query The Odds API for the sport keys your account covers")
    providers.add_argument("--api-key", help="API key, if not in the environment")
    providers.set_defaults(func=cmd_providers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # surface a clean message, not a traceback
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
