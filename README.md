# bettingedge

A data-driven football betting engine. It models goals, prices every market from
that model, compares its prices with the bookmaker's, and recommends singles,
doubles, trebles and accumulators — each with a written analysis explaining
where the number came from and what would make it wrong.

It is a modelling and discipline tool, not a tipping service. The honest
summary of what it does: it estimates probabilities, measures them against the
price on offer, sizes bets so a real edge can survive variance, and tells you
plainly when it cannot find one.

```
bettingedge demo                    # runs offline, no network, no setup
bettingedge recommend --league E0,E1,EC   # combined card across divisions
bettingedge verify    --league E0   # five-stage check that any of this is real
bettingedge backtest  --league E0   # walk-forward test on real history
bettingedge serve                   # web dashboard on localhost:8000
```

---

## Install

```bash
git clone https://github.com/arkwood-prog/accark.git
cd accark
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
```

Python 3.10+. Dependencies: numpy, scipy, requests, fastapi, uvicorn.

Check it works with no network and no data setup:

```bash
bettingedge demo
```

---

## How a price is made

Nine steps, no black boxes. Every intermediate value is visible in the
dashboard and in the JSON export.

**1. Fit the goal model.** A [Dixon-Coles](https://doi.org/10.1111/1467-9876.00065)
bivariate Poisson is fitted by maximum likelihood to every result in the
history window:

```
home goals ~ Poisson(λ),  λ = exp(attack_home − defence_away + γ)
away goals ~ Poisson(μ),  μ = exp(attack_away − defence_home)
```

Each team gets an attack and a defence rating. `γ` is home advantage. A
correction factor `τ(x,y)` is applied to the four low-score cells (0-0, 1-0,
0-1, 1-1) where independent Poissons are known to misprice — this is the whole
point of Dixon-Coles over naive Poisson. Matches are weighted by exponential
time decay (default half-life 180 days) so last month counts for more than last
year, ratings are mean-centred for identifiability, and an L2 penalty shrinks
them toward league average so newly-promoted sides do not take extreme values.

The attacking target is not raw goals. Goals are a noisy record of how a team
played — twelve chances and one goal reads as one goal. Shots on target are
about ten times more frequent and so a far steadier signal, so ratings are
fitted on a blend of goals and a shots-based expected-goals proxy
(`xg_weight`, default 0.5). Conversion rates are fitted per league from its own
history rather than assumed, because lower divisions take worse shots. Swept on
six Premier League seasons, 0.5 forecast better than either extreme: 1X2 log
loss 0.9829, against 0.9877 for pure goals and 0.9887 for pure expected goals.
Matches with no shot data fall back to goals automatically.

The likelihood is optimised with an **analytic gradient**. That is not a
micro-optimisation: a walk-forward backtest refits the model hundreds of times,
and finite differences would make it unusable.

**2. Build a score matrix.** For each fixture the model produces the full joint
distribution over scorelines — the probability of every 0-0, 2-1, 3-2, and so on.

**3. Derive every market from that one object.** Match result, double chance,
over/under and both-teams-to-score are all sums over regions of the same matrix,
expressed as boolean masks:

```python
P(selection)       = (matrix * mask).sum()
P(sel_a AND sel_b) = (matrix * mask_a * mask_b).sum()
```

Two consequences. Markets can never contradict each other — double chance
always equals the sum of its 1X2 parts, by construction. And same-match
combinations get an **exact** joint probability rather than the wrong answer you
get by multiplying two correlated legs.

**4. Strip the bookmaker's margin.** Prices imply probabilities summing to more
than 100%. Three methods are implemented — multiplicative, power, and
[Shin](https://doi.org/10.2307/2234526) — because the margin is *not* spread
evenly across outcomes; books load more of it onto longshots. Shin is the
default for 3-way markets.

**5. Blend model with market.** The two opinions are averaged in **logit
space**, which is the right space for combining beliefs about probabilities, and
renormalised so each market still sums to one.

The default weight is 35% model / 65% market. This is the most important
judgement in the project and it is deliberately humble: closing prices in liquid
football markets are close to efficient, and a model that ignores them will
find "edges" that are really just model error. The model is a tilt on the
market, not a replacement for it.

**6. Re-anchor the score matrix.** The blended probabilities are pushed back
into the joint distribution by iterative proportional fitting, so correct scores
and same-game combinations inherit the market-anchored view rather than the raw
model's.

**7. Measure the edge.** `EV = p × odds − 1`, against the best price available.
Only selections clearing the threshold survive, and each is scored 0–100 for
confidence from data coverage, market liquidity, margin, model-market
disagreement and edge size. That last one *saturates and then penalises*: an
implausibly large edge usually means the model is wrong or the price is stale.

**8. Build the slips.** Singles, then doubles, trebles and accumulators from
different matches. Same-match legs are priced from the joint distribution
instead of multiplied.

**9. Size the bets.** Fractional Kelly (default a quarter) with per-bet caps —
tighter for multis — and a slate-wide exposure ceiling, because a Saturday card
does not settle one bet at a time.

---

## On accumulators, honestly

Accumulators are in here because they were asked for and because the maths is
worth showing. They are not where the edge is, and the engine says so:

- **The margin compounds.** A book holding 5% per leg holds roughly 16% on a
  treble. A multi built from break-even legs is a losing bet. Legs must
  therefore clear a *higher* individual bar than a single would.
- **Correlated legs cannot be multiplied.** "Home win" and "Over 2.5" in the
  same match are not independent. Cross-match legs are multiplied; same-match
  legs are priced from the joint score distribution, and the engine reports the
  correlation factor so you can see how wrong the naive price is.
- **Ranking is by expected log growth**, not raw EV. Two slips can have
  identical expected value while one lands 60% of the time and the other 4%.
  Log growth is what the Kelly criterion actually maximises and it correctly
  prefers the first.

An empty accumulator section is the normal, correct answer on most cards.

---

## Reading the output

```
1. Single  [High]  |  odds 2.17  |  51.0% to land  |  edge +10.82%  |  confidence 89/100
     - Southgate Vale v Kingsmere (16 Aug): Kingsmere to win @ 2.17
       [model 57.3% | market 47.4% | blend 51.0% -> fair 1.96]
     Stake 6.00 -> returns 14.12 (profit +8.12)
```

| Field | Meaning |
|---|---|
| `model` | The goal model's probability, before it sees the price |
| `market` | The bookmaker's probability with the margin stripped out |
| `blend` | The two combined in logit space — this is what the stake is based on |
| `fair` | Fair odds implied by the blend. Bet only when the price beats this |
| `edge` | Expected profit per unit staked |
| `confidence` | 0–100, from data coverage, liquidity, margin and model-market agreement |

Each recommendation also carries a written analysis: verdict, model read,
market read (including whether the edge is real disagreement or just price
shopping), form, staking, and a specific list of risks.

---

## Verifying with live data

```bash
bettingedge verify --league E0 --seasons 6
```

One command runs the whole ladder, cheapest and most diagnostic first, and
exits non-zero if anything fails so you can gate a script on it.

| Stage | What it answers | Example checks |
|---|---|---|
| 1. Data integrity | Did the right numbers arrive? | goals/game 2.2–3.6, home win rate 35–55%, price coverage, median overround |
| 2. Model sanity | Does the fit describe football? | home advantage +0.10 to +0.45, ρ in range, rating spread, top/bottom teams printed for your eyes |
| 3. Forecast quality | Does the model know anything? | log loss vs the market, Brier, worst calibration band |
| 4. Price realism | Is the edge real or just shopping? | yield at best price vs **sharp price only**, hit rate vs implied |
| 5. Statistical power | Does the number mean anything? | yield ± 2 standard errors, bets needed to prove a 2% edge |

Several checks fire on results that look **too good**, because on this problem
an implausible number is far more likely to be a leak than an edge. A model
beating the closing line by more than 0.02 nats, or a yield above 8%, gets
warned about rather than celebrated.

Stage 2 is the one to read with your own football knowledge. It prints the
strongest and weakest teams; if that list looks wrong to you, stop — no
downstream maths fixes a model that disagrees with the table.

### Price modes

Each football-data.co.uk file carries two snapshots per market: a pre-closing
price and a closing price. Which pair you read decides what a backtest actually
measures, so it is an explicit flag rather than a buried default.

| `--price-mode` | You bet at | Scored against | What it measures |
|---|---|---|---|
| `best-closing` *(default)* | best closing price across all books | Pinnacle closing | Optimistic. Both sides are closing prices, so its CLV number reflects price shopping, not market movement |
| `early` | best price *before* the close | Pinnacle closing | **The honest CLV test.** Positive means the market moved toward your bet after you placed it |
| `sharp-only` | Pinnacle closing | itself | Pessimistic. No shopping at all — whatever survives is model edge |

```bash
bettingedge verify   --league E0 --seasons 6 --price-mode early
bettingedge backtest --league E0 --seasons 6 --pessimistic
```

`--pessimistic` applies the same stripping to any source, including your own
CSVs, by settling every bet at the sharp closing price. Expect the bet count to
fall sharply and CLV to go negative — you cannot beat a line you are taking.

The gap between the two runs is the number to internalise: if the edge only
exists at the best price across ten books, it is real money but it is shopping,
not modelling, and it is what gets accounts limited.

## Backtesting

```bash
bettingedge backtest --league E0 --seasons 6
```

The rules that make it honest are enforced in code:

- The model is refitted using **only matches already played** at the time of
  each simulated bet.
- Bets settle at the **best price actually on offer**, taken from the same
  historical record as the result.
- Results are reported as flat-stake yield *and* Kelly bankroll growth.
- The model's forecasts are scored against the **market's own** margin-free
  forecasts on log loss and Brier score.
- **Closing-line value** is tracked separately.

The last two matter more than the ROI. If the model cannot beat the closing
line on log loss, a positive backtest ROI is noise, and the tool says so
outright rather than letting you talk yourself into it:

```
VERDICT
  The market's probabilities scored BETTER than the model's on log loss.
  Any positive ROI above is very likely variance, not skill. Do not bet
  this configuration with real money.
```

Beating the closing price is the single most reliable predictor of long-run
profitability — more reliable than the ROI of any sample you are likely to have.

---

## Data

**Default source: [football-data.co.uk](https://www.football-data.co.uk)** —
free, and uniquely useful because it carries the result *and* the closing prices
for the same match, which is what makes honest backtesting possible. Files are
cached under `~/.cache/bettingedge`.

22 league codes are supported (`bettingedge leagues`): England E0–E3/EC, Scotland
SC0, Germany D1/D2, Italy I1/I2, Spain SP1/SP2, France F1/F2, Netherlands N1,
Belgium B1, Portugal P1, Turkey T1, Greece G1.

Upcoming fixtures come from that site's weekly fixtures feed, which only covers
the next few days and is empty between seasons.

### Live odds providers

The free CSV feed carries only 1X2 and over/under 2.5, covers a few days ahead,
and is empty between seasons. For a real live card, plug in a provider:

```bash
bettingedge providers                      # setup instructions and quotas
export ODDS_API_KEY=your-key
bettingedge recommend --league E0 --odds-provider theoddsapi
```

| Provider | Free tier | Markets | Key |
|---|---|---|---|
| `footballdata` *(default)* | unlimited, no key | 1X2, O/U 2.5 | — |
| `theoddsapi` | ~500 requests/month | 1X2, all O/U lines, BTTS on some plans | `ODDS_API_KEY` |
| `apifootball` | ~100 requests/day | 1X2, O/U, BTTS | `API_FOOTBALL_KEY` |

A live provider returns *every* book's price per fixture, which suits this
engine better than the CSV feed: best price and sharp reference come from one
snapshot rather than being approximated by column choice. Pinnacle or an
exchange is used as the sharp line where available; otherwise the thinnest
margin wins.

**History still comes from football-data.co.uk**, because it is the only free
source carrying results and closing prices together — which is what makes
backtesting honest. Providers supply the upcoming card only.

Adding another source means implementing one method (`fixtures()`) against the
`OddsProvider` protocol in `data/providers/base.py`.

### When the machine cannot reach the API

Locked-down CI runners, corporate egress policies and hosted agent sandboxes
often allow package registries and nothing else. Capture on a machine that has
access, replay anywhere:

```bash
# where the API is reachable
bettingedge capture --league E0 --odds-provider theoddsapi --out odds.json

# anywhere, no network at all
bettingedge recommend --league E0 --replay-raw odds.json
```

`capture` prints what the payload actually contains — which markets parsed, and
the team names exactly as the provider spells them, which is what team matching
has to cope with. The file holds the provider's untouched response and contains
**no API key** (keys travel in the request, not the reply), so it is safe to
commit or hand over for debugging.

A capture is a snapshot: use it to verify parsing, team matching and pricing,
not to place bets. Prices go stale within minutes.

### Several divisions at once

```bash
bettingedge recommend --league E0,E1,EC --seasons 16
```

Each division is fitted separately — attack and defence ratings are only
meaningful relative to the league they were estimated in, so a Championship
+0.3 is not a Premier League +0.3. Combination bets and staking then run across
the whole card, since legs in different matches are independent whichever
division they are in.

### Confidence floor

`--min-confidence` drops bets scoring below a threshold out of 100. It is 0 by
default so you see everything that clears the edge test and can judge for
yourself.

It tightens on its own when it needs to. Early in a season a 180-day decay
window is mostly off-season, so ratings are stale last-season values and the
model produces its largest and least trustworthy disagreements with the market
— 14–17% "edges" at 30/100 confidence. When the effective sample falls below
about 20 matches per team the floor rises to the bottom-tier boundary
automatically, and the card says so:

```
!! THIN SAMPLE WARNING
   EC 16 time-weighted matches per team.
   Early in a season the decay window is mostly off-season, so ratings are stale
   last-season values. The confidence floor has been raised automatically to 34
   for the affected leagues, so fewer bets survive than usual.
```

Raising `--half-life` does not fix this; it just averages in more stale data.
Tested on sixteen National League seasons.

### Team names — the silent killer

Your odds source says "Manchester United". Your results source says "Man
United". The model has never heard of the former, so it treats it as a
league-average side and prices the game anyway — producing a full card of
confident-looking recommendations built on nothing.

Two defences, both on by default:

1. **Names are reconciled** before pricing, via a curated alias table for the
   major European leagues plus similarity matching. The matcher **refuses to
   guess** when two candidates score too closely, because "Man United" and
   "Man City" are similar strings and a wrong match silently prices the wrong
   team. Anything unresolved is reported, never assumed.
2. **The engine refuses to price a team it has never seen** (`require_known_teams`).
   Such fixtures land in `skipped` with the reason attached.

```
Team name matching: 3 fixture(s) resolved, 1 dropped.
  Could not resolve:
    'Real Betis' matched nothing in the model
  Fix with --team-alias "Provider Name=Model Name", or check the league code matches.
```

Fix stragglers with a repeatable flag:

```bash
bettingedge recommend --league E0 --odds-provider theoddsapi \
  --team-alias "Nottingham Forest=Nott'm Forest"
```

**Your own prices.** For markets no feed carries, or an exchange price you can
actually get, supply a CSV:

```csv
date,home,away,H,D,A,O2.5,U2.5,BTTS_Y,BTTS_N,sharp_H,sharp_D,sharp_A
2026-08-15,Arsenal,Chelsea,2.10,3.50,3.60,1.85,1.95,1.70,2.10,2.05,3.40,3.55
```

```bash
bettingedge recommend --league E0 --fixtures-csv my-odds.csv
```

Columns are named either by friendly alias (`H`, `D`, `A`, `O2.5`, `BTTS_Y`) or
by canonical selection id (`1X2:H`, `OU2.5:O`, `BTTS:Y`). A `sharp_` prefix marks
a column as the sharp reference line rather than the best price you can take.

**Offline demo data.** `bettingedge demo` generates a synthetic league from a
known Dixon-Coles process, so the app runs with no network at all and the test
suite can check the estimator recovers parameters it was never told. The
simulated books price off the *true* probabilities, so the demo backtest
measures whether the machinery works — not whether the strategy prints money.

---

## Dashboard

```bash
bettingedge serve --league E0        # then open http://127.0.0.1:8000
bettingedge serve --synthetic        # offline demo
```

Five tabs: **Recommendations** (expandable slips with full analysis),
**Fixture analysis** (model read on every match, including the ones with no
bet), **Model & ratings** (team strength table, fitted parameters),
**Backtest** (run it live, with bankroll curve and calibration table), and
**Method**. Bankroll, Kelly fraction, minimum edge, model weight, form half-life
and maximum acca legs are all live controls — moving them refits and reprices.

---

## CLI reference

| Command | What it does |
|---|---|
| `demo` | Full run on generated offline data |
| `recommend` | Price the upcoming card and recommend bets |
| `verify` | Five-stage verification ladder on real data |
| `scan` | Compare model performance across several leagues |
| `providers` | List live odds sources and how to set them up |
| `capture` | Save a provider payload for replay on a machine with no network |
| `backtest` | Walk-forward test on historical results |
| `ratings` | Current team strength table |
| `serve` | Web dashboard |
| `leagues` | List league codes |

Common options: `--league`, `--seasons`, `--bankroll`, `--kelly`, `--min-edge`,
`--model-weight`, `--half-life`, `--max-legs`, `--price-mode`, `--odds-provider`,
`--api-key`, `--team-alias`, `--days-ahead`, `--dump-raw`, `--json`, `--markdown`,
`--fixtures-csv`, `--results-csv`, `--offline`, `--synthetic`.

```bash
# Before trusting anything: does the pipeline survive a hostile read?
bettingedge verify --league E0 --seasons 6 --price-mode early

# Conservative: trust the market more, demand a bigger edge, stake smaller
bettingedge recommend --league I1 --model-weight 0.2 --min-edge 0.05 --kelly 0.15

# Export a slate for later
bettingedge recommend --league SP1 --json slate.json --markdown slate.md
```

---

## Layout

```
bettingedge/
├── config.py            every tunable, in one place
├── pipeline.py          the engine: history in, ranked slips out
├── verify.py            the five-stage verification ladder
├── report.py            terminal and Markdown rendering
├── cli.py               command line interface
├── data/                schemas, football-data.co.uk, CSV import, synthetic league
│   ├── providers/       live odds clients (The Odds API, API-Football)
│   └── teams.py         cross-source team name matching
├── models/              Dixon-Coles fit, market masks, form/context
├── market/              devigging (multiplicative/power/Shin), consensus fair prices
├── betting/             value detection, multi construction, Kelly staking
├── analysis/            fixture context and written analysis
├── backtest/            walk-forward engine and forecast scoring
├── api/                 FastAPI backend
└── web/                 dashboard (no build step)
```

---

## Tests

```bash
pip install -e ".[dev]"
pytest
```

279 tests. The ones that matter most:

- the analytic gradient is verified against finite differences
- the fitter recovers known parameters from a simulated league
- markets derived from the score matrix are internally consistent, and
  same-game legs are provably not independent
- a full settlement truth table for every market
- staking caps, exposure ceilings and fractional-Kelly behaviour
- backtest accounting reconciles bet by bet, and the hit rate is checked
  against the implied probability of the prices taken
- each price mode reads the columns it claims to, and degrades gracefully on
  older seasons that have no closing prices
- the verification ladder catches deliberately corrupted data — swapped home
  and away teams, impossible scorelines
- provider payloads parse defensively: outcomes are matched by team name rather
  than position, and malformed events are skipped rather than crashing
- team matching never confuses Manchester United with Manchester City, and
  refuses ambiguous matches instead of guessing
- a replayed capture reproduces exactly what the live client would have parsed,
  with no API key and no network
- the shot-conversion fit recovers a known rate from generated data, and the
  likelihood accepts the continuous expected-goals target
- a thin early-season sample is detected, surfaced, and automatically raises
  the confidence floor
- a multi-league card fits each division separately but stakes as one portfolio

---

## What would make this wrong

Stated plainly, because a model that hides its failure modes is a liability.

- **Goals-only modelling knows nothing about football.** Injuries, suspensions,
  rotation before a cup tie, a manager sacked on Thursday, a team already safe
  in mid-table in May. The market knows all of it. Adding xG-based ratings and
  an injury feed is the highest-value next step.
- **A price you saw is not a price you can get.** Edges built on the best quote
  across many books shrink or vanish at the book you actually hold an account
  with. The analysis flags when an edge is mostly price shopping.
- **Backtest ROI over a few hundred bets is noise.** Football edges are small;
  you need thousands of bets before ROI means anything. Closing-line value and
  log loss against the market are the honest tests.
- **The blend weight is a judgement, not a fact.** Turn it up and you will find
  more "value" that is really model error.
- **Limits and account restriction are real.** A strategy that beats the closing
  line reliably will get its stakes cut.

---

## Responsible gambling

This is a statistical modelling tool, not financial advice and not a guarantee
of profit. A positive expected value is an estimate, not a promise; even a
genuine edge comes with long losing runs, and this backtest shows drawdowns
over 40% on configurations that finished in profit.

Never stake money you cannot afford to lose. If gambling stops being fun, stop.

- UK: [BeGambleAware](https://www.begambleaware.org) — 0808 8020 133
- US: 1-800-GAMBLER
- Ireland: [Gambling Care](https://www.gamblingcare.ie) — 089 241 5401

## Licence

MIT.
