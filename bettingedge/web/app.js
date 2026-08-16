'use strict';

/* ------------------------------------------------------------------ utils */
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const signed = (v, d = 2) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(d)}%`;
const money = (v) => v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const cls = (v) => (v > 0 ? 'pos' : v < 0 ? 'neg' : 'neutral');

/** Minimal markdown: **bold**, "- " bullets and blank-line paragraphs. */
function md(text) {
  return esc(text)
    .split(/\n\s*\n/)
    .map((block) => {
      // Bullets only at the start of a line — "0.82 - 1.54" is a scoreline,
      // not a list item.
      const html = block
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/(^|\n)- /g, '<br>· ')
        .replace(/\n/g, ' ');
      return `<p>${html}</p>`;
    })
    .join('');
}

const state = { slate: null, health: null, backtest: null, tab: 'bets', league: null };

/* ------------------------------------------------------------------ params */
function params() {
  const p = new URLSearchParams({
    bankroll: $('bankroll').value,
    kelly: $('kelly').value,
    min_edge: $('minedge').value,
    model_weight: $('weight').value,
    half_life: $('halflife').value,
    max_legs: $('maxlegs').value,
    min_leg_edge: $('minedge').value,
  });
  if (state.league) p.set('league', state.league);
  return p;
}

function syncLabels() {
  $('kelly-v').textContent = Number($('kelly').value).toFixed(2);
  $('minedge-v').textContent = pct(Number($('minedge').value), 1);
  $('weight-v').textContent = pct(Number($('weight').value), 0);
  $('halflife-v').textContent = `${$('halflife').value}d`;
  $('maxlegs-v').textContent = $('maxlegs').value;
}

/* ------------------------------------------------------------------ render */
function tiles(items) {
  return `<div class="tiles">${items.map((t) => `
    <div class="tile">
      <div class="k">${esc(t.k)}</div>
      <div class="v ${t.cls || ''}">${t.v}</div>
      ${t.s ? `<div class="s">${esc(t.s)}</div>` : ''}
    </div>`).join('')}</div>`;
}

function legNumbers(leg) {
  return `model ${pct(leg.model_probability)} · market ${pct(leg.market_probability)}
    · blend ${pct(leg.probability)} → fair odds ${leg.fair_odds.toFixed(2)}
    · edge <span class="${cls(leg.edge)}">${signed(leg.edge)}</span>
    · ${leg.book_count} book${leg.book_count === 1 ? '' : 's'} at ${pct(leg.overround)} margin`;
}

function legRow(leg, compact) {
  if (compact) {
    // A single already names the match and pick in its header — just the maths.
    return `<div class="leg"><span class="nums">${legNumbers(leg)}</span></div>`;
  }
  return `<div class="leg">
    <span class="match">${esc(leg.match)} · ${esc(leg.date)}</span>
    <span class="pick">${esc(leg.label)} @ ${leg.odds.toFixed(2)}</span>
    <span class="nums">${legNumbers(leg)}</span>
  </div>`;
}

function slipCard(slip, index) {
  const tier = (slip.correlated ? 'corr' : (slip.legs[0]?.tier || 'medium')).toLowerCase();
  const title = slip.size === 1
    ? `${esc(slip.legs[0].label)}`
    : `${esc(slip.kind)} — ${slip.legs.map((l) => esc(l.short_label)).join(' + ')}`;
  const sub = slip.size === 1
    ? `${esc(slip.legs[0].match)} · ${esc(slip.legs[0].date)}`
    : `${slip.size} legs · ${slip.legs.map((l) => esc(l.match)).join(' | ')}`;

  return `<article class="slip" data-index="${index}">
    <div class="slip-head">
      <span class="chev">▶</span>
      <span class="badge ${tier}">${esc(slip.profile || slip.kind)}</span>
      <div class="slip-title">${title}<span class="sub">${sub}</span></div>
      <div class="slip-metrics">
        <div class="m"><span class="lbl">odds</span>${slip.odds.toFixed(2)}</div>
        <div class="m"><span class="lbl">lands</span>${slip.probability_pct.toFixed(1)}%</div>
        <div class="m"><span class="lbl">edge</span><span class="${cls(slip.edge)}">${signed(slip.edge)}</span></div>
        <div class="m"><span class="lbl">conf</span>${slip.confidence.toFixed(0)}</div>
        <div class="m"><span class="lbl">stake</span>${money(slip.stake)}</div>
        <div class="m"><span class="lbl">returns</span>${money(slip.potential_return)}</div>
      </div>
    </div>
    <div class="legs">${slip.legs.map((l) => legRow(l, slip.size === 1)).join('')}</div>
    <div class="analysis">
      ${md(slip.analysis)}
      ${slip.leg_analysis && slip.leg_analysis.length
        ? `<div class="leg-detail"><h4>Leg-by-leg</h4>${slip.leg_analysis.map(md).join('<hr style="border:none;border-top:1px solid #1c2634;margin:14px 0">')}</div>`
        : ''}
    </div>
  </article>`;
}

function section(title, note, slips, emptyText) {
  const body = slips.length
    ? slips.map(slipCard).join('')
    : `<div class="empty">${esc(emptyText)}</div>`;
  return `<h2 class="section">${esc(title)}</h2>
    <p class="section-note">${esc(note)}</p>${body}`;
}

function renderBets() {
  const s = state.slate;
  const el = $('tab-bets');
  if (!s) { el.innerHTML = '<div class="loading">Loading…</div>'; return; }

  const p = s.portfolio;
  const head = tiles([
    { k: 'Qualifying bets', v: p.bet_count, s: `${s.singles.length} singles · ${s.multis.length} multis` },
    { k: 'Total staked', v: money(p.total_staked), s: `${p.exposure_pct.toFixed(1)}% of bankroll` },
    { k: 'Expected profit', v: money(p.expected_profit), cls: cls(p.expected_profit), s: `${p.expected_roi_pct >= 0 ? '+' : ''}${p.expected_roi_pct}% of turnover` },
    { k: 'Fixtures priced', v: s.fixtures_analysed, s: `${esc(s.league_name || s.league)}` },
    { k: 'Model blend', v: pct(s.config.market.model_weight, 0), s: 'weight on model vs market' },
  ]);

  el.innerHTML = head
    + section('Singles', 'One selection, one match. This is where a real edge is most likely to survive contact with reality — the bookmaker\'s margin is charged once.',
      s.singles, 'No single cleared the edge threshold. Lower "Min edge", or accept that this card has no value in it.')
    + section('Doubles, trebles and accumulators',
      'Legs from different matches, combined by multiplication. Ranked by expected log growth rather than raw expected value, so a combination that actually lands is preferred to a lottery ticket with the same headline EV.',
      s.multis, 'No multi cleared the threshold. Multis need every leg to be strongly positive because the margin compounds — an empty list here is the normal, correct answer.')
    + section('Same-game combinations',
      'Two legs from one match, priced from the joint score distribution rather than multiplied, because they are correlated. Only worth taking at a book that prices same-game multis by multiplying the legs.',
      s.same_game, 'No same-game combination showed enough correlation edge.');
}

function probBar(probs) {
  const h = probs['1X2:H'] || 0, d = probs['1X2:D'] || 0, a = probs['1X2:A'] || 0;
  const total = h + d + a || 1;
  return `<div class="pbar">
    <i class="home" style="width:${(h / total) * 100}%"></i>
    <i class="draw" style="width:${(d / total) * 100}%"></i>
    <i class="away" style="width:${(a / total) * 100}%"></i>
  </div>`;
}

/** "1X2:A" -> "Chelsea to win", so the table reads like a betting slip. */
function selLabel(sel, home, away) {
  const [head, pick] = sel.split(':');
  if (head === '1X2') return { H: `${home} to win`, D: 'Draw', A: `${away} to win` }[pick] || sel;
  if (head === 'DC') return { '1X': `${home} or draw`, '12': 'Either team (no draw)', X2: `${away} or draw` }[pick] || sel;
  if (head === 'BTTS') return pick === 'Y' ? 'Both teams to score' : 'Both teams to score — no';
  if (head.startsWith('OU')) return `${pick === 'O' ? 'Over' : 'Under'} ${head.slice(2)} goals`;
  return sel;
}

const MARKET_ORDER = ['1X2:H', '1X2:D', '1X2:A', 'DC:1X', 'DC:12', 'DC:X2'];
function marketRank(sel) {
  const i = MARKET_ORDER.indexOf(sel);
  return i >= 0 ? i : 100 + sel.charCodeAt(0);
}

function marketTable(preview) {
  const entries = Object.entries(preview.market)
    .sort((a, b) => marketRank(a[0]) - marketRank(b[0]) || a[0].localeCompare(b[0]));
  const rows = entries.map(([sel, q]) => {
    const blend = preview.blended_probabilities[sel];
    const model = preview.model_probabilities[sel];
    const edge = blend != null ? blend * q.best_odds - 1 : null;
    return `<tr>
      <td class="name">${esc(selLabel(sel, preview.home, preview.away))}
        <span style="color:var(--dim);font-size:11px"> ${esc(sel)}</span></td>
      <td class="num">${q.best_odds.toFixed(2)}</td>
      <td class="num">${model != null ? pct(model) : '—'}</td>
      <td class="num">${pct(q.fair_probability)}</td>
      <td class="num">${blend != null ? pct(blend) : '—'}</td>
      <td class="num ${edge != null ? cls(edge) : ''}">${edge != null ? signed(edge) : '—'}</td>
    </tr>`;
  }).join('');
  return `<div class="table-wrap" style="margin-top:14px"><table>
    <thead><tr><th>Selection</th><th class="num">Best odds</th><th class="num">Model</th>
      <th class="num">Market fair</th><th class="num">Blend</th><th class="num">Edge</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

function renderFixtures() {
  const s = state.slate;
  const el = $('tab-fixtures');
  if (!s) { el.innerHTML = '<div class="loading">Loading…</div>'; return; }
  if (!s.previews || !s.previews.length) {
    el.innerHTML = '<div class="empty">No fixture previews available.</div>';
    return;
  }
  el.innerHTML = `<h2 class="section">Fixture analysis</h2>
    <p class="section-note">The model's read on every fixture on the card, whether or not it
      produced a bet. A fixture with no bet is not a fixture with no opinion — it usually
      means the market has it right.</p>`
    + s.previews.map((f) => `
      <div class="fixture">
        <div class="fixture-head">
          <div>
            <div class="teams">${esc(f.home)} v ${esc(f.away)}</div>
            <div style="color:var(--dim);font-size:12px">${esc(f.date)}${f.kickoff ? ' · ' + esc(f.kickoff) : ''} · ${esc(f.league)}</div>
          </div>
          <div class="xg">xG ${f.expected_goals.home.toFixed(2)} – ${f.expected_goals.away.toFixed(2)}
            <span style="color:var(--dim)">(${f.expected_goals.total.toFixed(2)} total)</span></div>
        </div>
        <div style="margin-top:12px">${probBar(f.blended_probabilities)}</div>
        <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin-top:5px;font-family:var(--mono)">
          <span>${esc(f.home)} ${pct(f.blended_probabilities['1X2:H'] || 0)}</span>
          <span>Draw ${pct(f.blended_probabilities['1X2:D'] || 0)}</span>
          <span>${esc(f.away)} ${pct(f.blended_probabilities['1X2:A'] || 0)}</span>
        </div>
        <div class="scores">${f.likely_scores.map((s2) =>
          `<span class="score-chip"><b>${esc(s2.score)}</b> ${pct(s2.probability)}</span>`).join('')}</div>
        <p class="preview">${esc(f.preview)}</p>
        ${marketTable(f)}
      </div>`).join('');
}

function renderModel() {
  const s = state.slate;
  const el = $('tab-model');
  if (!s) { el.innerHTML = '<div class="loading">Loading…</div>'; return; }
  const m = s.model;
  const max = Math.max(...m.teams.map((t) => Math.abs(t.net)), 0.1);

  el.innerHTML = tiles([
    { k: 'Matches fitted', v: m.n_matches, s: `effective sample ${m.effective_sample}` },
    { k: 'Home advantage', v: `${m.home_advantage >= 0 ? '+' : ''}${m.home_advantage.toFixed(3)}`, s: 'log-goals' },
    // Not "ρ" as the key: the uppercase styling renders it as a capital Rho,
    // which reads as a Latin P.
    { k: 'Low-score rho', v: m.rho.toFixed(3), s: 'Dixon-Coles ρ correction' },
    { k: 'Form half-life', v: `${m.half_life_days}d`, s: 'time-decay weighting' },
    { k: 'Converged', v: m.converged ? 'yes' : 'no', cls: m.converged ? 'pos' : 'neg', s: `fitted through ${esc(m.fitted_through || '—')}` },
  ])
    + `<h2 class="section">Team ratings</h2>
       <p class="section-note">Attack and defence in log-goals relative to an average team in
         this league; higher is better for both. A team on +0.30 attack scores about 35% more
         than average against the same opponent.</p>
       <div class="table-wrap"><table>
         <thead><tr><th>#</th><th>Team</th><th class="num">Attack</th><th class="num">Defence</th>
           <th class="num">Net</th><th class="num">Played</th><th style="width:180px">Strength</th></tr></thead>
         <tbody>${m.teams.map((t, i) => `
           <tr>
             <td>${i + 1}</td>
             <td class="name">${esc(t.team)}</td>
             <td class="num ${cls(t.attack)}">${t.attack >= 0 ? '+' : ''}${t.attack.toFixed(3)}</td>
             <td class="num ${cls(t.defence)}">${t.defence >= 0 ? '+' : ''}${t.defence.toFixed(3)}</td>
             <td class="num ${cls(t.net)}">${t.net >= 0 ? '+' : ''}${t.net.toFixed(3)}</td>
             <td class="num">${t.matches}</td>
             <td><div class="pbar"><i class="${t.net >= 0 ? 'home' : 'away'}"
               style="width:${(Math.abs(t.net) / max) * 100}%"></i></div></td>
           </tr>`).join('')}</tbody></table></div>`;
}

function sparkline(curve) {
  if (!curve || curve.length < 2) return '';
  const values = curve.map((c) => c[1]);
  const min = Math.min(...values), max = Math.max(...values);
  const span = (max - min) || 1;
  const points = curve.map((c, i) => {
    const x = (i / (curve.length - 1)) * 100;
    const y = 100 - ((c[1] - min) / span) * 100;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(' ');
  const up = values[values.length - 1] >= values[0];
  const colour = up ? '#35d0a5' : '#ef6461';
  const start = curve[0], end = curve[curve.length - 1];
  return `<div class="table-wrap" style="padding:16px">
    <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--muted);font-family:var(--mono)">
      <span>${esc(start[0])} · ${money(start[1])}</span>
      <span>peak ${money(max)} · trough ${money(min)}</span>
      <span>${esc(end[0])} · ${money(end[1])}</span>
    </div>
    <svg class="spark" viewBox="0 0 100 100" preserveAspectRatio="none" style="margin-top:10px">
      <polyline points="${points}" fill="none" stroke="${colour}" stroke-width="0.7"
        vector-effect="non-scaling-stroke"/>
    </svg>
  </div>`;
}

function renderBacktest() {
  const el = $('tab-backtest');
  const b = state.backtest;

  const controls = `<h2 class="section">Walk-forward backtest</h2>
    <p class="section-note">The model is refitted using only matches played before each
      simulated bet, and bets settle at the best price that was actually on offer in the same
      historical record. It uses the tuning values set above.</p>
    <p><button class="primary" id="run-backtest" ${b === 'running' ? 'disabled' : ''}>
      ${b === 'running' ? 'Running…' : 'Run backtest'}</button>
      <span style="color:var(--dim);font-size:13px;margin-left:12px">Takes 10–90 seconds
      depending on how much history is loaded.</span></p>`;

  if (!b || b === 'running') {
    el.innerHTML = controls + (b === 'running'
      ? '<div class="loading">Refitting the model across every matchday in the window…</div>'
      : '');
    wireBacktestButton();
    return;
  }
  if (b.error) {
    el.innerHTML = controls + `<div class="error">${esc(b.error)}</div>`;
    wireBacktestButton();
    return;
  }

  const beatsMarket = b.model_log_loss < b.market_log_loss;
  const verdict = beatsMarket
    ? 'The model\'s probabilities scored better than the market\'s on log loss. That is the necessary condition for a real edge — necessary, not sufficient.'
    : 'The market\'s probabilities scored BETTER than the model\'s. Any positive ROI above is very likely variance rather than skill. Do not bet this configuration with real money.';

  el.innerHTML = controls
    + tiles([
      { k: 'Bets placed', v: b.bets, s: `avg odds ${b.average_odds}` },
      { k: 'Flat-stake yield', v: `${b.flat_yield_pct >= 0 ? '+' : ''}${b.flat_yield_pct}%`, cls: cls(b.flat_yield_pct), s: 'profit per unit staked' },
      { k: 'Bankroll', v: money(b.final_bankroll), cls: cls(b.final_bankroll - b.starting_bankroll), s: `from ${money(b.starting_bankroll)} (${b.growth_pct >= 0 ? '+' : ''}${b.growth_pct}%)` },
      { k: 'Max drawdown', v: `${b.max_drawdown_pct}%`, cls: 'neg', s: 'peak to trough' },
      { k: 'Hit rate', v: `${b.hit_rate_pct}%`, s: `${b.bets} settled` },
      { k: 'Closing-line value', v: `${b.average_clv_pct >= 0 ? '+' : ''}${b.average_clv_pct}%`, cls: cls(b.average_clv_pct), s: `beat close on ${b.beat_closing_rate_pct}%` },
    ])
    + `<div class="${beatsMarket ? 'table-wrap' : 'error'}" style="padding:16px;margin-bottom:18px">
        <b style="color:${beatsMarket ? 'var(--accent)' : '#ffb3b1'}">Verdict.</b> ${esc(verdict)}
        <div style="margin-top:10px;font-family:var(--mono);font-size:12.5px;color:var(--muted)">
          model log loss ${b.model_log_loss} · market log loss ${b.market_log_loss} ·
          difference ${b.log_loss_edge >= 0 ? '+' : ''}${b.log_loss_edge} ·
          model Brier ${b.model_brier} · market Brier ${b.market_brier} ·
          ${b.forecasts} forecasts
        </div>
      </div>`
    + `<h2 class="section">Bankroll</h2>` + sparkline(b.bankroll_curve)
    + `<h2 class="section">Calibration</h2>
       <p class="section-note">If the probabilities are honest, predicted and actual should
         track each other. Systematic gaps mean the model is over- or under-confident in that
         band.</p>
       <div class="table-wrap"><table>
         <thead><tr><th>Probability band</th><th class="num">Bets</th>
           <th class="num">Predicted</th><th class="num">Actual</th><th class="num">Gap</th></tr></thead>
         <tbody>${(b.calibration || []).map((r) => `
           <tr><td class="name">${esc(r.bucket)}</td><td class="num">${r.n}</td>
             <td class="num">${pct(r.predicted)}</td><td class="num">${pct(r.actual)}</td>
             <td class="num ${cls(r.actual - r.predicted)}">${signed(r.actual - r.predicted)}</td></tr>`).join('')
           || '<tr><td colspan="5" class="name">No bets to calibrate.</td></tr>'}</tbody>
       </table></div>`;
  wireBacktestButton();
}

function wireBacktestButton() {
  const button = $('run-backtest');
  if (button) button.onclick = runBacktest;
}

async function runBacktest() {
  state.backtest = 'running';
  renderBacktest();
  try {
    const response = await fetch(`/api/backtest?${params()}`);
    const data = await response.json();
    state.backtest = response.ok ? data : { error: data.detail || 'Backtest failed.' };
  } catch (err) {
    state.backtest = { error: String(err) };
  }
  renderBacktest();
}

/* ------------------------------------------------------------------ wiring */
function renderActive() {
  ({ bets: renderBets, fixtures: renderFixtures, model: renderModel,
     backtest: renderBacktest, method: () => {} })[state.tab]();
}

async function loadSlate() {
  $('tab-bets').innerHTML = '<div class="loading">Fitting the model and pricing the card…</div>';
  try {
    const response = await fetch(`/api/slate?${params()}`);
    const data = await response.json();
    if (!response.ok) {
      $('tab-bets').innerHTML = `<div class="error">${esc(data.detail || 'Could not build a slate.')}</div>`;
      return;
    }
    state.slate = data;
    $('disclaimer').textContent = data.disclaimer || '';
    renderActive();
  } catch (err) {
    $('tab-bets').innerHTML = `<div class="error">${esc(String(err))}</div>`;
  }
}

let debounce;
function onControlChange() {
  syncLabels();
  clearTimeout(debounce);
  debounce = setTimeout(loadSlate, 320);
}

document.addEventListener('click', (event) => {
  const head = event.target.closest('.slip-head');
  if (head) head.parentElement.classList.toggle('open');
});

document.querySelectorAll('.tab').forEach((tab) => {
  tab.onclick = () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    tab.classList.add('active');
    state.tab = tab.dataset.tab;
    ['bets', 'fixtures', 'model', 'backtest', 'method'].forEach((name) =>
      $(`tab-${name}`).classList.toggle('hidden', name !== state.tab));
    $('controls').classList.toggle('hidden', state.tab === 'method');
    renderActive();
  };
});

['bankroll', 'kelly', 'minedge', 'weight', 'halflife', 'maxlegs'].forEach((id) =>
  $(id).addEventListener('input', onControlChange));

/** "3 min ago" — so it is obvious at a glance whether this is current. */
function freshness(isoTimestamp) {
  if (!isoTimestamp) return '';
  const minutes = Math.max(0, Math.round((Date.now() - new Date(isoTimestamp)) / 60000));
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

function renderStatus(health) {
  const age = freshness(health.refreshed_at);
  // A refresh that has been failing is exactly what you would want to know
  // before trusting a price, so it is called out rather than buried.
  const stale = health.refresh_error
    ? `<br><span style="color:var(--bad)">refresh failing — prices may be stale</span>`
    : '';
  $('status').innerHTML = `<b>${esc(health.league_name)}</b><br>${health.matches} matches`
    + ` · ${health.fixtures} fixtures<br><span style="color:var(--dim)">${esc(health.source)}`
    + (age ? ` · updated ${age}` : '') + `</span>${stale}`;
}

async function pollHealth() {
  try {
    const health = await (await fetch(`/api/health?${params()}`)).json();
    const previous = state.health && state.health.refreshed_at;
    state.health = health;
    renderStatus(health);
    // The server refreshed underneath us — pull the new card in so a phone
    // left open overnight is not showing yesterday's fixtures.
    if (previous && health.refreshed_at !== previous) loadSlate();
  } catch (err) {
    /* transient; the next poll will pick it up */
  }
}

/** Everything that changes when a different league is picked. */
async function switchLeague(code) {
  state.league = code;
  state.slate = null;
  state.backtest = null;
  const url = new URL(window.location);
  url.searchParams.set('league', code);
  window.history.replaceState({}, '', url);   // bookmarkable, survives a refresh
  renderActive();
  try {
    const health = await (await fetch(`/api/health?${params()}`)).json();
    state.health = health;
    renderStatus(health);
  } catch (err) { /* status line just goes stale; not fatal */ }
  loadSlate();
}

/**
 * Populate the league dropdown from what the server actually has loaded.
 * Only one league loaded -> the picker stays hidden; nothing to choose.
 */
async function loadLeagueOptions() {
  const select = $('league-select');
  try {
    const leagues = await (await fetch('/api/leagues')).json();
    if (!Array.isArray(leagues) || leagues.length < 2) return;

    const fromUrl = new URLSearchParams(window.location.search).get('league');
    const primary = leagues.find((l) => l.is_primary) || leagues[0];
    const startCode = leagues.some((l) => l.code === fromUrl) ? fromUrl : primary.code;
    state.league = startCode;

    select.innerHTML = leagues.map((l) =>
      `<option value="${esc(l.code)}" ${l.code === startCode ? 'selected' : ''}>
        ${esc(l.name)} (${l.fixtures} fixtures)</option>`).join('');
    select.classList.remove('hidden');
    select.onchange = () => switchLeague(select.value);
  } catch (err) {
    /* a single-league server has no /api/leagues concept worth failing over */
  }
}

(async function init() {
  syncLabels();
  await loadLeagueOptions();
  try {
    const health = await (await fetch(`/api/health?${params()}`)).json();
    state.health = health;
    renderStatus(health);
    $('disclaimer').textContent = health.disclaimer || '';
  } catch (err) {
    $('status').textContent = 'backend unreachable';
  }
  loadSlate();
  // Cheap: one small JSON request a minute, and it keeps the age indicator
  // honest on a phone that has been sitting open.
  setInterval(pollHealth, 60000);
})();
