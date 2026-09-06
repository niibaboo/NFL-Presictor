#!/usr/bin/env python3
"""
Blitz IQ — NFL Team Points Over/Under Predictor
Same architecture as the MLB (Strike Zone) and soccer (Goal IQ) tools, adapted
for NFL scoring:

- Uses ESPN's public (undocumented but documented-elsewhere and no-key-needed)
  API — no signup required.
- Team point totals are modeled with a NORMAL distribution, not Poisson.
  Poisson fits well for low-count events like goals or strikeouts; NFL scores
  are larger, non-1-point increments (3, 6, 7, 8...) and behave much closer
  to a bell curve in practice.
- Recency-weighted last-5-games average, shrunk toward the league average
  for small samples (same small-sample protection built for MLB/soccer,
  since NFL teams only play ~17 games/season — "last 5" is meaningfully
  more of the season than in MLB or soccer).

Setup:
    pip3 install requests --break-system-packages
    python3 nfl_model.py

Output:
    docs/index.html, docs/blitz_iq_predictions.csv

Note: ESPN doesn't publish an official rate limit for this endpoint, but
"excessive requests may be blocked" per their own docs — this script paces
itself conservatively (1.2s between calls) to stay well clear of that,
rather than assuming no limit means no risk.
"""

import os
import sys
import time
import math
import csv
import json
from datetime import datetime, timedelta
import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
REQUEST_DELAY = 1.2  # no official ESPN rate limit is published — paced conservatively anyway
RECENT_GAMES = 5
PRIOR_STRENGTH = 3  # same "worth" of league-average games blended in for small samples as MLB/soccer
DEFAULT_TEAM_STD = 10.0  # rough per-team point stdev; used until we compute a real one from data


def _get(url, params=None, timeout=15):
    try:
        r = requests.get(url, params=params or {}, timeout=timeout)
        time.sleep(REQUEST_DELAY)
        if r.status_code != 200:
            print(f"  [!] {r.status_code} on {url}")
            return None
        return r
    except Exception as e:
        time.sleep(REQUEST_DELAY)
        print(f"  [!] request failed: {url} ({e})")
        return None


def norm_cdf(x, mean, std):
    """Standard normal CDF via the error function — no scipy dependency."""
    if std <= 0:
        return 1.0 if x >= mean else 0.0
    z = (x - mean) / (std * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


def get_teams():
    r = _get(f"{BASE}/teams", params={"limit": 40})
    if r is None:
        return []
    try:
        return r.json()['sports'][0]['leagues'][0]['teams']
    except Exception as e:
        print(f"  [!] couldn't parse teams response: {e}")
        return []


team_form_cache = {}


def get_team_form(team_id):
    """Last N completed games for a team — points scored and allowed."""
    if team_id in team_form_cache:
        return team_form_cache[team_id]
    r = _get(f"{BASE}/teams/{team_id}/schedule")
    if r is None:
        return None
    try:
        events = r.json().get('events', [])
        completed = [e for e in events if e.get('competitions', [{}])[0].get('status', {})
                     .get('type', {}).get('completed')]
        completed.sort(key=lambda e: e.get('date', ''))
        recent = completed[-RECENT_GAMES:]
        scored, allowed = [], []
        for e in recent:
            comp = e['competitions'][0]
            competitors = comp.get('competitors', [])
            me = next((c for c in competitors if str(c['team']['id']) == str(team_id)), None)
            opp = next((c for c in competitors if str(c['team']['id']) != str(team_id)), None)
            if not me or not opp:
                continue
            try:
                scored.append(float(me['score']['value']))
                allowed.append(float(opp['score']['value']))
            except (KeyError, TypeError, ValueError):
                continue
        if not scored:
            return None
        n = len(scored)
        form = {
            'avg_scored': round(sum(scored) / n, 1),
            'avg_allowed': round(sum(allowed) / n, 1),
            'n_games': n,
            'scored_list': scored,
            'allowed_list': allowed,
        }
        team_form_cache[team_id] = form
        return form
    except Exception as e:
        print(f"  [!] couldn't parse schedule for team {team_id}: {e}")
        return None


def recency_weighted(values):
    n = len(values)
    if n == 0:
        return None
    wts = [1.3 ** i for i in range(n)]  # most recent (last index) weighted highest; gentler than the
    return sum(w * v for w, v in zip(wts, values)) / sum(wts)  # 1.4 used elsewhere — NFL samples are smaller


def shrink(value, n, league_avg, prior=PRIOR_STRENGTH):
    """Same small-sample protection as MLB/soccer — a team with only 1-2
    games on record leans mostly on the league average; by 5 games their
    own form dominates."""
    return (n * value + prior * league_avg) / (n + prior)


def league_averages(all_forms):
    scored = [f['avg_scored'] for f in all_forms if f]
    allowed = [f['avg_allowed'] for f in all_forms if f]
    lg_scored = sum(scored) / len(scored) if scored else 22.0  # ~league-average NFL score, sane fallback
    lg_allowed = sum(allowed) / len(allowed) if allowed else 22.0
    return lg_scored, lg_allowed


def predict(h_form, a_form, lg_scored, lg_allowed):
    h_recent_scored = recency_weighted(h_form['scored_list'])
    h_recent_allowed = recency_weighted(h_form['allowed_list'])
    a_recent_scored = recency_weighted(a_form['scored_list'])
    a_recent_allowed = recency_weighted(a_form['allowed_list'])

    h_scored = shrink(h_recent_scored, h_form['n_games'], lg_scored)
    h_allowed = shrink(h_recent_allowed, h_form['n_games'], lg_allowed)
    a_scored = shrink(a_recent_scored, a_form['n_games'], lg_scored)
    a_allowed = shrink(a_recent_allowed, a_form['n_games'], lg_allowed)

    exp_home = h_scored * (a_allowed / lg_allowed)
    exp_away = a_scored * (h_allowed / lg_allowed)
    exp_total = round(exp_home + exp_away, 1)

    # Combined-game stdev from two independent-ish team stdevs (rough
    # approximation — real NFL totals correlate slightly by pace/weather,
    # not modeled here).
    total_std = math.sqrt(DEFAULT_TEAM_STD ** 2 + DEFAULT_TEAM_STD ** 2)

    return {
        'exp_home': round(exp_home, 1), 'exp_away': round(exp_away, 1),
        'exp_total': exp_total, 'total_std': round(total_std, 1),
    }


def over_under_prob(exp_total, total_std, line):
    p_under = norm_cdf(line, exp_total, total_std)
    return round((1 - p_under) * 100), round(p_under * 100)


def get_week_scoreboard():
    r = _get(f"{BASE}/scoreboard")
    if r is None:
        return []
    try:
        return r.json().get('events', [])
    except Exception as e:
        print(f"  [!] couldn't parse scoreboard: {e}")
        return []


# ---------------------------------------------------------------------------
# Player props — QB passing yards, RB rushing yards, WR/TE receptions.
# This section is the least-tested part of the whole script: ESPN's docs
# confirm these endpoints exist for NFL, but not their exact JSON field
# names. Every parse step below is defensive (try a few plausible field
# names, print a diagnostic and return None rather than crashing if none
# match) so a wrong guess fails safely and shows up clearly in the Actions
# log instead of silently producing garbage or stopping the whole run.
# ---------------------------------------------------------------------------

PLAYER_STAT_CONFIG = {
    'QB': {'stat': 'passingYards', 'label': 'Passing Yards', 'dist': 'normal', 'std': 55, 'prior': 235},
    'RB': {'stat': 'rushingYards', 'label': 'Rushing Yards', 'dist': 'normal', 'std': 28, 'prior': 60},
    'WR': {'stat': 'receptions', 'label': 'Receptions', 'dist': 'poisson', 'std': None, 'prior': 4.0},
    'TE': {'stat': 'receptions', 'label': 'Receptions', 'dist': 'poisson', 'std': None, 'prior': 3.5},
}

depth_chart_cache = {}


def get_starters(team_id):
    """Best-effort read of a team's starting QB/RB/top WR/top TE. ESPN's
    depth chart schema isn't documented in detail, so this tries a couple
    of plausible shapes and gives up cleanly (returning {}) if neither
    matches, rather than guessing wrong silently."""
    if team_id in depth_chart_cache:
        return depth_chart_cache[team_id]
    r = _get(f"{BASE}/teams/{team_id}/depthcharts")
    starters = {}
    if r is None:
        depth_chart_cache[team_id] = starters
        return starters
    try:
        data = r.json()
        groups = data.get('items') or data.get('athletes') or []
        for group in groups:
            positions = group.get('positions', {})
            for pos_key, pos_data in positions.items():
                pos_abbr = (pos_data.get('position', {}).get('abbreviation')
                            or pos_key or '').upper()
                if pos_abbr not in PLAYER_STAT_CONFIG:
                    continue
                if pos_abbr in starters:
                    continue  # already have this position's starter
                slots = pos_data.get('athletes', [])
                if slots:
                    athlete = slots[0].get('athlete', slots[0])
                    starters[pos_abbr] = {
                        'id': athlete.get('id'), 'name': athlete.get('displayName', athlete.get('fullName', '?')),
                    }
    except Exception as e:
        print(f"  [!] couldn't parse depth chart for team {team_id}: {e}")
    depth_chart_cache[team_id] = starters
    return starters


player_gamelog_cache = {}


def get_player_gamelog(athlete_id, stat_key):
    """Last N games' value for one stat (passing yards, rushing yards,
    receptions) for a given player. Tries the common v3 gamelog endpoint
    first since it's documented as NFL-supported."""
    cache_key = (athlete_id, stat_key)
    if cache_key in player_gamelog_cache:
        return player_gamelog_cache[cache_key]

    url = f"https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{athlete_id}/gamelog"
    r = _get(url)
    values = []
    if r is not None:
        try:
            data = r.json()
            events = data.get('events', {})
            # ESPN gamelog responses are typically keyed by event id with a
            # parallel 'labels'/'names' array describing which stat each
            # position in the per-game array corresponds to — exact shape
            # unconfirmed, so this tries the most likely structure and
            # bails cleanly if it doesn't match.
            season_data = data.get('seasonTypes', [])
            for st in season_data:
                for cat in st.get('categories', []):
                    for game in cat.get('events', []):
                        stats = game.get('stats', [])
                        labels = cat.get('labels', []) or cat.get('names', [])
                        if stat_key in labels:
                            idx = labels.index(stat_key)
                            try:
                                values.append(float(stats[idx]))
                            except (IndexError, ValueError, TypeError):
                                continue
        except Exception as e:
            print(f"  [!] couldn't parse gamelog for athlete {athlete_id}: {e}")

    values = values[-RECENT_GAMES:]
    player_gamelog_cache[cache_key] = values
    return values


def project_player_stat(pos_abbr, athlete_id, name):
    cfg = PLAYER_STAT_CONFIG[pos_abbr]
    values = get_player_gamelog(athlete_id, cfg['stat'])
    if not values:
        return None
    n = len(values)
    recent = recency_weighted(values)
    shrunk = shrink(recent, n, cfg['prior'])
    return {
        'name': name, 'position': pos_abbr, 'label': cfg['label'],
        'dist': cfg['dist'], 'std': cfg['std'],
        'projected': round(shrunk, 1), 'n_games': n, 'recent_values': values,
    }


def get_team_player_props(team_id):
    starters = get_starters(team_id)
    props = []
    for pos_abbr, athlete in starters.items():
        if not athlete.get('id'):
            continue
        proj = project_player_stat(pos_abbr, athlete['id'], athlete['name'])
        if proj:
            props.append(proj)
    return props


def player_over_under_prob(proj, line):
    if proj['dist'] == 'poisson':
        p_under = poisson_cdf(math.floor(line), proj['projected'])
    else:
        p_under = norm_cdf(line, proj['projected'], proj['std'])
    return round((1 - p_under) * 100), round(p_under * 100)


def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_cdf(k, lam):
    return sum(poisson_pmf(i, lam) for i in range(int(k) + 1))


def build_predictions():
    print("Fetching teams…")
    teams = get_teams()
    if not teams:
        print("No teams returned — aborting.")
        return []

    print(f"Fetching form for {len(teams)} teams…")
    all_forms = {}
    for t in teams:
        tid = t['team']['id']
        print(f"  {t['team']['displayName']}")
        all_forms[tid] = get_team_form(tid)

    lg_scored, lg_allowed = league_averages(all_forms.values())
    print(f"League averages: {lg_scored:.1f} scored/game, {lg_allowed:.1f} allowed/game")

    print("Fetching this week's scoreboard…")
    events = get_week_scoreboard()
    print(f"{len(events)} games this week")

    predictions = []
    for e in events:
        comp = e.get('competitions', [{}])[0]
        competitors = comp.get('competitors', [])
        home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
        away = next((c for c in competitors if c.get('homeAway') == 'away'), None)
        if not home or not away:
            continue
        if comp.get('status', {}).get('type', {}).get('completed'):
            continue  # already played — skip, this is a predictor not a results page

        h_id, a_id = home['team']['id'], away['team']['id']
        h_form, a_form = all_forms.get(h_id), all_forms.get(a_id)
        if not h_form or not a_form:
            print(f"  skipping {away['team']['displayName']} @ {home['team']['displayName']}: missing form data")
            continue

        proj = predict(h_form, a_form, lg_scored, lg_allowed)
        print(f"  Player props: {away['team']['displayName']} @ {home['team']['displayName']}")
        home_props = get_team_player_props(h_id)
        away_props = get_team_player_props(a_id)
        predictions.append({
            'date': e.get('date', ''),
            'match': f"{away['team']['displayName']} @ {home['team']['displayName']}",
            'home_team': home['team']['displayName'], 'away_team': away['team']['displayName'],
            'home_form': h_form, 'away_form': a_form,
            'home_props': home_props, 'away_props': away_props,
            **proj,
        })

    predictions.sort(key=lambda x: x['exp_total'], reverse=True)
    return predictions


HTML_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Blitz IQ — NFL</title></head>
<body style="background:#0b0f14;color:white;font-family:Arial;padding:12px;max-width:600px;margin:auto">
<h2 style="text-align:center">🏈 BLITZ IQ — NFL Team Points</h2>
<p style="text-align:center;color:#888;font-size:11px">Recency-weighted scoring/allowed rates, Normal-distribution projected · {generated}</p>
<p style="text-align:center;margin-bottom:16px"><a href="blitz_iq_predictions.csv" download style="background:#222;border:1px solid #444;color:white;padding:8px 14px;border-radius:8px;text-decoration:none;font-size:13px">⬇ Download CSV</a></p>
{cards}
<p style="text-align:center;color:#666;font-size:10px;margin-top:20px">Enter your book's Over/Under line and odds to compute an edge the same way as the MLB/soccer tools — this page shows the model's own projection only.</p>
</body></html>"""

CARD_TEMPLATE = """<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:12px 0;border:1px solid #2a3038">
  <div style="font-size:11px;color:#999;margin-bottom:4px">{date}</div>
  <div style="font-size:17px;font-weight:bold;margin-bottom:10px">{match}</div>
  <div style="display:flex;justify-content:space-between;text-align:center">
    <div><div style="color:#aaa;font-size:11px">{away_team}</div><div style="color:#ffeb3b;font-size:20px;font-weight:bold">{exp_away}</div></div>
    <div><div style="color:#aaa;font-size:11px">TOTAL</div><div style="color:#7ec8ff;font-size:22px;font-weight:bold">{exp_total}</div></div>
    <div><div style="color:#aaa;font-size:11px">{home_team}</div><div style="color:#ffeb3b;font-size:20px;font-weight:bold">{exp_home}</div></div>
  </div>
  <div style="background:#0f1318;border-radius:8px;padding:8px;margin-top:10px;display:flex;justify-content:space-between;font-size:11px">
    <div>{away_team}: {away_scored} scored/gm • {away_allowed} allowed/gm ({away_n}gm sample)</div>
  </div>
  <div style="background:#0f1318;border-radius:8px;padding:8px;margin-top:6px;font-size:11px">
    {home_team}: {home_scored} scored/gm • {home_allowed} allowed/gm ({home_n}gm sample)
  </div>
  {player_props_html}
</div>"""

PLAYER_PROP_ROW = """<div style="display:flex;justify-content:space-between;font-size:11px;padding:5px 0;border-top:1px solid #232a33">
  <div>{name} ({position}) — {label}</div>
  <div style="color:#c792ea;font-weight:bold">{projected} <span style="color:#666;font-weight:normal">({n_games}gm)</span></div>
</div>"""


def player_props_section(team_label, props):
    if not props:
        return ""
    rows = "".join(PLAYER_PROP_ROW.format(**p) for p in props)
    return f'<div style="margin-top:8px"><div style="color:#888;font-size:10px;text-transform:uppercase;margin-bottom:2px">{team_label} Player Props</div>{rows}</div>'


def make_html(predictions):
    cards = "".join(CARD_TEMPLATE.format(
        date=p['date'][:16].replace('T', ' '), match=p['match'],
        away_team=p['away_team'], home_team=p['home_team'],
        exp_away=p['exp_away'], exp_home=p['exp_home'], exp_total=p['exp_total'],
        away_scored=p['away_form']['avg_scored'], away_allowed=p['away_form']['avg_allowed'],
        away_n=p['away_form']['n_games'],
        home_scored=p['home_form']['avg_scored'], home_allowed=p['home_form']['avg_allowed'],
        home_n=p['home_form']['n_games'],
        player_props_html=(
            player_props_section(p['away_team'], p.get('away_props', []))
            + player_props_section(p['home_team'], p.get('home_props', []))
        ),
    ) for p in predictions)
    if not cards:
        cards = '<p style="text-align:center;color:#888">No upcoming games with usable form data this week.</p>'
    return HTML_TEMPLATE.format(generated=datetime.now().strftime('%d %b %H:%M'), cards=cards)


def write_csv(predictions, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Date', 'Match', 'HomeTeam', 'AwayTeam',
            'HomeScoredAvg', 'HomeAllowedAvg', 'HomeSampleSize',
            'AwayScoredAvg', 'AwayAllowedAvg', 'AwaySampleSize',
            'ExpHome', 'ExpAway', 'ExpTotal', 'TotalStd',
            'Line', 'OverOdds', 'UnderOdds',
            'ActualHomeScore', 'ActualAwayScore', 'HitOrMiss',
        ])
        for p in predictions:
            writer.writerow([
                p['date'], p['match'], p['home_team'], p['away_team'],
                p['home_form']['avg_scored'], p['home_form']['avg_allowed'], p['home_form']['n_games'],
                p['away_form']['avg_scored'], p['away_form']['avg_allowed'], p['away_form']['n_games'],
                p['exp_home'], p['exp_away'], p['exp_total'], p['total_std'],
                '', '', '',  # blank — fill in the book's line/odds yourself
                '', '', '',  # blank — fill in after the game
            ])


def write_player_props_csv(predictions, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Date', 'Match', 'Team', 'Player', 'Position', 'PropType',
            'Projected', 'SampleSize', 'RecentValues',
            'Line', 'OverOdds', 'UnderOdds', 'ActualValue', 'HitOrMiss',
        ])
        for p in predictions:
            for team_label, props in [(p['away_team'], p.get('away_props', [])),
                                       (p['home_team'], p.get('home_props', []))]:
                for prop in props:
                    writer.writerow([
                        p['date'], p['match'], team_label, prop['name'], prop['position'], prop['label'],
                        prop['projected'], prop['n_games'], '; '.join(str(v) for v in prop['recent_values']),
                        '', '', '', '', '',
                    ])


if __name__ == "__main__":
    predictions = build_predictions()
    os.makedirs('docs', exist_ok=True)
    with open('docs/index.html', 'w') as f:
        f.write(make_html(predictions))
    write_csv(predictions, 'docs/blitz_iq_predictions.csv')
    write_player_props_csv(predictions, 'docs/blitz_iq_player_props.csv')
    with open('docs/blitz_iq.json', 'w') as f:
        json.dump(predictions, f, indent=2, default=str)
    print(f"\nDone — {len(predictions)} games projected.")
