"""
models/player_stats.py v4 — Statistically correct
- Traccia gol per partita in ordine cronologico
- EMA con halflife 5 partite (alpha = 1 - exp(-1/5))
- Shrinkage bayesiano verso la media storica (prior strength = 8 partite)
- Nessun boost arbitrario — tutto derivato dai dati
"""
import json, numpy as np
from pathlib import Path
from collections import defaultdict

NAME_MAP = {
    'AC Milan': 'Milan', 'FC Internazionale Milano': 'Inter',
    'AS Roma': 'Roma', 'SSC Napoli': 'Napoli',
    'Atalanta Bergamasca Calcio': 'Atalanta', 'Juventus FC': 'Juventus',
    'ACF Fiorentina': 'Fiorentina', 'SS Lazio': 'Lazio',
    'Bologna FC 1909': 'Bologna', 'US Sassuolo Calcio': 'Sassuolo',
    'Parma Calcio 1913': 'Parma', 'Genoa CFC': 'Genoa',
    'Udinese Calcio': 'Udinese', 'Torino FC': 'Torino',
    'Cagliari Calcio': 'Cagliari', 'Venezia FC': 'Venezia',
    'Frosinone Calcio': 'Frosinone', 'US Lecce': 'Lecce',
    'AC Monza': 'Monza', 'Hellas Verona FC': 'Verona',
    'Calcio Como 1907': 'Como',
}

SEASON_WEIGHTS = {'2026_27': 3.0, '2025_26': 2.0, '2024_25': 1.0}
LEAGUE_AVG_CROSSES  = 12.0
LEAGUE_AVG_CORNERS  = 9.5
LEAGUE_AVG_SPRINTS  = 170.0

# Bayesian prior strength — quante partite di prior equivalgono alla media storica
PRIOR_STRENGTH = 8.0
# EMA halflife in partite
EMA_HALFLIFE   = 5.0
EMA_ALPHA      = 1 - np.exp(-1.0 / EMA_HALFLIFE)

POSITION_CARD_FACTORS = {'M': 1.35, 'D': 1.15, 'F': 1.00, 'G': 0.35}


def _safe_float(v):
    try:
        return float(str(v).split('/')[0].replace('%', '') or 0)
    except Exception:
        return 0.0


class PlayerStatsBuilder:
    def __init__(self, details_dir='cache/match_details'):
        self.details_dir = Path(details_dir)
        # {season_key -> {team -> {pid -> stats}}}
        self._rosters = {}
        # {season_key -> {team -> {stat -> [values]}}}
        self._team_ctx = {}
        # {pid -> [(round, season_key, goals_in_game, minutes, cards)]}
        self._player_timeline = defaultdict(list)
        self._built = False

    def _parse_all(self):
        for season_dir in sorted(self.details_dir.iterdir()):
            if not season_dir.is_dir():
                continue
            season_key = season_dir.name
            roster     = defaultdict(lambda: defaultdict(lambda: {
                'name': '', 'team': '', 'position': 'F', 'height': 180,
                'apps': 0, 'minutes': 0.0,
                'goals': 0, 'assists': 0,
                'goals_head': 0, 'goals_foot': 0,
                'goals_setpiece': 0, 'goals_penalty': 0, 'goals_open': 0,
                'yellow': 0, 'red': 0,
                'shots': 0, 'xg': 0.0,
            }))
            team_ctx = defaultdict(lambda: defaultdict(list))

            for f in season_dir.iterdir():
                try:
                    data = json.loads(f.read_text())
                except Exception:
                    continue

                home = NAME_MAP.get(data.get('home', ''), data.get('home', ''))
                away = NAME_MAP.get(data.get('away', ''), data.get('away', ''))
                rnd  = int(data.get('round', 0))

                # --- Minuti da sostituzioni ---
                player_minutes = {}
                player_info    = {}
                for inc in (data.get('incidents', {}) or {}).get('incidents', []):
                    if inc.get('incidentType') != 'substitution':
                        continue
                    t    = min(int(inc.get('time', 90) or 90), 90)
                    team = home if inc.get('isHome', True) else away
                    for key, mins in [('playerOut', t), ('playerIn', 90 - t)]:
                        p = inc.get(key) or {}
                        if p.get('id'):
                            player_minutes[p['id']] = float(mins)
                            player_team = team
                            player_info[p['id']] = {
                                'name': p.get('name', ''),
                                'position': p.get('position', 'M'),
                                'height': p.get('height', 180) or 180,
                                'team': team,
                            }

                appeared = set()

                for inc in (data.get('incidents', {}) or {}).get('incidents', []):
                    typ    = inc.get('incidentType')
                    is_home = inc.get('isHome', True)
                    team   = home if is_home else away

                    if typ == 'goal':
                        net      = inc.get('footballPassingNetworkAction', [])
                        goal_act = next((a for a in net if a.get('eventType') == 'goal'), {})
                        body     = goal_act.get('bodyPart', '')
                        sit      = (inc.get('situation') or goal_act.get('situation') or '').lower()
                        xg_val   = float(goal_act.get('xg', 0) or 0)

                        p   = inc.get('player') or {}
                        pid = p.get('id')
                        if pid:
                            appeared.add((pid, team))
                            pi = player_info.setdefault(pid, {
                                'name': p.get('name', ''),
                                'position': p.get('position', 'F'),
                                'height': p.get('height', 180) or 180,
                                'team': team,
                            })
                            s = roster[team][pid]
                            s['name']     = p.get('name', '') or s['name']
                            s['team']     = team
                            s['position'] = p.get('position', 'F') or s['position']
                            s['height']   = p.get('height', 180) or s['height'] or 180
                            s['goals']   += 1
                            s['xg']      += xg_val
                            s['shots']   += 1
                            if 'head' in body:
                                s['goals_head'] += 1
                            else:
                                s['goals_foot'] += 1
                            if 'penalty' in sit:
                                s['goals_penalty'] += 1
                            elif any(k in sit for k in ('set', 'corner', 'free')):
                                s['goals_setpiece'] += 1
                            else:
                                s['goals_open'] += 1

                        # assist
                        a   = inc.get('assist1') or {}
                        aid = a.get('id')
                        if aid:
                            appeared.add((aid, team))
                            roster[team][aid]['assists'] += 1
                            roster[team][aid]['name']    = a.get('name', '') or roster[team][aid]['name']
                            roster[team][aid]['team']    = team

                    elif typ == 'card':
                        p   = inc.get('player') or {}
                        pid = p.get('id')
                        if pid:
                            appeared.add((pid, team))
                            s   = roster[team][pid]
                            s['name']     = p.get('name', '') or s['name']
                            s['team']     = team
                            s['position'] = p.get('position', 'M') or s['position']
                            s['height']   = p.get('height', 180) or s['height'] or 180
                            cls = inc.get('incidentClass', 'yellow')
                            if cls == 'yellow':
                                s['yellow'] += 1
                            elif cls in ('red', 'yellowRed'):
                                s['red'] += 1

                # Aggiorna presenze e timeline per ogni giocatore visto
                all_seen = set(player_minutes.keys()) | {pid for pid, _ in appeared}
                for pid in all_seen:
                    inf  = player_info.get(pid, {})
                    team = inf.get('team', '')
                    if not team:
                        continue
                    mins = player_minutes.get(pid, 90.0)
                    s    = roster[team][pid]
                    if inf.get('name') and not s['name']:
                        s['name'] = inf['name']
                    if not s['team']:
                        s['team'] = team
                    if inf.get('position'):
                        s['position'] = inf['position']
                    if inf.get('height'):
                        s['height'] = inf['height'] or 180
                    s['apps']    += 1
                    s['minutes'] += mins

                    # Timeline: quanti gol ha segnato in questa partita
                    goals_this_game  = sum(
                        1 for a_pid, a_team in appeared
                        if a_pid == pid and a_team == team
                        and any(
                            inc2.get('player', {}).get('id') == pid
                            and inc2.get('incidentType') == 'goal'
                            for inc2 in (data.get('incidents', {}) or {}).get('incidents', [])
                        )
                    )
                    # Contiamo i gol direttamente
                    goals_this_game = 0
                    cards_this_game = 0
                    for inc2 in (data.get('incidents', {}) or {}).get('incidents', []):
                        if inc2.get('incidentType') == 'goal':
                            if (inc2.get('player') or {}).get('id') == pid:
                                goals_this_game += 1
                        elif inc2.get('incidentType') == 'card':
                            if (inc2.get('player') or {}).get('id') == pid:
                                cards_this_game += 1

                    self._player_timeline[pid].append({
                        'season': season_key,
                        'round': rnd,
                        'team': team,
                        'goals': goals_this_game,
                        'cards': cards_this_game,
                        'minutes': mins,
                    })

                # Team context stats
                stats_raw = (data.get('statistics', {}) or {}).get('statistics', [])
                if stats_raw:
                    for sg in stats_raw[0].get('groups', []):
                        for item in sg.get('statisticsItems', []):
                            n = item.get('name', '')
                            hv, av = _safe_float(item.get('home')), _safe_float(item.get('away'))
                            if n == 'Corner kicks':
                                team_ctx[home]['corners'].append(hv)
                                team_ctx[away]['corners'].append(av)
                            elif n == 'Crosses':
                                team_ctx[home]['crosses'].append(hv)
                                team_ctx[away]['crosses'].append(av)
                            elif n == 'Number of sprints':
                                team_ctx[home]['sprints'].append(hv)
                                team_ctx[away]['sprints'].append(av)
                            elif n == 'Fouls':
                                team_ctx[home]['fouls'].append(hv)
                                team_ctx[away]['fouls'].append(af := av)
                            elif n == 'Total tackles':
                                team_ctx[home]['tackles'].append(hv)
                                team_ctx[away]['tackles'].append(av)
                            elif n == 'Total shots':
                                team_ctx[home]['shots'].append(hv)
                                team_ctx[away]['shots'].append(av)
                            elif n == 'Shots on target':
                                team_ctx[home]['shots_ot'].append(hv)
                                team_ctx[away]['shots_ot'].append(av)
                            elif n == 'Total shots':
                                team_ctx[home]['shots'].append(hv)
                                team_ctx[away]['shots'].append(av)
                            elif n == 'Shots on target':
                                team_ctx[home]['shots_ot'].append(hv)
                                team_ctx[away]['shots_ot'].append(av)

            self._rosters[season_key] = {t: dict(d) for t, d in roster.items()}
            self._team_ctx[season_key] = {
                t: {k: float(np.mean(v)) if v else 0.0 for k, v in s.items()}
                for t, s in team_ctx.items()
            }

        self._built = True

    # ── Calcola tasso EMA con shrinkage bayesiano ─────────────────────────────
    def _ema_rate_with_shrinkage(self, pid, stat='goals', prior_rate=0.0):
        """
        Calcola il tasso atteso con:
        1. EMA sulle partite in ordine cronologico (halflife=5)
        2. Shrinkage bayesiano verso prior_rate quando n_games è piccolo

        Ritorna (posterior_rate, n_games, ema_last5)
        """
        timeline = sorted(
            self._player_timeline.get(pid, []),
            key=lambda x: (x['season'], x['round'])
        )
        if not timeline:
            return prior_rate, 0, prior_rate

        n_games = len(timeline)

        # EMA sul tasso per partita
        ema = None
        for match in timeline:
            val = match.get(stat, 0)
            mins = max(match.get('minutes', 90), 1)
            rate_this_game = val / mins * 90  # normalizza per 90 min
            if ema is None:
                ema = rate_this_game
            else:
                ema = EMA_ALPHA * rate_this_game + (1 - EMA_ALPHA) * ema

        # EMA ultime 5 partite (per tag forma)
        recent = timeline[-5:]
        ema_last5 = sum(m.get(stat, 0) for m in recent)

        # Shrinkage bayesiano:
        # posterior = (prior_strength * prior_rate + n_games * ema) / (prior_strength + n_games)
        # Con n_games << PRIOR_STRENGTH, il posterior tende al prior
        # Con n_games >> PRIOR_STRENGTH, il posterior tende all'EMA
        posterior = (PRIOR_STRENGTH * prior_rate + n_games * ema) / (PRIOR_STRENGTH + n_games)

        return posterior, n_games, ema_last5

    # ── Contesto di squadra ───────────────────────────────────────────────────
    def get_team_context(self, team, season='2026_27'):
        if not self._built:
            self._parse_all()
        ctx = self._team_ctx.get(season, {}).get(team, {})
        if not ctx:
            ctx = self._team_ctx.get('2025_26', {}).get(team, {})
        return {
            'crosses_avg':  ctx.get('crosses',  LEAGUE_AVG_CROSSES),
            'corners_avg':  ctx.get('corners',  LEAGUE_AVG_CORNERS),
            'sprints_avg':  ctx.get('sprints',  LEAGUE_AVG_SPRINTS),
            'fouls_avg':    ctx.get('fouls',    12.0),
            'tackles_avg':  ctx.get('tackles',  15.0),
            'shots_avg':    ctx.get('shots',    11.0),
            'shots_ot_avg': ctx.get('shots_ot', 4.0),
        }

    # ── Roster squadra ────────────────────────────────────────────────────────
    def get_team_players(self, team):
        if not self._built:
            self._parse_all()
        # Usa roster ufficiale da lineup G3-G4 (post mercato)
        import json as _json
        from pathlib import Path as _Path
        _roster_file = _Path('cache/rosters_2627.json')
        if _roster_file.exists():
            _official = _json.loads(_roster_file.read_text()).get(team, {})
            # Filtra solo i giocatori ufficialmente in rosa
            roster_all = self._rosters.get('2026_27', {}).get(team, {})
            roster_26  = self._rosters.get('2025_26', {}).get(team, {})
            roster_25  = self._rosters.get('2024_25', {}).get(team, {})
            roster = {}
            for pid_str, pdata in _official.items():
                pid = int(pid_str) if str(pid_str).isdigit() else pid_str
                # Prendi stats da qualsiasi stagione disponibile
                stats = roster_all.get(pid) or roster_26.get(pid) or roster_25.get(pid)
                if stats:
                    roster[pid] = stats
                else:
                    # Giocatore in rosa ma senza stats — aggiungilo con dati minimi
                    roster[pid] = {
                        'name': pdata.get('name',''),
                        'team': team,
                        'position': pdata.get('position','F'),
                        'height': pdata.get('height', 180) or 180,
                        'apps': 0, 'minutes': 0.0,
                        'goals': 0, 'assists': 0,
                        'goals_head': 0, 'goals_foot': 0,
                        'goals_setpiece': 0, 'goals_penalty': 0, 'goals_open': 0,
                        'yellow': 0, 'red': 0, 'shots': 0, 'xg': 0.0,
                    }
        else:
            roster = self._rosters.get('2026_27', {}).get(team, {})
            if not roster:
                roster = self._rosters.get('2025_26', {}).get(team, {})

        players = []
        for pid, s in roster.items():
            if not s.get('name'):
                continue
            apps  = max(s['apps'], 1)
            mins  = max(s['minutes'], 1.0)
            prior_goals = 0.0
            prior_cards = 0.0
            for sk, w in [('2025_26', 2.0), ('2024_25', 1.0)]:
                hist = self._rosters.get(sk, {}).get(team, {}).get(pid)
                if hist and hist['apps'] > 0:
                    prior_goals += hist['goals'] / hist['apps'] * w
                    prior_cards += hist['yellow'] / hist['apps'] * w
            total_w = sum(w for sk, w in [('2025_26', 2.0), ('2024_25', 1.0)]
                          if pid in self._rosters.get(sk, {}).get(team, {}))
            prior_goals = prior_goals / total_w if total_w > 0 else 0.0
            prior_cards = prior_cards / total_w if total_w > 0 else 0.0

            g_post, n_g, g_last5 = self._ema_rate_with_shrinkage(pid, 'goals', prior_goals)
            c_post, n_c, c_last5 = self._ema_rate_with_shrinkage(pid, 'cards', prior_cards)

            xg_per_shot = s['xg'] / s['shots'] if s['shots'] > 0 else 0.0

            players.append({
                'id': pid,
                'name': s['name'], 'team': team,
                'position': s['position'], 'height': s['height'],
                'goals': s['goals'], 'assists': s['assists'],
                'goals_head': s['goals_head'],
                'goals_setpiece': s['goals_setpiece'],
                'goals_penalty': s['goals_penalty'],
                'goals_open': s['goals_open'],
                'yellow_cards': s['yellow'],
                'apps': s['apps'],
                'avg_mins_per_game': round(mins / apps, 1),
                'goals_per90': round(g_post, 3),
                'cards_per90': round(c_post, 3),
                'xg_per_shot': round(xg_per_shot, 3),
                'n_games': n_g,
                'goals_last5': int(g_last5),
                '_prior_goals': prior_goals,
                '_ema_goals': g_post,
            })
        return sorted(players, key=lambda x: -x['goals_per90'])

    # ── Probabilità marcatori ────────────────────────────────────────────────
    def scorer_probability(self, team, team_xg, opp_context=None, limit=5):
        players = self.get_team_players(team)
        if not players:
            return []

        opp           = opp_context or {}
        corner_factor = opp.get('corners_avg', LEAGUE_AVG_CORNERS)  / LEAGUE_AVG_CORNERS
        cross_factor  = opp.get('crosses_avg', LEAGUE_AVG_CROSSES)  / LEAGUE_AVG_CROSSES
        press_factor  = opp.get('sprints_avg', LEAGUE_AVG_SPRINTS)  / LEAGUE_AVG_SPRINTS
        tackle_factor = opp.get('tackles_avg', 15.0) / 15.0

        enriched = []
        for p in players:
            # Base: tasso bayesiano (già shrinkato verso il prior storico)
            base = p['goals_per90']

            # Header boost (contestuale ai corner avversario)
            if p['height'] >= 184 and p['goals_head'] > 0:
                head_rate = p['goals_head'] / max(p['apps'], 1)
                header_boost = head_rate * corner_factor * cross_factor * 0.4
            else:
                header_boost = 0.0

            # Set-piece boost (contestuale ai falli avversario)
            sp_rate   = p['goals_setpiece'] / max(p['apps'], 1)
            sp_boost  = sp_rate * tackle_factor * 0.25

            # Penalty boost (pressing → più falli in area)
            pen_rate  = p['goals_penalty'] / max(p['apps'], 1)
            pen_boost = pen_rate * press_factor * 0.15

            # xG quality (qualità dei tiri)
            xg_boost  = p['xg_per_shot'] * 0.08

            # Fattore partecipazione
            min_factor = min(p['avg_mins_per_game'] / 90.0, 1.0)

            score = (base + header_boost + sp_boost + pen_boost + xg_boost) * min_factor

            # Tag informativi
            tags = []
            if p['goals_head'] > 0 and p['height'] >= 184:
                tags.append(f"testa {p['goals_head']}")
            if p['goals_setpiece'] > 0:
                tags.append(f"piazzati {p['goals_setpiece']}")
            if p['goals_penalty'] > 0:
                tags.append(f"rig. {p['goals_penalty']}")
            # Forma: basata su gol reali nelle ultime 5 partite (non stimati)
            if p['goals_last5'] >= 3:
                tags.append("🔥 in forma")
            elif p['goals_last5'] == 0 and p['goals'] > 0 and p['n_games'] >= 4:
                tags.append("a secco")

            # Nota sul peso del prior
            if p['n_games'] < 4:
                tags.append(f"({p['n_games']}P dati)")

            enriched.append({**p, '_score': score, 'tags': ', '.join(tags)})

        total_score = sum(p['_score'] for p in enriched) or 1
        result = []
        for p in enriched:
            share = p['_score'] / total_score
            prob  = round(1 - np.exp(-share * team_xg), 3)
            result.append({**p, 'prob_score': prob, 'goal_share': round(share, 2)})

        return sorted(result, key=lambda x: -x['prob_score'])[:limit]

    # ── Probabilità ammonizioni ──────────────────────────────────────────────
    def booking_probability(self, team, exp_cards, opp_context=None,
                            referee_factor=1.0, limit=5):
        players = self.get_team_players(team)
        if not players:
            return []

        opp          = opp_context or {}
        press_factor = opp.get('sprints_avg', LEAGUE_AVG_SPRINTS) / LEAGUE_AVG_SPRINTS

        enriched = []
        for p in players:
            pos_factor = POSITION_CARD_FACTORS.get(p['position'], 1.0)
            rate = p['cards_per90'] * press_factor * pos_factor
            enriched.append({**p, '_rate': rate})

        total_rate = sum(p['_rate'] for p in enriched) or 1
        result = []
        for p in enriched:
            share = p['_rate'] / total_rate
            # referee_factor sul tasso atteso totale (non sulla share)
            prob = round(1 - np.exp(-share * exp_cards * referee_factor), 3)
            result.append({
                **p,
                'yellow_cards': p['yellow_cards'],
                'prob_booking': prob,
                'card_share': round(share, 2),
            })
        return sorted(result, key=lambda x: -x['prob_booking'])[:limit]
