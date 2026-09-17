"""
models/player_stats.py v3
Modello marcatori completo:
- Minuti giocati esatti (da eventi sostituzione)
- Gol/cartellini per 90 min normalizzati
- Tiri tentati (da footballPassingNetworkAction)
- xG per 90 min e xG per tiro
- Contesto avversario: stile difensivo, cross concessi, pressing
- Top 5 marcatori per team con probabilità contestuale
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
    'Spezia Calcio': 'Spezia', 'Empoli FC': 'Empoli',
    'US Salernitana 1919': 'Salernitana', 'US Cremonese': 'Cremonese',
    'UC Sampdoria': 'Sampdoria', 'Calcio Como 1907': 'Como',
}

SEASON_WEIGHTS = {'2026_27': 3.0, '2025_26': 2.0, '2024_25': 1.0}
LEAGUE_AVG_CROSSES = 12.0
LEAGUE_AVG_CORNERS = 9.5
LEAGUE_AVG_SPRINTS = 170.0


def _empty():
    return {
        'name': '', 'team': '', 'position': 'F', 'height': 180,
        'minutes': 0.0, 'apps': 0,
        'goals': 0, 'assists': 0,
        'goals_head': 0, 'goals_foot': 0,
        'goals_setpiece': 0, 'goals_penalty': 0, 'goals_open': 0,
        'yellow': 0, 'red': 0,
        'shots': 0, 'xg': 0.0,
    }


class PlayerStatsBuilder:
    def __init__(self, details_dir='cache/match_details'):
        self.details_dir = Path(details_dir)
        self._rosters = {}
        self._team_stats = {}   # {season -> {team -> {crosses, corners, sprints, ...}}}
        self._built = False

    def _parse_season(self, season_dir):
        players = defaultdict(_empty)
        team_stats = defaultdict(lambda: {
            'crosses': [], 'corners': [], 'sprints': [],
            'fouls': [], 'clearances': [], 'tackles': []
        })

        for f in season_dir.iterdir():
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue

            home = NAME_MAP.get(data.get('home', ''), data.get('home', ''))
            away = NAME_MAP.get(data.get('away', ''), data.get('away', ''))
            game_id = str(data.get('game_id', f.stem))
            incs = data.get('incidents', {})
            inc_list = incs.get('incidents', []) if isinstance(incs, dict) else []

            # ── Minuti giocati da sostituzioni ───────────────────────────────
            player_minutes = {}   # pid -> minutes played
            player_team = {}      # pid -> team
            player_info = {}      # pid -> {name, position, height}

            for inc in inc_list:
                if inc.get('incidentType') == 'substitution':
                    t = min(int(inc.get('time', 90)), 90 + int(inc.get('addedTime', 0) or 0))
                    is_home = inc.get('isHome', True)
                    team = home if is_home else away
                    p_out = inc.get('playerOut') or {}
                    p_in = inc.get('playerIn') or {}
                    if p_out.get('id'):
                        pid = p_out['id']
                        player_minutes[pid] = float(t)
                        player_team[pid] = team
                        player_info[pid] = {
                            'name': p_out.get('name', ''),
                            'position': p_out.get('position', 'M'),
                            'height': p_out.get('height', 180) or 180,
                        }
                    if p_in.get('id'):
                        pid = p_in['id']
                        player_minutes[pid] = float(90 - t)
                        player_team[pid] = team
                        player_info[pid] = {
                            'name': p_in.get('name', ''),
                            'position': p_in.get('position', 'M'),
                            'height': p_in.get('height', 180) or 180,
                        }

            # ── Goal incidents ────────────────────────────────────────────────
            appeared = set()   # pid

            for inc in inc_list:
                typ = inc.get('incidentType')
                is_home = inc.get('isHome', True)
                team = home if is_home else away

                if typ == 'goal':
                    net = inc.get('footballPassingNetworkAction', [])
                    goal_act = next((a for a in net if a.get('eventType') == 'goal'), {})
                    body_part = goal_act.get('bodyPart', '')
                    situation = (inc.get('situation') or goal_act.get('situation') or '').lower()
                    xg_val = float(goal_act.get('xg', 0) or 0)

                    p = inc.get('player') or {}
                    pid = p.get('id')
                    if pid:
                        appeared.add(pid)
                        player_team[pid] = team
                        player_info[pid] = {
                            'name': p.get('name', ''),
                            'position': p.get('position', 'F'),
                            'height': p.get('height', 180) or 180,
                        }
                        s = players[pid]
                        s['goals'] += 1
                        s['xg'] += xg_val
                        s['shots'] += 1
                        if 'head' in body_part:
                            s['goals_head'] += 1
                        else:
                            s['goals_foot'] += 1
                        if 'penalty' in situation:
                            s['goals_penalty'] += 1
                        elif any(k in situation for k in ('set', 'corner', 'free', 'piazzato')):
                            s['goals_setpiece'] += 1
                        else:
                            s['goals_open'] += 1

                    # Assist
                    a = inc.get('assist1') or {}
                    aid = a.get('id')
                    if aid:
                        appeared.add(aid)
                        player_team[aid] = team
                        player_info[aid] = {
                            'name': a.get('name', ''),
                            'position': a.get('position', 'M'),
                            'height': a.get('height', 180) or 180,
                        }
                        players[aid]['assists'] += 1

                    # Tiri tentati da passing network
                    for act in net:
                        if act.get('eventType') in ('goal', 'shot'):
                            shot_pid = act.get('player', {}).get('id')
                            if shot_pid and shot_pid != pid:
                                appeared.add(shot_pid)
                                players[shot_pid]['shots'] += 1

                elif typ == 'card':
                    p = inc.get('player') or {}
                    pid = p.get('id')
                    if pid:
                        appeared.add(pid)
                        player_team[pid] = team
                        player_info[pid] = {
                            'name': p.get('name', ''),
                            'position': p.get('position', 'M'),
                            'height': p.get('height', 180) or 180,
                        }
                        cls = inc.get('incidentClass', 'yellow')
                        if cls == 'yellow':
                            players[pid]['yellow'] += 1
                        elif cls in ('red', 'yellowRed'):
                            players[pid]['red'] += 1

            # Aggiorna info giocatori e minuti
            all_pids = set(player_minutes.keys()) | appeared
            for pid in all_pids:
                info = player_info.get(pid, {})
                team = player_team.get(pid, '')
                if not team:
                    continue
                mins = player_minutes.get(pid, 90.0)
                p = players[pid]
                if not p['name'] and info.get('name'):
                    p['name'] = info['name']
                if not p['team']:
                    p['team'] = team
                if info.get('position'):
                    p['position'] = info['position']
                if info.get('height'):
                    p['height'] = info['height'] or 180
                p['minutes'] += mins
                p['apps'] += 1

            # ── Statistiche di squadra da statistics ─────────────────────────
            stats_data = data.get('statistics', {}).get('statistics', [])
            if stats_data:
                groups = stats_data[0].get('groups', [])
                for sg in groups:
                    for item in sg.get('statisticsItems', []):
                        name = item.get('name', '')
                        hv = item.get('home', '0') or '0'
                        av = item.get('away', '0') or '0'
                        try:
                            hf = float(str(hv).split('/')[0].replace('%','').strip())
                            af = float(str(av).split('/')[0].replace('%','').strip())
                        except Exception:
                            continue
                        if name == 'Corner kicks':
                            team_stats[home]['corners'].append(hf)
                            team_stats[away]['corners'].append(af)
                        elif name == 'Crosses':
                            team_stats[home]['crosses'].append(hf)
                            team_stats[away]['crosses'].append(af)
                        elif name == 'Number of sprints':
                            team_stats[home]['sprints'].append(hf)
                            team_stats[away]['sprints'].append(af)
                        elif name == 'Fouls':
                            team_stats[home]['fouls'].append(hf)
                            team_stats[away]['fouls'].append(af)
                        elif name == 'Clearances':
                            team_stats[home]['clearances'].append(hf)
                            team_stats[away]['clearances'].append(af)
                        elif name == 'Total tackles':
                            team_stats[home]['tackles'].append(hf)
                            team_stats[away]['tackles'].append(af)

        # Consolida
        roster = defaultdict(dict)
        for pid, p in players.items():
            team = p.get('team', '')
            if team and p['apps'] > 0:
                roster[team][pid] = dict(p)

        # Media team stats
        ts_avg = {}
        for team, stats in team_stats.items():
            ts_avg[team] = {
                k: float(np.mean(v)) if v else 0.0
                for k, v in stats.items()
            }

        return dict(roster), ts_avg

    def _build(self):
        for season_dir in sorted(self.details_dir.iterdir()):
            if not season_dir.is_dir():
                continue
            season_key = season_dir.name
            roster, ts = self._parse_season(season_dir)
            self._rosters[season_key] = roster
            self._team_stats[season_key] = ts
        self._built = True

    def get_team_context(self, team, season='2026_27'):
        """Statistiche medie di stile di gioco di una squadra."""
        if not self._built:
            self._build()
        ts = self._team_stats.get(season, {}).get(team, {})
        if not ts:
            # Fallback alla stagione precedente
            ts = self._team_stats.get('2025_26', {}).get(team, {})
        return {
            'crosses_avg': ts.get('crosses', LEAGUE_AVG_CROSSES),
            'corners_avg': ts.get('corners', LEAGUE_AVG_CORNERS),
            'sprints_avg': ts.get('sprints', LEAGUE_AVG_SPRINTS),
            'fouls_avg': ts.get('fouls', 12.0),
            'clearances_avg': ts.get('clearances', 15.0),
            'tackles_avg': ts.get('tackles', 15.0),
        }

    def get_team_players(self, team, seasons=None):
        if not self._built:
            self._build()
        if seasons is None:
            seasons = ['2026_27', '2025_26', '2024_25']

        # Prima cerca nel roster 2026/27
        current = self._rosters.get('2026_27', {}).get(team, {})

        merged = {}
        for season_key in seasons:
            weight = SEASON_WEIGHTS.get(season_key, 1.0)
            s_data = self._rosters.get(season_key, {}).get(team, {})
            for pid, stats in s_data.items():
                # Solo giocatori nel roster corrente (se disponibile)
                if current and pid not in current:
                    continue
                if pid not in merged:
                    merged[pid] = {
                        'name': stats['name'],
                        'team': team,
                        'position': stats['position'],
                        'height': stats['height'],
                        'goals': 0, 'assists': 0,
                        'goals_head': 0, 'goals_setpiece': 0,
                        'goals_penalty': 0, 'goals_open': 0,
                        'yellow_cards': 0,
                        'shots': 0, 'xg': 0.0,
                        'minutes': 0.0, 'apps': 0,
                        '_g_rate': 0.0, '_h_rate': 0.0,
                        '_sp_rate': 0.0, '_pen_rate': 0.0,
                        '_c_rate': 0.0, '_xg90': 0.0,
                        '_shots90': 0.0, '_weight': 0.0,
                    }
                mins = max(stats['minutes'], 1.0)
                w = weight
                merged[pid]['goals'] += stats['goals']
                merged[pid]['assists'] += stats['assists']
                merged[pid]['goals_head'] += stats['goals_head']
                merged[pid]['goals_setpiece'] += stats['goals_setpiece']
                merged[pid]['goals_penalty'] += stats['goals_penalty']
                merged[pid]['goals_open'] += stats['goals_open']
                merged[pid]['yellow_cards'] += stats['yellow']
                merged[pid]['shots'] += stats['shots']
                merged[pid]['xg'] += stats['xg']
                merged[pid]['minutes'] += stats['minutes']
                merged[pid]['apps'] += stats['apps']
                merged[pid]['_g_rate'] += stats['goals'] / mins * 90 * w
                merged[pid]['_h_rate'] += stats['goals_head'] / mins * 90 * w
                merged[pid]['_sp_rate'] += stats['goals_setpiece'] / mins * 90 * w
                merged[pid]['_pen_rate'] += stats['goals_penalty'] / mins * 90 * w
                merged[pid]['_c_rate'] += stats['yellow'] / mins * 90 * w
                merged[pid]['_xg90'] += stats['xg'] / mins * 90 * w
                merged[pid]['_shots90'] += stats['shots'] / mins * 90 * w
                merged[pid]['_weight'] += w

        result = []
        for pid, p in merged.items():
            w = p['_weight'] or 1.0
            xg_per_shot = p['xg'] / p['shots'] if p['shots'] > 0 else 0.0
            result.append({
                **p,
                'goals_per90': round(p['_g_rate'] / w, 3),
                'head_per90': round(p['_h_rate'] / w, 3),
                'setpiece_per90': round(p['_sp_rate'] / w, 3),
                'penalty_per90': round(p['_pen_rate'] / w, 3),
                'cards_per90': round(p['_c_rate'] / w, 3),
                'xg_per90': round(p['_xg90'] / w, 3),
                'shots_per90': round(p['_shots90'] / w, 3),
                'xg_per_shot': round(xg_per_shot, 3),
                'avg_mins_per_game': round(p['minutes'] / max(p['apps'], 1), 1),
            })
        return sorted(result, key=lambda x: -x['goals_per90'])

    def scorer_probability(self, team, team_xg,
                           opp_context=None,
                           limit=5):
        """
        Probabilità di segnare con contesto completo:
        opp_context: dict con statistiche avversario (crosses, corners, sprints, tackles)
        """
        players = self.get_team_players(team)
        if not players:
            return []

        opp = opp_context or {}
        opp_crosses = opp.get('crosses_avg', LEAGUE_AVG_CROSSES)
        opp_corners = opp.get('corners_avg', LEAGUE_AVG_CORNERS)
        opp_sprints = opp.get('sprints_avg', LEAGUE_AVG_SPRINTS)
        opp_tackles = opp.get('tackles_avg', 15.0)

        # Fattori contestuali
        cross_factor = opp_crosses / LEAGUE_AVG_CROSSES        # >1 = avversario crossa molto
        corner_factor = opp_corners / LEAGUE_AVG_CORNERS       # >1 = avversario concede corner
        press_factor = opp_sprints / LEAGUE_AVG_SPRINTS        # >1 = pressing alto
        tackle_factor = opp_tackles / 15.0                     # >1 = avversario molto aggressivo

        enriched = []
        for p in players:
            base = p['goals_per90']

            # Header boost: avversario concede molti corner/cross + giocatore alto
            if p['height'] >= 184 and p['head_per90'] > 0:
                header_boost = p['head_per90'] * corner_factor * cross_factor * 0.5
            else:
                header_boost = 0

            # Set-piece boost: avversario fallo molto
            sp_boost = p['setpiece_per90'] * tackle_factor * 0.3

            # Penalty boost: avversario pressing aggressivo = più falli in area
            pen_boost = p['penalty_per90'] * press_factor * 0.2

            # xG quality boost: giocatori con alto xG/tiro sono più letali
            xg_boost = p['xg_per_shot'] * 0.1

            # Partecipazione: se gioca pochi minuti pesiamo di meno
            min_factor = min(p['avg_mins_per_game'] / 90.0, 1.0)

            score = (base + header_boost + sp_boost + pen_boost + xg_boost) * min_factor

            # Tag contestuali
            tags = []
            if p['goals_head'] > 0 and p['height'] >= 184:
                tags.append(f"testa {p['goals_head']}")
            if p['goals_setpiece'] > 0:
                tags.append(f"piazzati {p['goals_setpiece']}")
            if p['goals_penalty'] > 0:
                tags.append(f"rig. {p['goals_penalty']}")
            if p['goals_open'] > p['goals']:
                tags.append("open play")

            enriched.append({**p, '_score': score, 'tags': ', '.join(tags)})

        total_score = sum(p['_score'] for p in enriched) or 1
        result = []
        for p in enriched:
            share = p['_score'] / total_score
            prob = round(1 - np.exp(-share * team_xg), 3)
            result.append({
                **p,
                'prob_score': prob,
                'goal_share': round(share, 2),
            })

        return sorted(result, key=lambda x: -x['prob_score'])[:limit]

    def booking_probability(self, team, exp_cards, opp_context=None,
                             referee_factor=1.0, limit=5):
        """
        Probabilità ammonizione con:
        - Fattore posizione: M > D > F > G
        - Fattore pressing avversario
        - Fattore arbitro (da CardsModel)
        """
        # Fattori per posizione (statistiche Serie A storiche)
        POSITION_FACTORS = {
            'M': 1.35,   # Centrocampisti — più falli, più cartellini
            'D': 1.15,   # Difensori — tackle duri
            'F': 1.00,   # Attaccanti — baseline
            'G': 0.35,   # Portieri — raramente ammoniti
        }

        players = self.get_team_players(team)
        if not players:
            return []

        opp = opp_context or {}
        press_factor = opp.get('sprints_avg', LEAGUE_AVG_SPRINTS) / LEAGUE_AVG_SPRINTS

        # Calcola score ponderato per ogni giocatore
        enriched = []
        for p in players:
            pos = p.get('position', 'F')
            pos_factor = POSITION_FACTORS.get(pos, 1.0)
            rate = p['cards_per90'] * press_factor * pos_factor * referee_factor
            enriched.append({**p, '_rate': rate})

        total_rate = sum(p['_rate'] for p in enriched) or 1
        result = []
        for p in enriched:
            share = p['_rate'] / total_rate
            prob = round(1 - np.exp(-share * exp_cards * referee_factor), 3)
            pos = p.get('position', 'F')
            result.append({
                **p,
                'yellow_cards': p['yellow_cards'],
                'prob_booking': prob,
                'card_share': round(share, 2),
                'position': pos,
            })
        return sorted(result, key=lambda x: -x['prob_booking'])[:limit]
