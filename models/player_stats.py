"""
models/player_stats.py v2
Modello marcatori avanzato con:
- bodyPart (head/foot) per identificare bomber di testa
- situation (set-piece/penalty/open-play) per specialisti
- xG per shot → qualità del tiratore
- posizione e altezza → difensori pericolosi su angoli
- probabilità adattiva in base al contesto (angoli avversario, situazione partita)
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


def _empty_player():
    return {
        'name': '', 'team': '', 'position': 'F', 'height': 180,
        'apps': set(), 'goals': 0, 'assists': 0,
        'goals_head': 0, 'goals_foot': 0,
        'goals_setpiece': 0, 'goals_penalty': 0, 'goals_open': 0,
        'yellow': 0, 'red': 0,
        'xg_total': 0.0, 'shots': 0,
    }


class PlayerStatsBuilder:
    def __init__(self, details_dir='cache/match_details'):
        self.details_dir = Path(details_dir)
        self._rosters = {}       # season_key -> {team -> {pid -> stats}}
        self._built = False

    def _parse_all(self):
        for season_dir in sorted(self.details_dir.iterdir()):
            if not season_dir.is_dir():
                continue
            season_key = season_dir.name
            roster = defaultdict(lambda: defaultdict(_empty_player))

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

                appeared = set()

                for inc in inc_list:
                    typ = inc.get('incidentType')
                    is_home = inc.get('isHome', True)
                    team = home if is_home else away

                    if typ == 'substitution':
                        for key in ('playerIn', 'playerOut'):
                            p = inc.get(key) or {}
                            if p.get('id'):
                                appeared.add((p['id'], p.get('name', ''), team,
                                              p.get('position', 'M'), p.get('height', 180)))

                    elif typ == 'goal':
                        p = inc.get('player') or {}
                        pid = p.get('id')
                        if pid:
                            appeared.add((pid, p.get('name', ''), team,
                                          p.get('position', 'F'), p.get('height', 180)))
                            s = roster[team][pid]
                            s['name'] = p.get('name', '')
                            s['team'] = team
                            s['position'] = p.get('position', 'F')
                            s['height'] = p.get('height', 180) or 180
                            s['goals'] += 1

                            # Dati dal footballPassingNetworkAction
                            net_actions = inc.get('footballPassingNetworkAction', [])
                            goal_action = next((a for a in net_actions if a.get('eventType') == 'goal'), {})

                            body_part = goal_action.get('bodyPart', '')
                            situation = inc.get('situation', goal_action.get('situation', ''))
                            xg_val = goal_action.get('xg', 0) or 0

                            if 'head' in body_part:
                                s['goals_head'] += 1
                            else:
                                s['goals_foot'] += 1

                            sit = situation.lower() if situation else ''
                            if 'penalty' in sit:
                                s['goals_penalty'] += 1
                            elif 'set' in sit or 'corner' in sit or 'free' in sit or 'piazzato' in sit:
                                s['goals_setpiece'] += 1
                            else:
                                s['goals_open'] += 1

                            s['xg_total'] += float(xg_val)
                            s['shots'] += 1

                        # Assist
                        a = inc.get('assist1') or {}
                        aid = a.get('id')
                        if aid:
                            appeared.add((aid, a.get('name', ''), team,
                                          a.get('position', 'M'), a.get('height', 180)))
                            s = roster[team][aid]
                            s['name'] = a.get('name', '')
                            s['team'] = team
                            s['position'] = a.get('position', 'M')
                            s['height'] = a.get('height', 180) or 180
                            s['assists'] += 1

                    elif typ == 'card':
                        p = inc.get('player') or {}
                        pid = p.get('id')
                        if pid:
                            appeared.add((pid, p.get('name', ''), team,
                                          p.get('position', 'M'), p.get('height', 180)))
                            s = roster[team][pid]
                            s['name'] = p.get('name', '')
                            s['team'] = team
                            s['position'] = p.get('position', 'M')
                            s['height'] = p.get('height', 180) or 180
                            cls = inc.get('incidentClass', 'yellow')
                            if cls == 'yellow':
                                s['yellow'] += 1
                            elif cls in ('red', 'yellowRed'):
                                s['red'] += 1

                # Registra apparizioni
                for pid, pname, team, pos, height in appeared:
                    s = roster[team][pid]
                    s['apps'].add(game_id)
                    if pname and not s['name']:
                        s['name'] = pname
                    if not s['position'] or s['position'] == 'F':
                        s['position'] = pos
                    if not s['height'] or s['height'] == 180:
                        s['height'] = height or 180

            # Converti set in conteggio
            for team_data in roster.values():
                for pid in team_data:
                    team_data[pid]['apps'] = len(team_data[pid]['apps'])

            self._rosters[season_key] = {t: dict(d) for t, d in roster.items()}

        self._built = True

    def get_team_players(self, team, seasons=None):
        if not self._built:
            self._parse_all()
        if seasons is None:
            seasons = ['2026_27', '2025_26']

        merged = {}
        for season_key in reversed(seasons):
            weight = SEASON_WEIGHTS.get(season_key, 1.0)
            s_data = self._rosters.get(season_key, {}).get(team, {})
            for pid, stats in s_data.items():
                if pid not in merged:
                    merged[pid] = {
                        'name': stats['name'], 'team': team,
                        'position': stats['position'], 'height': stats['height'],
                        'goals': 0, 'assists': 0,
                        'goals_head': 0, 'goals_foot': 0,
                        'goals_setpiece': 0, 'goals_penalty': 0, 'goals_open': 0,
                        'yellow_cards': 0, 'red_cards': 0,
                        'xg_total': 0.0, 'shots': 0, 'apps': 0,
                        'goals_rate': 0.0, 'head_rate': 0.0,
                        'setpiece_rate': 0.0, 'penalty_rate': 0.0,
                        'cards_rate': 0.0, 'xg_per_shot': 0.0,
                    }
                apps = max(stats['apps'], 1)
                w = weight
                merged[pid]['goals'] += stats['goals']
                merged[pid]['assists'] += stats['assists']
                merged[pid]['goals_head'] += stats['goals_head']
                merged[pid]['goals_foot'] += stats['goals_foot']
                merged[pid]['goals_setpiece'] += stats['goals_setpiece']
                merged[pid]['goals_penalty'] += stats['goals_penalty']
                merged[pid]['goals_open'] += stats['goals_open']
                merged[pid]['yellow_cards'] += stats['yellow']
                merged[pid]['red_cards'] += stats['red']
                merged[pid]['xg_total'] += stats['xg_total']
                merged[pid]['shots'] += stats['shots']
                merged[pid]['apps'] += stats['apps']
                merged[pid]['goals_rate'] += stats['goals'] / apps * w
                merged[pid]['head_rate'] += stats['goals_head'] / apps * w
                merged[pid]['setpiece_rate'] += stats['goals_setpiece'] / apps * w
                merged[pid]['penalty_rate'] += stats['goals_penalty'] / apps * w
                merged[pid]['cards_rate'] += stats['yellow'] / apps * w

        # Normalizza per peso totale
        total_weight = sum(SEASON_WEIGHTS.get(s, 1.0) for s in seasons if s in self._rosters)
        for pid in merged:
            tw = total_weight or 1
            merged[pid]['goals_rate'] /= tw
            merged[pid]['head_rate'] /= tw
            merged[pid]['setpiece_rate'] /= tw
            merged[pid]['penalty_rate'] /= tw
            merged[pid]['cards_rate'] /= tw
            shots = merged[pid]['shots']
            merged[pid]['xg_per_shot'] = round(merged[pid]['xg_total'] / shots, 3) if shots > 0 else 0.0

        # Filtra: usa solo giocatori presenti nella stagione corrente
        current = self._rosters.get('2026_27', {}).get(team, {})
        if current:
            merged = {pid: data for pid, data in merged.items() if pid in current}

        players = list(merged.values())
        return sorted(players, key=lambda x: -x['goals_rate'])

    def scorer_probability(self, team, team_xg, opp_corners_avg=9.5, limit=8):
        """
        Probabilità di segnare con contesto:
        - opp_corners_avg: corner medi subiti dall'avversario → boost per giocatori alti/testa
        """
        players = self.get_team_players(team)
        if not players:
            return []

        enriched = []
        for p in players:
            # Score base = goals_rate
            score = p['goals_rate']

            # Boost header se avversario concede molti corner (>10)
            corner_factor = max(0, (opp_corners_avg - 9.5) / 9.5)
            if p['height'] >= 186 and p['goals_head'] > 0:
                score += p['head_rate'] * corner_factor * 1.5

            # Boost set-piece
            score += p['setpiece_rate'] * 0.3

            enriched.append({**p, '_score': score})

        total_score = sum(p['_score'] for p in enriched) or 1
        result = []
        for p in enriched:
            share = p['_score'] / total_score
            prob = round(1 - np.exp(-share * team_xg), 3)
            # Etichetta speciale per tipi di gol
            tags = []
            if p['goals_head'] > 0 and p['height'] >= 185:
                tags.append(f"testa {p['goals_head']}")
            if p['goals_setpiece'] > 0:
                tags.append(f"piazzati {p['goals_setpiece']}")
            if p['goals_penalty'] > 0:
                tags.append(f"rig. {p['goals_penalty']}")
            result.append({
                **p,
                'prob_score': prob,
                'goal_share': round(share, 2),
                'tags': ', '.join(tags) if tags else '',
                'xg_per_shot': p['xg_per_shot'],
            })

        return sorted(result, key=lambda x: -x['prob_score'])[:limit]

    def booking_probability(self, team, exp_cards, limit=6):
        players = self.get_team_players(team)
        if not players:
            return []
        total_rate = sum(p['cards_rate'] for p in players) or 1
        result = []
        for p in players:
            share = p['cards_rate'] / total_rate
            prob = round(1 - np.exp(-share * exp_cards), 3)
            result.append({
                **p,
                'yellow_cards': p['yellow_cards'],
                'prob_booking': prob,
                'card_share': round(share, 2),
            })
        return sorted(result, key=lambda x: -x['prob_booking'])[:limit]
