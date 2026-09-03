"""
Bet Tracker — registra e chiude automaticamente le giocate.
Salva in cache/bet_tracker.json
"""
import json, uuid
from pathlib import Path
from datetime import datetime

TRACKER_PATH = Path('cache/bet_tracker.json')

def load_tracker() -> dict:
    if not TRACKER_PATH.exists():
        return {'bets': []}
    return json.loads(TRACKER_PATH.read_text(encoding='utf-8'))

def save_tracker(data: dict):
    TRACKER_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def add_bets(home: str, away: str, match_date: str, vbs: list, round_num: int = None):
    """Aggiunge le value bet trovate al tracker."""
    data = load_tracker()
    added = 0
    for vb in vbs:
        if not vb.get('affidabile', True):
            continue  # Skip mercati con ROI storico negativo
        mercato = vb.get('mercato', '')
        # Solo mercati positivi con stellina
        if not any(m in mercato for m in ['Pareggio', 'Under 2.5', 'Over 2.5',
                                           'Vittoria Ospite', 'Under 3.5', 'Over 3.5',
                                           'Under 1.5', 'Over 1.5', 'Goal/Goal', 'No Goal',
                                           'HT Over', 'HT Under']):
            continue
        # Evita duplicati
        exists = any(
            b['home'] == home and b['away'] == away and
            b['mercato'] == mercato and b['status'] == 'pending'
            for b in data['bets']
        )
        if exists:
            continue
        bet = {
            'id': str(uuid.uuid4())[:8],
            'home': home,
            'away': away,
            'match_date': match_date,
            'round': round_num,
            'mercato': mercato,
            'quota': vb.get('quota', 0),
            'edge_pct': vb.get('edge_%', 0),
            'stake': 4.0,
            'status': 'pending',
            'profitto': None,
            'added_at': datetime.now().isoformat()[:19],
        }
        data['bets'].append(bet)
        added += 1
    save_tracker(data)
    return added

def check_bet_result(bet: dict, home_score: int, away_score: int,
                     ht_home: int = 0, ht_away: int = 0) -> bool | None:
    """Determina se una giocata è vinta o persa."""
    mercato = bet['mercato']
    total = home_score + away_score
    ht_total = ht_home + ht_away

    rules = {
        '1 - Vittoria Casa':    home_score > away_score,
        'X - Pareggio':         home_score == away_score,
        '2 - Vittoria Ospite':  home_score < away_score,
        'Over 1.5 Gol':         total > 1.5,
        'Under 1.5 Gol':        total < 1.5,
        'Over 2.5 Gol':         total > 2.5,
        'Under 2.5 Gol':        total < 2.5,
        'Over 3.5 Gol':         total > 3.5,
        'Under 3.5 Gol':        total < 3.5,
        'Over 4.5 Gol':         total > 4.5,
        'Under 4.5 Gol':        total < 4.5,
        'Goal/Goal':            home_score > 0 and away_score > 0,
        'GG - Goal/Goal':       home_score > 0 and away_score > 0,
        'No Goal':              home_score == 0 or away_score == 0,
        'NG - No Goal':         home_score == 0 or away_score == 0,
        'HT Over 0.5 Gol':      ht_total > 0.5,
        'HT Under 0.5 Gol':     ht_total < 0.5,
        'HT Over 1.5 Gol':      ht_total > 1.5,
        'HT Under 1.5 Gol':     ht_total < 1.5,
        'HT Over 2.5 Gol':      ht_total > 2.5,
        'HT Under 2.5 Gol':     ht_total < 2.5,
    }
    for key, result in rules.items():
        if key in mercato:
            return result
    return None

def close_bets_from_sofascore(round_num: int, dry_run: bool = False):
    """
    Scarica risultati Sofascore e chiude le giocate pendenti.
    Chiamata automaticamente dal weekly update.
    """
    import requests
    try:
        resp = requests.get(
            f'http://localhost:8000/sofascore/serie-a/results?round={round_num}',
            timeout=15)
        events = resp.json().get('events', [])
    except Exception as e:
        print(f'  [tracker] Errore Sofascore: {e}')
        return 0

    # Mappa nomi Sofascore -> nomi modello
    NAME_MAP = {
        'AS Roma': 'Roma', 'FC Internazionale Milano': 'Inter', 'Inter': 'Inter',
        'AC Milan': 'Milan', 'Calcio Como 1907': 'Como', 'Como': 'Como',
        'SSC Napoli': 'Napoli', 'Atalanta Bergamasca Calcio': 'Atalanta',
        'Juventus FC': 'Juventus', 'ACF Fiorentina': 'Fiorentina',
        'SS Lazio': 'Lazio', 'Bologna FC 1909': 'Bologna',
        'US Sassuolo Calcio': 'Sassuolo', 'Parma Calcio 1913': 'Parma',
        'Genoa CFC': 'Genoa', 'Udinese Calcio': 'Udinese',
        'Torino FC': 'Torino', 'Cagliari Calcio': 'Cagliari',
        'Venezia FC': 'Venezia', 'Frosinone Calcio': 'Frosinone',
        'US Lecce': 'Lecce', 'AC Monza': 'Monza',
    }

    data = load_tracker()
    closed = 0

    for event in events:
        status = event.get('status', {}).get('type', '')
        if status != 'finished':
            continue

        home_raw = event['homeTeam']['name']
        away_raw = event['awayTeam']['name']
        home = NAME_MAP.get(home_raw, home_raw)
        away = NAME_MAP.get(away_raw, away_raw)
        hs = event.get('homeScore', {}).get('current', 0)
        as_ = event.get('awayScore', {}).get('current', 0)
        ht_h = event.get('homeScore', {}).get('period1', 0)
        ht_a = event.get('awayScore', {}).get('period1', 0)

        for bet in data['bets']:
            if bet['status'] != 'pending':
                continue
            if bet['home'] != home or bet['away'] != away:
                continue

            won = check_bet_result(bet, hs, as_, ht_h, ht_a)
            if won is None:
                continue

            if not dry_run:
                bet['status'] = 'won' if won else 'lost'
                bet['profitto'] = round(
                    bet['stake'] * (bet['quota'] - 1) if won else -bet['stake'], 2)
                bet['closed_at'] = datetime.now().isoformat()[:19]
                bet['result'] = f'{hs}-{as_}'

            emoji = '✅' if won else '❌'
            prof = bet['stake'] * (bet['quota'] - 1) if won else -bet['stake']
            print(f'  {emoji} {home} vs {away} ({hs}-{as_}) | {bet["mercato"]} @ {bet["quota"]} → {prof:+.2f}€')
            closed += 1

    if not dry_run:
        save_tracker(data)
    print(f'  [tracker] {closed} giocate chiuse per giornata {round_num}')
    return closed

def get_stats() -> dict:
    """Statistiche del tracker per la dashboard."""
    data = load_tracker()
    bets = data['bets']
    closed = [b for b in bets if b['status'] != 'pending']
    pending = [b for b in bets if b['status'] == 'pending']
    won = [b for b in closed if b['status'] == 'won']
    lost = [b for b in closed if b['status'] == 'lost']

    total_stake = sum(b['stake'] for b in closed)
    total_profit = sum(b['profitto'] for b in closed if b['profitto'])

    return {
        'total': len(bets),
        'pending': len(pending),
        'closed': len(closed),
        'won': len(won),
        'lost': len(lost),
        'win_rate': len(won) / max(len(closed), 1) * 100,
        'total_stake': total_stake,
        'total_profit': total_profit,
        'roi': total_profit / max(total_stake, 1) * 100,
        'bets': bets,
    }

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'close':
        round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        print(f'Chiudo giocate giornata {round_num}...')
        close_bets_from_sofascore(round_num)
    else:
        stats = get_stats()
        print(f'Giocate totali: {stats["total"]}')
        print(f'Pendenti: {stats["pending"]}')
        print(f'Chiuse: {stats["closed"]} ({stats["won"]} vinte, {stats["lost"]} perse)')
        print(f'Win rate: {stats["win_rate"]:.1f}%')
        print(f'ROI: {stats["roi"]:+.1f}%')
        print(f'Profitto: €{stats["total_profit"]:+.2f}')
