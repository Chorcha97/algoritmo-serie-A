"""
utils/update_dataset_2627.py
Scarica le nuove partite 2026/27 da Sofascore e le integra nel dataset.
Da chiamare nel weekly_update.py dopo il download dei match_details.
"""
import json, requests, time, pandas as pd
from pathlib import Path

BASE = Path('/Users/davideforcato/Desktop/football_predictor')
SEASON_ID = 95836

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

def update_dataset_2627():
    print("[dataset] Integrazione partite 2026/27...")

    details_dir = BASE / 'cache/match_details/2026_27'
    details_dir.mkdir(exist_ok=True)
    cal = pd.read_csv(BASE / 'cache/calendario_2627.csv')
    existing = {f.stem for f in details_dir.iterdir()}

    # 1. Scarica nuovi match_details
    new_downloads = 0
    for rnd in range(1, 39):
        try:
            r = requests.get('http://localhost:8000/sofascore/serie-a/results',
                             params={'round': rnd, 'season_id': SEASON_ID}, timeout=10)
            if r.status_code != 200:
                break
            events = r.json().get('events', [])
            if not events:
                break
            for e in events:
                gid = str(e.get('id', ''))
                if gid in existing:
                    continue
                dr = requests.get('http://localhost:8000/sofascore/match/details',
                                  params={'match_id': gid}, timeout=10)
                if dr.status_code == 200:
                    (details_dir / f'{gid}.json').write_text(json.dumps(dr.json()))
                    existing.add(gid)
                    new_downloads += 1
                    time.sleep(0.4)
            time.sleep(0.8)
        except Exception as ex:
            print(f"  [WARN] Giornata {rnd}: {ex}")
            break

    print(f"  Nuovi match_details scaricati: {new_downloads}")
    print(f"  Totale partite 2026/27: {len(list(details_dir.iterdir()))}")

    # 2. Costruisci righe dataset da tutti i match_details 2026/27
    rows = []
    for f in details_dir.iterdir():
        try:
            data = json.loads(f.read_text())
        except:
            continue

        home = NAME_MAP.get(data.get('home', ''), data.get('home', ''))
        away = NAME_MAP.get(data.get('away', ''), data.get('away', ''))
        hg = int(data.get('home_score', 0) or 0)
        ag = int(data.get('away_score', 0) or 0)
        ftr = 'H' if hg > ag else ('A' if ag > hg else 'D')
        rnd = data.get('round', 0)

        # Estrai statistiche da Sofascore
        hc = ac = hy = ay = hs = as_ = hxg = axg = 0.0
        stats = data.get('statistics', {}).get('statistics', [])
        if stats:
            for sg in stats[0].get('groups', []):
                for item in sg.get('statisticsItems', []):
                    n = item.get('name', '')
                    try:
                        hv = float(str(item.get('home','0')).split('/')[0].replace('%','') or 0)
                        av = float(str(item.get('away','0')).split('/')[0].replace('%','') or 0)
                    except:
                        continue
                    if n == 'Corner kicks':       hc, ac = hv, av
                    elif n == 'Yellow cards':     hy, ay = hv, av
                    elif n == 'Total shots':      hs, as_ = hv, av
                    elif n == 'Shots on target':  hst, ast = hv, av
                    elif n == 'Expected goals':   hxg, axg = hv, av
                    elif n == 'Fouls':            hf, af = hv, av

        # Data dal calendario
        match = cal[(cal['home'] == home) & (cal['away'] == away)]
        date_str = match['data'].values[0] if len(match) > 0 else '2026-09-01'

        rows.append({
            'Date': date_str, 'HomeTeam': home, 'AwayTeam': away,
            'FTHG': hg, 'FTAG': ag, 'FTR': ftr,
            'HTHG': 0, 'HTAG': 0, 'HTR': 'D',
            'HS': hs, 'AS': as_, 'HST': 0, 'AST': 0,
            'HC': hc, 'AC': ac,
            'HF': 0, 'AF': 0,
            'HY': hy, 'AY': ay, 'HR': 0, 'AR': 0,
            'season': '2026-27', 'league': 'Serie A',
            'xg_home': hxg, 'xg_away': axg,
            'elo_home': 0, 'elo_away': 0, 'elo_diff': 0,
            'round': rnd,
        })

    if not rows:
        print("  Nessuna partita da aggiungere")
        return

    df_new = pd.DataFrame(rows)

    # 3. Integra nel dataset principale
    df_old = pd.read_csv(BASE / 'serie_a_dataset.csv')
    df_old = df_old[df_old['season'] != '2026-27']  # rimuovi vecchio 2026-27
    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all['Date'] = pd.to_datetime(df_all['Date'], errors='coerce')
    df_all = df_all.sort_values('Date').reset_index(drop=True)
    df_all.to_csv(BASE / 'serie_a_dataset.csv', index=False)

    print(f"  Dataset: {len(df_old)} storiche + {len(df_new)} del 2026/27 = {len(df_all)} totali")
    o25 = (df_new['FTHG'] + df_new['FTAG'] > 2.5).mean() * 100
    print(f"  Over 2.5 stagione corrente: {o25:.0f}%")


def update_standings_from_results():
    """Aggiorna la classifica dai risultati 2026/27 nel dataset."""
    import json
    from pathlib import Path as _P

    standings_path = BASE / 'cache/standings_detailed.json'
    if not standings_path.exists():
        return

    data = json.loads(standings_path.read_text())
    table = data.get('api_table', [])
    if not table:
        return

    df = pd.read_csv(BASE / 'serie_a_dataset.csv')
    df_curr = df[df['season'] == '2026-27'].copy()
    if df_curr.empty:
        return

    # Ricostruisci classifica da zero dai risultati
    teams = set(df_curr['HomeTeam'].tolist() + df_curr['AwayTeam'].tolist())
    standings = {t: {'team': t, 'position': 0, 'playedGames': 0, 'won': 0,
                     'draw': 0, 'lost': 0, 'points': 0,
                     'goalsFor': 0, 'goalsAgainst': 0, 'goalDifference': 0}
                 for t in teams}

    for _, r in df_curr.iterrows():
        h, a = r['HomeTeam'], r['AwayTeam']
        hg, ag = int(r['FTHG']), int(r['FTAG'])
        for team, gf, ga in [(h, hg, ag), (a, ag, hg)]:
            if team not in standings:
                continue
            standings[team]['playedGames'] += 1
            standings[team]['goalsFor'] += gf
            standings[team]['goalsAgainst'] += ga
            standings[team]['goalDifference'] += gf - ga
            if gf > ga:
                standings[team]['won'] += 1
                standings[team]['points'] += 3
            elif gf == ga:
                standings[team]['draw'] += 1
                standings[team]['points'] += 1
            else:
                standings[team]['lost'] += 1

    sorted_teams = sorted(standings.values(),
                          key=lambda x: (-x['points'], -x['goalDifference'], -x['goalsFor']))
    for i, t in enumerate(sorted_teams):
        t['position'] = i + 1
        # Mantieni formato originale con team come dict se era dict
        for row in table:
            team_name = row['team'].get('name','') if isinstance(row['team'], dict) else row['team']
            if team_name == t['team']:
                row.update({k: v for k, v in t.items() if k != 'team'})
                break

    data['api_table'] = table
    standings_path.write_text(json.dumps(data, indent=2))
    print(f"  Classifica aggiornata: {len(sorted_teams)} squadre")


if __name__ == '__main__':
    import os
    os.chdir(BASE)
    update_dataset_2627()
    update_standings_from_results()
    print("\nOra rigenera il modello: rm -f model_cache.pkl && python3 main.py")
