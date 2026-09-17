"""
Aggiunge al weekly update il re-download del calendario da Sofascore
con date e orari ufficiali aggiornati.
Da aggiungere a utils/weekly_update.py
"""
import requests, json, pandas as pd, datetime
from pathlib import Path

BASE = Path('/Users/davideforcato/Desktop/football_predictor')

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
    'Calcio Como 1907': 'Como', 'US Sassuolo Calcio': 'Sassuolo',
}

SEASON_ID = 95836  # 2026/27

def update_calendario():
    """Aggiorna il calendario CSV con date/orari ufficiali da Sofascore."""
    print("[calendario] Aggiornamento orari da Sofascore...")
    
    cal_path = BASE / 'cache/calendario_2627.csv'
    if not cal_path.exists():
        print("  [WARN] calendario_2627.csv non trovato")
        return
    
    df = pd.read_csv(cal_path)
    updated = 0
    
    for round_num in range(1, 39):
        try:
            resp = requests.get(
                'http://localhost:8000/sofascore/serie-a/results',
                params={'round': round_num, 'season_id': SEASON_ID},
                timeout=10
            )
            if resp.status_code != 200:
                continue
            
            events = resp.json().get('events', [])
            if not events:
                break
            
            for e in events:
                home_raw = e.get('homeTeam', {}).get('name', '')
                away_raw = e.get('awayTeam', {}).get('name', '')
                home = NAME_MAP.get(home_raw, home_raw)
                away = NAME_MAP.get(away_raw, away_raw)
                
                ts = e.get('startTimestamp', 0)
                if not ts:
                    continue
                
                # Converti UTC -> UTC+2 (ora italiana)
                dt_it = datetime.datetime.utcfromtimestamp(ts)
                data_str = dt_it.strftime('%Y-%m-%d')
                ora_str = dt_it.strftime('%H:%M')
                
                mask = (df['home'] == home) & (df['away'] == away) & (df['giornata'] == round_num)
                if mask.any():
                    df.loc[mask, 'data'] = data_str
                    df.loc[mask, 'ora'] = ora_str
                    updated += 1
            
            import time
            time.sleep(0.3)
            
        except Exception as ex:
            print(f"  [WARN] Giornata {round_num}: {ex}")
            break
    
    df.to_csv(cal_path, index=False)
    print(f"  OK {updated} partite aggiornate")

if __name__ == '__main__':
    update_calendario()

# ── Integra in weekly_update.py ──────────────────────────────────────────────
wu_path = BASE / 'utils/weekly_update.py'
content = wu_path.read_text()

calendario_call = '''
    # [4/4] Aggiorna calendario con orari ufficiali Sofascore
    print("\\n[4/4] Calendario orari...")
    try:
        from update_calendario_sofascore import update_calendario
        update_calendario()
    except Exception as e:
        print(f"  [WARN] Calendario: {e}")
'''

# Aggiungi alla fine del main se non già presente
if 'update_calendario' not in content:
    # Trova la fine della funzione main
    if 'if __name__' in content:
        content = content.replace('if __name__', calendario_call + '\nif __name__')
    else:
        content += calendario_call
    wu_path.write_text(content)
    print('OK update_calendario aggiunto a weekly_update.py')

# Copia lo script nella cartella utils
import shutil
shutil.copy(__file__, BASE / 'utils/update_calendario_sofascore.py')
print('OK script copiato in utils/')
