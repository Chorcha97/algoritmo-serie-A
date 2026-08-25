"""
Tracker quote giornaliero
Salva le quote correnti per tutte le partite in programma.
Va lanciato ogni giorno — costruisce automaticamente lo storico
delle quote di apertura vs chiusura.
"""

import requests
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

CACHE_DIR = Path("cache/odds_history")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = "39aeb7b663b54fe3d5d42fc6d7accde3"

SPORTS = {
    "serie_a":   "soccer_italy_serie_a",
    "mondiale":  "soccer_fifa_world_cup",
    "premier":   "soccer_epl",
    "bundesliga":"soccer_germany_bundesliga",
    "la_liga":   "soccer_spain_la_liga",
    "ligue_1":   "soccer_france_ligue_one",
}

MARKETS = "h2h,totals"


def fetch_odds(sport_key: str) -> list:
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey":      API_KEY,
        "regions":     "eu",
        "markets":     MARKETS,
        "oddsFormat":  "decimal",
        "dateFormat":  "iso",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        remaining = resp.headers.get("x-requests-remaining", "?")
        print(f"  [{sport_key}] status: {resp.status_code} | richieste rimaste: {remaining}")
        if resp.status_code == 200:
            return resp.json()
        return []
    except Exception as e:
        print(f"  [{sport_key}] errore: {e}")
        return []


def parse_match_odds(event: dict, sport: str) -> dict:
    """Estrae le quote migliori per ogni mercato da un evento."""
    result = {
        "id":         event.get("id"),
        "sport":      sport,
        "home":       event.get("home_team"),
        "away":       event.get("away_team"),
        "kickoff":    event.get("commence_time"),
        "timestamp":  datetime.utcnow().isoformat(),
        "H": None, "D": None, "A": None,
        "over25": None, "under25": None,
    }

    best = {"H": 0, "D": 0, "A": 0, "over25": 0, "under25": 0}

    for bk in event.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt["key"] == "h2h":
                for o in mkt.get("outcomes", []):
                    if o["name"] == event["home_team"]:
                        best["H"] = max(best["H"], o["price"])
                    elif o["name"] == event["away_team"]:
                        best["A"] = max(best["A"], o["price"])
                    elif o["name"] == "Draw":
                        best["D"] = max(best["D"], o["price"])
            elif mkt["key"] == "totals":
                for o in mkt.get("outcomes", []):
                    pt = o.get("point", 0)
                    if pt == 2.5:
                        if o["name"] == "Over":
                            best["over25"] = max(best["over25"], o["price"])
                        elif o["name"] == "Under":
                            best["under25"] = max(best["under25"], o["price"])

    for k in best:
        result[k] = best[k] if best[k] > 0 else None
    return result


def save_snapshot(records: list, sport: str):
    """Salva lo snapshot giornaliero."""
    if not records:
        return
    today = datetime.utcnow().strftime("%Y-%m-%d")
    path = CACHE_DIR / f"{sport}_{today}.json"
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  Salvati {len(records)} eventi in {path.name}")


def load_opening_odds(match_id: str) -> dict:
    """
    Carica la prima quotazione salvata per una partita.
    Questa è la 'quota di apertura' che abbiamo tracciato.
    """
    files = sorted(CACHE_DIR.glob("*.json"))
    for fpath in files:
        try:
            with open(fpath) as f:
                data = json.load(f)
            for record in data:
                if record.get("id") == match_id:
                    return record
        except:
            continue
    return {}


def compute_drift(opening: dict, current: dict) -> dict:
    """
    Calcola il drift (movimento) delle quote tra apertura e ora.
    Negativo = quota scesa (soldi entrati su quell'esito).
    """
    drift = {}
    for market in ["H", "D", "A", "over25", "under25"]:
        o = opening.get(market)
        c = current.get(market)
        if o and c and o > 0:
            drift[f"drift_{market}"] = round(c - o, 3)
            drift[f"drift_{market}_pct"] = round((c - o) / o * 100, 1)
        else:
            drift[f"drift_{market}"] = None
            drift[f"drift_{market}_pct"] = None
    return drift


def get_current_odds_with_drift(sport: str = "serie_a") -> pd.DataFrame:
    """
    Restituisce le quote attuali con il drift rispetto alla prima
    quotazione salvata. Pronto per essere usato nella dashboard.
    """
    sport_key = SPORTS.get(sport)
    if not sport_key:
        return pd.DataFrame()

    events = fetch_odds(sport_key)
    if not events:
        return pd.DataFrame()

    rows = []
    for event in events:
        current = parse_match_odds(event, sport)
        opening = load_opening_odds(event.get("id", ""))
        drift   = compute_drift(opening, current) if opening else {}
        rows.append({**current, **drift})

    return pd.DataFrame(rows)


def run_daily_snapshot():
    """
    Lancia ogni giorno per salvare le quote correnti.
    Configura come cron job o lancio manuale.
    """
    print(f"\n=== ODDS TRACKER — {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC ===")
    total = 0
    for sport, key in SPORTS.items():
        print(f"\n[{sport}]")
        events = fetch_odds(key)
        if events:
            records = [parse_match_odds(e, sport) for e in events]
            save_snapshot(records, sport)
            total += len(records)
        else:
            print(f"  Nessuna partita in programma")

    print(f"\nTotale: {total} partite salvate")
    print("Richieste API rimanenti: vedi output sopra")


def get_serie_a_odds() -> dict:
    """
    Scarica le quote Pinnacle attuali per tutte le partite Serie A.
    Include 1X2 e Over/Under 2.5.
    """
    # Scarica sia h2h che totals
    events_h2h    = fetch_odds("soccer_italy_serie_a")
    
    # Fetch separato per totals
    import requests
    params_totals = {
        "apiKey": API_KEY, "regions": "eu",
        "markets": "totals", "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    try:
        resp = requests.get(
            "https://api.the-odds-api.com/v4/sports/soccer_italy_serie_a/odds/",
            params=params_totals, timeout=15)
        events_totals = resp.json() if resp.status_code == 200 else []
    except:
        events_totals = []
    
    # Costruisci dizionario totals
    totals_by_id = {}
    for event in events_totals:
        eid = event.get("id")
        for bk in event.get("bookmakers", []):
            if bk["key"] == "pinnacle":
                for mkt in bk.get("markets", []):
                    if mkt["key"] == "totals":
                        for o in mkt.get("outcomes", []):
                            pt = o.get("point", 0)
                            if pt == 2.5:
                                if eid not in totals_by_id:
                                    totals_by_id[eid] = {}
                                if o["name"] == "Over":
                                    totals_by_id[eid]["over25"] = o["price"]
                                elif o["name"] == "Under":
                                    totals_by_id[eid]["under25"] = o["price"]

    result = {}
    for event in events_h2h:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        key  = f"{home} vs {away}"
        eid  = event.get("id")
        for bk in event.get("bookmakers", []):
            if bk["key"] == "pinnacle":
                for mkt in bk.get("markets", []):
                    if mkt["key"] == "h2h":
                        odds = {}
                        for o in mkt.get("outcomes", []):
                            if o["name"] == home:
                                odds["H"] = o["price"]
                            elif o["name"] == away:
                                odds["A"] = o["price"]
                            else:
                                odds["D"] = o["price"]
                        # Aggiungi over/under se disponibili
                        if eid in totals_by_id:
                            odds.update(totals_by_id[eid])
                        if odds:
                            result[key] = odds
    return result


# Mappa nomi API -> nomi dataset
API_NAME_MAP = {
    "Inter Milan":  "Inter",
    "AC Milan":     "Milan",
    "Atalanta BC":  "Atalanta",
    "AS Roma":      "Roma",
    "SSC Napoli":   "Napoli",
    "Hellas Verona":"Verona",
    "Udinese":      "Udinese",
    "Frosinone":    "Frosinone",
    "Venezia":      "Venezia",
    "AC Monza":     "Monza",
}

def normalize_team_name(name: str) -> str:
    """Normalizza il nome squadra dall API al formato del dataset."""
    return API_NAME_MAP.get(name, name)


def get_odds_for_match(home: str, away: str) -> dict:
    """
    Scarica le quote Pinnacle per una partita specifica.
    Gestisce le differenze nei nomi delle squadre.
    """
    all_odds = get_serie_a_odds()

    for match_key, odds in all_odds.items():
        h_api, a_api = match_key.split(" vs ")
        h_norm = normalize_team_name(h_api)
        a_norm = normalize_team_name(a_api)

        # Match esatto o parziale
        h_match = (home.lower() == h_norm.lower() or
                   home.lower() in h_norm.lower() or
                   h_norm.lower() in home.lower())
        a_match = (away.lower() == a_norm.lower() or
                   away.lower() in a_norm.lower() or
                   a_norm.lower() in away.lower())

        # Prova anche match invertito (casa/ospite scambiati)
        h_match_inv = (away.lower() == h_norm.lower() or
                       away.lower() in h_norm.lower() or
                       h_norm.lower() in away.lower())
        a_match_inv = (home.lower() == a_norm.lower() or
                       home.lower() in a_norm.lower() or
                       a_norm.lower() in home.lower())

        if h_match and a_match:
            return odds
        if h_match_inv and a_match_inv:
            # Match invertito - scambia H e A
            return {"H": odds.get("A"), "D": odds.get("D"), "A": odds.get("H")}

    return {}


if __name__ == "__main__":
    run_daily_snapshot()
