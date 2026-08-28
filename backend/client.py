"""
Client per il backend FastAPI.
La dashboard importa questo invece di chiamare i moduli direttamente.
Fallback automatico al modello locale se il backend non e' disponibile.
"""
import requests
from typing import Optional

BACKEND_URL = "http://localhost:8000"
TIMEOUT = 10

def _get(endpoint: str, params: dict = None) -> dict:
    try:
        resp = requests.get(f"{BACKEND_URL}{endpoint}", params=params, timeout=TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None

def _post(endpoint: str, data: dict) -> dict:
    try:
        resp = requests.post(f"{BACKEND_URL}{endpoint}", json=data, timeout=TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None

def is_backend_available() -> bool:
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return resp.status_code == 200
    except:
        return False

def predict(home: str, away: str, match_date: str = None,
            match_time: str = None, odds_h: float = None,
            odds_d: float = None, odds_a: float = None) -> dict:
    """Previsione per una partita."""
    params = {"home": home, "away": away}
    if match_date: params["match_date"] = match_date
    if match_time: params["match_time"] = match_time
    if odds_h: params["odds_h"] = odds_h
    if odds_d: params["odds_d"] = odds_d
    if odds_a: params["odds_a"] = odds_a
    result = _get("/predict", params)
    return result.get("predictions", {}) if result else {}

def value_bets(home: str, away: str, odds: dict,
               min_edge: float = 0.07, bankroll: float = 300.0,
               match_date: str = None) -> list:
    """Value bet per una partita con le quote inserite."""
    result = _post("/value_bets", {
        "home": home, "away": away, "odds": odds,
        "min_edge": min_edge, "bankroll": bankroll,
        "match_date": match_date,
    })
    return result.get("value_bets", []) if result else []

def get_calendar(giornata: int = None) -> list:
    """Calendario con previsioni e orari italiani."""
    result = _get("/calendar", {"giornata": giornata} if giornata else None)
    return result.get("matches", []) if result else []

def get_standings() -> dict:
    """Classifica con stats casa/trasferta."""
    return _get("/standings") or {}

def get_injuries(team: str) -> list:
    """Infortuni attivi per una squadra."""
    result = _get(f"/update/injuries/{team}")
    return result.get("injuries", []) if result else []

def get_all_injuries() -> dict:
    """Tutti gli infortuni attivi."""
    result = _get("/injuries")
    return result.get("injuries", {}) if result else {}

def get_lineups(match_date: str, home: str, away: str) -> dict:
    """Formazioni per una partita."""
    return _get(f"/update/lineups/{match_date}/{home}/{away}") or {}

def post_injury(team: str, player: str, status: str,
                return_date: str = None, source: str = "manual") -> dict:
    """Aggiorna stato infortunio."""
    return _post("/update/injuries", {
        "team": team, "player": player, "status": status,
        "return_date": return_date, "source": source,
    })

def post_lineups(home: str, away: str, match_date: str,
                 home_lineup: list, away_lineup: list, source: str = "scraper") -> dict:
    """Invia formazioni titolari."""
    return _post("/update/lineups", {
        "home": home, "away": away, "match_date": match_date,
        "home_lineup": home_lineup, "away_lineup": away_lineup,
        "source": source,
    })

def post_stats(home: str, away: str, match_date: str,
               data: dict, match_id: str = None, source: str = "opta") -> dict:
    """Invia statistiche Opta."""
    return _post("/update/stats", {
        "home": home, "away": away, "match_date": match_date,
        "data": data, "match_id": match_id,
    })
