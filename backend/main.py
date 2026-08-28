"""
Serie A Predictor — FastAPI Backend v2
Tutte le correzioni integrate:
- UTC+2 per orari italiani
- Nomi italiani giorni/mesi
- Integrazione infortuni nelle previsioni
- Quote Pinnacle automatiche
- Calibrazione per fascia, blend Pinnacle, derby, temporali
"""
import os, json, pickle
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from babel.dates import format_date

from backend.models.Input import OddsInput
from backend.models.Request import PredictRequest
from backend.models.Update import LineupUpdate, InjuryUpdate, StatsUpdate
from routers import theanalyst

BASE_DIR = Path(__file__).parent.parent
os.chdir(BASE_DIR)
import sys; sys.path.insert(0, str(BASE_DIR))

from models.markets import compute_all_markets, find_value_bets_extended
from contextlib import asynccontextmanager
from clientHelper import refresh_session
from curl_cffi.requests import AsyncSession

@asynccontextmanager
async def lifespan(app: FastAPI):
    # impersonate="chrome" imita l'handshake TLS reale di Chrome
    async with AsyncSession(impersonate="chrome") as client:
        app.state.http_client = client
        await refresh_session(client)
        yield


app = FastAPI(title="Serie A Predictor API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(theanalyst.router)

def utc_to_italy(ora_str: str) -> str:
    """Converte orario UTC in ora italiana (UTC+2)."""
    try:
        h, m = int(ora_str[:2]), int(ora_str[3:5])
        h_it = (h + 2) % 24
        return f"{h_it:02d}:{m:02d}"
    except:
        return ora_str

def data_italiana(date_str: str) -> str:
    """Formatta data in italiano: Sabato 28 agosto 2026"""
    try:
        ts = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
        # EEEE = giorno settimana (Sabato), d = giorno, MMMM = mese (agosto), yyyy = anno (2026)
        formatted = format_date(ts, format="EEEE d MMMM yyyy", locale='it_IT')
        return formatted.capitalize()
    except Exception:
        return str(date_str)

# ── Carica modello ────────────────────────────────────────────────────────────
model = None

def load_model():
    global model
    cache = BASE_DIR / "model_cache.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            model = pickle.load(f)
        print("[OK] Modello caricato")
    else:
        print("[WARN] model_cache.pkl non trovato - lancia python3 main.py")

load_model()

# ── Helper: carica infortuni ──────────────────────────────────────────────────
def get_team_injuries(team: str) -> list:
    path = BASE_DIR / "cache/injuries.json"
    if not path.exists():
        return []
    injuries = json.loads(path.read_text(encoding="utf-8"))
    return [i for i in injuries.get(team, []) if i.get("status") in ["out", "doubt"]]

def get_pinnacle_odds(home: str, away: str) -> dict:
    """Scarica quote Pinnacle automaticamente."""
    try:
        from data.odds_tracker import get_odds_for_match
        return get_odds_for_match(home, away)
    except:
        return {}

# ── Endpoints base ────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "version": "2.0.0",
        "endpoints": {
            "previsioni": ["GET /predict", "POST /predict", "POST /value_bets"],
            "dati": ["GET /calendar", "GET /standings", "GET /teams", "GET /arbitri"],
            "aggiornamenti": ["POST /update/lineups", "POST /update/injuries", "POST /update/stats"],
            "lettura": ["GET /update/injuries/{team}", "GET /update/lineups/{date}/{home}/{away}"],
        }
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
        "cache_files": [f.name for f in (BASE_DIR/"cache").glob("*.json") if f.is_file()]
    }

# ── Previsioni ────────────────────────────────────────────────────────────────
def _predict(home, away, match_date=None, match_time=None,
             odds_h=None, odds_d=None, odds_a=None):
    if model is None:
        raise HTTPException(503, "Modello non caricato")

    # Quote Pinnacle automatiche se non fornite
    if not odds_h:
        pinnacle = get_pinnacle_odds(home, away)
        odds_h = pinnacle.get("H")
        odds_d = pinnacle.get("D")
        odds_a = pinnacle.get("A")

    # Infortuni attivi
    infortuni_home = get_team_injuries(home)
    infortuni_away = get_team_injuries(away)

    kwargs = {}
    if match_date: kwargs["match_date"] = match_date
    if match_time:
        # Converti orario italiano in UTC per i moltiplicatori
        try:
            h, m = int(match_time[:2]), int(match_time[3:5])
            h_utc = (h - 2) % 24
            kwargs["match_time"] = f"{h_utc:02d}:{m:02d}"
        except:
            kwargs["match_time"] = match_time
    if odds_h: kwargs["odds_h"] = odds_h
    if odds_d: kwargs["odds_d"] = odds_d
    if odds_a: kwargs["odds_a"] = odds_a

    preds = compute_all_markets(model.poisson, home, away, **kwargs)
    if not preds:
        raise HTTPException(404, f"Squadre non trovate: {home} vs {away}")

    # Top risultati esatti
    import numpy as np
    mat = model.poisson.predict_score_matrix(home, away)
    if mat is not None:
        flat = [(mat[i,j]*100, i, j) for i in range(mat.shape[0]) for j in range(mat.shape[1])]
        top5 = sorted(flat, reverse=True)[:5]
        preds["exact_scores"] = [{"score": f"{i}-{j}", "prob": round(p,2)} for p,i,j in top5]

    return {
        "home": home,
        "away": away,
        "match_date": match_date,
        "match_time_italy": match_time,
        "pinnacle_odds": {"H": odds_h, "D": odds_d, "A": odds_a} if odds_h else None,
        "injuries": {
            "home": infortuni_home,
            "away": infortuni_away,
        },
        "predictions": preds,
    }

@app.get("/predict")
def predict_get(home: str, away: str,
                match_date: Optional[str]=None, match_time: Optional[str]=None,
                odds_h: Optional[float]=None, odds_d: Optional[float]=None,
                odds_a: Optional[float]=None):
    return _predict(home, away, match_date, match_time, odds_h, odds_d, odds_a)

@app.post("/predict")
def predict_post(req: PredictRequest):
    return _predict(req.home, req.away, req.match_date, req.match_time,
                    req.odds_h, req.odds_d, req.odds_a)

@app.post("/value_bets")
def value_bets(req: OddsInput):
    if model is None:
        raise HTTPException(503, "Modello non caricato")
    preds = compute_all_markets(model.poisson, req.home, req.away)
    if not preds:
        raise HTTPException(404, "Squadre non trovate")
    vbs = find_value_bets_extended(
        preds, req.odds,
        min_edge=req.min_edge,
        bankroll=req.bankroll,
        match_date=req.match_date,
    )
    return {
        "home": req.home, "away": req.away,
        "value_bets": vbs, "count": len(vbs),
    }

# ── Dati ──────────────────────────────────────────────────────────────────────
@app.get("/calendar")
def calendar(giornata: Optional[int]=None):
    cal_path = BASE_DIR / "cache/calendario_2627.csv"
    if not cal_path.exists():
        raise HTTPException(404, "Calendario non disponibile")
    import pandas as pd
    cal = pd.read_csv(cal_path)
    if giornata:
        cal = cal[cal["giornata"] == giornata]
    cal = cal.sort_values(["data","ora"])

    matches = []
    for _, row in cal.iterrows():
        ora_it = utc_to_italy(str(row.get("ora","")))
        data_it = data_italiana(str(row["data"])[:10])
        m = {
            "giornata": int(row["giornata"]),
            "data": str(row["data"])[:10],
            "data_italiana": data_it,
            "ora_utc": str(row.get("ora","")),
            "ora_italia": ora_it,
            "home": row["home"],
            "away": row["away"],
        }
        # Aggiungi previsioni
        if model:
            try:
                preds = compute_all_markets(
                    model.poisson, row["home"], row["away"],
                    match_date=str(row["data"])[:10])
                m["prob_H"] = round(preds.get("prob_H",0)*100, 1)
                m["prob_D"] = round(preds.get("prob_D",0)*100, 1)
                m["prob_A"] = round(preds.get("prob_A",0)*100, 1)
                m["prob_over25"] = round(preds.get("prob_over25",0)*100, 1)
                # Risultato piu probabile
                import numpy as np
                mat = model.poisson.predict_score_matrix(row["home"], row["away"])
                if mat is not None:
                    idx = np.unravel_index(mat.argmax(), mat.shape)
                    m["best_score"] = f"{idx[0]}-{idx[1]}"
                    m["best_score_prob"] = round(mat[idx]*100, 1)
            except:
                pass
        matches.append(m)

    return {"giornata": giornata, "matches": matches, "count": len(matches)}

@app.get("/standings")
def standings():
    path = BASE_DIR / "cache/standings_detailed.json"
    if not path.exists():
        raise HTTPException(404, "Classifica non disponibile")
    return json.loads(path.read_text(encoding="utf-8"))

@app.get("/teams")
def teams():
    if model is None:
        raise HTTPException(503, "Modello non caricato")
    return {"teams": sorted(model.poisson.attack.keys()), "count": len(model.poisson.attack)}

@app.get("/arbitri")
def arbitri():
    path = BASE_DIR / "cache/referee_stats_seria_a.json"
    if not path.exists():
        raise HTTPException(404, "Non disponibile")
    return json.loads(path.read_text(encoding="utf-8"))

@app.get("/injuries")
def all_injuries():
    path = BASE_DIR / "cache/injuries.json"
    if not path.exists():
        return {"injuries": {}}
    return {"injuries": json.loads(path.read_text(encoding="utf-8"))}

# ── Aggiornamenti ─────────────────────────────────────────────────────────────
@app.post("/update/lineups")
def update_lineups(data: LineupUpdate):
    path = BASE_DIR / "cache/lineups.json"
    lineups = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    key = f"{data.match_date}_{data.home}_vs_{data.away}"
    lineups[key] = {
        "home": data.home, "away": data.away, "match_date": data.match_date,
        "home_lineup": data.home_lineup, "away_lineup": data.away_lineup,
        "source": data.source, "updated_at": datetime.now().isoformat(),
    }
    path.write_text(json.dumps(lineups, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", "key": key,
            "players_home": len(data.home_lineup), "players_away": len(data.away_lineup)}

@app.post("/update/injuries")
def update_injuries(data: InjuryUpdate):
    path = BASE_DIR / "cache/injuries.json"
    injuries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if data.team not in injuries:
        injuries[data.team] = []
    injuries[data.team] = [i for i in injuries[data.team] if i["player"] != data.player]
    if data.status != "available":  # non salvare se disponibile
        injuries[data.team].append({
            "player": data.player, "status": data.status,
            "return_date": data.return_date, "source": data.source,
            "updated_at": datetime.now().isoformat(),
        })
    path.write_text(json.dumps(injuries, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", "team": data.team, "player": data.player, "injury_status": data.status}

@app.post("/update/stats")
def update_stats(data: StatsUpdate):
    """Endpoint principale per i dati Opta dalla tua amica."""
    path = BASE_DIR / "cache/opta_stats.json"
    stats = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    key = f"{data.match_date}_{data.home}_vs_{data.away}"
    stats[key] = {
        "home": data.home, "away": data.away, "match_date": data.match_date,
        "match_id": data.match_id, "data": data.data,
        "updated_at": datetime.now().isoformat(),
    }
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", "key": key, "fields_received": list(data.data.keys())}

# ── Lettura dati aggiornati ───────────────────────────────────────────────────
@app.get("/update/injuries/{team}")
def get_injuries(team: str):
    return {"team": team, "injuries": get_team_injuries(team)}

@app.get("/update/lineups/{match_date}/{home}/{away}")
def get_lineups(match_date: str, home: str, away: str):
    path = BASE_DIR / "cache/lineups.json"
    if not path.exists():
        raise HTTPException(404, "Nessuna formazione disponibile")
    lineups = json.loads(path.read_text(encoding="utf-8"))
    key = f"{match_date}_{home}_vs_{away}"
    if key not in lineups:
        raise HTTPException(404, f"Formazione non trovata per {key}")
    return lineups[key]

@app.get("/update/stats/{match_date}/{home}/{away}")
def get_stats(match_date: str, home: str, away: str):
    path = BASE_DIR / "cache/opta_stats.json"
    if not path.exists():
        raise HTTPException(404, "Nessun dato disponibile")
    stats = json.loads(path.read_text(encoding="utf-8"))
    key = f"{match_date}_{home}_vs_{away}"
    if key not in stats:
        raise HTTPException(404, f"Dati non trovati per {key}")
    return stats[key]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
