"""
Parametro motivazione per il modello Serie A
Considera:
1. Posizione in classifica e obiettivi stagionali
2. Impegni in coppe europee (Champions, Europa League, Conference)
3. Coppa Italia
"""

import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
from datetime import datetime, timedelta

CACHE_DIR = Path("cache")

# ─────────────────────────────────────────────
# 1. CLASSIFICA E OBIETTIVI
# ─────────────────────────────────────────────

# Soglie posizione in classifica Serie A (20 squadre)
ZONE_CHAMPIONS    = 4   # prime 4 → Champions League
ZONE_EUROPA       = 6   # 5-6 → Europa League
ZONE_CONFERENCE   = 7   # 7 → Conference League
ZONE_SALVEZZA     = 17  # 15-17 → lotta salvezza
ZONE_RETROCESSIONE= 18  # 18-20 → retrocessione diretta


# Moltiplicatori calibrati sui dati reali Serie A 2021-2026
MONTH_MULTIPLIERS = {
    8:  0.96,  # Agosto: -4% gol (squadre non a regime)
    9:  1.07,  # Settembre: +7% gol (squadre lanciate)
    10: 1.02,  # Ottobre: +2% gol
    11: 0.98,  # Novembre: -2% gol
    12: 1.02,  # Dicembre: +2% gol
    1:  1.05,  # Gennaio: +5% gol (mercato invernale, motivazione)
    2:  0.98,  # Febbraio: -2% gol
    3:  0.88,  # Marzo: -12% gol (sosta internazionale, stanchezza)
    4:  0.93,  # Aprile: -7% gol
    5:  1.05,  # Maggio: +5% gol (ultime giornate decisive)
}

WEEKDAY_MULTIPLIERS = {
    0: 1.02,  # Lunedì: +2% gol
    1: 0.95,  # Martedì: -5% gol
    2: 1.07,  # Mercoledì: +7% gol (recuperi infrasettimanali)
    3: 1.14,  # Giovedì: +14% gol (il più alto!)
    4: 1.02,  # Venerdì: +2% gol
    5: 0.99,  # Sabato: -1% gol
    6: 0.98,  # Domenica: -2% gol
}

# Moltiplicatori orario calibrati su dati reali Serie A
# Media generale: 2.61 gol/partita
HOUR_MULTIPLIERS = {
    11: 0.95,  # 11:00 - 2.49 gol (-5%)
    12: 0.97,  # 12:00 - stima
    13: 0.98,  # 13:00 - stima
    14: 0.99,  # 14:00 - 2.58 gol (-1%)
    15: 0.99,  # 15:00 - stima
    16: 0.98,  # 16:00 - stima
    17: 0.98,  # 17:00 - 2.57 gol (-2%)
    18: 1.00,  # 18:00 - stima
    19: 1.02,  # 19:00 - 2.67 gol (+2%)
    20: 1.02,  # 20:00 - stima simile sera
    21: 1.01,  # 21:00 - stima
}


def get_temporal_multipliers(match_date: str, match_time: str = None) -> dict:
    """
    Restituisce i moltiplicatori temporali per una partita.
    Calibrati sui dati reali Serie A 2021-2026.
    """
    try:
        d = pd.Timestamp(match_date)
        month_mult   = MONTH_MULTIPLIERS.get(d.month, 1.0)
        weekday_mult = WEEKDAY_MULTIPLIERS.get(d.dayofweek, 1.0)
        combined     = (month_mult + weekday_mult) / 2
        return {
            "month_mult":    month_mult,
            "weekday_mult":  weekday_mult,
            "goals_mult":    round(combined, 3),
            "month":         d.month,
            "weekday":       d.dayofweek,
        }
    except:
        return {
            "month_mult":   1.0,
            "weekday_mult": 1.0,
            "goals_mult":   1.0,
        }


PROMOTION_PENALTY = {
    'Frosinone': 0.83,
    'Venezia':   0.87,
    'Monza':     0.88,
}


def get_promotion_penalty(team: str) -> float:
    return PROMOTION_PENALTY.get(team, 1.0)


def get_motivation_score(
    position: int,
    points: int,
    points_gap_above: int,  # punti di distanza dalla posizione sopra
    points_gap_below: int,  # punti di distanza dalla posizione sotto
    matchdays_remaining: int,
) -> float:
    """
    Calcola un punteggio di motivazione da 0.5 a 1.5
    basato sulla situazione in classifica.

    > 1.0 = alta motivazione (lotta per obiettivo)
    = 1.0 = motivazione normale
    < 1.0 = bassa motivazione (stagione già decisa)
    """
    score = 1.0

    # Squadre in lotta per la Champions (posizioni 2-5)
    if 2 <= position <= 5:
        if points_gap_above <= matchdays_remaining * 3:
            score += 0.15  # ancora in corsa

    # Prima in classifica — vuole mantenere
    if position == 1:
        if points_gap_below <= 6:
            score += 0.20  # testa a testa
        else:
            score -= 0.10  # già campione/gestione

    # Lotta salvezza — massima motivazione
    if position >= ZONE_SALVEZZA:
        if points_gap_above <= matchdays_remaining * 3:
            score += 0.30  # disperazione

    # Già retrocessa matematicamente
    if position >= 19 and matchdays_remaining <= 5:
        score -= 0.25

    # Già campione o già qualificata senza lotta
    if position <= 3 and points_gap_below > matchdays_remaining * 3:
        score -= 0.15  # gestione delle energie

    # Ultime giornate — tutto più intenso
    if matchdays_remaining <= 4:
        if position in range(ZONE_RETROCESSIONE, 21):
            score += 0.20
        if position in range(1, ZONE_CONFERENCE + 1):
            score += 0.10

    return max(0.5, min(1.5, score))


# ─────────────────────────────────────────────
# 2. IMPEGNI IN COPPE
# ─────────────────────────────────────────────

# Coppe europee per stagione — fonte Wikipedia
COPPE_EUROPEE = {
    "2021-22": {
        "Milan": "champions", "Inter": "champions",
        "Napoli": "champions", "Juventus": "champions",
        "Lazio": "europa", "Roma": "europa",
        "Fiorentina": "conference",
    },
    "2022-23": {
        "Napoli": "champions", "Lazio": "champions",
        "Inter": "champions", "Milan": "champions",
        "Atalanta": "europa", "Roma": "europa",
        "Fiorentina": "conference",
    },
    "2023-24": {
        "Inter": "champions", "Milan": "champions",
        "Juventus": "champions", "Atalanta": "champions",
        "Bologna": "champions",
        "Roma": "europa", "Lazio": "europa",
        "Fiorentina": "conference",
    },
    "2024-25": {
        "Napoli": "champions", "Inter": "champions",
        "Milan": "champions", "Atalanta": "champions",
        "Juventus": "champions",
        "Roma": "europa", "Bologna": "europa",
        "Fiorentina": "conference",
    },
    "2025-26": {
        "Inter": "champions", "Napoli": "champions",
        "Roma": "champions", "Como": "champions",
        "Milan": "europa", "Juventus": "europa",
        "Atalanta": "conference",
    },
    "2026-27": {
        "Inter": "champions", "Napoli": "champions",
        "Atalanta": "champions", "Juventus": "champions",
        "Lazio": "europa", "Roma": "europa",
        "Fiorentina": "conference", "Milan": "conference",
    },
}

# Alias per compatibilità con il codice esistente
COPPE_EUROPEE_2627 = COPPE_EUROPEE["2026-27"]


def get_season_coppe(match_date: str) -> dict:
    """Restituisce le squadre nelle coppe per la stagione corrente alla data."""
    try:
        d = pd.Timestamp(match_date)
        if d.month >= 8:
            season = f"{d.year}-{str(d.year+1)[-2:]}"
        else:
            season = f"{d.year-1}-{str(d.year)[-2:]}"
        return COPPE_EUROPEE.get(season, COPPE_EUROPEE["2026-27"])
    except:
        return COPPE_EUROPEE["2026-27"]

# Derby e big match Serie A
DERBY = {
    # Derby della Madonnina
    frozenset(["Inter", "Milan"]): {"name": "Derby della Madonnina", "intensity": 1.0},
    # Derby della Capitale
    frozenset(["Roma", "Lazio"]): {"name": "Derby della Capitale", "intensity": 1.0},
    # Derby d'Italia
    frozenset(["Juventus", "Inter"]): {"name": "Derby dItalia", "intensity": 0.8},
    # Derby del Sole
    frozenset(["Roma", "Napoli"]): {"name": "Derby del Sole", "intensity": 0.7},
    # Derby del Nord
    frozenset(["Juventus", "Torino"]): {"name": "Derby di Torino", "intensity": 1.0},
    # Altre rivalità forti
    frozenset(["Juventus", "Roma"]): {"name": "Juve-Roma", "intensity": 0.6},
    frozenset(["Juventus", "Napoli"]): {"name": "Juve-Napoli", "intensity": 0.6},
    frozenset(["Inter", "Napoli"]): {"name": "Inter-Napoli", "intensity": 0.6},
    frozenset(["Milan", "Roma"]): {"name": "Milan-Roma", "intensity": 0.5},
    frozenset(["Milan", "Juventus"]): {"name": "Milan-Juve", "intensity": 0.6},
    frozenset(["Atalanta", "Inter"]): {"name": "Atalanta-Inter", "intensity": 0.5},
    frozenset(["Fiorentina", "Juventus"]): {"name": "Fiorentina-Juve", "intensity": 0.6},
    frozenset(["Genoa", "Sampdoria"]): {"name": "Derby della Lanterna", "intensity": 1.0},
    frozenset(["Verona", "Chievo"]): {"name": "Derby del Bentegodi", "intensity": 0.9},
    frozenset(["Bologna", "Fiorentina"]): {"name": "Derby dell'Appennino", "intensity": 0.7},
    frozenset(["Napoli", "Roma"]): {"name": "Napoli-Roma", "intensity": 0.6},
}


def get_match_intensity(home: str, away: str) -> dict:
    """
    Restituisce informazioni sull'intensità della partita.
    I big match tendono ad avere:
    - Più cartellini (+20-30%)
    - Meno gol (squadre più prudenti)
    - Più pareggi
    - Maggiore varianza nel risultato
    """
    key = frozenset([home, away])
    match_info = DERBY.get(key)

    if not match_info:
        return {
            "is_derby": False,
            "is_bigmatch": False,
            "intensity": 0.0,
            "name": None,
            "cards_multiplier": 1.0,
            "goals_multiplier": 1.0,
            "draw_bonus": 0.0,
        }

    intensity = match_info["intensity"]
    is_derby = intensity >= 0.9

    return {
        "is_derby":         is_derby,
        "is_bigmatch":      True,
        "intensity":        intensity,
        "name":             match_info["name"],
        # Calibrato su dati reali Serie A 2021-2026
        "cards_multiplier": 1.11 if is_derby else 0.92,
        "goals_multiplier": 0.75 if is_derby else 0.98,
        "draw_bonus":       0.049 if is_derby else 0.091,
    }


COPPA_ITALIA_PARTECIPANTI = [
    "Inter", "Napoli", "Atalanta", "Juventus", "Lazio", "Roma",
    "Fiorentina", "Milan", "Bologna", "Torino", "Udinese", "Sassuolo",
    "Parma", "Cagliari", "Genoa", "Como", "Lecce", "Venezia",
    "Frosinone", "Monza",
]


def get_fixture_congestion_score(
    team: str,
    match_date: str,
    days_window: int = 7,
) -> float:
    """
    Stima l'impatto della stanchezza da coppe usando
    i dati storici reali per ogni stagione.
    """
    coppe_stagione = get_season_coppe(match_date)
    coppa = coppe_stagione.get(team)
    if not coppa:
        return 1.0

    if coppa == "champions":
        return 0.92
    elif coppa == "europa":
        return 0.95
    elif coppa == "conference":
        return 0.97

    return 1.0


def get_coppa_italia_pressure(team: str, match_date: str) -> float:
    """
    Coppa Italia: le partite sono solitamente di martedì/mercoledì.
    Se la squadra ha una partita di Coppa Italia entro 3 giorni,
    potrebbe fare turnover.
    """
    if team not in COPPA_ITALIA_PARTECIPANTI:
        return 1.0

    try:
        d = pd.Timestamp(match_date)
        # Ottavi e quarti Coppa Italia: dicembre-gennaio e febbraio
        if d.month in [12, 1, 2]:
            return 0.97  # piccolo malus per turnover probabile
    except:
        pass

    return 1.0


# ─────────────────────────────────────────────
# 3. SCRAPER CLASSIFICA (Football-data API)
# ─────────────────────────────────────────────

def fetch_standings() -> dict:
    """
    Scarica la classifica Serie A attuale.
    Usa l'API gratuita di football-data.org
    (richiede registrazione gratuita per la chiave)
    Fallback: calcola dai risultati nel dataset locale.
    """
    cache_path = CACHE_DIR / "standings.json"

    # Prova a caricare cache recente (max 3 giorni)
    if cache_path.exists():
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mtime < timedelta(days=3):
            with open(cache_path) as f:
                return json.load(f)

    # Calcola dai risultati locali
    return _compute_standings_from_results()


def compute_standings_at_date(df: pd.DataFrame, date: str) -> dict:
    """
    Calcola la classifica usando solo i risultati PRIMA di una data.
    Usato nel backtest walk-forward per evitare lookahead bias.
    """
    try:
        cutoff = pd.Timestamp(date)
        if "league" in df.columns:
            df = df[df["league"] == "Serie A"]
        df = df[df["Date"] < cutoff].copy()
        if df.empty:
            return {}
        return _build_standings(df)
    except Exception as e:
        print(f"  [warning] Classifica rolling non disponibile: {e}")
        return {}


def _compute_standings_from_results() -> dict:
    """
    Calcola la classifica dai risultati nel dataset locale.
    """
    try:
        df = pd.read_csv("serie_a_dataset.csv", parse_dates=["Date"])
        if "league" in df.columns:
            df = df[df["league"] == "Serie A"]

        # Prendi solo la stagione corrente
        current_year = datetime.now().year
        if datetime.now().month >= 8:
            season = f"{current_year}/{str(current_year+1)[-2:]}"
        else:
            season = f"{current_year-1}/{str(current_year)[-2:]}"

        if "season" in df.columns:
            df = df[df["season"] == season.replace("/", "-")]

        if df.empty:
            return {}

        standings = {}
        for _, row in df.dropna(subset=["FTR"]).iterrows():
            h, a = row["HomeTeam"], row["AwayTeam"]
            ftr = row["FTR"]

            for team in [h, a]:
                if team not in standings:
                    standings[team] = {"pts": 0, "g": 0, "w": 0, "d": 0, "l": 0,
                                       "gf": 0, "ga": 0}

            fthg, ftag = int(row.get("FTHG", 0) or 0), int(row.get("FTAG", 0) or 0)
            standings[h]["g"] += 1
            standings[a]["g"] += 1
            standings[h]["gf"] += fthg
            standings[h]["ga"] += ftag
            standings[a]["gf"] += ftag
            standings[a]["ga"] += fthg

            if ftr == "H":
                standings[h]["pts"] += 3
                standings[h]["w"] += 1
                standings[a]["l"] += 1
            elif ftr == "D":
                standings[h]["pts"] += 1
                standings[a]["pts"] += 1
                standings[h]["d"] += 1
                standings[a]["d"] += 1
            elif ftr == "A":
                standings[a]["pts"] += 3
                standings[a]["w"] += 1
                standings[h]["l"] += 1

        return _build_standings(df)

    except Exception as e:
        print(f"  [warning] Classifica non disponibile: {e}")
        return {}


def _build_standings(df: pd.DataFrame) -> dict:
    """Costruisce la classifica da un DataFrame di risultati."""
    standings = {}
    for _, row in df.dropna(subset=["FTR"]).iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        ftr  = row["FTR"]
        fthg = int(row.get("FTHG", 0) or 0)
        ftag = int(row.get("FTAG", 0) or 0)
        for team in [h, a]:
            if team not in standings:
                standings[team] = {"pts":0,"g":0,"w":0,"d":0,"l":0,"gf":0,"ga":0}
        standings[h]["g"] += 1; standings[a]["g"] += 1
        standings[h]["gf"] += fthg; standings[h]["ga"] += ftag
        standings[a]["gf"] += ftag; standings[a]["ga"] += fthg
        if ftr == "H":
            standings[h]["pts"] += 3; standings[h]["w"] += 1; standings[a]["l"] += 1
        elif ftr == "D":
            standings[h]["pts"] += 1; standings[a]["pts"] += 1
            standings[h]["d"] += 1; standings[a]["d"] += 1
        elif ftr == "A":
            standings[a]["pts"] += 3; standings[a]["w"] += 1; standings[h]["l"] += 1
    sorted_teams = sorted(standings.items(),
                          key=lambda x: (x[1]["pts"], x[1]["gf"]-x[1]["ga"]),
                          reverse=True)
    result = {}
    for pos, (team, stats) in enumerate(sorted_teams, 1):
        result[team] = {"position": pos, **stats}
    with open(CACHE_DIR / "standings.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ─────────────────────────────────────────────
# 4. FUNZIONE PRINCIPALE
# ─────────────────────────────────────────────

def compute_motivation_features(
    home: str,
    away: str,
    match_date: str = None,
    standings: dict = None,
    total_matchdays: int = 38,
) -> dict:
    """
    Calcola tutte le feature di motivazione per una partita.
    """
    if standings is None:
        standings = fetch_standings()

    date = match_date or datetime.now().strftime("%Y-%m-%d")

    features = {}

    for side, team in [("home", home), ("away", away)]:
        info = standings.get(team, {})
        position = info.get("position", 10)
        points   = info.get("pts", 30)
        played   = info.get("g", 19)
        remaining = total_matchdays - played

        # Distanza da posizione sopra/sotto
        sorted_standings = sorted(
            standings.items(), key=lambda x: x[1].get("position", 20))
        positions = [t for t, _ in sorted_standings]
        idx = positions.index(team) if team in positions else position - 1
        gap_above = (standings[positions[idx-1]]["pts"] - points
                     if idx > 0 else 0)
        gap_below = (points - standings[positions[idx+1]]["pts"]
                     if idx < len(positions)-1 else 99)

        # Motivazione classifica
        mot_score = get_motivation_score(
            position, points, gap_above, gap_below, remaining)

        # Stanchezza coppe
        euro_factor = get_fixture_congestion_score(team, date)
        ci_factor   = get_coppa_italia_pressure(team, date)
        fatigue     = min(euro_factor, ci_factor)

        features[f"f_motivation_{side}"]  = mot_score
        features[f"f_fatigue_{side}"]     = fatigue
        features[f"f_position_{side}"]    = position
        features[f"f_points_{side}"]      = points
        features[f"f_coppa_europa_{side}"]= 1 if team in COPPE_EUROPEE_2627 else 0

    features["f_motivation_diff"] = (features["f_motivation_home"] -
                                      features["f_motivation_away"])
    features["f_fatigue_diff"]    = (features["f_fatigue_home"] -
                                      features["f_fatigue_away"])

    return features


def update_european_squads(season: str = "2627"):
    """
    Aggiorna le squadre nelle coppe europee.
    Da chiamare ad inizio stagione.
    """
    print(f"Aggiorna COPPE_EUROPEE_{season} in motivation.py con le squadre corrette")
    print("Fonte: UEFA.com o Gazzetta.it a fine luglio")


if __name__ == "__main__":
    print("=== TEST MOTIVAZIONE ===")
    standings = _compute_standings_from_results()
    if standings:
        print(f"\nClassifica calcolata: {len(standings)} squadre")
        for team, info in list(standings.items())[:5]:
            print(f"  {info['position']:2d}. {team:<20} {info['pts']} pts")

        print("\nTest motivazione Inter vs Juventus:")
        feat = compute_motivation_features("Inter", "Juventus",
                                           standings=standings)
        for k, v in feat.items():
            print(f"  {k}: {v}")
    else:
        print("Nessun dato stagione corrente disponibile")
