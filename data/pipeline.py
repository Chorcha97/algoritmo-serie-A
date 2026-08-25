"""
Pipeline di raccolta dati Serie A
Fonti: football-data.co.uk (storico + quote) + Understat (xG) + Club Elo (rating)
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import asyncio
import os
from io import StringIO
from datetime import datetime
from pathlib import Path

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# 1. FOOTBALL-DATA.CO.UK — storico + quote
# ─────────────────────────────────────────────

FDCO_SEASONS = {
    "2021-22": "2122", "2022-23": "2223",
    "2023-24": "2324", "2024-25": "2425",
    "2025-26": "2526",
}

FDCO_LEAGUES = {
    "serie_a":    ("I1",  "Serie A"),
}

# Colonne che ci interessano da football-data.co.uk
FDCO_COLS = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",          # gol e risultato finale
    "HTHG", "HTAG", "HTR",          # primo tempo
    "HS", "AS",                      # tiri
    "HST", "AST",                    # tiri in porta
    "HC", "AC",
    "HF", "AF",
    "Referee",
    
    "Time",
    "AHh",
    "B365AHH", "B365AHA",
    "PAHH", "PAHA",
    "MaxH", "MaxD", "MaxA",
    "AvgH", "AvgD", "AvgA",
    "Referee",                      # corner
    "HY", "AY", "HR", "AR",         # cartellini
    # Quote medie bookmaker (se disponibili)
    "B365H", "B365D", "B365A",       # Bet365
    "BWH", "BWD", "BWA",             # BetWin
    "IWH", "IWD", "IWA",             # Interwetten
    "PSH", "PSD", "PSA",             # Pinnacle (le più efficienti)
    "B365>2.5", "B365<2.5",          # over/under Bet365
    "P>2.5", "P<2.5",                # over/under Pinnacle
    "Avg>2.5", "Avg<2.5",            # over/under media
    "Referee",                       # arbitro
    "HF", "AF",                      # falli
]


def fetch_fdco(season_key: str, season_code: str, league_code: str = "I1", league_name: str = "Serie A") -> pd.DataFrame:
    cache_path = CACHE_DIR / f"fdco_{league_code}_{season_code}.csv"
    if cache_path.exists():
        print(f"  [cache] football-data {season_key}")
        return pd.read_csv(cache_path)

    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"
    print(f"  [download] football-data {season_key} ...")
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        # Tieni solo le colonne disponibili
        cols = [c for c in FDCO_COLS if c in df.columns]
        df = df[cols].copy()
        # Rinomina colonne con simboli speciali
        rename_map = {}
        for c in df.columns:
            if ">2.5" in c:
                rename_map[c] = c.replace(">2.5", "_over25")
            elif "<2.5" in c:
                rename_map[c] = c.replace("<2.5", "_under25")
        if rename_map:
            df = df.rename(columns=rename_map)
        rename_map = {c: c.replace(">2.5","_over25").replace("<2.5","_under25") for c in df.columns if "2.5" in c}
        if rename_map:
            df = df.rename(columns=rename_map)
        df = df.loc[:, ~df.columns.duplicated()]
        df["season"] = season_key
        df["league"] = league_name
        df.to_csv(cache_path, index=False)
        time.sleep(0.5)
        return df
    except Exception as e:
        print(f"  [errore] {url}: {e}")
        return pd.DataFrame()


def load_fdco_all() -> pd.DataFrame:
    frames = []
    for league_key, (league_code, league_name) in FDCO_LEAGUES.items():
        for season_key, season_code in FDCO_SEASONS.items():
            df = fetch_fdco(season_key, season_code, league_code, league_name)
            if not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    # Rinomina colonne con simboli speciali e rimuovi duplicati
    cleaned = []
    for f in frames:
        rename_map = {c: c.replace(">2.5","_over25").replace("<2.5","_under25") for c in f.columns if "2.5" in c}
        f = f.rename(columns=rename_map)
        f = f.loc[:, ~f.columns.duplicated()]
        cleaned.append(f)
    combined = pd.concat(cleaned, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"], dayfirst=True, errors="coerce")
    combined = combined.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    combined = combined.sort_values("Date").reset_index(drop=True)
    print(f"  football-data: {len(combined)} partite caricate")
    return combined


# ─────────────────────────────────────────────
# 2. UNDERSTAT — xG per partita (Serie A)
# ─────────────────────────────────────────────


import asyncio
from understatapi import UnderstatClient

UNDERSTAT_SEASONS = [str(y) for y in range(2018, 2025)]

def _fetch_understat_async(year: str) -> list:
    with UnderstatClient() as client:
        data = client.league(league="Serie_A").get_match_data(season=year)
    return data


def _parse_understat(data: list, year) -> pd.DataFrame:
    rows = []
    for match in data:
        if not match.get('isResult'):
            continue
        try:
            rows.append({
                'date_understat': pd.to_datetime(match.get('datetime', ''), errors='coerce'),
                'home_understat': match.get('h', {}).get('title', ''),
                'away_understat': match.get('a', {}).get('title', ''),
                'xg_home': float(match.get('xG', {}).get('h', 0) or 0),
                'xg_away': float(match.get('xG', {}).get('a', 0) or 0),
                'goals_home_us': int(match.get('goals', {}).get('h', 0) or 0),
                'goals_away_us': int(match.get('goals', {}).get('a', 0) or 0),
                'season_understat': f"{year}-{str(int(year)+1)[-2:]}",
            })
        except:
            continue
    return pd.DataFrame(rows)

def fetch_understat_season(year: int) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"understat_{year}.json"
    if cache_path.exists():
        print(f"  [cache] understat {year}/{int(year)+1}")
        with open(cache_path) as f:
            data = json.load(f)
        return _parse_understat(data, year)
    print(f"  [download] understat {year}/{int(year)+1} ...")
    try:
        data = _fetch_understat_async(str(year))
        with open(cache_path, "w") as f:
            json.dump(data, f)
        time.sleep(1)
        return _parse_understat(data, year)
    except Exception as e:
        print(f"  [errore] understat {year}: {e}")
        return pd.DataFrame()


def load_understat_all() -> pd.DataFrame:
    frames = []
    for year in UNDERSTAT_SEASONS:
        df = fetch_understat_season(year)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["date_understat"])
    print(f"  understat: {len(combined)} partite con xG caricate")
    return combined


# ─────────────────────────────────────────────
# 3. CLUB ELO — rating ELO squadre
# ─────────────────────────────────────────────

# Mappa nomi squadra football-data → Club Elo
TEAM_NAME_MAP = {
    "Inter": "Internazionale", "AC Milan": "Milan", "Juventus": "Juventus",
    "Napoli": "Napoli", "Roma": "Roma", "Lazio": "Lazio",
    "Atalanta": "Atalanta", "Fiorentina": "Fiorentina", "Bologna": "Bologna",
    "Torino": "Torino", "Udinese": "Udinese", "Sampdoria": "Sampdoria",
    "Genoa": "Genoa", "Sassuolo": "Sassuolo", "Cagliari": "Cagliari",
    "Verona": "Verona", "Spezia": "Spezia", "Venezia": "Venezia",
    "Salernitana": "Salernitana", "Empoli": "Empoli", "Cremonese": "Cremonese",
    "Lecce": "Lecce", "Monza": "Monza", "Frosinone": "Frosinone",
}

ELO_TEAMS = list(set(TEAM_NAME_MAP.values()))


def fetch_club_elo(team: str) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"elo_{team.replace(' ', '_')}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["date_elo"])

    url = f"http://api.clubelo.com/{team.replace(' ', '-')}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df = df.rename(columns={"From": "date_elo", "Elo": "elo", "Club": "team_elo"})
        df["date_elo"] = pd.to_datetime(df["date_elo"], errors="coerce")
        df = df[["date_elo", "team_elo", "elo"]].dropna()
        df.to_csv(cache_path, index=False)
        time.sleep(0.3)
        return df
    except Exception as e:
        print(f"  [warning] ELO non trovato per {team}: {e}")
        return pd.DataFrame()


def load_elo_all() -> pd.DataFrame:
    print("  [download] Club Elo ratings ...")
    frames = []
    for team in ELO_TEAMS:
        df = fetch_club_elo(team)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    print(f"  club elo: {combined['team_elo'].nunique()} squadre caricate")
    return combined


def get_elo_at_date(elo_df: pd.DataFrame, team_elo_name: str, date: pd.Timestamp) -> float:
    """Restituisce il rating ELO della squadra alla data della partita."""
    subset = elo_df[elo_df["team_elo"] == team_elo_name]
    past = subset[subset["date_elo"] <= date]
    if past.empty:
        return np.nan
    return past.iloc[-1]["elo"]


# ─────────────────────────────────────────────
# 4. FUZZY MATCH nomi squadra
# ─────────────────────────────────────────────

# Mappa nomi Understat → football-data.co.uk
UNDERSTAT_TO_FDCO = {
    "Inter": "Inter", "Juventus": "Juventus", "Milan": "AC Milan",
    "Napoli": "Napoli", "Roma": "Roma", "Lazio": "Lazio",
    "Atalanta": "Atalanta", "Fiorentina": "Fiorentina", "Bologna": "Bologna",
    "Torino": "Torino", "Udinese": "Udinese", "Sampdoria": "Sampdoria",
    "Genoa": "Genoa", "Sassuolo": "Sassuolo", "Cagliari": "Cagliari",
    "Verona": "Verona", "Spezia": "Spezia", "Venezia": "Venezia",
    "Salernitana": "Salernitana", "Empoli": "Empoli", "Cremonese": "Cremonese",
    "Lecce": "Lecce", "Monza": "Monza", "Frosinone": "Frosinone",
    "Hellas Verona": "Verona", "SPAL": "SPAL", "Brescia": "Brescia",
    "Parma": "Parma", "Benevento": "Benevento", "Crotone": "Crotone",
}


def normalize_understat(df_us: pd.DataFrame) -> pd.DataFrame:
    if df_us.empty or "home_understat" not in df_us.columns:
        return pd.DataFrame(columns=["home_understat","away_understat","home_norm","away_norm","date_understat","xg_home","xg_away"])
    df = df_us.copy()
    df["home_norm"] = df["home_understat"].map(UNDERSTAT_TO_FDCO).fillna(df["home_understat"])
    df["away_norm"] = df["away_understat"].map(UNDERSTAT_TO_FDCO).fillna(df["away_understat"])
    return df


# ─────────────────────────────────────────────
# 5. MERGE PRINCIPALE
# ─────────────────────────────────────────────

def merge_all(fdco: pd.DataFrame, understat: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    """Unisce le tre fonti in un unico DataFrame per partita."""
    print("\n[merge] Unione dataset ...")

    # Normalizza nomi understat
    us = normalize_understat(understat)

    # Merge FDCO + Understat (match per data ±2 giorni + squadre)
    merged_rows = []
    fdco_indexed = fdco.copy()

    for _, row in fdco_indexed.iterrows():
        date = row["Date"]
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        # Cerca corrispondenza in understat (finestra ±2 giorni)
        mask = (
            (us["date_understat"] >= date - pd.Timedelta(days=2)) &
            (us["date_understat"] <= date + pd.Timedelta(days=2)) &
            (us["home_norm"] == home) &
            (us["away_norm"] == away)
        )
        us_match = us[mask]

        base = row.to_dict()
        if not us_match.empty:
            base["xg_home"] = us_match.iloc[0]["xg_home"]
            base["xg_away"] = us_match.iloc[0]["xg_away"]
        else:
            base["xg_home"] = np.nan
            base["xg_away"] = np.nan

        # ELO alla data della partita
        home_elo_name = TEAM_NAME_MAP.get(home, home)
        away_elo_name = TEAM_NAME_MAP.get(away, away)
        if not elo.empty:
            base["elo_home"] = get_elo_at_date(elo, home_elo_name, date)
            base["elo_away"] = get_elo_at_date(elo, away_elo_name, date)
            base["elo_diff"] = base["elo_home"] - base["elo_away"]
        else:
            base["elo_home"] = base["elo_away"] = base["elo_diff"] = np.nan

        merged_rows.append(base)

    df = pd.DataFrame(merged_rows)
    xg_coverage = df["xg_home"].notna().mean() * 100
    elo_coverage = df["elo_home"].notna().mean() * 100
    print(f"  copertura xG: {xg_coverage:.1f}% delle partite")
    print(f"  copertura ELO: {elo_coverage:.1f}% delle partite")
    print(f"  totale partite nel dataset: {len(df)}")
    return df


# ─────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────

def build_dataset(save_path: str = "serie_a_dataset.csv") -> pd.DataFrame:
    print("=" * 55)
    print("  PIPELINE DATI SERIE A")
    print("=" * 55)

    print("\n[1/3] Football-data.co.uk ...")
    fdco = load_fdco_all()

    print("\n[2/3] Understat (xG) ...")
    understat = load_understat_all()

    print("\n[3/3] Club Elo (rating) ...")
    elo = load_elo_all()

    df = merge_all(fdco, understat, elo)

    df.to_csv(save_path, index=False)
    print(f"\n[✓] Dataset salvato in: {save_path}")
    print("=" * 55)
    return df


if __name__ == "__main__":
    df = build_dataset()
    print("\nAnteprima colonne disponibili:")
    print(df.dtypes.to_string())
    print(f"\nRange date: {df['Date'].min().date()} → {df['Date'].max().date()}")
