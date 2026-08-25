"""
ELO dinamico calcolato dai risultati Serie A
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

CACHE_DIR = Path("cache")
ELO_START = 1500.0
K_FACTOR  = 32.0
HOME_ADV  = 100.0


def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def update_elo(elo_a, elo_b, score_a, k=K_FACTOR):
    exp_a = expected_score(elo_a, elo_b)
    new_a = elo_a + k * (score_a - exp_a)
    new_b = elo_b + k * ((1 - score_a) - (1 - exp_a))
    return new_a, new_b


def compute_dynamic_elo(df):
    df = df.sort_values("Date").copy()
    elo_ratings = {}
    elo_home_list = []
    elo_away_list = []

    for _, row in df.iterrows():
        home = row.get("HomeTeam")
        away = row.get("AwayTeam")
        ftr  = str(row.get("FTR", "") or "")

        if not home or not away:
            elo_home_list.append(ELO_START)
            elo_away_list.append(ELO_START)
            continue

        elo_h = elo_ratings.get(home, ELO_START)
        elo_a = elo_ratings.get(away, ELO_START)
        elo_home_list.append(elo_h)
        elo_away_list.append(elo_a)

        if ftr == "H":
            score = 1.0
        elif ftr == "D":
            score = 0.5
        elif ftr == "A":
            score = 0.0
        else:
            continue

        new_h, new_a = update_elo(elo_h + HOME_ADV/2, elo_a - HOME_ADV/2, score)
        elo_ratings[home] = new_h
        elo_ratings[away] = new_a

    df["elo_home_dyn"] = elo_home_list
    df["elo_away_dyn"] = elo_away_list
    df["elo_diff_dyn"] = df["elo_home_dyn"] - df["elo_away_dyn"]
    return df


def get_current_elo(df):
    df_elo = compute_dynamic_elo(df)
    current = {}
    for _, row in df_elo.sort_values("Date").iterrows():
        current[row["HomeTeam"]] = row["elo_home_dyn"]
        current[row["AwayTeam"]] = row["elo_away_dyn"]
    return current


def save_elo_snapshot(elo_dict):
    path = CACHE_DIR / "elo_dynamic_current.json"
    with open(path, "w") as f:
        json.dump(elo_dict, f, indent=2)
    print("  ELO dinamico salvato: " + str(len(elo_dict)) + " squadre")


def load_elo_snapshot():
    path = CACHE_DIR / "elo_dynamic_current.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    df = pd.read_csv("serie_a_dataset.csv", parse_dates=["Date"])
    sa = df[df["league"] == "Serie A"] if "league" in df.columns else df

    print("Calcolo ELO dinamico Serie A...")
    elo = get_current_elo(sa)

    print("\nTop 10 squadre per ELO:")
    for team, rating in sorted(elo.items(), key=lambda x: -x[1])[:10]:
        print("  " + str(team) + ": " + str(round(rating, 0)))

    save_elo_snapshot(elo)
