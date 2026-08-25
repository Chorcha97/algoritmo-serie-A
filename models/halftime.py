"""
Modulo mercati primo tempo Serie A
Usa il Poisson del tempo pieno scalato al 44% per il primo tempo.
Mercati: 1X2 HT, Over/Under 0.5 / 1.5 / 2.5 HT
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson

HT_SCALE = 0.44  # frazione gol nel primo tempo


def compute_ht_matrix(poisson_model, home: str, away: str,
                      max_goals: int = 6) -> np.ndarray:
    """
    Genera la matrice Poisson per il primo tempo
    scalando i parametri del tempo pieno al 44%.
    """
    if home not in poisson_model.attack:
        return None

    lam_h_ft = (poisson_model.attack[home] /
                poisson_model.defense[away] *
                poisson_model.avg_goals *
                np.exp(poisson_model.home_adv))
    lam_a_ft = (poisson_model.attack[away] /
                poisson_model.defense[home] *
                poisson_model.avg_goals)

    lam_h = lam_h_ft * HT_SCALE
    lam_a = lam_a_ft * HT_SCALE

    matrix = np.outer(
        [poisson.pmf(i, lam_h) for i in range(max_goals + 1)],
        [poisson.pmf(j, lam_a) for j in range(max_goals + 1)]
    )
    return matrix


def compute_ht_markets(poisson_model, home: str, away: str) -> dict:
    """
    Calcola tutte le probabilità per i mercati primo tempo.
    """
    mat = compute_ht_matrix(poisson_model, home, away)
    if mat is None:
        return {}

    # 1X2 primo tempo
    prob_h = float(np.tril(mat, -1).sum())
    prob_d = float(np.trace(mat))
    prob_a = float(np.triu(mat, 1).sum())

    # Doppia chance primo tempo
    prob_1x = prob_h + prob_d
    prob_x2 = prob_d + prob_a
    prob_12 = prob_h + prob_a

    # Over/Under primo tempo
    goals_matrix = np.array([
        [i + j for j in range(mat.shape[1])]
        for i in range(mat.shape[0])
    ])

    prob_over05  = float(np.sum(mat[goals_matrix > 0.5]))
    prob_under05 = 1 - prob_over05
    prob_over15  = float(np.sum(mat[goals_matrix > 1.5]))
    prob_under15 = 1 - prob_over15
    prob_over25  = float(np.sum(mat[goals_matrix > 2.5]))
    prob_under25 = 1 - prob_over25

    # GG primo tempo
    prob_gg_ht = float(np.sum(mat[1:, 1:]))
    prob_ng_ht = 1 - prob_gg_ht

    return {
        "ht_prob_H":       prob_h,
        "ht_prob_D":       prob_d,
        "ht_prob_A":       prob_a,
        "ht_prob_1X":      prob_1x,
        "ht_prob_X2":      prob_x2,
        "ht_prob_12":      prob_12,
        "ht_prob_over05":  prob_over05,
        "ht_prob_under05": prob_under05,
        "ht_prob_over15":  prob_over15,
        "ht_prob_under15": prob_under15,
        "ht_prob_over25":  prob_over25,
        "ht_prob_under25": prob_under25,
        "ht_prob_gg":      prob_gg_ht,
        "ht_prob_ng":      prob_ng_ht,
    }


def find_ht_value_bets(
    poisson_model,
    home: str,
    away: str,
    odds_dict: dict,
    min_edge: float = 0.07,
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.125,
) -> list:
    """
    Trova value bet sui mercati primo tempo.

    odds_dict esempio:
    {
        'ht_H': 2.80, 'ht_D': 2.10, 'ht_A': 3.20,
        'ht_over05': 1.65, 'ht_under05': 2.20,
        'ht_over15': 3.10, 'ht_under15': 1.35,
    }
    """
    preds = compute_ht_markets(poisson_model, home, away)
    if not preds:
        return []

    markets = [
        ("ht_H",       "ht_prob_H",       "HT 1 — Vittoria Casa"),
        ("ht_D",       "ht_prob_D",       "HT X — Pareggio"),
        ("ht_A",       "ht_prob_A",       "HT 2 — Vittoria Ospite"),
        ("ht_1X",      "ht_prob_1X",      "HT 1X — Doppia Chance Casa"),
        ("ht_X2",      "ht_prob_X2",      "HT X2 — Doppia Chance Ospite"),
        ("ht_12",      "ht_prob_12",      "HT 12 — No Pareggio"),
        ("ht_over05",  "ht_prob_over05",  "HT Over 0.5 Gol"),
        ("ht_under05", "ht_prob_under05", "HT Under 0.5 Gol"),
        ("ht_over15",  "ht_prob_over15",  "HT Over 1.5 Gol"),
        ("ht_under15", "ht_prob_under15", "HT Under 1.5 Gol"),
        ("ht_over25",  "ht_prob_over25",  "HT Over 2.5 Gol"),
        ("ht_under25", "ht_prob_under25", "HT Under 2.5 Gol"),
        ("ht_gg",      "ht_prob_gg",      "HT Goal/Goal"),
        ("ht_ng",      "ht_prob_ng",      "HT No Goal"),
    ]

    value_bets = []
    for mkt_key, prob_key, label in markets:
        odds = odds_dict.get(mkt_key)
        prob = preds.get(prob_key)
        if not odds or not prob or odds <= 1.0:
            continue
        edge = prob * odds - 1
        if edge < min_edge:
            continue
        b = odds - 1
        q = 1 - prob
        kelly = max(0, (b * prob - q) / b * kelly_fraction)
        stake = round(kelly * bankroll, 2)
        value_bets.append({
            "mercato":          label,
            "quota":            round(odds, 2),
            "prob_modello_%":   round(prob * 100, 1),
            "prob_implicita_%": round(100 / odds, 1),
            "edge_%":           round(edge * 100, 1),
            "stake_€":          stake,
        })

    return sorted(value_bets, key=lambda x: x["edge_%"], reverse=True)


def backtest_ht(df: pd.DataFrame, poisson_model,
                min_edge: float = 0.07) -> pd.DataFrame:
    """
    Backtest sui mercati primo tempo usando i dati HTR, HTHG, HTAG.
    Non abbiamo le quote storiche per il primo tempo quindi usiamo
    le quote eque del modello come riferimento — il backtest
    mostra la calibrazione del modello, non il P&L reale.
    """
    results = []
    required = ["HTR", "HTHG", "HTAG"]
    df_clean = df.dropna(subset=required).copy()
    print(f"  Partite con dati HT: {len(df_clean)}")

    calibration = {
        "H": {"correct": 0, "total": 0, "prob_sum": 0},
        "D": {"correct": 0, "total": 0, "prob_sum": 0},
        "A": {"correct": 0, "total": 0, "prob_sum": 0},
    }

    for _, row in df_clean.iterrows():
        home = row.get("HomeTeam")
        away = row.get("AwayTeam")
        if not home or not away:
            continue

        preds = compute_ht_markets(poisson_model, home, away)
        if not preds:
            continue

        htr   = row["HTR"]
        hthg  = int(row["HTHG"])
        htag  = int(row["HTAG"])
        total = hthg + htag

        actual = {
            "H": htr == "H",
            "D": htr == "D",
            "A": htr == "A",
            "over05": total > 0.5,
            "under05": total <= 0.5,
            "over15": total > 1.5,
            "under15": total <= 1.5,
        }

        for outcome, happened in actual.items():
            prob_key = f"ht_prob_{outcome}"
            prob = preds.get(prob_key, 0)
            if prob <= 0:
                continue
            fair_odds = 1 / prob

            if outcome in ["H", "D", "A"]:
                calibration[outcome]["total"] += 1
                calibration[outcome]["prob_sum"] += prob
                if happened:
                    calibration[outcome]["correct"] += 1

            results.append({
                "partita":      f"{home} vs {away}",
                "mercato":      f"HT {outcome}",
                "prob_%":       round(prob * 100, 1),
                "quota_equa":   round(fair_odds, 2),
                "esito":        "✓" if happened else "✗",
            })

    print("\n=== CALIBRAZIONE MODELLO PRIMO TEMPO ===")
    for outcome, stats in calibration.items():
        if stats["total"] > 0:
            actual_rate = stats["correct"] / stats["total"] * 100
            pred_rate   = stats["prob_sum"] / stats["total"] * 100
            print(f"  {outcome}: previsto {pred_rate:.1f}%  reale {actual_rate:.1f}%  "
                  f"diff {actual_rate-pred_rate:+.1f}pp")

    return pd.DataFrame(results)


if __name__ == "__main__":
    import sys, pickle
    sys.path.insert(0, "..")
    from pathlib import Path

    print("Carico modello...")
    with open("model_cache.pkl", "rb") as f:
        model = pickle.load(f)

    print("\nTest Inter vs Juventus — Mercati Primo Tempo:")
    ht = compute_ht_markets(model.poisson, "Inter", "Juventus")
    print(f"  1X2:  H={ht['ht_prob_H']:.3f}  D={ht['ht_prob_D']:.3f}  A={ht['ht_prob_A']:.3f}")
    print(f"  Over/Under 0.5: {ht['ht_prob_over05']:.3f} / {ht['ht_prob_under05']:.3f}")
    print(f"  Over/Under 1.5: {ht['ht_prob_over15']:.3f} / {ht['ht_prob_under15']:.3f}")
    print(f"  GG/NG: {ht['ht_prob_gg']:.3f} / {ht['ht_prob_ng']:.3f}")

    print("\nBacktest calibrazione...")
    df_raw = pd.read_csv("serie_a_dataset.csv", parse_dates=["Date"])
    sa = df_raw[df_raw["league"]=="Serie A"] if "league" in df_raw.columns else df_raw
    bt = backtest_ht(sa, model.poisson, min_edge=0.07)
