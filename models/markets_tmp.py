"""
Mercati estesi per il sistema predittivo Serie A
Fix v3:
- Kelly 1/8
- Home advantage correction (+0.097)
- Edge minimo 3%
- Warning vittoria casa
- Moltiplicatori temporali (mese, giorno, orario)
- Derby e big match calibrati su dati reali
- Matrice correlazione mercati
- Flag ultime giornate (-50% stake maggio)
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson

# Correzione home advantage calibrata su dati reali Serie A post-Covid
# Calibrazione per fascia - basata su analisi di 1900 partite Serie A 2021-2026
# Mappa prob. modello -> prob. reale osservata
HOME_WIN_CORRECTION = 0.0  # rimossa - usiamo calibrazione per fascia

# Calibrazione isotonica casa (raw Poisson -> reale)
# Basata su 1900 partite Serie A 2021-2026
CALIBRATION_HOME = [
    (0.20, 0.275),  # raw 20-30% -> reale 27.5%
    (0.30, 0.347),  # raw 30-40% -> reale 34.7%
    (0.40, 0.417),  # raw 40-50% -> reale 41.7%
    (0.50, 0.570),  # raw 50-60% -> reale 57.0%
    (0.60, 0.615),  # raw 60-70% -> reale 61.5%
]

# Calibrazione isotonica pareggio (raw Poisson -> reale)
CALIBRATION_DRAW = [
    (0.20, 0.279),  # raw 20-30% -> reale 27.9%
    (0.30, 0.370),  # raw 30-40% -> reale 37.0%
]

# Calibrazione isotonica ospite (raw Poisson -> reale)
CALIBRATION_AWAY = [
    (0.20, 0.236),  # raw 20-30% -> reale 23.6%
    (0.30, 0.310),  # raw 30-40% -> reale 31.0%
    (0.40, 0.397),  # raw 40-50% -> reale 39.7%
    (0.50, 0.521),  # raw 50-60% -> reale 52.1%
]


def calibrate_prob(prob: float, calibration: list) -> float:
    """
    Calibra una probabilita usando interpolazione lineare
    tra i punti di calibrazione osservati.
    """
    if not calibration:
        return prob
    
    # Sotto il primo punto
    if prob < calibration[0][0]:
        return calibration[0][1] * (prob / calibration[0][0])
    
    # Sopra l ultimo punto
    if prob >= calibration[-1][0] + 0.10:
        last_calib = calibration[-1][1]
        return min(0.95, last_calib + (prob - calibration[-1][0]) * 0.8)
    
    # Interpolazione lineare tra i punti
    for i in range(len(calibration) - 1):
        low_p,  low_r  = calibration[i]
        high_p, high_r = calibration[i+1]
        if low_p <= prob < high_p:
            t = (prob - low_p) / (high_p - low_p)
            return low_r + t * (high_r - low_r)
    
    return calibration[-1][1]

# Kelly frazionato 1/8
KELLY_FRACTION = 0.125

# Edge minimo consigliato
RECOMMENDED_MIN_EDGE = 0.07

# Edge minimo assoluto - sotto non si gioca mai
MIN_EDGE_HARD = 0.03

# Riduzione stake per mercati correlati
CORRELATION_REDUCTION = 0.40

# Riduzione stake ultime giornate (maggio)
LATE_SEASON_REDUCTION = 0.50
LATE_SEASON_MATCHDAYS = 4

# Mercati con ROI storico negativo
UNRELIABLE_MARKETS = {"1 - Vittoria Casa", "1X - Doppia Chance Casa"}

# Mercati correlati
CORRELATED_MARKETS = [
    {"Under 2.5 Gol", "No Goal"},
    {"Over 2.5 Gol", "Goal/Goal"},
    {"Under 1.5 Gol", "No Goal"},
    {"1 - Vittoria Casa", "1X - Doppia Chance Casa"},
    {"2 - Vittoria Ospite", "X2 - Doppia Chance Ospite"},
]


def are_correlated(market1: str, market2: str) -> bool:
    for group in CORRELATED_MARKETS:
        if market1 in group and market2 in group:
            return True
    return False


def is_late_season(date, season_end_month: int = 5) -> bool:
    try:
        d = pd.Timestamp(date)
        return d.month == season_end_month and d.day >= 1
    except:
        return False


def compute_all_markets(poisson_model, home: str, away: str, **kwargs) -> dict:
    """
    Calcola le probabilita per tutti i mercati supportati.
    Applica correzione home advantage, moltiplicatori temporali e derby.
    """
    mat = poisson_model.predict_score_matrix(home, away)
    if mat is None:
        return {}

    # 1X2 raw
    prob_h_raw = float(np.tril(mat, -1).sum())
    prob_d_raw = float(np.trace(mat))
    prob_a_raw = float(np.triu(mat, 1).sum())

    # Calibrazione per fascia basata su dati reali Serie A 2021-2026
    prob_h = calibrate_prob(prob_h_raw, CALIBRATION_HOME)
    prob_d = calibrate_prob(prob_d_raw, CALIBRATION_DRAW)
    prob_a = calibrate_prob(prob_a_raw, CALIBRATION_AWAY)

    # Normalizza a somma 1
    total = prob_h + prob_d + prob_a
    prob_h, prob_d, prob_a = prob_h/total, prob_d/total, prob_a/total

    # Blend con probabilita implicita Pinnacle
    # 1X2: 50/50 (Pinnacle molto piu preciso sulla vittoria casa)
    # Over/Under e GG: 80/20 (nostro modello competitivo)
    W_MODEL_1X2 = 0.50
    W_BOOK_1X2  = 0.50
    bk_h = kwargs.get("odds_h") if kwargs else None
    bk_d = kwargs.get("odds_d") if kwargs else None
    bk_a = kwargs.get("odds_a") if kwargs else None
    if bk_h and bk_d and bk_a and bk_h > 1.0 and bk_d > 1.0 and bk_a > 1.0:
        imp_h = 1/bk_h; imp_d = 1/bk_d; imp_a = 1/bk_a
        tot_imp = imp_h + imp_d + imp_a
        imp_h, imp_d, imp_a = imp_h/tot_imp, imp_d/tot_imp, imp_a/tot_imp
        prob_h = W_MODEL_1X2 * prob_h + W_BOOK_1X2 * imp_h
        prob_d = W_MODEL_1X2 * prob_d + W_BOOK_1X2 * imp_d
        prob_a = W_MODEL_1X2 * prob_a + W_BOOK_1X2 * imp_a
        total2 = prob_h + prob_d + prob_a
        prob_h, prob_d, prob_a = prob_h/total2, prob_d/total2, prob_a/total2

    # Doppia chance
    prob_1x = prob_h + prob_d
    prob_x2 = prob_d + prob_a
    prob_12 = prob_h + prob_a

    # Gol totali
    goals_matrix = np.array([
        [i + j for j in range(mat.shape[1])]
        for i in range(mat.shape[0])
    ])

    def over_prob(threshold):
        return float(np.sum(mat[goals_matrix > threshold]))

    prob_over15  = over_prob(1.5)
    prob_over25  = over_prob(2.5)
    prob_over35  = over_prob(3.5)
    prob_over45  = over_prob(4.5)
    prob_over55  = over_prob(5.5)

    # GG/NG
    prob_gg = float(np.sum(mat[1:, 1:]))
    prob_ng = 1 - prob_gg

    preds = {
        "prob_H":      prob_h,
        "prob_D":      prob_d,
        "prob_A":      prob_a,
        "prob_1X":     prob_1x,
        "prob_X2":     prob_x2,
        "prob_12":     prob_12,
        "prob_over15": prob_over15,
        "prob_over25": prob_over25,
        "prob_over35": prob_over35,
        "prob_over45": prob_over45,
        "prob_over55": prob_over55,
        "prob_under15": 1 - prob_over15,
        "prob_under25": 1 - prob_over25,
        "prob_under35": 1 - prob_over35,
        "prob_under45": 1 - prob_over45,
        "prob_under55": 1 - prob_over55,
        "prob_gg":     prob_gg,
        "prob_ng":     prob_ng,
        "prob_cards_over25":  0.65,
        "prob_cards_over35":  0.45,
        "prob_cards_over45":  0.28,
        "prob_cards_over55":  0.15,
        "prob_cards_under25": 0.35,
        "prob_cards_under35": 0.55,
        "prob_cards_under45": 0.72,
        "prob_cards_under55": 0.85,
    }

    # Mercati primo tempo
    try:
        from models.halftime import compute_ht_markets
        ht = compute_ht_markets(poisson_model, home, away)
        preds.update(ht)
    except Exception:
        pass

    # Moltiplicatori temporali (mese, giorno, orario)
    _md = kwargs.get("match_date") if kwargs else None
    _mt = kwargs.get("match_time") if kwargs else None
    if _md:
        try:
            from models.motivation import get_temporal_multipliers
            _temp = get_temporal_multipliers(_md, _mt)
            _gm = _temp["goals_mult"]
            for _k in ["prob_over15","prob_over25","prob_over35",
                       "prob_over45","prob_over55","prob_gg"]:
                if _k in preds:
                    preds[_k] = max(0.0, min(1.0, preds[_k] * _gm))
            for _k in ["prob_under15","prob_under25","prob_under35",
                       "prob_under45","prob_under55","prob_ng"]:
                if _k in preds:
                    preds[_k] = max(0.0, min(1.0, preds[_k] * (2.0 - _gm)))
        except Exception:
            pass

    # Aggiustamenti derby/big match
    try:
        from models.motivation import get_match_intensity
        match_info = get_match_intensity(home, away)

        if match_info["is_bigmatch"]:
            g_mult  = match_info["goals_multiplier"]
            d_bonus = match_info["draw_bonus"]

            for key in ["prob_over15","prob_over25","prob_over35",
                        "prob_over45","prob_over55","prob_gg"]:
                if key in preds:
                    preds[key] = max(0, preds[key] * g_mult)

            for key in ["prob_under15","prob_under25","prob_under35",
                        "prob_under45","prob_under55","prob_ng"]:
                if key in preds:
                    preds[key] = min(1, preds[key] * (2 - g_mult))

            if "prob_D" in preds:
                old_d = preds["prob_D"]
                new_d = min(0.95, old_d + d_bonus)
                diff  = new_d - old_d
                preds["prob_D"] = new_d
                preds["prob_H"] = max(0.02, preds["prob_H"] - diff/2)
                preds["prob_A"] = max(0.02, preds["prob_A"] - diff/2)

            preds["prob_1X"] = preds["prob_H"] + preds["prob_D"]
            preds["prob_X2"] = preds["prob_D"] + preds["prob_A"]
            preds["prob_12"] = preds["prob_H"] + preds["prob_A"]

            preds["match_name"]       = match_info["name"]
            preds["match_intensity"]  = match_info["intensity"]
            preds["is_derby"]         = match_info["is_derby"]
            preds["cards_multiplier"] = match_info["cards_multiplier"]
        else:
            preds["match_name"]       = None
            preds["is_derby"]         = False
            preds["cards_multiplier"] = 1.0
    except Exception:
        pass

    return preds


def find_value_bets_extended(
    predictions: dict,
    odds_dict: dict,
    min_edge: float = RECOMMENDED_MIN_EDGE,
    bankroll: float = 1000.0,
    match_date=None,
) -> list:
    """
    Confronta probabilita del modello con le quote su tutti i mercati.
    """
    late_season = is_late_season(match_date) if match_date else False
    late_reduction = LATE_SEASON_REDUCTION if late_season else 1.0

    markets = [
        ("H",             "prob_H",            "1 - Vittoria Casa"),
        ("D",             "prob_D",             "X - Pareggio"),
        ("A",             "prob_A",             "2 - Vittoria Ospite"),
        ("1X",            "prob_1X",            "1X - Doppia Chance Casa"),
        ("X2",            "prob_X2",            "X2 - Doppia Chance Ospite"),
        ("12",            "prob_12",            "12 - Doppia Chance No Pareggio"),
        ("over15",        "prob_over15",        "Over 1.5 Gol"),
        ("under15",       "prob_under15",       "Under 1.5 Gol"),
        ("over25",        "prob_over25",        "Over 2.5 Gol"),
        ("under25",       "prob_under25",       "Under 2.5 Gol"),
        ("over35",        "prob_over35",        "Over 3.5 Gol"),
        ("under35",       "prob_under35",       "Under 3.5 Gol"),
        ("over45",        "prob_over45",        "Over 4.5 Gol"),
        ("under45",       "prob_under45",       "Under 4.5 Gol"),
        ("over55",        "prob_over55",        "Over 5.5 Gol"),
        ("under55",       "prob_under55",       "Under 5.5 Gol"),
        ("gg",            "prob_gg",            "Goal/Goal"),
        ("ng",            "prob_ng",            "No Goal"),
        ("cards_over25",  "prob_cards_over25",  "Cartellini Over 2.5"),
        ("cards_under25", "prob_cards_under25", "Cartellini Under 2.5"),
        ("cards_over35",  "prob_cards_over35",  "Cartellini Over 3.5"),
        ("cards_under35", "prob_cards_under35", "Cartellini Under 3.5"),
        ("cards_over45",  "prob_cards_over45",  "Cartellini Over 4.5"),
        ("cards_under45", "prob_cards_under45", "Cartellini Under 4.5"),
        ("cards_over55",  "prob_cards_over55",  "Cartellini Over 5.5"),
        ("cards_under55", "prob_cards_under55", "Cartellini Under 5.5"),
        ("ht_H",       "ht_prob_H",       "HT 1 - Vittoria Casa"),
        ("ht_D",       "ht_prob_D",       "HT X - Pareggio"),
        ("ht_A",       "ht_prob_A",       "HT 2 - Vittoria Ospite"),
        ("ht_1X",      "ht_prob_1X",      "HT 1X - Doppia Chance"),
        ("ht_X2",      "ht_prob_X2",      "HT X2 - Doppia Chance"),
        ("ht_over05",  "ht_prob_over05",  "HT Over 0.5 Gol"),
        ("ht_under05", "ht_prob_under05", "HT Under 0.5 Gol"),
        ("ht_over15",  "ht_prob_over15",  "HT Over 1.5 Gol"),
        ("ht_under15", "ht_prob_under15", "HT Under 1.5 Gol"),
        ("ht_gg",      "ht_prob_gg",      "HT Goal/Goal"),
        ("ht_ng",      "ht_prob_ng",      "HT No Goal"),
    ]

    value_bets = []
    for mkt_key, prob_key, label in markets:
        odds = odds_dict.get(mkt_key)
        prob = predictions.get(prob_key)
        if odds is None or prob is None:
            continue
        if pd.isna(odds) or pd.isna(prob) or odds <= 1.0:
            continue

        edge = prob * odds - 1

        if edge < MIN_EDGE_HARD:
            continue
        if edge < min_edge:
            continue

        b = odds - 1
        q = 1 - prob
        kelly_full = (b * prob - q) / b
        kelly_frac = max(0, kelly_full * KELLY_FRACTION)

        corr_reduction = 1.0
        for existing in value_bets:
            if are_correlated(label, existing["mercato"]):
                corr_reduction = 1.0 - CORRELATION_REDUCTION
                break

        stake = round(kelly_frac * bankroll * corr_reduction * late_reduction, 2)
        is_unreliable = label in UNRELIABLE_MARKETS

        value_bets.append({
            "mercato":          label,
            "quota":            round(odds, 2),
            "prob_modello_%":   round(prob * 100, 1),
            "prob_implicita_%": round(100 / odds, 1),
            "edge_%":           round(edge * 100, 1),
            "kelly_%":          round(kelly_frac * 100, 2),
            "stake_euro":       stake,
            "affidabile":       not is_unreliable,
            "correlato":        corr_reduction < 1.0,
        })

    # Aggiusta chiave stake per compatibilita
    for vb in value_bets:
        vb["stake_€"] = vb.pop("stake_euro")

    return sorted(value_bets, key=lambda x: (x["affidabile"], x["edge_%"]),
                  reverse=True)


def backtest_extended(df: pd.DataFrame, poisson_model,
                      min_edge: float = RECOMMENDED_MIN_EDGE) -> pd.DataFrame:
    results = []

    for _, row in df.iterrows():
        home, away = row.get("HomeTeam"), row.get("AwayTeam")
        if not home or not away:
            continue

        preds = compute_all_markets(poisson_model, home, away)
        if not preds:
            continue

        ftr   = row.get("FTR", "")
        fthg  = int(row.get("FTHG", 0) or 0)
        ftag  = int(row.get("FTAG", 0) or 0)
        total = fthg + ftag
        hy    = int(float(row.get("HY", 0) or 0))
        ay    = int(float(row.get("AY", 0) or 0))
        cards = hy + ay

        def q(col):
            try:
                v = float(row.get(col, 0) or 0)
                return v if v > 1.0 else None
            except:
                return None

        odds_dict = {}
        h = q("PSH") or q("B365H")
        d = q("PSD") or q("B365D")
        a = q("PSA") or q("B365A")
        o25 = q("P_over25") or q("B365_over25")
        u25 = q("P_under25") or q("B365_under25")
        if h:   odds_dict["H"]       = h
        if d:   odds_dict["D"]       = d
        if a:   odds_dict["A"]       = a
        if o25: odds_dict["over25"]  = o25
        if u25: odds_dict["under25"] = u25

        vbs = find_value_bets_extended(
            preds, odds_dict, min_edge=min_edge, bankroll=1000,
            match_date=row.get("Date")
        )

        for vb in vbs:
            won = _check_result(vb["mercato"], ftr, total, cards)
            if won is None:
                continue
            profit = vb["stake_€"] * (vb["quota"] - 1) if won else -vb["stake_€"]
            results.append({
                "data":       row.get("Date"),
                "partita":    f"{home} vs {away}",
                "mercato":    vb["mercato"],
                "quota":      vb["quota"],
                "prob_%":     vb["prob_modello_%"],
                "edge_%":     vb["edge_%"],
                "stake_€":    vb["stake_€"],
                "affidabile": vb["affidabile"],
                "esito":      "Vinta" if won else "Persa",
                "profitto_€": round(profit, 2),
            })

    return pd.DataFrame(results)


def _check_result(mercato: str, ftr: str, total_goals: int, total_cards: int):
    m = mercato.lower()
    if "vittoria casa"   in m and "doppia" not in m and "ht" not in m: return ftr == "H"
    if "pareggio"        in m and "doppia" not in m and "ht" not in m: return ftr == "D"
    if "vittoria ospite" in m and "doppia" not in m and "ht" not in m: return ftr == "A"
    if "doppia"          in m and "casa"   in m:     return ftr in ("H","D")
    if "doppia"          in m and "ospite" in m:     return ftr in ("A","D")
    if "no pareggio"     in m:                       return ftr in ("H","A")
    if "over 1.5"  in m and "cart" not in m and "ht" not in m: return total_goals > 1.5
    if "under 1.5" in m and "cart" not in m and "ht" not in m: return total_goals < 1.5
    if "over 2.5"  in m and "cart" not in m and "ht" not in m: return total_goals > 2.5
    if "under 2.5" in m and "cart" not in m and "ht" not in m: return total_goals < 2.5
    if "over 3.5"  in m and "cart" not in m: return total_goals > 3.5
    if "under 3.5" in m and "cart" not in m: return total_goals < 3.5
    if "over 4.5"  in m and "cart" not in m: return total_goals > 4.5
    if "under 4.5" in m and "cart" not in m: return total_goals < 4.5
    if "over 5.5"  in m and "cart" not in m: return total_goals > 5.5
    if "under 5.5" in m and "cart" not in m: return total_goals < 5.5
    if "goal/goal" in m and "ht" not in m: return total_goals > 0
    if "no goal"   in m and "ht" not in m: return total_goals == 0
    if "cartellini over 2.5"  in m: return total_cards > 2.5
    if "cartellini under 2.5" in m: return total_cards < 2.5
    if "cartellini over 3.5"  in m: return total_cards > 3.5
    if "cartellini under 3.5" in m: return total_cards < 3.5
    if "cartellini over 4.5"  in m: return total_cards > 4.5
    if "cartellini under 4.5" in m: return total_cards < 4.5
    if "cartellini over 5.5"  in m: return total_cards > 5.5
    if "cartellini under 5.5" in m: return total_cards < 5.5
    return None