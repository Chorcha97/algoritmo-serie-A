"""
Modulo Handicap Asiatico
Calcola probabilità e value bet per l'handicap asiatico
usando la matrice di Poisson bivariato.

Linee comuni in Serie A:
  -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
   0.0 (Draw No Bet),
  +0.25, +0.5, +0.75, +1.0, +1.25, +1.5, +1.75, +2.0
"""

import numpy as np
import pandas as pd


def prob_asian_handicap(matrix: np.ndarray, line: float) -> tuple:
    """
    Calcola le probabilità per l'handicap asiatico dato una linea.
    
    line: valore dell'handicap dalla prospettiva della squadra di casa
          es. -1.0 = casa deve vincere con 2+ gol
              +0.5 = casa vince o pareggia
    
    Restituisce (prob_home, prob_away) dove:
    - prob_home = prob. che la casa copra l'handicap
    - prob_away = prob. che l'ospite copra l'handicap
    
    Per le linee a quarto (0.25, 0.75 ecc.) usa la media di due linee.
    """
    n = matrix.shape[0]
    
    # Gestisci linee a quarto (split bet)
    if line % 0.5 != 0:
        # Linea a quarto = media di due linee intere/mezze
        line_low  = line - 0.25
        line_high = line + 0.25
        p_h_low,  p_a_low  = prob_asian_handicap(matrix, line_low)
        p_h_high, p_a_high = prob_asian_handicap(matrix, line_high)
        return (p_h_low + p_h_high) / 2, (p_a_low + p_a_high) / 2
    
    prob_home = 0.0
    prob_away = 0.0
    prob_push = 0.0  # rimborso (solo per linee intere)
    
    for i in range(n):
        for j in range(n):
            margin = i - j  # gol casa - gol ospite
            adjusted = margin + line  # margine aggiustato
            
            p = matrix[i, j]
            
            if line % 1 == 0:  # linea intera o mezza
                if adjusted > 0:
                    prob_home += p
                elif adjusted < 0:
                    prob_away += p
                else:
                    prob_push += p  # rimborso
            else:  # linea mezza (0.5, 1.5 ecc.)
                if adjusted > 0:
                    prob_home += p
                else:
                    prob_away += p
    
    # Con rimborso: distribuisci push proporzionalmente
    if prob_push > 0:
        total = prob_home + prob_away
        if total > 0:
            prob_home += prob_push * (prob_home / total)
            prob_away += prob_push * (prob_away / total)
    
    return float(prob_home), float(prob_away)


def compute_ah_markets(poisson_model, home: str, away: str,
                       lines: list = None) -> dict:
    """
    Calcola probabilità e quote eque per tutte le linee AH.
    
    lines: lista di linee da calcolare. Se None usa le più comuni.
    """
    mat = poisson_model.predict_score_matrix(home, away)
    if mat is None:
        return {}
    
    if lines is None:
        lines = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
                  0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    
    result = {}
    for line in lines:
        p_h, p_a = prob_asian_handicap(mat, line)
        line_str = f"{line:+.2f}".replace("+", "")
        result[f"ah_{line_str}_home"] = p_h
        result[f"ah_{line_str}_away"] = p_a
        if p_h > 0:
            result[f"ah_{line_str}_home_fair_odds"] = round(1/p_h, 3)
        if p_a > 0:
            result[f"ah_{line_str}_away_fair_odds"] = round(1/p_a, 3)
    
    return result


def find_ah_value_bets(
    poisson_model,
    home: str,
    away: str,
    ah_line: float,
    odds_home: float,
    odds_away: float,
    min_edge: float = 0.07,
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.125,
) -> list:
    """
    Trova value bet sull'handicap asiatico per una linea specifica.
    """
    mat = poisson_model.predict_score_matrix(home, away)
    if mat is None:
        return []
    
    p_h, p_a = prob_asian_handicap(mat, ah_line)
    value_bets = []
    
    for prob, odds, label in [
        (p_h, odds_home, f"AH {ah_line:+.2f} Casa ({home})"),
        (p_a, odds_away, f"AH {ah_line:+.2f} Ospite ({away})"),
    ]:
        if not odds or odds <= 1.0 or prob <= 0:
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
            "linea":            ah_line,
            "quota":            round(odds, 2),
            "prob_modello_%":   round(prob * 100, 1),
            "prob_implicita_%": round(100 / odds, 1),
            "edge_%":           round(edge * 100, 1),
            "stake_€":          stake,
        })
    
    return sorted(value_bets, key=lambda x: x["edge_%"], reverse=True)


def backtest_ah(df: pd.DataFrame, poisson_model,
                min_edge: float = 0.07) -> pd.DataFrame:
    """
    Backtest sul mercato handicap asiatico.
    Usa le quote B365AHH/B365AHA e la linea AHh.
    """
    results = []
    
    required = ["AHh", "B365AHH", "B365AHA", "FTHG", "FTAG"]
    df_clean = df.dropna(subset=required).copy()
    print(f"  Partite con dati AH: {len(df_clean)}/{len(df)}")
    
    for _, row in df_clean.iterrows():
        home = row.get("HomeTeam")
        away = row.get("AwayTeam")
        if not home or not away:
            continue
        
        line      = float(row["AHh"])
        odds_home = float(row.get("PAHH") or row["B365AHH"])
        odds_away = float(row.get("PAHA") or row["B365AHA"])
        fthg      = int(row["FTHG"])
        ftag      = int(row["FTAG"])
        
        vbs = find_ah_value_bets(
            poisson_model, home, away,
            line, odds_home, odds_away,
            min_edge=min_edge, bankroll=1000
        )
        
        for vb in vbs:
            # Determina se home o away ha vinto l'AH
            margin = fthg - ftag
            adjusted = margin + line
            
            if line % 1 == 0:  # linea intera
                if adjusted > 0:
                    winner = "home"
                elif adjusted < 0:
                    winner = "away"
                else:
                    winner = "push"  # rimborso
            else:  # linea mezza
                winner = "home" if adjusted > 0 else "away"
            
            is_home_bet = "Casa" in vb["mercato"]
            
            if winner == "push":
                profit = 0.0
                esito = "↩ Rimborso"
            elif (is_home_bet and winner == "home") or \
                 (not is_home_bet and winner == "away"):
                profit = vb["stake_€"] * (vb["quota"] - 1)
                esito = "✓ Vinta"
            else:
                profit = -vb["stake_€"]
                esito = "✗ Persa"
            
            results.append({
                "data":       row.get("Date"),
                "partita":    f"{home} vs {away}",
                "mercato":    vb["mercato"],
                "linea":      line,
                "quota":      vb["quota"],
                "prob_%":     vb["prob_modello_%"],
                "edge_%":     vb["edge_%"],
                "stake_€":    vb["stake_€"],
                "esito":      esito,
                "profitto_€": round(profit, 2),
            })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Test rapido
    import sys
    sys.path.insert(0, "..")
    import pandas as pd
    from models.features import build_features
    from models.model import EnsembleModel
    
    print("Carico modello...")
    import pickle
    from pathlib import Path
    if Path("model_cache.pkl").exists():
        with open("model_cache.pkl", "rb") as f:
            model = pickle.load(f)
    else:
        df_raw = pd.read_csv("serie_a_dataset.csv", parse_dates=["Date"])
        df = build_features(df_raw, verbose=False)
        model = EnsembleModel()
        model.fit(df)
    
    print("\nTest Inter vs Juventus:")
    ah = compute_ah_markets(model.poisson, "Inter", "Juventus")
    for line in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        line_str = f"{line:+.2f}".replace("+","")
        ph = ah.get(f"ah_{line_str}_home", 0)
        pa = ah.get(f"ah_{line_str}_away", 0)
        fh = ah.get(f"ah_{line_str}_home_fair_odds", 0)
        fa = ah.get(f"ah_{line_str}_away_fair_odds", 0)
        print(f"  AH {line:+.2f}  "
              f"Casa: {ph:.3f} (quota equa {fh:.2f})  "
              f"Ospite: {pa:.3f} (quota equa {fa:.2f})")
    
    print("\nBacktest AH su dataset...")
    df_raw = pd.read_csv("serie_a_dataset.csv", parse_dates=["Date"])
    sa = df_raw[df_raw["league"]=="Serie A"] if "league" in df_raw.columns else df_raw
    bt = backtest_ah(sa, model.poisson, min_edge=0.07)
    if not bt.empty:
        tot_s = bt["stake_€"].sum()
        tot_p = bt["profitto_€"].sum()
        roi   = tot_p/tot_s*100 if tot_s > 0 else 0
        wr    = (bt["esito"]=="✓ Vinta").mean()*100
        print(f"  Giocate: {len(bt)}  Win rate: {wr:.1f}%  "
              f"Stake: €{tot_s:.0f}  Profitto: €{tot_p:+.0f}  ROI: {roi:+.1f}%")
