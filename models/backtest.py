"""
Walk-forward backtest con arbitro rolling
==========================================
1. WALK-FORWARD: il modello si allena solo sulle stagioni precedenti
2. REFEREE ROLLING: le statistiche arbitro vengono aggiornate partita per partita
"""

import pandas as pd
import numpy as np
from pathlib import Path

MIN_TRAIN_SEASONS = 2

def run_walkforward_backtest(
    df: pd.DataFrame,
    min_edge: float = 0.07,
    bankroll: float = 1000.0,
    verbose: bool = True,
) -> pd.DataFrame:
    from models.model import EnsembleModel
    from models.features import build_features
    from models.markets import compute_all_markets, find_value_bets_extended
    from models.referee import compute_cards_markets

    df = df.sort_values("Date").reset_index(drop=True)
    if "season" not in df.columns:
        df = _infer_season(df)
    seasons = sorted(df["season"].unique())

    if len(seasons) < MIN_TRAIN_SEASONS + 1:
        print(f"[warning] Servono almeno {MIN_TRAIN_SEASONS+1} stagioni")
        return pd.DataFrame()

    all_results = []

    for i in range(MIN_TRAIN_SEASONS, len(seasons)):
        train_seasons = seasons[:i]
        test_season   = seasons[i]
        train_df = df[df["season"].isin(train_seasons)].copy()
        test_df  = df[df["season"] == test_season].copy()

        if verbose:
            print(f"  Fold {i-MIN_TRAIN_SEASONS+1}: "
                  f"train {train_seasons[0]}→{train_seasons[-1]}  "
                  f"test {test_season}  "
                  f"({len(train_df)} train, {len(test_df)} test)")

        try:
            train_feat = build_features(train_df, verbose=False)
            fold_model = EnsembleModel()
            fold_model.fit(train_feat, halflife_days=90)
        except Exception as e:
            print(f"  [errore fold {i}]: {e}")
            continue

        rolling_ref_history = train_df.copy()
        fold_results = []

        for _, row in test_df.iterrows():
            home = row.get("HomeTeam")
            away = row.get("AwayTeam")
            if not home or not away:
                continue

            kwargs_preds = {}
            try:
                psh = row.get("PSH"); psd = row.get("PSD"); psa = row.get("PSA")
                if psh and psd and psa and float(str(psh)) > 1.0:
                    kwargs_preds = {"odds_h": float(str(psh)), "odds_d": float(str(psd)), "odds_a": float(str(psa))}
            except:
                pass
            preds = compute_all_markets(fold_model.poisson, home, away, **kwargs_preds)
            if not preds:
                continue

            referee = str(row.get("Referee", "") or "").strip()
            ref_stats = _compute_referee_stats_rolling(rolling_ref_history, referee)
            ref_feat  = _get_referee_features_rolling(ref_stats, referee)

            # Classifica rolling — usa solo partite precedenti a questa data
            try:
                from models.motivation import compute_standings_at_date, compute_motivation_features
                match_date = str(row.get("Date", ""))
                standings_rolling = compute_standings_at_date(rolling_ref_history, match_date)
                mot_feat = compute_motivation_features(
                    home, away, match_date=match_date, standings=standings_rolling)
            except:
                mot_feat = {}

            cards_preds = compute_cards_markets(
                preds["prob_D"], preds["prob_over25"], ref_feat)
            preds.update(cards_preds)

            odds_dict = _extract_odds(row)
            # Passa quote Pinnacle per il blend 80/20
            kwargs_preds = {}
            try:
                psh = row.get('PSH'); psd = row.get('PSD'); psa = row.get('PSA')
                if psh and psd and psa and float(psh) > 1.0:
                    kwargs_preds = {'odds_h': float(psh), 'odds_d': float(psd), 'odds_a': float(psa)}
            except:
                pass
            vbs = find_value_bets_extended(
                preds, odds_dict, min_edge=min_edge, bankroll=bankroll)

            ftr   = str(row.get("FTR", "") or "")
            total = int(row.get("FTHG", 0) or 0) + int(row.get("FTAG", 0) or 0)
            hy = row.get("HY", 0); ay = row.get("AY", 0)
            cards = (0 if (hy is None or str(hy)=="nan") else int(float(hy))) + (0 if (ay is None or str(ay)=="nan") else int(float(ay)))

            for vb in vbs:
                won = _check_result(vb["mercato"], ftr, total, cards)
                if won is None:
                    continue
                profit = vb["stake_€"] * (vb["quota"] - 1) if won else -vb["stake_€"]
                fold_results.append({
                    "stagione":           test_season,
                    "data":               row.get("Date"),
                    "partita":            f"{home} vs {away}",
                    "arbitro":            referee or "—",
                    "mercato":            vb["mercato"],
                    "quota":              vb["quota"],
                    "prob_%":             vb["prob_modello_%"],
                    "edge_%":             vb["edge_%"],
                    "stake_€":            vb["stake_€"],
                    "affidabile":         vb["affidabile"],
                    "esito":              "✓ Vinta" if won else "✗ Persa",
                    "profitto_€":         round(profit, 2),
                    "ref_gialli_storici": round(ref_feat.get("ref_gialli_media", 3.8), 2),
                    "ref_found":          ref_feat.get("ref_found", False),
                })

            rolling_ref_history = pd.concat(
                [rolling_ref_history, pd.DataFrame([row])], ignore_index=True)

        if fold_results:
            fold_df = pd.DataFrame(fold_results)
            stake  = fold_df["stake_€"].sum()
            profit = fold_df["profitto_€"].sum()
            roi    = profit / stake * 100 if stake > 0 else 0
            emoji  = "✅" if profit > 0 else "❌"
            if verbose:
                print(f"    {emoji} {test_season}: {len(fold_df)} giocate  "
                      f"ROI {roi:+.1f}%  €{profit:+,.0f}")
            all_results.extend(fold_results)

    if not all_results:
        return pd.DataFrame()

    result_df = pd.DataFrame(all_results)

    if verbose:
        print("\n" + "="*50)
        tot_stake  = result_df["stake_€"].sum()
        tot_profit = result_df["profitto_€"].sum()
        roi_tot    = tot_profit / tot_stake * 100 if tot_stake > 0 else 0
        wr         = (result_df["esito"] == "✓ Vinta").mean() * 100
        print(f"  WALK-FORWARD TOTALE")
        print(f"  Giocate:  {len(result_df)}")
        print(f"  Win rate: {wr:.1f}%")
        print(f"  Profitto: €{tot_profit:+,.2f}")
        print(f"  ROI:      {roi_tot:+.2f}%")
        print("="*50)

    return result_df


def _compute_referee_stats_rolling(history: pd.DataFrame, referee: str) -> dict:
    if not referee or "Referee" not in history.columns:
        return {}
    past = history[
        history["Referee"].astype(str).str.strip().str.upper() == referee.upper()
    ]
    if past.empty:
        return {}
    hy = past["HY"].fillna(0).astype(float)
    ay = past["AY"].fillna(0).astype(float)
    hr = past.get("HR", pd.Series(0, index=past.index)).fillna(0).astype(float)
    ar = past.get("AR", pd.Series(0, index=past.index)).fillna(0).astype(float)
    total_yellow = hy + ay
    total_red    = hr + ar
    return {
        "referee":           referee,
        "partite":           len(past),
        "gialli_media":      float(total_yellow.mean()),
        "rossi_media":       float(total_red.mean()),
        "gialli_casa_avg":   float(hy.mean()),
        "gialli_ospite_avg": float(ay.mean()),
        "bias_casa":         float(ay.mean() - hy.mean()),
        "severita_pct":      _severity_pct(float(total_yellow.mean()),
                                            float(total_red.mean())),
    }


def _severity_pct(gialli: float, rossi: float) -> float:
    from scipy.stats import norm
    z = 0.7 * (gialli - 3.8) / 1.2 + 0.3 * (rossi - 0.18) / 0.15
    return float(norm.cdf(z) * 100)


def _get_referee_features_rolling(stats: dict, referee: str) -> dict:
    if not stats or stats.get("partite", 0) < 3:
        return {"ref_gialli_media": 3.8, "ref_rossi_media": 0.18,
                "ref_severita_pct": 50.0, "ref_bias_casa": 0.0, "ref_found": False}
    return {
        "ref_gialli_media":  stats["gialli_media"],
        "ref_rossi_media":   stats["rossi_media"],
        "ref_severita_pct":  stats["severita_pct"],
        "ref_bias_casa":     stats["bias_casa"],
        "ref_partite":       stats["partite"],
        "ref_found":         stats["partite"] >= 5,
    }


def _infer_season(df: pd.DataFrame) -> pd.DataFrame:
    def s(d):
        try:
            d = pd.Timestamp(d)
            return f"{d.year}/{str(d.year+1)[-2:]}" if d.month >= 8 \
                   else f"{d.year-1}/{str(d.year)[-2:]}"
        except:
            return "unknown"
    df = df.copy()
    df["season"] = df["Date"].apply(s)
    return df


def _extract_odds(row: pd.Series) -> dict:
    def q(col):
        try:
            v = float(row.get(col, 0) or 0)
            return v if v > 1.0 else None
        except:
            return None
    odds = {}
    h = q("PSH") or q("B365H")
    d = q("PSD") or q("B365D")
    a = q("PSA") or q("B365A")
    o25 = q("P_over25") or q("B365_over25")
    u25 = q("P_under25") or q("B365_under25")
    if h:   odds["H"]       = h
    if d:   odds["D"]       = d
    if a:   odds["A"]       = a
    if o25: odds["over25"]  = o25
    if u25: odds["under25"] = u25
    return odds


def _check_result(mercato: str, ftr: str, total_goals: int, total_cards: int):
    m = mercato.lower()
    if "vittoria casa"   in m and "doppia" not in m: return ftr == "H"
    if "pareggio"        in m and "doppia" not in m: return ftr == "D"
    if "vittoria ospite" in m and "doppia" not in m: return ftr == "A"
    if "doppia"          in m and "casa"   in m:     return ftr in ("H","D")
    if "doppia"          in m and "ospite" in m:     return ftr in ("A","D")
    if "no pareggio"     in m:                       return ftr in ("H","A")
    if "over 1.5"  in m and "cart" not in m: return total_goals > 1.5
    if "under 1.5" in m and "cart" not in m: return total_goals < 1.5
    if "over 2.5"  in m and "cart" not in m: return total_goals > 2.5
    if "under 2.5" in m and "cart" not in m: return total_goals < 2.5
    if "over 3.5"  in m and "cart" not in m: return total_goals > 3.5
    if "under 3.5" in m and "cart" not in m: return total_goals < 3.5
    if "over 4.5"  in m and "cart" not in m: return total_goals > 4.5
    if "under 4.5" in m and "cart" not in m: return total_goals < 4.5
    if "over 5.5"  in m and "cart" not in m: return total_goals > 5.5
    if "under 5.5" in m and "cart" not in m: return total_goals < 5.5
    if "goal/goal" in m: return total_goals > 0
    if "no goal"   in m: return total_goals == 0
    if "cartellini over 2.5"  in m: return total_cards > 2.5
    if "cartellini under 2.5" in m: return total_cards < 2.5
    if "cartellini over 3.5"  in m: return total_cards > 3.5
    if "cartellini under 3.5" in m: return total_cards < 3.5
    if "cartellini over 4.5"  in m: return total_cards > 4.5
    if "cartellini under 4.5" in m: return total_cards < 4.5
    if "cartellini over 5.5"  in m: return total_cards > 5.5
    if "cartellini under 5.5" in m: return total_cards < 5.5
    return None


def summarize_walkforward(bt: pd.DataFrame) -> pd.DataFrame:
    if bt.empty:
        return pd.DataFrame()
    g = bt.groupby("stagione").agg(
        giocate    = ("stake_€", "count"),
        stake_eur  = ("stake_€", "sum"),
        profitto_eur = ("profitto_€", "sum"),
        win_rate   = ("esito", lambda x: (x == "✓ Vinta").mean() * 100),
    ).reset_index()
    g["roi_%"] = (g["profitto_eur"] / g["stake_eur"] * 100).round(2)
    g["profitto_€"] = g["profitto_eur"].round(2)
    g["win_rate"] = g["win_rate"].round(1)
    return g.sort_values("stagione")


def summarize_by_referee(bt: pd.DataFrame) -> pd.DataFrame:
    if bt.empty or "arbitro" not in bt.columns:
        return pd.DataFrame()
    sub = bt[bt["arbitro"] != "—"].copy()
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for arb, grp in sub.groupby("arbitro"):
        if len(grp) < 10:
            continue
        s = grp["stake_€"].sum()
        p = grp["profitto_€"].sum()
        rows.append({
            "Arbitro": arb,
            "Giocate": len(grp),
            "Stake": round(s, 2),
            "Profitto": round(p, 2),
            "Gialli medi": round(grp["ref_gialli_storici"].mean(), 2),
            "ROI %": round(p / s * 100, 1) if s > 0 else 0,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("ROI %", ascending=False)
