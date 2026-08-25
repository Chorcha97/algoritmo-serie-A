"""
Modello predittivo Serie A — Stack Ensemble
  1. Poisson bivariato  → probabilità risultati esatti + 1X2/OU/GG
  2. XGBoost classifier → 1X2, Over/Under, GG/NG
  3. Meta-modello       → combina le due stime con calibrazione isotonica
  4. Value bet engine   → confronta prob. modello vs quote bookmaker
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False
    print("[warning] xgboost non installato, uso GradientBoosting sklearn")

from models.features import get_feature_columns


# ─────────────────────────────────────────────
# 1. MODELLO DI POISSON BIVARIATO
# ─────────────────────────────────────────────

class PoissonModel:
    """
    Stima la forza offensiva/difensiva di ogni squadra via max likelihood.
    Genera la distribuzione di probabilità su tutti i risultati esatti.
    """

    def __init__(self, home_advantage: float = 0.3):
        self.attack: dict = {}
        self.defense: dict = {}
        self.home_adv = home_advantage
        self.avg_goals = 2.7  # media Serie A
        self.teams: list = []

    def fit(self, df: pd.DataFrame, halflife_days: int = 90):
        teams = list(set(df["HomeTeam"]) | set(df["AwayTeam"]))
        self.teams = sorted(teams)
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}

        # Calibra avg_goals sui dati reali invece di usare valore fisso
        if "FTHG" in df.columns and "FTAG" in df.columns:
            total_goals = (df["FTHG"].fillna(0) + df["FTAG"].fillna(0)).sum()
            total_matches = df[["FTHG","FTAG"]].dropna().shape[0]
            if total_matches > 50:
                self.avg_goals = float(total_goals / total_matches)
                # Clamp tra 2.3 e 3.2 per sicurezza
                self.avg_goals = max(2.3, min(3.2, self.avg_goals))

        # Calcola pesi temporali: partite recenti pesano di piu
        # halflife_days = giorni dopo cui il peso si dimezza
        if "Date" in df.columns:
            dates = pd.to_datetime(df["Date"], errors="coerce")
            max_date = dates.max()
            days_ago = (max_date - dates).dt.days.fillna(365).values
            weights = np.exp(-days_ago * np.log(2) / halflife_days)
        else:
            weights = np.ones(len(df))

        # Parametri iniziali: attack=1, defense=1, home_adv=0.3
        x0 = np.ones(2 * n + 1)
        x0[-1] = self.home_adv

        df_reset = df.reset_index(drop=True)

        home_idx_arr = np.array([idx.get(h, -1) for h in df_reset["HomeTeam"]])
        away_idx_arr = np.array([idx.get(a, -1) for a in df_reset["AwayTeam"]])
        gh_arr = pd.to_numeric(df_reset["FTHG"], errors="coerce").values
        ga_arr = pd.to_numeric(df_reset["FTAG"], errors="coerce").values
        valid  = (home_idx_arr >= 0) & (away_idx_arr >= 0) & ~np.isnan(gh_arr) & ~np.isnan(ga_arr)
        h_idx  = home_idx_arr[valid].astype(int)
        a_idx  = away_idx_arr[valid].astype(int)
        gh_v   = gh_arr[valid].astype(int)
        ga_v   = ga_arr[valid].astype(int)
        w_v    = weights[valid]

        def neg_log_likelihood(params):
            attack  = np.exp(params[:n])
            defense = np.exp(params[n:2*n])
            home    = params[-1]
            lam_h = attack[h_idx] / defense[a_idx] * self.avg_goals * np.exp(home)
            lam_a = attack[a_idx] / defense[h_idx] * self.avg_goals
            ll = -np.sum(w_v * (poisson.logpmf(gh_v, lam_h) + poisson.logpmf(ga_v, lam_a)))
            return ll

        res = minimize(neg_log_likelihood, x0, method="L-BFGS-B",
                       options={"maxiter": 50, "disp": False})
        params = res.x
        for i, t in enumerate(teams):
            self.attack[t]  = np.exp(params[i])
            self.defense[t] = np.exp(params[n + i])
        self.home_adv = params[-1]
        return self

    def predict_score_matrix(self, home: str, away: str, max_goals: int = 8) -> np.ndarray:
        """Restituisce matrice [home_goals x away_goals] di probabilità."""
        if home not in self.attack or away not in self.attack:
            return None
        lam_h = self.attack[home] / self.defense[away] * self.avg_goals * np.exp(self.home_adv)
        lam_a = self.attack[away] / self.defense[home] * self.avg_goals
        matrix = np.outer(
            [poisson.pmf(i, lam_h) for i in range(max_goals + 1)],
            [poisson.pmf(j, lam_a) for j in range(max_goals + 1)]
        )
        return matrix

    def predict_proba(self, home: str, away: str) -> dict:
        """Restituisce dizionario con prob. per tutti i mercati."""
        mat = self.predict_score_matrix(home, away)
        if mat is None:
            return {}
        prob_h = np.tril(mat, -1).sum()
        prob_d = np.trace(mat)
        prob_a = np.triu(mat, 1).sum()

        # Over/Under
        goals_matrix = np.array([[i + j for j in range(mat.shape[1])]
                                  for i in range(mat.shape[0])])
        prob_ou25 = float((goals_matrix > 2.5).astype(float) * mat).sum() if False else \
                    float(np.sum(mat[goals_matrix > 2.5]))
        prob_gg = float(np.sum(mat[1:, 1:]))

        return {
            "poisson_h": prob_h, "poisson_d": prob_d, "poisson_a": prob_a,
            "poisson_ou25": prob_ou25, "poisson_gg": prob_gg,
        }


# ─────────────────────────────────────────────
# 2. XGBOOST / GRADIENT BOOSTING CLASSIFIER
# ─────────────────────────────────────────────

class XGBModel:
    """
    Tre classificatori separati: 1X2, Over/Under 2.5, GG/NG.
    Calibrati con calibrazione isotonica.
    """

    def __init__(self):
        self.models: dict = {}
        self.feature_cols = get_feature_columns()

    def _make_clf(self):
        if HAS_XGB:
            base = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, verbosity=0,
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            base = GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
        return CalibratedClassifierCV(base, cv=3, method="isotonic")

    def fit(self, df: pd.DataFrame, halflife_days: int = 90):
        feats = [c for c in self.feature_cols if c in df.columns]
        X = df[feats].fillna(df[feats].median())

        targets = {
            "1x2":  ("target_1x2", [0, 1, 2]),
            "ou25": ("target_ou25", [0, 1]),
            "gg":   ("target_gg", [0, 1]),
        }

        for name, (tcol, _) in targets.items():
            if tcol not in df.columns:
                continue
            mask = df[tcol].notna()
            clf = self._make_clf()
            clf.fit(X[mask], df.loc[mask, tcol].astype(int))
            self.models[name] = (clf, feats)
        return self

    def predict_proba(self, row: pd.Series) -> dict:
        result = {}
        for name, (clf, feats) in self.models.items():
            x = row.reindex(feats).fillna(0).values.reshape(1, -1)
            proba = clf.predict_proba(x)[0]
            classes = clf.classes_
            p = {str(c): float(proba[i]) for i, c in enumerate(classes)}
            if name == "1x2":
                result["xgb_h"] = p.get("0", np.nan)
                result["xgb_d"] = p.get("1", np.nan)
                result["xgb_a"] = p.get("2", np.nan)
            elif name == "ou25":
                result["xgb_ou25"] = p.get("1", np.nan)
            elif name == "gg":
                result["xgb_gg"] = p.get("1", np.nan)
        return result


# ─────────────────────────────────────────────
# 3. META-MODELLO ENSEMBLE
# ─────────────────────────────────────────────

class EnsembleModel:
    """
    Combina Poisson + XGBoost con pesi ottimizzati (logistic regression).
    Le probabilità finali sono calibrate e sommate a 1.
    """

    def __init__(self):
        self.poisson = PoissonModel()
        self.xgb = XGBModel()
        self.meta_1x2: Optional[LogisticRegression] = None
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, halflife_days: int = 90):
        # Fit modelli base
        print("  → Training Poisson model ...")
        self.poisson.fit(df, halflife_days=halflife_days)
        print("  → Training XGBoost model ...")
        self.xgb.fit(df)

        # Genera predizioni in-sample per il meta-modello
        print("  → Training meta-model ...")
        rows = []
        for _, row in df.iterrows():
            p = self.poisson.predict_proba(row["HomeTeam"], row["AwayTeam"])
            x = self.xgb.predict_proba(row)
            if not p or not x:
                continue
            rows.append({
                "pois_h": p.get("poisson_h", np.nan),
                "pois_d": p.get("poisson_d", np.nan),
                "pois_a": p.get("poisson_a", np.nan),
                "xgb_h":  x.get("xgb_h", np.nan),
                "xgb_d":  x.get("xgb_d", np.nan),
                "xgb_a":  x.get("xgb_a", np.nan),
                "target": row.get("target_1x2", np.nan),
            })

        meta_df = pd.DataFrame(rows).dropna()
        if len(meta_df) > 50:
            X_meta = meta_df[["pois_h", "pois_d", "pois_a", "xgb_h", "xgb_d", "xgb_a"]].values
            y_meta = meta_df["target"].astype(int).values
            self.meta_1x2 = LogisticRegression(C=1.0, max_iter=500, random_state=42)
            self.meta_1x2.fit(X_meta, y_meta)

        self.is_fitted = True
        return self

    def predict(self, home: str, away: str, features_row: pd.Series) -> dict:
        """Restituisce probabilità finali per tutti i mercati."""
        p = self.poisson.predict_proba(home, away)
        x = self.xgb.predict_proba(features_row)

        # 1X2 ensemble
        # Media pesata: 70% Poisson (piu stabile) + 30% GBM
        prob_h = 0.7 * p.get("poisson_h", 0.4) + 0.3 * x.get("xgb_h", p.get("poisson_h", 0.4))
        prob_d = 0.7 * p.get("poisson_d", 0.25) + 0.3 * x.get("xgb_d", p.get("poisson_d", 0.25))
        prob_a = 0.7 * p.get("poisson_a", 0.35) + 0.3 * x.get("xgb_a", p.get("poisson_a", 0.35))
        total = prob_h + prob_d + prob_a
        prob_h, prob_d, prob_a = prob_h/total, prob_d/total, prob_a/total

        # OU/GG: media Poisson + XGB
        prob_ou25 = (p.get("poisson_ou25", 0) + x.get("xgb_ou25", 0)) / 2
        prob_gg   = (p.get("poisson_gg", 0)   + x.get("xgb_gg", 0))   / 2

        return {
            "prob_H": prob_h, "prob_D": prob_d, "prob_A": prob_a,
            "prob_over25": prob_ou25, "prob_gg": prob_gg,
            # Dettaglio per debug
            "_poisson_h": p.get("poisson_h"), "_xgb_h": x.get("xgb_h"),
        }


# ─────────────────────────────────────────────
# 4. VALUE BET ENGINE
# ─────────────────────────────────────────────

def calculate_edge(prob_model: float, odds: float) -> float:
    """Edge = prob_modello × quota - 1. Positivo = value bet."""
    if pd.isna(prob_model) or pd.isna(odds) or odds <= 1:
        return np.nan
    return prob_model * odds - 1


def kelly_fraction(prob: float, odds: float, fraction: float = 0.25) -> float:
    """
    Kelly Criterion frazionato (default ¼ Kelly per gestione rischio).
    Restituisce la % del bankroll da puntare.
    """
    b = odds - 1  # profitto netto per unità puntata
    q = 1 - prob
    kelly = (b * prob - q) / b
    return max(0, kelly * fraction)


def find_value_bets(
    predictions: dict,
    odds_dict: dict,
    min_edge: float = 0.05,
    bankroll: float = 1000.0,
) -> list:
    """
    Confronta le probabilità del modello con le quote bookmaker
    e restituisce le giocate con edge positivo.

    predictions: {"prob_H": .., "prob_D": .., "prob_A": .., "prob_over25": .., "prob_gg": ..}
    odds_dict:   {"H": 2.1, "D": 3.3, "A": 3.8, "over25": 1.85, "gg": 1.90}
    """
    markets = [
        ("H",       "prob_H",      "1 (vittoria casa)"),
        ("D",       "prob_D",      "X (pareggio)"),
        ("A",       "prob_A",      "2 (vittoria ospite)"),
        ("over25",  "prob_over25", "Over 2.5"),
        ("gg",      "prob_gg",     "Goal/Goal"),
    ]

    value_bets = []
    for mkt_key, prob_key, label in markets:
        odds = odds_dict.get(mkt_key)
        prob = predictions.get(prob_key)
        if odds is None or prob is None:
            continue
        edge = calculate_edge(prob, odds)
        if pd.isna(edge) or edge < min_edge:
            continue
        stake_pct = kelly_fraction(prob, odds)
        value_bets.append({
            "mercato":    label,
            "quota":      round(odds, 2),
            "prob_modello": round(prob * 100, 1),
            "prob_implicita": round(100 / odds, 1),
            "edge_%":     round(edge * 100, 1),
            "kelly_pct":  round(stake_pct * 100, 2),
            "stake_€":    round(stake_pct * bankroll, 2),
        })

    return sorted(value_bets, key=lambda x: x["edge_%"], reverse=True)


# ─────────────────────────────────────────────
# 5. BACKTESTING
# ─────────────────────────────────────────────

def backtest(model: EnsembleModel, df: pd.DataFrame, min_edge: float = 0.05) -> pd.DataFrame:
    """
    Walk-forward backtest: addestra sul passato, predice il futuro.
    Usa TimeSeriesSplit per rispettare l'ordine temporale.
    """
    from models.features import get_feature_columns
    feature_cols = [c for c in get_feature_columns() if c in df.columns]
    df = df.dropna(subset=["FTR"]).copy()

    results = []
    tscv = TimeSeriesSplit(n_splits=4)
    indices = list(range(len(df)))

    for fold, (train_idx, test_idx) in enumerate(tscv.split(indices)):
        train_df = df.iloc[train_idx]
        test_df  = df.iloc[test_idx]

        fold_model = EnsembleModel()
        fold_model.fit(train_df)

        for _, row in test_df.iterrows():
            feat_row = row.reindex(feature_cols).fillna(0)
            preds = fold_model.predict(row["HomeTeam"], row["AwayTeam"], feat_row)

            # Raccolta risultati per metriche
            actual_1x2 = row.get("FTR", "")
            total_gol  = row.get("FTHG", 0) + row.get("FTAG", 0)
            actual_ou  = int(total_gol > 2.5)
            actual_gg  = int(row.get("FTHG", 0) > 0 and row.get("FTAG", 0) > 0)

            results.append({
                "fold": fold,
                "date": row.get("Date"),
                "home": row["HomeTeam"],
                "away": row["AwayTeam"],
                "actual_1x2": actual_1x2,
                "actual_ou":  actual_ou,
                "actual_gg":  actual_gg,
                "prob_H":  preds.get("prob_H", np.nan),
                "prob_D":  preds.get("prob_D", np.nan),
                "prob_A":  preds.get("prob_A", np.nan),
                "prob_ou": preds.get("prob_over25", np.nan),
                "prob_gg": preds.get("prob_gg", np.nan),
                # Quote Pinnacle se disponibili
                "odds_H": row.get("PSH", row.get("B365H", np.nan)),
                "odds_D": row.get("PSD", row.get("B365D", np.nan)),
                "odds_A": row.get("PSA", row.get("B365A", np.nan)),
            })

    bt = pd.DataFrame(results)

    # Calcola metriche
    if len(bt) > 0:
        actual_1x2_enc = bt["actual_1x2"].map({"H": 0, "D": 1, "A": 2})
        valid = bt[["prob_H","prob_D","prob_A"]].notna().all(axis=1) & actual_1x2_enc.notna()
        if valid.sum() > 0:
            y_true_1x2 = pd.get_dummies(actual_1x2_enc[valid]).values
            y_pred_1x2 = bt.loc[valid, ["prob_H","prob_D","prob_A"]].values
            ll = log_loss(actual_1x2_enc[valid], y_pred_1x2)
            print(f"\n[backtest] Log-Loss 1X2: {ll:.4f}  (baseline ~1.09)")

        # ROI simulato sulle value bet
        roi_rows = []
        for _, r in bt.iterrows():
            for mkt, prob_col, odds_col, actual_fn in [
                ("H", "prob_H", "odds_H", lambda r: r["actual_1x2"] == "H"),
                ("D", "prob_D", "odds_D", lambda r: r["actual_1x2"] == "D"),
                ("A", "prob_A", "odds_A", lambda r: r["actual_1x2"] == "A"),
            ]:
                edge = calculate_edge(r[prob_col], r[odds_col])
                if pd.isna(edge) or edge < min_edge:
                    continue
                stake = kelly_fraction(r[prob_col], r[odds_col])
                won = actual_fn(r)
                profit = stake * (r[odds_col] - 1) if won else -stake
                roi_rows.append({"mercato": mkt, "stake": stake, "profit": profit})

        if roi_rows:
            roi_df = pd.DataFrame(roi_rows)
            total_stake = roi_df["stake"].sum()
            total_profit = roi_df["profit"].sum()
            roi = total_profit / total_stake * 100 if total_stake > 0 else 0
            n_bets = len(roi_df)
            print(f"[backtest] Value bet: {n_bets} giocate, ROI: {roi:.1f}%")
            print(f"           (stake medio: {roi_df['stake'].mean()*100:.2f}% bankroll)")

    return bt


# ─────────────────────────────────────────────
# DEMO RAPIDO
# ─────────────────────────────────────────────

from typing import Optional  # fix import order

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from models.features import build_features
    import random
    random.seed(0)

    teams = ["Juventus","Inter","Milan","Napoli","Roma","Lazio","Atalanta","Fiorentina"]
    rows = []
    for i in range(300):
        h, a = random.sample(teams, 2)
        gh, ga = np.random.poisson(1.4), np.random.poisson(1.1)
        ftr = "H" if gh > ga else ("A" if ga > gh else "D")
        rows.append({
            "Date": pd.Timestamp("2021-08-01") + pd.Timedelta(days=i*3),
            "HomeTeam": h, "AwayTeam": a,
            "FTHG": gh, "FTAG": ga, "FTR": ftr,
            "HST": random.randint(2,8), "AST": random.randint(1,6),
            "HS": random.randint(5,18), "AS": random.randint(3,14),
            "HC": random.randint(2,10), "AC": random.randint(1,8),
            "xg_home": round(np.random.gamma(2, 0.7), 2),
            "xg_away": round(np.random.gamma(1.5, 0.7), 2),
            "elo_home": random.randint(1550, 1850),
            "elo_away": random.randint(1550, 1850),
            "elo_diff": 0,
            "PSH": round(random.uniform(1.5,4.0),2),
            "PSD": round(random.uniform(2.8,4.2),2),
            "PSA": round(random.uniform(1.8,5.0),2),
            "season": "test",
        })
    df_raw = pd.DataFrame(rows)
    df_raw["elo_diff"] = df_raw["elo_home"] - df_raw["elo_away"]

    df = build_features(df_raw, verbose=False)

    model = EnsembleModel()
    model.fit(df.iloc[:200])

    # Predizione esempio
    feat_row = df.iloc[200].reindex(get_feature_columns()).fillna(0)
    preds = model.predict("Juventus", "Inter", feat_row)
    print("\n=== Predizione Juventus vs Inter ===")
    for k, v in preds.items():
        if not k.startswith("_") and v is not None:
            print(f"  {k:15s}: {v:.3f}")

    # Value bet
    odds = {"H": 2.20, "D": 3.40, "A": 3.10, "over25": 1.85, "gg": 1.92}
    vb = find_value_bets(preds, odds, min_edge=0.03, bankroll=1000)
    if vb:
        print("\n=== Value Bets trovate ===")
        for b in vb:
            print(f"  {b['mercato']:20s} quota:{b['quota']}  edge:{b['edge_%']}%  stake:€{b['stake_€']}")
    else:
        print("\nNessuna value bet con edge sufficiente")
