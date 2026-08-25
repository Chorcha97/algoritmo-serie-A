"""
MAIN — Pipeline completa Serie A
Uso:
    python main.py                      # scarica dati, addestra, analizza
    python main.py --predict "Juventus" "Inter"  --odds 2.20 3.40 3.10
    python main.py --backtest
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Fix import path
sys.path.insert(0, str(Path(__file__).parent))

from data.pipeline import build_dataset
from models.features import build_features, get_feature_columns
from models.model import EnsembleModel, find_value_bets, backtest


def print_banner():
    print("""
╔══════════════════════════════════════════════╗
║   SERIE A PREDICTION SYSTEM                 ║
║   Poisson + XGBoost + Kelly Value Bet       ║
╚══════════════════════════════════════════════╝
""")


def load_or_build(dataset_path: str = "serie_a_dataset.csv") -> pd.DataFrame:
    if Path(dataset_path).exists():
        print(f"[✓] Dataset trovato: {dataset_path}")
        df_raw = pd.read_csv(dataset_path, parse_dates=["Date"])
    else:
        print("[→] Dataset non trovato, avvio download ...")
        df_raw = build_dataset(dataset_path)
    return df_raw


def run_predict(model: EnsembleModel, df: pd.DataFrame,
                home: str, away: str, odds_input: list = None):
    """Predice una singola partita e mostra le value bet."""
    feat_cols = get_feature_columns()
    # Usa l'ultima riga disponibile per le feature (approssimazione senza roster live)
    feat_row = df.iloc[-1].reindex(feat_cols).fillna(0)

    from models.markets import compute_all_markets
    preds = compute_all_markets(model.poisson, home, away)

    print(f"\n{'═'*50}")
    print(f"  {home}  vs  {away}")
    print(f"{'═'*50}")
    print(f"  Vittoria {home:<20s}: {preds['prob_H']*100:.1f}%")
    print(f"  Pareggio {'X':<20s}: {preds['prob_D']*100:.1f}%")
    print(f"  Vittoria {away:<20s}: {preds['prob_A']*100:.1f}%")
    print(f"  Over 2.5             : {preds['prob_over25']*100:.1f}%")
    print(f"  Goal/Goal            : {preds['prob_gg']*100:.1f}%")
    print(f"{'─'*50}")

    if odds_input and len(odds_input) >= 3:
        odds_dict = {
            "H": odds_input[0], "D": odds_input[1], "A": odds_input[2]
        }
        if len(odds_input) >= 4: odds_dict["over25"] = odds_input[3]
        if len(odds_input) >= 5: odds_dict["gg"]     = odds_input[4]

        vb = find_value_bets(preds, odds_dict, min_edge=0.04)
        if vb:
            print("\n  VALUE BET TROVATE:")
            for b in vb:
                star = "★" if b["edge_%"] >= 10 else " "
                print(f"  {star} {b['mercato']:<22s} "
                      f"quota: {b['quota']}  "
                      f"edge: +{b['edge_%']}%  "
                      f"prob modello: {b['prob_modello']}%  "
                      f"stake: €{b['stake_€']}")
        else:
            print("\n  Nessuna value bet significativa (edge < 4%)")
    else:
        print("\n  [tip] Aggiungi le quote con --odds Q_H Q_D Q_A")
    print()


def main():
    print_banner()
    parser = argparse.ArgumentParser(description="Serie A Prediction System")
    parser.add_argument("--predict", nargs=2, metavar=("HOME", "AWAY"),
                        help="Predici una partita. Es: --predict Juventus Inter")
    parser.add_argument("--odds", nargs="+", type=float,
                        metavar="Q",
                        help="Quote bookmaker: H D A [over25] [gg]")
    parser.add_argument("--backtest", action="store_true",
                        help="Esegui walk-forward backtest")
    parser.add_argument("--dataset", default="serie_a_dataset.csv",
                        help="Percorso file dataset")
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="Bankroll per calcolo stake Kelly (default: 1000)")
    args = parser.parse_args()

    # 1. Carica / scarica dati
    df_raw = load_or_build(args.dataset)
    if df_raw.empty:
        print("[errore] Dataset vuoto. Controlla la connessione internet.")
        sys.exit(1)

    # 2. Feature engineering
    print("[→] Feature engineering ...")
    df = build_features(df_raw)

    # 3. Training
    print("[→] Training modello ensemble ...")
    model = EnsembleModel()
    model.fit(df)
    print("[✓] Modello pronto\n")

    # 4. Azioni
    if args.predict:
        run_predict(model, df, args.predict[0], args.predict[1], args.odds)
    elif args.backtest:
        print("[→] Backtesting walk-forward ...")
        bt = backtest(model, df)
        bt.to_csv("backtest_results.csv", index=False)
        print("[✓] Risultati salvati in: backtest_results.csv")
    else:
        # Modalità interattiva
        print("Squadre disponibili nel dataset:")
        teams = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))
        for i, t in enumerate(teams):
            print(f"  {i+1:2d}. {t}")

        print("\nEsempio d'uso:")
        print("  python main.py --predict Juventus Inter --odds 2.20 3.40 3.10 1.85 1.90")
        print("  python main.py --backtest")

        # Demo automatica
        print("\n--- DEMO: Juventus vs Inter ---")
        if "Juventus" in teams and "Inter" in teams:
            run_predict(model, df, "Juventus", "Inter",
                        odds_input=[2.20, 3.40, 3.10, 1.85, 1.90])


if __name__ == "__main__":
    main()
