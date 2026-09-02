"""
Alert Runner — gira ogni 30 minuti
Controlla le partite in programma e manda alert su Telegram
quando trova value bet con edge > 15%
"""

import sys, json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.odds_tracker import fetch_odds, parse_match_odds, load_opening_odds, compute_drift, SPORTS, normalize_team_name
from models.model import EnsembleModel
from models.features import build_features
from models.markets import compute_all_markets, find_value_bets_extended
from models.referee import compute_cards_markets, get_referee_features
from utils.telegram_bot import send_value_bet_alert, send_daily_summary
import pickle

MIN_EDGE    = 0.15
BANKROLL    = 100.0
ALERT_LOG   = Path("cache/alert_log.json")


def load_model():
    model_path = Path("model_cache.pkl")
    if model_path.exists():
        with open(model_path, "rb") as f:
            return pickle.load(f)
    df_raw = pd.read_csv("serie_a_dataset.csv", parse_dates=["Date"])
    df = build_features(df_raw, verbose=False)
    model = EnsembleModel()
    model.fit(df)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model


def load_alert_log() -> set:
    """Tiene traccia degli alert già inviati per non duplicarli."""
    if ALERT_LOG.exists():
        with open(ALERT_LOG) as f:
            return set(json.load(f))
    return set()


def save_alert_log(sent: set):
    with open(ALERT_LOG, "w") as f:
        json.dump(list(sent), f)


def run_alerts(sport: str = "serie_a", hours_ahead: int = 96):
    """
    Controlla le partite nelle prossime X ore e manda alert.
    """
    print(f"\n=== ALERT RUNNER — {datetime.now().strftime('%d/%m %H:%M')} ===")

    sport_key = SPORTS.get(sport)
    if not sport_key:
        print(f"Sport non trovato: {sport}")
        return

    # Carica modello
    print("Carico modello...")
    model = load_model()

    # Carica log alert già inviati
    sent_alerts = load_alert_log()

    # Scarica quote attuali
    print("Scarico quote...")
    events = fetch_odds(sport_key)
    if not events:
        print("Nessuna partita trovata")
        return

    print(f"Partite trovate: {len(events)}")
    now = datetime.utcnow()
    alerts_sent = 0

    for event in events:
        # Controlla solo partite nelle prossime X ore
        try:
            kickoff = pd.Timestamp(event.get("commence_time"))
            kickoff_naive = kickoff.tz_localize(None) if kickoff.tzinfo else kickoff
            hours_to_kickoff = (kickoff_naive - now).total_seconds() / 3600
            if hours_to_kickoff < 0 or hours_to_kickoff > hours_ahead:
                continue
        except:
            continue

        home = normalize_team_name(event.get("home_team", ""))
        away = normalize_team_name(event.get("away_team", ""))
        match_id = event.get("id")

        print(f"\n  {home} vs {away} (tra {hours_to_kickoff:.1f}h)")

        # Calcola probabilità modello
        preds = compute_all_markets(model.poisson, home, away)
        if not preds:
            print(f"    Squadre non nel modello")
            continue

        # Aggiorna con modello cartellini
        ref_features = get_referee_features(pd.DataFrame(), "")
        cards_preds = compute_cards_markets(preds["prob_D"], preds["prob_over25"], ref_features)
        preds.update(cards_preds)

        # Estrai quote
        current = parse_match_odds(event, sport)
        odds_dict = {
            k: current[k] for k in ["H","D","A","over25","under25"]
            if current.get(k)
        }

        if not odds_dict:
            print(f"    Nessuna quota disponibile")
            continue

        # Trova value bet
        vbs = find_value_bets_extended(
            preds, odds_dict,
            min_edge=MIN_EDGE,
            bankroll=BANKROLL,
            match_date=kickoff
        )

        if not vbs:
            print(f"    Nessuna value bet (edge < {MIN_EDGE*100:.0f}%)")
            continue

        # Calcola drift quote
        opening = load_opening_odds(match_id)
        drift = compute_drift(opening, current) if opening else {}
        drift_filtered = {
            k.replace("drift_","").replace("_pct","%"): v
            for k, v in drift.items()
            if "_pct" in k and v and abs(v) >= 5
        }

        # Chiave univoca per evitare duplicati
        alert_key = f"{match_id}_{','.join(v['mercato'] for v in vbs)}"
        if alert_key in sent_alerts:
            print(f"    Alert già inviato")
            continue

        # Invia alert
        kickoff_str = kickoff.strftime("%d/%m %H:%M")
        print(f"    ✅ {len(vbs)} value bet trovate — invio alert")
        for vb in vbs:
            print(f"       {vb['mercato']} @ {vb['quota']} edge +{vb['edge_%']}%")

        send_value_bet_alert(home, away, kickoff_str, vbs, drift_filtered)
        sent_alerts.add(alert_key)
        alerts_sent += 1

    save_alert_log(sent_alerts)
    print(f"\nAlert inviati: {alerts_sent}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default="serie_a")
    parser.add_argument("--hours", type=int, default=96)
    parser.add_argument("--mondiale", action="store_true")
    args = parser.parse_args()

    sport = "mondiale" if args.mondiale else args.sport
    run_alerts(sport=sport, hours_ahead=args.hours)
