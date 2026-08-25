"""
Bot Telegram — Alert Value Bet Serie A
"""

import requests
from datetime import datetime

TOKEN   = "8984001199:AAEuWQ-yp0rKbVUFY7yeLWXhsVKff9f_Lis"
CHAT_ID = "744310830"


def send_message(text: str) -> bool:
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"Errore Telegram: {e}")
        return False


def send_value_bet_alert(home, away, kickoff, value_bets, drift_info=None):
    if not value_bets:
        return
    ora = datetime.now().strftime("%H:%M")
    msg = f"⚽ <b>{home} vs {away}</b>\n🕐 {kickoff} | Alert: {ora}\n"
    msg += "─" * 30 + "\n"
    for vb in value_bets:
        edge = vb.get("edge_%", 0)
        star = "⭐" if edge >= 20 else "✅"
        correlato = " ⚠️ correlato" if vb.get("correlato") else ""
        msg += f"\n{star} <b>{vb['mercato']}</b>{correlato}\n"
        msg += f"   Quota: <b>{vb['quota']}</b> | Edge: <b>+{edge}%</b> | Stake: <b>€{vb['stake_€']}</b>\n"
        msg += f"   Prob.: {vb['prob_modello_%']}% | Implicita: {vb['prob_implicita_%']}%\n"
    if drift_info:
        msg += "\n📉 <b>Movimento quote:</b>\n"
        for mkt, drift in drift_info.items():
            if drift and abs(drift) >= 0.05:
                direzione = "↓" if drift < 0 else "↑"
                msg += f"   {mkt}: {direzione} {drift:+.2f}\n"
    msg += "\n<i>Modello Serie A — edge min 15%</i>"
    send_message(msg)


def send_daily_summary(results, bankroll=100.0):
    if not results:
        send_message("📊 <b>Nessuna value bet trovata oggi</b>")
        return
    today = datetime.now().strftime("%d/%m/%Y")
    msg = f"📊 <b>Riepilogo giocate — {today}</b>\n"
    msg += f"Bankroll: €{bankroll} | Giocate: {len(results)}\n"
    msg += "─" * 30 + "\n"
    tot_stake = sum(r.get("stake_€", 0) for r in results)
    msg += f"\n💰 Stake totale: €{tot_stake:.2f}\n\n"
    for r in results:
        edge = r.get("edge_%", 0)
        star = "⭐" if edge >= 20 else "✅"
        msg += f"{star} {r['partita']}\n"
        msg += f"   {r['mercato']} @ {r['quota']} | +{edge}% | €{r['stake_€']}\n"
    send_message(msg)


def send_test():
    msg = (
        "🤖 <b>Bot Serie A Predictor attivo!</b>\n\n"
        "Riceverai notifiche quando il modello trova value bet.\n\n"
        "Mercati monitorati:\n"
        "✅ Under 2.5 Gol\n"
        "✅ Over 2.5 Gol\n"
        "✅ X — Pareggio\n\n"
        "<i>Edge minimo: 15% | Kelly: 1/8</i>"
    )
    return send_message(msg)


if __name__ == "__main__":
    print("Test bot Telegram...")
    ok = send_test()
    print("Messaggio inviato!" if ok else "Errore invio")
