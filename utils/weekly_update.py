"""
Aggiornamento settimanale automatico
- Scarica nuovi risultati da football-data.co.uk
- Aggiorna il dataset
- Rigenera il model cache
- Aggiorna i valori Transfermarkt (inizio stagione)
- Invia riepilogo su Telegram
"""

import sys, os, pickle
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.pipeline import build_dataset
from models.features import build_features
from models.model import EnsembleModel
from models.referee import compute_referee_stats
from utils.telegram_bot import send_message


def run_weekly_update():
    print(f"\n=== AGGIORNAMENTO SETTIMANALE — {datetime.now().strftime('%d/%m/%Y %H:%M')} ===")
    
    # 1. Cancella cache FDCO per forzare download fresco
    print("\n[1/4] Aggiorno dati football-data.co.uk...")
    cache_dir = Path("cache")
    fdco_files = list(cache_dir.glob("fdco_*.csv"))
    for f in fdco_files:
        f.unlink()
    print(f"  Cancellati {len(fdco_files)} file cache")

    # 2. Riscari il dataset completo
    print("\n[2/4] Scarico dataset aggiornato...")
    try:
        dataset_path = "serie_a_dataset.csv"
        if Path(dataset_path).exists():
            Path(dataset_path).unlink()
        df_raw = build_dataset(dataset_path)
        n_partite = len(df_raw)
        print(f"  Dataset aggiornato: {n_partite} partite")
    except Exception as e:
        print(f"  Errore download: {e}")
        send_message(f"⚠️ Errore aggiornamento settimanale: {e}")
        return

    # 3. Rigenera features e modello
    print("\n[3/4] Rigenero modello...")
    try:
        df = build_features(df_raw, verbose=False)
        model = EnsembleModel()
        model.fit(df)
        with open("model_cache.pkl", "wb") as f:
            pickle.dump(model, f)
        print("  Modello rigenerato e salvato")
    except Exception as e:
        print(f"  Errore training: {e}")
        send_message(f"⚠️ Errore training modello: {e}")
        return

    # 4. Aggiorna classifica da football-data.org
    print("\n[3c/4] Aggiorno classifica Serie A...")
    update_standings_from_api()

    # 5. Aggiorna Transfermarkt se siamo a luglio/agosto (mercato estivo)
    now = datetime.now()
    if now.month in [7, 8]:
        print("\n[3b/4] Aggiorno valori Transfermarkt...")
        try:
            import asyncio, re, json
            from playwright.async_api import async_playwright

            async def scrape_tm():
                leagues = {
                    "Serie A":       "https://www.transfermarkt.it/serie-a/startseite/wettbewerb/IT1",
                    "Premier League":"https://www.transfermarkt.it/premier-league/startseite/wettbewerb/GB1",
                    "Bundesliga":    "https://www.transfermarkt.it/bundesliga/startseite/wettbewerb/L1",
                }
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    all_teams = {}
                    for league_name, url in leagues.items():
                        page = await browser.new_page(extra_http_headers={
                            "User-Agent": "Mozilla/5.0"})
                        await page.goto(url, timeout=30000)
                        await asyncio.sleep(3)
                        body = await page.inner_text("body")
                        for line in body.split("\n"):
                            m = re.match(r"^(.+?)	\d+	[\d,]+	\d+	[\d,.]+ mln €	([\d,.]+) mln €", line.strip())
                            if m:
                                team = m.group(1).strip()
                                value = float(m.group(2).replace(".","").replace(",","."))
                                all_teams[team] = value
                        await page.close()
                    await browser.close()
                    return all_teams

            teams = asyncio.run(scrape_tm())
            with open("cache/transfermarkt_values.json", "w") as f:
                json.dump(teams, f, ensure_ascii=False, indent=2)
            print(f"  Aggiornate {len(teams)} squadre")
        except Exception as e:
            print(f"  Errore Transfermarkt: {e}")

    # 4. Statistiche aggiornamento
    print("\n[4/4] Invio riepilogo Telegram...")
    ultima = df_raw["Date"].max()
    stagione = df_raw["season"].value_counts().index[0] if "season" in df_raw.columns else "—"

    msg = (
        f"✅ <b>Aggiornamento settimanale completato</b>\n\n"
        f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
        f"⚽ Partite nel dataset: <b>{n_partite}</b>\n"
        f"📊 Stagione attiva: <b>{stagione}</b>\n"
        f"🗓 Ultima partita: <b>{pd.Timestamp(ultima).strftime('%d/%m/%Y')}</b>\n\n"
        f"Il modello è aggiornato e pronto per la prossima giornata."
    )
    send_message(msg)
    print("  Riepilogo inviato!")
    print("\n=== AGGIORNAMENTO COMPLETATO ===")


def update_standings_from_api():
    """
    Scarica la classifica aggiornata da football-data.org
    e la salva per il parametro motivazione.
    """
    import requests, json
    from pathlib import Path

    TOKEN = "8e7fc33124f84f528285986cece5a1bf"
    headers = {"X-Auth-Token": TOKEN}

    name_map = {
        "FC Internazionale Milano": "Inter",
        "Juventus FC": "Juventus",
        "AC Milan": "Milan",
        "Como 1907": "Como",
        "AS Roma": "Roma",
        "SSC Napoli": "Napoli",
        "Atalanta BC": "Atalanta",
        "ACF Fiorentina": "Fiorentina",
        "SS Lazio": "Lazio",
        "Bologna FC 1909": "Bologna",
        "US Sassuolo Calcio": "Sassuolo",
        "Parma Calcio 1913": "Parma",
        "Genoa CFC": "Genoa",
        "Udinese Calcio": "Udinese",
        "Torino FC": "Torino",
        "Cagliari Calcio": "Cagliari",
        "Venezia FC": "Venezia",
        "Frosinone Calcio": "Frosinone",
        "US Lecce": "Lecce",
        "AC Monza": "Monza",
    }

    try:
        resp = requests.get(
            "https://api.football-data.org/v4/competitions/SA/standings",
            headers=headers, timeout=15
        )
        data = resp.json()
        table = data.get("standings", [{}])[0].get("table", [])

        standings = {}
        for row in table:
            team_api = row["team"]["name"]
            team = name_map.get(team_api, team_api)
            standings[team] = {
                "position": row["position"],
                "pts":      row["points"],
                "g":        row["playedGames"],
                "w":        row["won"],
                "d":        row["draw"],
                "l":        row["lost"],
                "gf":       row["goalsFor"],
                "ga":       row["goalsAgainst"],
            }

        path = Path("cache/standings.json")
        with open(path, "w") as f:
            json.dump(standings, f, indent=2)

        print(f"  Classifica aggiornata: {len(standings)} squadre")
        if standings:
            top = min(standings.items(), key=lambda x: x[1]["position"])
            print(f"  1° posto: {top[0]} ({top[1]['pts']} pts)")
        return standings

    except Exception as e:
        print(f"  Errore aggiornamento classifica: {e}")
        return {}


if __name__ == "__main__":
    run_weekly_update()
