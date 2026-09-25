"""
Riaggiorna cache/rosters_2627.json (rosa ufficiale di ogni squadra Serie A)
scaricando la rosa corrente da Sofascore tramite il backend locale.

Perché serve: cache/rosters_2627.json era stato generato una volta sola a
mano ed è rimasto congelato alla formazione della giornata 3-4 — i
trasferimenti successivi (mercato di gennaio, svincolati, nuovi arrivi)
non vengono mai recepiti, quindi models/player_stats.py può continuare a
considerare "in rosa" un giocatore che ha cambiato squadra, o ignorare un
nuovo acquisto.

Uso:
    python3 utils/update_rosters.py

Richiede il backend locale attivo (uvicorn, la stessa API che la dashboard
usa per Marathonbet/Sofascore) perché è lui ad avere accesso a internet;
questo script parla solo con http://localhost:8000.

Integrato anche in utils/weekly_update.py, così la rosa si riallinea da
sola a ogni aggiornamento settimanale, senza bisogno di ricordarsene.
"""
import json
import sys
from pathlib import Path

import requests

BASE_DIR = str(Path(__file__).parent.parent)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ID squadra Sofascore (stabili stagione su stagione per lo stesso club) →
# nome breve usato ovunque nel progetto (stesso elenco di SERIE_A_2627_TEAMS
# in dashboard.py). Estratti da cache/sofascore_team_stats.json, dove i nomi
# sono già nella forma breve tranne le 3 eccezioni qui sotto.
TEAM_IDS = {
    "Milan": 2692,       # "AC Milan" su Sofascore
    "Roma": 2702,        # "AS Roma" su Sofascore
    "Atalanta": 2686,
    "Bologna": 2685,
    "Cagliari": 2719,
    "Como": 2704,
    "Fiorentina": 2693,
    "Frosinone": 2801,
    "Genoa": 2713,
    "Inter": 2697,
    "Juventus": 2687,
    "Lazio": 2699,
    "Lecce": 2689,
    "Monza": 2729,
    "Parma": 2690,
    "Napoli": 2714,      # "SSC Napoli" su Sofascore
    "Sassuolo": 2793,
    "Torino": 2696,
    "Udinese": 2695,
    "Venezia": 2688,
}

ROSTERS_PATH = Path(BASE_DIR) / "cache" / "rosters_2627.json"
BACKEND_URL = "http://localhost:8000/sofascore/serie-a/squad"


def _extract_player(entry: dict) -> dict | None:
    """
    Estrae un giocatore da un elemento della risposta Sofascore
    /team/{id}/players. La struttura esatta (se jerseyNumber/position/height
    stanno dentro "player" o a fianco) non è stata ancora verificata con una
    risposta reale in questa sessione (nessun accesso a internet da qui) —
    per questo si prova prima dentro "player" e poi, come ripiego, allo
    stesso livello di "player" nell'elemento stesso, così lo script non
    scrive silenziosamente dati vuoti se la forma reale è leggermente
    diversa da quella attesa.
    """
    p = entry.get("player") if isinstance(entry.get("player"), dict) else entry
    pid = p.get("id")
    name = p.get("name")
    if not pid or not name:
        return None
    position = p.get("position") or entry.get("position") or ""
    height = p.get("height") or entry.get("height")
    jersey = p.get("jerseyNumber") or entry.get("jerseyNumber") or ""
    return {
        "name": name,
        "position": position,
        "height": height,
        "jerseyNumber": str(jersey) if jersey != "" else "",
        "id": pid,
    }


def update_rosters(verbose: bool = True) -> dict:
    """Scarica la rosa corrente di ogni squadra e riscrive rosters_2627.json.

    Ritorna il nuovo dizionario {squadra: {player_id: {...}}}. Se una
    squadra fallisce (rete, backend giù, risposta inattesa) mantiene la
    rosa precedentemente salvata per quella squadra invece di svuotarla,
    così un errore temporaneo non cancella dati buoni.
    """
    rosters = {}
    if ROSTERS_PATH.exists():
        try:
            rosters = json.loads(ROSTERS_PATH.read_text(encoding="utf-8"))
        except Exception:
            rosters = {}

    ok_teams, failed_teams = [], []

    for team, team_id in TEAM_IDS.items():
        try:
            resp = requests.get(BACKEND_URL, params={"team_id": team_id}, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            entries = data.get("players", data if isinstance(data, list) else [])
            squad = {}
            for entry in entries:
                player = _extract_player(entry)
                if player:
                    squad[str(player["id"])] = player
            if not squad:
                raise ValueError("risposta senza giocatori riconoscibili")
            rosters[team] = squad
            ok_teams.append((team, len(squad)))
            if verbose:
                print(f"  OK {team}: {len(squad)} giocatori")
        except Exception as e:
            failed_teams.append((team, str(e)))
            if verbose:
                print(f"  [WARN] {team}: {e} — mantengo la rosa precedente")

    ROSTERS_PATH.write_text(
        json.dumps(rosters, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if verbose:
        print(f"\nRose aggiornate: {len(ok_teams)}/{len(TEAM_IDS)} squadre "
              f"({sum(n for _, n in ok_teams)} giocatori totali)")
        if failed_teams:
            print(f"Squadre non aggiornate (rosa precedente mantenuta): "
                  f"{', '.join(t for t, _ in failed_teams)}")

    return rosters


if __name__ == "__main__":
    update_rosters()
