"""
Trova gli id_aggregata Marathonbet per i mercati mancanti
(2 Tempo, 1X2 Primo Tempo, Corner per squadra).

USO:
1. Avvia il backend come al solito (in un altro terminale):
   cd ~/Desktop/football_predictor && python3 main.py
   (o come lo avvii di solito - deve rispondere su localhost:8000)
2. In un terminale nuovo, lancia questo script:
   cd ~/Desktop/football_predictor && python3 utils/find_marathonbet_ids.py
3. Copia l'output e mandalo a Claude in chat.
"""
import requests

def _walk(items, depth=0):
    for it in items:
        pad = "  " * depth
        print(f"{pad}- {it.get('name','?')}  (id={it.get('id','?')})")
        sub = it.get("submenu") or []
        if sub:
            _walk(sub, depth + 1)

if __name__ == "__main__":
    try:
        r = requests.get("http://localhost:8000/marathonbet/serie-a-bet/menu", timeout=15)
        r.raise_for_status()
        data = r.json()
        print("=== MENU MERCATI MARATHONBET SERIE A ===\n")
        items = data if isinstance(data, list) else data.get("items", data)
        _walk(items)
        print("\n=== FINE ===")
        print("Cerca nell'elenco sopra le voci tipo '2 Tempo', 'Primo Tempo 1X2', 'Corner squadra'")
        print("e mandami il loro (id=...) in chat.")
    except Exception as e:
        print(f"Errore: {e}")
        print("Controlla che il backend sia acceso su localhost:8000")
