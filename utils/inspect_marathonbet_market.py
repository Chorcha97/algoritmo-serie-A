"""
Ispeziona la struttura grezza di un mercato Marathonbet per una partita specifica.
Serve per capire come mappare correttamente le quote (Claude lo usa per scrivere
il codice di estrazione senza indovinare).

USO:
1. Avvia il backend come al solito.
2. python3 utils/inspect_marathonbet_market.py <id_aggregata> <squadra_casa> <squadra_ospite>

Esempio:
   python3 utils/inspect_marathonbet_market.py 341 Inter Juventus
   python3 utils/inspect_marathonbet_market.py 362 Inter Juventus
   python3 utils/inspect_marathonbet_market.py 2719 Inter Juventus

Copia TUTTO l'output (anche se lungo) e mandalo a Claude in chat.
"""
import sys, json, requests

def _find_match(avs, h, a):
    for p in avs:
        n = p.get("dsl", {}).get("IT", "").upper()
        if any(x in n for x in [h[:4].upper(), h[:5].upper()]) and any(x in n for x in [a[:4].upper(), a[:5].upper()]):
            return p
    return None

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Uso: python3 utils/inspect_marathonbet_market.py <id_aggregata> <squadra_casa> <squadra_ospite>")
        sys.exit(1)
    id_agg, home, away = sys.argv[1], sys.argv[2], sys.argv[3]
    try:
        r = requests.get(f"http://localhost:8000/marathonbet/serie-a-bet/pre-match/eventi?id_aggregata={id_agg}", timeout=15)
        r.raise_for_status()
        data = r.json()
        p = _find_match(data.get("avs", []), home, away)
        if not p:
            print(f"Partita {home}-{away} non trovata su id_aggregata={id_agg}.")
            print("Partite disponibili:", [x.get("dsl", {}).get("IT") for x in data.get("avs", [])][:10])
            sys.exit(1)
        print(f"=== id_aggregata={id_agg} — {home} vs {away} ===\n")
        print(json.dumps(p.get("scs", []), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Errore: {e}")
