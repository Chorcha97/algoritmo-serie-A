"""
Modulo statistiche arbitri Serie A
- Calcola medie storiche per arbitro dal dataset football-data.co.uk
- Scraper designazioni AIA per partite future
- Feature arbitro per il modello cartellini
"""

import pandas as pd
import numpy as np
import requests
import re
import time
from pathlib import Path

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# 1. STATISTICHE STORICHE ARBITRI
# ─────────────────────────────────────────────

def load_serie_a_referee_stats() -> dict:
    """Carica statistiche arbitri Serie A da Transfermarkt."""
    import json
    from pathlib import Path
    path = Path("cache/referee_stats_seria_a.json")
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_referee_adjustments(referee_name: str) -> dict:
    """
    Restituisce moltiplicatori cartellini per l'arbitro designato.
    Calibrato su dati reali Serie A 2021-2026.
    """
    stats = load_serie_a_referee_stats()
    media = stats.get("_media_serie_a", {"gialli_media": 4.18, "rossi_media": 0.144})

    if not referee_name or referee_name in ["Non specificato", "", "—"]:
        return {
            "cards_mult":    1.0,
            "gialli_attesi": media["gialli_media"],
            "rossi_attesi":  media["rossi_media"],
            "affidabile":    False,
        }

    ref_data = None
    for name, data in stats.items():
        if name.startswith("_"):
            continue
        if (referee_name.lower() == name.lower() or
            referee_name.lower() in name.lower() or
            name.lower() in referee_name.lower()):
            ref_data = data
            break

    if not ref_data:
        return {
            "cards_mult":    1.0,
            "gialli_attesi": media["gialli_media"],
            "rossi_attesi":  media["rossi_media"],
            "affidabile":    False,
        }

    cards_mult = ref_data["gialli_media"] / media["gialli_media"]
    return {
        "cards_mult":    round(cards_mult, 3),
        "gialli_attesi": ref_data["gialli_media"],
        "rossi_attesi":  ref_data["rossi_media"],
        "partite":       ref_data.get("partite", 0),
        "affidabile":    ref_data.get("partite", 0) >= 10,
    }


def compute_referee_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola le statistiche medie per arbitro dal dataset storico.
    Richiede colonne: Referee, HY, AY, HR, AR, FTHG, FTAG

    Restituisce un DataFrame con una riga per arbitro.
    """
    required = ["Referee", "HY", "AY"]
    if not all(c in df.columns for c in required):
        print("  [warning] Colonne arbitro non disponibili nel dataset")
        return pd.DataFrame()

    df = df.dropna(subset=["Referee"]).copy()
    df["total_yellow"]  = df["HY"].fillna(0) + df["AY"].fillna(0)
    df["total_red"]     = df.get("HR", pd.Series(0, index=df.index)).fillna(0) + \
                          df.get("AR", pd.Series(0, index=df.index)).fillna(0)
    df["total_goals"]   = df["FTHG"].fillna(0) + df["FTAG"].fillna(0)
    df["home_yellow"]   = df["HY"].fillna(0)
    df["away_yellow"]   = df["AY"].fillna(0)
    df["home_red"]      = df.get("HR", pd.Series(0, index=df.index)).fillna(0)
    df["away_red"]      = df.get("AR", pd.Series(0, index=df.index)).fillna(0)

    stats = df.groupby("Referee").agg(
        partite          = ("total_yellow", "count"),
        gialli_media     = ("total_yellow", "mean"),
        rossi_media      = ("total_red", "mean"),
        gialli_casa_avg  = ("home_yellow", "mean"),
        gialli_ospite_avg= ("away_yellow", "mean"),
        rossi_casa_avg   = ("home_red", "mean"),
        rossi_ospite_avg = ("away_red", "mean"),
        gol_media        = ("total_goals", "mean"),
    ).reset_index()

    # Calcola percentile severità (0-100, più alto = più severo)
    stats["severita_pct"] = (
        stats["gialli_media"].rank(pct=True) * 0.7 +
        stats["rossi_media"].rank(pct=True) * 0.3
    ) * 100

    # Bias casa/ospite: positivo = favorisce casa, negativo = favorisce ospite
    stats["bias_casa"] = stats["gialli_ospite_avg"] - stats["gialli_casa_avg"]

    stats = stats.sort_values("gialli_media", ascending=False).reset_index(drop=True)
    return stats


def get_referee_features(ref_stats: pd.DataFrame, referee: str) -> dict:
    """
    Restituisce le feature per un arbitro specifico.
    Se non trovato, usa la media del dataset.
    """
    if ref_stats.empty or not referee:
        return {
            "ref_gialli_media":    3.8,
            "ref_rossi_media":     0.18,
            "ref_severita_pct":    50.0,
            "ref_bias_casa":       0.0,
            "ref_found":           False,
        }

    # Cerca corrispondenza esatta o parziale
    match = ref_stats[ref_stats["Referee"].str.upper() == referee.upper()]
    if match.empty:
        match = ref_stats[ref_stats["Referee"].str.upper().str.contains(
            referee.upper().split()[-1], na=False
        )]

    if match.empty:
        # Usa media globale
        return {
            "ref_gialli_media":    float(ref_stats["gialli_media"].mean()),
            "ref_rossi_media":     float(ref_stats["rossi_media"].mean()),
            "ref_severita_pct":    50.0,
            "ref_bias_casa":       0.0,
            "ref_found":           False,
        }

    row = match.iloc[0]
    return {
        "ref_gialli_media":    float(row["gialli_media"]),
        "ref_rossi_media":     float(row["rossi_media"]),
        "ref_severita_pct":    float(row["severita_pct"]),
        "ref_bias_casa":       float(row["bias_casa"]),
        "ref_partite":         int(row["partite"]),
        "ref_found":           True,
    }


# ─────────────────────────────────────────────
# 2. SCRAPER DESIGNAZIONI AIA
# ─────────────────────────────────────────────

def fetch_aia_designations() -> list:
    """
    Scarica designazioni arbitrali AIA per la Serie A.
    Usa requests per trovare l URL e scaricare le designazioni.
    """
    import requests, re, json
    from pathlib import Path

    cache_path = Path("cache/aia_designations.json")
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
    }

    # Step 1: Trova ID giornata corrente dalla sitemap o dal feed RSS
    url = None
    try:
        # Prova sitemap
        resp = requests.get("https://www.aia-figc.it/sitemap.xml",
                           headers=headers, timeout=10)
        if resp.status_code == 200:
            ids = re.findall(
                r'serie-a[^"]*designazioni[^"]*-(\d+)/',
                resp.text, re.IGNORECASE)
            if ids:
                latest_id = max(int(i) for i in ids)
                url = f"https://www.aia-figc.it/dettaglio.asp?ID={latest_id}"
                print(f"  [AIA] ID trovato via sitemap: {latest_id}")
    except Exception:
        pass

    # Fallback: usa cache per trovare l ultimo ID noto e prova quelli successivi
    if not url:
        try:
            last_id = 27569  # ID prima giornata 2026/27
            # Prova IDs recenti
            for test_id in range(last_id + 10, last_id - 1, -1):
                test_url = f"https://www.aia-figc.it/dettaglio.asp?ID={test_id}"
                r = requests.get(test_url, headers=headers, timeout=5)
                if r.status_code == 200 and "SERIE A" in r.content.decode("latin-1", errors="ignore").upper():
                    content_text = r.content.decode("latin-1", errors="ignore")
                    if any(s in content_text.upper() for s in ["INTER","JUVENTUS","MILAN","NAPOLI","ROMA"]):
                        url = test_url
                        print(f"  [AIA] ID trovato via scan: {test_id}")
                        break
        except Exception:
            pass

    if not url:
        print("  [AIA] URL non trovato, uso cache")
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        return []

    # Converti URL slug in URL dettaglio.asp
    m_id = re.search(r"-(\d+)/?$", url)
    if m_id:
        detail_url = f"https://www.aia-figc.it/dettaglio.asp?ID={m_id.group(1)}"
    else:
        detail_url = url

    print(f"  [AIA] Scarico: {detail_url}")

    # Step 2: Scarica e parsa le designazioni
    try:
        resp = requests.get(detail_url, headers=headers, timeout=15)
        text = resp.content.decode("latin-1")
    except Exception as e:
        print(f"  [AIA] Errore download: {e}")
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        return []

    squadre_sa = [
        "INTER","MONZA","UDINESE","COMO","JUVENTUS","FROSINONE",
        "NAPOLI","GENOA","MILAN","TORINO","ROMA","FIORENTINA",
        "LAZIO","BOLOGNA","ATALANTA","SASSUOLO","PARMA","LECCE",
        "VENEZIA","CAGLIARI",
    ]

    paras = re.findall(r"<p[^>]*>(.*?)</p>", text, re.DOTALL)
    paras_clean = []
    for p in paras:
        clean = re.sub(r"<[^>]+>", "", p)
        clean = clean.replace("&ndash;", "-").replace("&nbsp;", " ")
        clean = clean.replace("&igrave;", "i").replace("&agrave;", "a")
        clean = clean.replace("&egrave;", "e").replace("&ograve;", "o")
        clean = re.sub(r"\s+", " ", clean).strip()
        if clean:
            paras_clean.append(clean)

    PARTICELLE = {"VAR:","AVAR:","IV:","ASSISTENTE"}
    designations = []
    for i, para in enumerate(paras_clean):
        m = re.match(
            r"([A-Z][A-Z\s\.]+?)\s*-\s*([A-Z][A-Z\s\.]+?)\s+"
            r"(?:\w+\s+)?(?:\d{1,2}/\d{2}\s+)?h\.\s*\d+",
            para)
        if not m: continue
        home = m.group(1).strip()
        away = m.group(2).strip()
        if not any(s in home.upper() for s in squadre_sa): continue
        if not any(s in away.upper() for s in squadre_sa): continue
        if i+1 >= len(paras_clean): continue
        ref_raw = paras_clean[i+1].strip()
        ref_raw = re.sub(r"\s*\(.*?\)", "", ref_raw).strip()
        if not re.match(r"^[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z\.]+){0,2}$", ref_raw): continue
        if any(x in ref_raw.upper() for x in PARTICELLE): continue
        tokens = ref_raw.upper().split()
        if len(tokens) == 1 and tokens[0] in {"AVAR","IV","DEL","LA","LO"}: continue
        designations.append({"home": home, "away": away, "referee": ref_raw})

    if designations:
        cache_path.write_text(
            json.dumps(designations, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [AIA] {len(designations)} designazioni trovate")
    elif cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    return designations


def fetch_aia_designations_OLD() -> list:
    """Scarica designazioni arbitrali AIA per la Serie A."""
    import json, re, asyncio
    from pathlib import Path

    cache_path = Path("cache/aia_designations.json")

    async def _scrape():
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(extra_http_headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
                await page.goto("https://www.aia-figc.it/news/?c=9", timeout=30000)
                import asyncio as _a; await _a.sleep(3)
                body = await page.inner_text("body")
                await browser.close()
                return body
        except Exception as e:
            print(f"  [AIA] Errore: {e}")
            return ""

    try:
        body = asyncio.run(_scrape())
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            body = pool.submit(asyncio.run, _scrape()).result()

    if not body:
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        return []

    squadre_sa = [
        "INTER","MONZA","UDINESE","COMO","JUVENTUS","FROSINONE",
        "NAPOLI","GENOA","MILAN","TORINO","ROMA","FIORENTINA",
        "LAZIO","BOLOGNA","ATALANTA","SASSUOLO","PARMA","LECCE",
        "VENEZIA","CAGLIARI",
    ]

    designations = []
    for raw_line in body.split("\n"):
        line = raw_line.strip()
        if not line: continue
        # Normalizza caratteri speciali
        line_c = line.replace("\xa0", " ").replace("\u00a0", " ")
        line_c = line_c.replace("\u2013", "-").replace("\u2014", "-")
        line_c = line_c.replace("\u2019", "'")
        # Rimuovi spazi multipli
        import re as _re
        line_c = _re.sub(r" +", " ", line_c)
        line_up = line_c.upper()
        found = [s for s in squadre_sa if s in line_up]
        if len(found) < 2: continue
        # Match squadre prima della data
        m = _re.search(
            r"([A-ZÀÈÉÌÒÙ][\w\s\.]+?)\s*[\-\u2013]\s*([A-ZÀÈÉÌÒÙ][\w\s\.]+?)\s+"
            r"(?:Sabato|Domenica|Venerd|Luned|Marted|Mercoled|Gioved|\d{1,2}/\d{2})",
            line_c, _re.IGNORECASE)
        if not m: continue
        home_raw = m.group(1).strip()
        away_raw = m.group(2).strip()
        # Arbitro: primo token MAIUSCOLO dopo h. XX.XX
        m2 = _re.search(r"h\.\s*\d+\.\d+\s+([A-ZÀÈÉÌÒÙ]{3,})", line_c)
        if not m2:
            m2 = _re.search(r"h\.\s*\d+:\d+\s+([A-ZÀÈÉÌÒÙ]{3,})", line_c)
        if not m2: continue
        referee = m2.group(1).strip("().,\'")
        if referee in ["VAR","AVAR","IV","DI","DEL","LA"]: continue
        designations.append({"home": home_raw, "away": away_raw, "referee": referee})

    if designations:
        cache_path.write_text(
            json.dumps(designations, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [AIA] {len(designations)} designazioni trovate")
    elif cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    return designations


    async def _scrape():
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(extra_http_headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
                await page.goto("https://www.aia-figc.it/news/?c=9", timeout=30000)
                await asyncio.sleep(3)
                body = await page.inner_text("body")
                await browser.close()
                return body
        except:
            return ""

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                body = pool.submit(asyncio.run, _scrape()).result()
        else:
            body = asyncio.run(_scrape())
    except:
        body = ""

    if not body:
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        return []

    # Parse designazioni
    designations = []
    squadre_sa = [
        "INTER","MONZA","UDINESE","COMO","JUVENTUS","FROSINONE",
        "NAPOLI","GENOA","MILAN","TORINO","ROMA","FIORENTINA",
        "LAZIO","BOLOGNA","ATALANTA","SASSUOLO","PARMA","LECCE",
        "VENEZIA","CAGLIARI",
    ]
    for raw_line in body.split("\n"):
        line = raw_line.strip()
        if not line: continue
        line_c = line.replace("\xa0"," ").replace("\u2013","-").replace("\u2014","-").replace("\u2013","-")
        line_c = line_c.replace("\xe2\x80\x93","-")
        line_up = line_c.upper()
        found = [s for s in squadre_sa if s in line_up]
        if len(found) < 2: continue
        m = re.search(r"([\w\s]+?)\s*-\s*([\w\s]+?)\s+(?:Sab|Dom|Ven|Lun|Mar|Mer|Gio|\d{1,2}/\d{2})", line_c, re.IGNORECASE)
        if not m: continue
        home_raw = m.group(1).strip()
        away_raw = m.group(2).strip()
        m2 = re.search(r"h\.?\s*\d+[.:]\d+\s+([A-ZÀÈÉÌÒÙ]{3,})", line_c)
        if not m2: continue
        referee = m2.group(1).strip("().,'")
        designations.append({"home": home_raw, "away": away_raw, "referee": referee})

    if designations:
        cache_path.write_text(json.dumps(designations, ensure_ascii=False, indent=2))
        print(f"  [AIA] {len(designations)} designazioni trovate")

    if not designations and cache_path.exists():
        return json.loads(cache_path.read_text())

    return designations

# ─────────────────────────────────────────────
# 3. MODELLO CARTELLINI CON ARBITRO
# ─────────────────────────────────────────────

def get_referee_for_match(home: str, away: str, designations: list) -> str:
    """
    Cerca l arbitro designato per una partita specifica.
    Usa matching parziale sul nome delle squadre.
    """
    if not designations:
        return None

    home_key = home.upper()[:5]
    away_key = away.upper()[:5]

    for d in designations:
        dh = d.get("home", "").upper()
        da = d.get("away", "").upper()
        if home_key in dh and away_key in da:
            return d.get("referee")
        if away_key in dh and home_key in da:
            return d.get("referee")
    return None


def estimate_cards_lambda_with_referee(
    prob_draw: float,
    prob_over25: float,
    ref_features: dict,
    cards_ema_total: float = 3.8,
) -> float:
    """
    Stima il numero atteso di cartellini usando:
    - EMA storica cartellini delle due squadre (se disponibile)
    - Media storica arbitro (se disponibile)
    - Equilibrio e intensità della partita
    """
    # Base: usa EMA squadre se disponibile, altrimenti arbitro, altrimenti media Serie A
    if cards_ema_total and cards_ema_total > 0:
        base = cards_ema_total
    elif ref_features.get("ref_found"):
        base = ref_features["ref_gialli_media"]
    else:
        base = 3.8

    # Aggiustamento partita
    equilibrio_bonus = (prob_draw - 0.25) * 2.0
    difensivo_bonus  = (0.5 - prob_over25) * 1.5

    lam = base + equilibrio_bonus + difensivo_bonus
    return max(1.5, min(7.5, lam))


def compute_cards_markets(
    prob_draw: float,
    prob_over25: float,
    ref_features: dict,
    cards_ema_total: float = 3.8,
) -> dict:
    """
    Calcola le probabilità per tutti i mercati cartellini
    usando EMA squadre + arbitro + dinamiche partita.
    """
    from scipy.stats import poisson

    lam = estimate_cards_lambda_with_referee(
        prob_draw, prob_over25, ref_features, cards_ema_total)

    # Bias casa/ospite dall'arbitro
    bias = ref_features.get("ref_bias_casa", 0.0)
    lam_home = max(0.5, lam / 2 - bias / 2)
    lam_away = max(0.5, lam / 2 + bias / 2)

    def over_p(threshold):
        # Usa distribuzione convoluta home+away
        prob = 0.0
        for h in range(12):
            for a in range(12):
                if h + a > threshold:
                    prob += poisson.pmf(h, lam_home) * poisson.pmf(a, lam_away)
        return float(prob)

    return {
        "cards_lambda":       round(lam, 2),
        "cards_lambda_home":  round(lam_home, 2),
        "cards_lambda_away":  round(lam_away, 2),
        "prob_cards_over15":  over_p(1.5),
        "prob_cards_over25":  over_p(2.5),
        "prob_cards_over35":  over_p(3.5),
        "prob_cards_over45":  over_p(4.5),
        "prob_cards_over55":  over_p(5.5),
        "prob_cards_under15": 1 - over_p(1.5),
        "prob_cards_under25": 1 - over_p(2.5),
        "prob_cards_under35": 1 - over_p(3.5),
        "prob_cards_under45": 1 - over_p(4.5),
        "prob_cards_under55": 1 - over_p(5.5),
    }


# ─────────────────────────────────────────────
# 4. UTILITY: LISTA ARBITRI
# ─────────────────────────────────────────────

# Arbitri Serie A più comuni (per autocompletamento dashboard)
ARBITRI_SERIE_A = [
    "Sozza", "Mariani", "Massa", "Chiffi", "Fabbri",
    "Doveri", "Guida", "Di Bello", "Abisso", "Orsato",
    "Colombo", "Piccinini", "Meraviglia", "Marchetti",
    "Sacchi", "Fourneau", "Manganiello", "Mazzoleni",
    "Feliciani", "Marcenaro", "Tremolada", "Dionisi",
    "Bonacina", "Mucera", "Giua", "Aureliano",
    # Prima giornata 2026/27
    "Marinelli", "Arena", "Perenzoni", "Ayroldi",
    "Rapuano", "Crezzini", "Zufferli", "Maresca F.",
    # Altri arbitri Serie A
    "Massimi", "La Penna", "Pairetto", "Irrati",
    "Prontera", "Collu", "Valeri",
]


if __name__ == "__main__":
    # Test con dati sintetici
    import random
    random.seed(42)

    refs = ["Sozza", "Massa", "Mariani", "Chiffi", "Fabbri", "Doveri"]
    rows = []
    for i in range(200):
        ref = random.choice(refs)
        # Sozza più severo, Fabbri più morbido
        base = {"Sozza": 5, "Massa": 4.5, "Mariani": 4,
                "Chiffi": 3.5, "Fabbri": 3, "Doveri": 4}[ref]
        hy = max(0, int(random.gauss(base/2, 1)))
        ay = max(0, int(random.gauss(base/2, 1)))
        rows.append({
            "Referee": ref, "HY": hy, "AY": ay,
            "HR": random.randint(0,1), "AR": random.randint(0,1),
            "FTHG": random.randint(0,4), "FTAG": random.randint(0,3),
        })

    df = pd.DataFrame(rows)
    stats = compute_referee_stats(df)
    print("Statistiche arbitri:")
    print(stats[["Referee","partite","gialli_media","rossi_media",
                 "severita_pct","bias_casa"]].to_string(index=False))

    print("\nFeature per Sozza:")
    feat = get_referee_features(stats, "Sozza")
    for k, v in feat.items():
        print(f"  {k}: {v}")

    print("\nMercati cartellini con arbitro Sozza:")
    cards = compute_cards_markets(0.28, 0.55, feat)
    for k, v in cards.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
