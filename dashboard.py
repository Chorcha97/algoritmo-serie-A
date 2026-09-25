"""
Dashboard Serie A — Streamlit v3
Avvio: streamlit run dashboard.py

Fix v3:
- Home advantage correction (-10pp)
- Kelly 1/8
- Edge minimo default 7%
- Warning su vittoria casa
- Tabella quote eque sempre visibile
"""

import streamlit as st

# Backend API client (con fallback automatico al modello locale)
try:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from backend.client import (is_backend_available, predict as api_predict,
                                 value_bets as api_value_bets, get_injuries as api_get_injuries,
                                 get_lineups as api_get_lineups)
    BACKEND_CLIENT_OK = True
except ImportError:
    BACKEND_CLIENT_OK = False
    def is_backend_available(): return False
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

SERIE_A_2627_TEAMS = [
    "Atalanta", "Bologna", "Cagliari", "Como", "Fiorentina",
    "Frosinone", "Genoa", "Inter", "Juventus", "Lazio",
    "Lecce", "Milan", "Monza", "Napoli", "Parma",
    "Roma", "Sassuolo", "Torino", "Udinese", "Venezia",
]

GIORNATA_1 = [
    ("Atalanta","Sassuolo"),("Bologna","Lazio"),("Frosinone","Juventus"),
    ("Genoa","Napoli"),("Inter","Monza"),("Parma","Cagliari"),
    ("Roma","Fiorentina"),("Torino","Milan"),("Udinese","Como"),("Venezia","Lecce"),
]

def get_stats():
    from utils.bet_tracker import load_tracker
    data = load_tracker()
    bets = data.get("bets", [])
    closed = [b for b in bets if b["status"] != "pending"]
    won = [b for b in closed if b["status"] == "won"]
    tot_profit = sum(b.get("profitto",0) or 0 for b in closed)
    tot_stake = sum(b.get("stake",4) for b in closed)
    return {
    "bets": bets, "total": len(bets), "closed": len(closed),
    "won": len(won), "lost": len(closed)-len(won),
    "win_rate": len(won)/len(closed)*100 if closed else 0,
    "roi": tot_profit/tot_stake*100 if tot_stake > 0 else 0,
    "total_profit": tot_profit,
    }

st.set_page_config(page_title="Serie A Predictor", page_icon="⚽",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.value-bet{background:#e6fff0;border-left:4px solid #40c040;border-radius:8px;padding:12px 16px;margin:6px 0;color:#1a4a2a}
.value-bet-warn{background:#fff8e6;border-left:4px solid #f0a500;border-radius:8px;padding:12px 16px;margin:6px 0;color:#7a5000}
.no-value{background:#1e1e2e;border-left:4px solid #555;border-radius:8px;padding:12px 16px;margin:6px 0}
.ref-card{background:#e8f0fe;border-left:4px solid #4080c0;border-radius:8px;padding:12px 16px;color:#1a3a6a;margin:6px 0}
.ref-severe{background:#fee8e8;border-left:4px solid #c04040;color:#6a1a1a;border-radius:8px;padding:12px 16px;margin:6px 0}
.fix-badge{background:#1a2a1a;border:1px solid #40c040;border-radius:4px;padding:2px 8px;font-size:0.75em;color:#40c040}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Caricamento statistiche giocatori...")
def load_player_stats_builder():
    from models.player_stats import PlayerStatsBuilder
    return PlayerStatsBuilder()

@st.cache_resource(show_spinner="Caricamento modello in corso...")
def load_model():
    import pickle
    from data.pipeline import build_dataset
    from models.features import build_features
    from models.model import EnsembleModel
    from models.referee import compute_referee_stats

    dataset_path = "serie_a_dataset.csv"
    model_path   = "model_cache.pkl"

    if Path(dataset_path).exists():
        df_raw = pd.read_csv(dataset_path, parse_dates=["Date"])
    else:
        df_raw = build_dataset(dataset_path)

    df = build_features(df_raw, verbose=False)
    ref_stats = compute_referee_stats(df_raw)

    # Carica modello da disco se esiste, altrimenti allena e salva
    if Path(model_path).exists():
        with open(model_path, "rb") as f:
            loaded = pickle.load(f)
            if isinstance(loaded, tuple):
                model = loaded[0]
            else:
                model = loaded
        model = EnsembleModel()
        model.fit(df)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    return model, df, df_raw, ref_stats

with st.sidebar:
    st.title("⚽ Serie A Predictor")
    st.caption("Poisson + GBM + Kelly Value Bet")
    st.divider()
    page = st.radio("Sezione", [
        "📅 Calendario 26/27", "🔮 Predizione", "📊 Classifica", "🎯 Tracker", "📈 Backtest", "ℹ️ Info modello"
    ], label_visibility="collapsed")
    st.divider()
    st.caption("Impostazioni")
    min_edge = st.slider("Edge minimo (%)", 1, 20, 7) / 100
    bankroll = st.number_input("Bankroll (€)", min_value=100,
                                max_value=100000, value=1000, step=100)
    st.divider()
    # Stato backend
    if BACKEND_CLIENT_OK and is_backend_available():
        st.sidebar.success("🟢 Backend online")
    else:
        st.sidebar.warning("🟡 Backend offline — modello locale")
    st.sidebar.divider()
    st.caption("Fix attivi v3")
    st.markdown("""
    ✅ Home corr. −10pp  
    ✅ Kelly 1/8  
    ✅ Edge ottimale 7%  
    ✅ Warning vittoria casa
    """)

try:
    model, df, df_raw, ref_stats = load_model()
    teams = SERIE_A_2627_TEAMS
except Exception as e:
    st.error(f"Errore caricamento modello: {e}")
    st.info("Assicurati di aver lanciato almeno una volta `python3 main.py`.")
    st.stop()

# ═══ PREDIZIONE ═══
if page == "🔮 Predizione":
    st.title("🔮 Analisi Partita")

    default_home = st.session_state.get("cal_home", "Juventus")
    default_away = st.session_state.get("cal_away", "Inter")
    default_ref  = st.session_state.get("cal_ref", "")

    col1, col2, col3 = st.columns([2,1,2])
    with col1:
        home_idx = teams.index(default_home) if default_home in teams else 0
        home = st.selectbox("Squadra Casa", teams, index=home_idx)
    with col2:
        st.markdown("<br><h3 style='text-align:center;color:#888'>VS</h3>",
                    unsafe_allow_html=True)
    with col3:
        away_teams = [t for t in teams if t != home]
        away_idx = away_teams.index(default_away) if default_away in away_teams else 0
        away = st.selectbox("Squadra Ospite", away_teams, index=away_idx)

    # ── Arbitro ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("👨‍⚖️ Arbitro")
    from models.referee import ARBITRI_SERIE_A, get_referee_features, \
        compute_cards_markets, fetch_aia_designations, get_referee_for_match

    col_r1, col_r2 = st.columns([3,1])
    with col_r1:
        opts = ["— Non specificato —"] + sorted(ARBITRI_SERIE_A)
        ref_idx = opts.index(default_ref) if default_ref in opts else 0
        referee_sel = st.selectbox("Arbitro designato", opts, index=ref_idx)
        referee = referee_sel if referee_sel != "— Non specificato —" else ""
    with col_r2:
        st.markdown("<br>", unsafe_allow_html=True)
        aia_id = st.text_input("ID AIA (opzionale)", placeholder="es. 27587", label_visibility="collapsed")
        if st.button("🔄 Cerca AIA"):
            with st.spinner("Cerco designazioni AIA..."):
                # Prima leggi dalla cache già popolata
                import json
                from pathlib import Path
                cache_path = Path("cache/aia_designations.json")
                desig = []
                if cache_path.exists():
                    try:
                        desig = json.loads(cache_path.read_text(encoding="utf-8"))
                    except:
                        pass
                # Se cache vuota, prova a scaricare
                if not desig:
                    if aia_id and aia_id.strip().isdigit():
                        import requests, re
                        headers = {"User-Agent": "Mozilla/5.0"}
                        resp = requests.get(f"https://www.aia-figc.it/dettaglio.asp?ID={aia_id.strip()}", headers=headers, timeout=15)
                        text = resp.content.decode("latin-1")
                        paras = re.findall(r"<p[^>]*>(.*?)</p>", text, re.DOTALL)
                        paras_clean = []
                        for p in paras:
                            clean = re.sub(r"<[^>]+>", "", p)
                            clean = clean.replace("&ndash;","-").replace("&nbsp;"," ")
                            clean = clean.replace("&igrave;","i").replace("&agrave;","a")
                            clean = re.sub(r"\s+", " ", clean).strip()
                            if clean: paras_clean.append(clean)
                        squadre_sa = ["INTER","MONZA","UDINESE","COMO","JUVENTUS","FROSINONE",
                                      "NAPOLI","GENOA","MILAN","TORINO","ROMA","FIORENTINA",
                                      "LAZIO","BOLOGNA","ATALANTA","SASSUOLO","PARMA","LECCE",
                                      "VENEZIA","CAGLIARI"]
                        desig = []
                        for idx_p, para in enumerate(paras_clean):
                            m = re.match(r"([A-Z][A-Z\s\.]+?)\s*-\s*([A-Z][A-Z\s\.]+?)\s+(?:\w+\s+)?(?:\d{1,2}/\d{2}\s+)?h\.\s*\d+", para)
                            if not m: continue
                            home_r = m.group(1).strip(); away_r = m.group(2).strip()
                            if not any(s in home_r.upper() for s in squadre_sa): continue
                            if not any(s in away_r.upper() for s in squadre_sa): continue
                            if idx_p+1 >= len(paras_clean): continue
                            ref_r = re.sub(r"\s*\(.*?\)", "", paras_clean[idx_p+1]).strip()
                            if not re.match(r"^[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z\.]+){0,2}$", ref_r): continue
                            desig.append({"home": home_r, "away": away_r, "referee": ref_r})
                        import json
                        from pathlib import Path
                        Path("cache/aia_designations.json").write_text(
                            json.dumps(desig, ensure_ascii=False, indent=2), encoding="utf-8")
                    else:
                        desig = fetch_aia_designations()
                found = get_referee_for_match(home, away, desig)
                # Converti MAIUSCOLO in Titlecase per il selectbox
                if found:
                    found_title = found.title()
                    # Cerca match nel selectbox
                    from models.referee import ARBITRI_SERIE_A
                    for a in ARBITRI_SERIE_A:
                        if a.upper() == found.upper() or a.upper() == found_title.upper():
                            found = a
                            break
                    if found:
                        st.session_state["cal_ref"] = found
                        st.success(f"Trovato: {found}")
                        st.rerun()
                else:
                    st.warning("Designazione non ancora disponibile")

    # Usa statistiche arbitri Serie A reali invece del dataset
    try:
        from models.referee import get_referee_adjustments
        ref_adj = get_referee_adjustments(referee) if referee and "Non specificato" not in referee else {"affidabile": False}
        if ref_adj["affidabile"]:
            gialli = ref_adj["gialli_attesi"]
            rossi  = ref_adj["rossi_attesi"]
            mult   = ref_adj["cards_mult"]
            css    = "ref-severe" if mult > 1.15 else "ref-card"
            st.markdown(f'''<div class="{css}"><b>📋 {referee}</b> — '''
                        f'''{ref_adj.get("partite","?")} partite Serie A<br>'''
                        f'''🟨 Gialli/partita: <b>{gialli:.1f}</b> (media SA: 4.18) &nbsp;|&nbsp;'''
                        f'''🟥 Rossi/partita: <b>{rossi:.3f}</b> &nbsp;|&nbsp;'''
                        f'''Moltiplicatore cartellini: <b>{mult:.2f}x</b></div>''',
                        unsafe_allow_html=True)
        else:
            st.info(f"ℹ️ {referee} — statistiche non disponibili, uso media Serie A")
    except Exception as _e:
        st.info("ℹ️ Nessun arbitro — usata media Serie A per i cartellini")
    else:
        st.caption("ℹ️ Nessun arbitro — usata media Serie A per i cartellini")

    # ── Quote ─────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Quote bookmaker")
    match_key = f"odds_{home}_{away}"
    # Carica quote dal file JSON (persistenti tra riavvii)
    import json as _json
    from pathlib import Path as _Path
    _odds_file = _Path("cache/saved_odds.json")
    _all_odds = _json.loads(_odds_file.read_text()) if _odds_file.exists() else {}
    if match_key not in st.session_state:
        st.session_state[match_key] = _all_odds.get(match_key, {})
    odds = dict(st.session_state[match_key])
    # ── Carica quote Marathonbet automaticamente ─────────────────────────────
    # Mappa nomi modello -> nomi Marathonbet
    MB_NAME_MAP = {
        "Inter": "Inter", "Milan": "Milan", "Juventus": "Juventus",
        "Roma": "Roma", "Napoli": "Napoli", "Lazio": "Lazi",
        "Fiorentina": "Fior", "Atalanta": "Atal", "Bologna": "Bolo",
        "Torino": "Tori", "Udinese": "Udin", "Sassuolo": "Sass",
        "Genoa": "Geno", "Cagliari": "Cagl", "Lecce": "Lecc",
        "Parma": "Parm", "Como": "Como", "Venezia": "Vene",
        "Monza": "Monz", "Frosinone": "Fros",
    }
    # Mappa ce -> chiave odds per 1X2
    CE_1X2 = {1: "H", 2: "D", 3: "A"}
    # Mappa etichette Over/Under (ce=1 -> over, ce=2 -> under per questi mercati)
    if st.button("\U0001f4e5 Carica quote Marathonbet", help="Scarica le quote in tempo reale da Marathonbet"):
        with st.spinner("Scarico quote Marathonbet..."):
            try:
                import requests as _req
                _home_mb = home.upper()
                _away_mb = away.upper()
                _new_odds = {}
                _loaded = []

                def _find_match(avs, h, a):
                    for _p in avs:
                        _n = _p.get("dsl",{}).get("IT","").upper()
                        if any(x in _n for x in [h[:4],h[:5]]) and any(x in _n for x in [a[:4],a[:5]]):
                            return _p
                    return None

                # 1X2 + Over/Under gol (id=319)
                _r1 = _req.get("http://localhost:8000/marathonbet/serie-a-bet/pre-match/eventi?id_aggregata=319",timeout=10).json()
                _p1 = _find_match(_r1.get("avs",[]), _home_mb, _away_mb)
                if _p1:
                    for _si, _sc in enumerate(_p1.get("scs",[])):
                        _eqs = _sc.get("eqs",[])
                        _ces = {_e.get("ce") for _e in _eqs}
                        _dsl = _sc.get("dsl",{}).get("IT","") if isinstance(_sc.get("dsl"),dict) else ""
                        for _eq in _eqs:
                            _ce = _eq.get("ce"); _q = round(_eq.get("q",0)/100,2)
                            if _q <= 1: continue
                            _csn = str(_eq.get("csn",""))
                            if {1,2,3}.issubset(_ces) and _si==0:
                                if _ce==1: _new_odds["H"]=_q
                                elif _ce==2: _new_odds["D"]=_q
                                elif _ce==3: _new_odds["A"]=_q
                            elif len(_eqs)==2 and _ces=={1,2}:
                                for _s,_k1,_k2 in [("15","over15","under15"),("25","over25","under25"),("35","over35","under35"),("45","over45","under45")]:
                                    if _s in _csn or _s[0]+"."+_s[1] in _dsl:
                                        if _ce==1: _new_odds[_k1]=_q
                                        elif _ce==2: _new_odds[_k2]=_q
                    if "H" in _new_odds: _loaded.append("1X2+O/U")

                # GG/NG (id=3141)
                _r2 = _req.get("http://localhost:8000/marathonbet/serie-a-bet/pre-match/eventi?id_aggregata=3141",timeout=10).json()
                _p2 = _find_match(_r2.get("avs",[]), _home_mb, _away_mb)
                if _p2:
                    for _sc in _p2.get("scs",[]):
                        _eqs = _sc.get("eqs",[])
                        if len(_eqs)==2:
                            for _eq in _eqs:
                                _ce=_eq.get("ce"); _q=round(_eq.get("q",0)/100,2)
                                if _q>1:
                                    if _ce==1: _new_odds["gg"]=_q
                                    elif _ce==2: _new_odds["ng"]=_q
                    if "gg" in _new_odds: _loaded.append("GG/NG")

                # U/O Corner (id=565)
                _r3 = _req.get("http://localhost:8000/marathonbet/serie-a-bet/pre-match/eventi?id_aggregata=565",timeout=10).json()
                _p3 = _find_match(_r3.get("avs",[]), _home_mb, _away_mb)
                if _p3:
                    for _sc in _p3.get("scs",[]):
                        _eqs = _sc.get("eqs",[])
                        if len(_eqs)==2:
                            for _eq in _eqs:
                                _ce=_eq.get("ce"); _q=round(_eq.get("q",0)/100,2)
                                _csn=str(_eq.get("csn",""))
                                if _q>1:
                                    for _s,_k1,_k2 in [("75","corn_over75","corn_under75"),("85","corn_over85","corn_under85"),("95","corn_over95","corn_under95"),("105","corn_over105","corn_under105"),("115","corn_over115","corn_under115")]:
                                        if _s in _csn:
                                            if _ce==1: _new_odds[_k1]=_q
                                            elif _ce==2: _new_odds[_k2]=_q
                    if any(k.startswith("corn_") for k in _new_odds): _loaded.append("Corner")

                # U/O Cartellini (id=1690)
                _r4 = _req.get("http://localhost:8000/marathonbet/serie-a-bet/pre-match/eventi?id_aggregata=1690",timeout=10).json()
                _p4 = _find_match(_r4.get("avs",[]), _home_mb, _away_mb)
                if _p4:
                    for _sc in _p4.get("scs",[]):
                        _eqs = _sc.get("eqs",[])
                        if len(_eqs)==2:
                            for _eq in _eqs:
                                _ce=_eq.get("ce"); _q=round(_eq.get("q",0)/100,2)
                                _csn=str(_eq.get("csn",""))
                                if _q>1:
                                    for _s,_k1,_k2 in [("25","cards_over25","cards_under25"),("35","cards_over35","cards_under35"),("45","cards_over45","cards_under45"),("55","cards_over55","cards_under55")]:
                                        if _s in _csn:
                                            if _ce==1: _new_odds[_k1]=_q
                                            elif _ce==2: _new_odds[_k2]=_q
                    if any("cards" in k for k in _new_odds): _loaded.append("Cartellini")

                # U/O Primo Tempo (id=344)
                _r5 = _req.get("http://localhost:8000/marathonbet/serie-a-bet/pre-match/eventi?id_aggregata=344",timeout=10).json()
                _p5 = _find_match(_r5.get("avs",[]), _home_mb, _away_mb)
                if _p5:
                    for _sc in _p5.get("scs",[]):
                        _eqs = _sc.get("eqs",[])
                        if len(_eqs)==2:
                            for _eq in _eqs:
                                _ce=_eq.get("ce"); _q=round(_eq.get("q",0)/100,2)
                                _csn=str(_eq.get("csn",""))
                                if _q>1:
                                    for _s,_k1,_k2 in [("05","ht_over05","ht_under05"),("15","ht_over15","ht_under15"),("25","ht_over25","ht_under25")]:
                                        if _s in _csn:
                                            if _ce==1: _new_odds[_k1]=_q
                                            elif _ce==2: _new_odds[_k2]=_q
                    if any("ht_" in k for k in _new_odds): _loaded.append("HT O/U")

                if _new_odds.get("H"):
                    if match_key not in st.session_state:
                        st.session_state[match_key] = {}
                    st.session_state[match_key].update(_new_odds)
                    st.success(f"\u2705 Quote caricate: {chr(44).join(_loaded)}")
                    st.rerun()
                else:
                    st.warning("Partita non trovata su Marathonbet")
            except Exception as _me:
                st.warning(f"Marathonbet non disponibile: {_me}")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "1X2 & DC", "Gol O/U", "GG/NG & Combo", "Cartellini",
        "1° Tempo", "2° Tempo", "Corner", "Marcatori", "Ammoniti"])

    def _inp(key, label):
        return st.number_input(label, min_value=0.0,
            value=st.session_state.get(match_key, {}).get(key, 0.0),
            step=0.05, format="%.2f", key=f"{match_key}_{key}")

    # ── TAB 1: 1X2 & Doppia Chance ───────────────────────────────────────────
    with tab1:
        c1,c2,c3 = st.columns(3)
        with c1: odds["H"]  = _inp("H",  "1 (Casa)")
        with c2: odds["D"]  = _inp("D",  "X (Pareggio)")
        with c3: odds["A"]  = _inp("A",  "2 (Ospite)")
        c4,c5,c6 = st.columns(3)
        with c4: odds["1X"] = _inp("1X", "1X")
        with c5: odds["X2"] = _inp("X2", "X2")
        with c6: odds["12"] = _inp("12", "12")
        st.divider()
        st.caption("Clean Sheet / Win to nil")
        c1,c2,c3,c4 = st.columns(4)
        with c1: odds["cs_home"]     = _inp("cs_home",     "CS Casa")
        with c2: odds["cs_away"]     = _inp("cs_away",     "CS Ospite")
        with c3: odds["wtn_home"]    = _inp("wtn_home",    "Win to nil Casa")
        with c4: odds["wtn_away"]    = _inp("wtn_away",    "Win to nil Ospite")

    # ── TAB 2: Gol Over/Under ─────────────────────────────────────────────────
    with tab2:
        c1,c2 = st.columns(2)
        with c1:
            st.caption("Over")
            for t in ["05","15","25","35","45","55"]:
                odds[f"over{t}"] = _inp(f"over{t}", f"Over {t[0]}.{t[1]}")
        with c2:
            st.caption("Under")
            for t in ["05","15","25","35","45","55"]:
                odds[f"under{t}"] = _inp(f"under{t}", f"Under {t[0]}.{t[1]}")

    # ── TAB 3: GG/NG & Combo ──────────────────────────────────────────────────
    with tab3:
        c1,c2 = st.columns(2)
        with c1: odds["gg"] = _inp("gg", "Goal/Goal")
        with c2: odds["ng"] = _inp("ng", "No Goal")
        st.caption("Combo GG + Over/Under")
        c1,c2,c3 = st.columns(3)
        with c1: odds["gg_over25"]  = _inp("gg_over25",  "GG + Over 2.5")
        with c2: odds["gg_under25"] = _inp("gg_under25", "GG + Under 2.5")
        with c3: odds["ng_under25"] = _inp("ng_under25", "NG + Under 2.5")
        st.caption("HT/FT")
        c1,c2,c3 = st.columns(3)
        with c1:
            odds["htft_11"] = _inp("htft_11", "1/1")
            odds["htft_1X"] = _inp("htft_1X", "1/X")
            odds["htft_12"] = _inp("htft_12", "1/2")
        with c2:
            odds["htft_X1"] = _inp("htft_X1", "X/1")
            odds["htft_XX"] = _inp("htft_XX", "X/X")
            odds["htft_X2"] = _inp("htft_X2", "X/2")
        with c3:
            odds["htft_21"] = _inp("htft_21", "2/1")
            odds["htft_2X"] = _inp("htft_2X", "2/X")
            odds["htft_22"] = _inp("htft_22", "2/2")

    # ── TAB 4: Cartellini ─────────────────────────────────────────────────────
    with tab4:
        try:
            from models.referee import get_referee_adjustments
            _adj = get_referee_adjustments(referee)
            if _adj["affidabile"]:
                st.info(f"🎯 Calibrato su {referee} ({_adj['gialli_attesi']:.1f} gialli/p)")
        except:
            pass
        c1,c2 = st.columns(2)
        with c1:
            st.caption("Over")
            for t in ["15","25","35","45","55","65"]:
                odds[f"cards_over{t}"] = _inp(f"cards_over{t}", f"Cart. Over {t[0]}.{t[1]}")
        with c2:
            st.caption("Under")
            for t in ["15","25","35","45","55","65"]:
                odds[f"cards_under{t}"] = _inp(f"cards_under{t}", f"Cart. Under {t[0]}.{t[1]}")

    # ── TAB 5: Primo Tempo (1HT) ──────────────────────────────────────────────
    with tab5:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("1X2 Primo Tempo")
            odds["ht_H"]  = _inp("ht_H",  "HT 1 (Casa)")
            odds["ht_D"]  = _inp("ht_D",  "HT X (Pari)")
            odds["ht_A"]  = _inp("ht_A",  "HT 2 (Ospite)")
            odds["ht_1X"] = _inp("ht_1X", "HT 1X")
            odds["ht_X2"] = _inp("ht_X2", "HT X2")
        with c2:
            st.caption("Gol Primo Tempo")
            for t in ["05","15","25"]:
                odds[f"ht_over{t}"]  = _inp(f"ht_over{t}",  f"HT Over {t[0]}.{t[1]}")
                odds[f"ht_under{t}"] = _inp(f"ht_under{t}", f"HT Under {t[0]}.{t[1]}")
            odds["ht_gg"] = _inp("ht_gg", "HT GG")
            odds["ht_ng"] = _inp("ht_ng", "HT NG")

    # ── TAB 6: Secondo Tempo (2HT) ────────────────────────────────────────────
    with tab6:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("1X2 Secondo Tempo")
            odds["st_H"]  = _inp("st_H",  "2HT 1 (Casa)")
            odds["st_D"]  = _inp("st_D",  "2HT X (Pari)")
            odds["st_A"]  = _inp("st_A",  "2HT 2 (Ospite)")
            odds["st_1X"] = _inp("st_1X", "2HT 1X")
            odds["st_X2"] = _inp("st_X2", "2HT X2")
        with c2:
            st.caption("Gol Secondo Tempo")
            for t in ["05","15","25"]:
                odds[f"st_over{t}"]  = _inp(f"st_over{t}",  f"2HT Over {t[0]}.{t[1]}")
                odds[f"st_under{t}"] = _inp(f"st_under{t}", f"2HT Under {t[0]}.{t[1]}")
            odds["st_gg"] = _inp("st_gg", "2HT GG")
            odds["st_ng"] = _inp("st_ng", "2HT NG")

    # ── TAB 7: Corner ─────────────────────────────────────────────────────────
    with tab7:
        st.caption("Corner totali")
        c1,c2 = st.columns(2)
        with c1:
            for t in ["75","85","95","105","115"]:
                odds[f"corn_over{t}"]  = _inp(f"corn_over{t}",  f"Corner Over {t[:-1]}.{t[-1]}")
        with c2:
            for t in ["75","85","95","105","115"]:
                odds[f"corn_under{t}"] = _inp(f"corn_under{t}", f"Corner Under {t[:-1]}.{t[-1]}")
        st.caption("Corner Casa / Ospite")
        c1,c2 = st.columns(2)
        with c1:
            st.caption(f"Corner {home[:12]}")
            for t in ["35","45","55"]:
                odds[f"corn_h_over{t}"]  = _inp(f"corn_h_over{t}",  f"Casa Over {t[0]}.{t[1]}")
                odds[f"corn_h_under{t}"] = _inp(f"corn_h_under{t}", f"Casa Under {t[0]}.{t[1]}")
        with c2:
            st.caption(f"Corner {away[:12]}")
            for t in ["35","45","55"]:
                odds[f"corn_a_over{t}"]  = _inp(f"corn_a_over{t}",  f"Ospite Over {t[0]}.{t[1]}")
                odds[f"corn_a_under{t}"] = _inp(f"corn_a_under{t}", f"Ospite Under {t[0]}.{t[1]}")

    # ── TAB 8: Marcatori top 5 ────────────────────────────────────────────────
    with tab8:
        try:
            _pb8 = load_player_stats_builder()
            _opp_h = _pb8.get_team_context(away)
            _opp_a = _pb8.get_team_context(home)
            _xg_h8 = 1.4
            _xg_a8 = 1.2
            _sc_h = _pb8.scorer_probability(home, _xg_h8, opp_context=_opp_h, limit=5)
            _sc_a = _pb8.scorer_probability(away, _xg_a8, opp_context=_opp_a, limit=5)
            c1, c2 = st.columns(2)
            with c1:
                st.caption(f"⚽ Marcatori {home}")
                for p in _sc_h:
                    pid = str(p['id'])
                    label = f"{p['name']} ({int(p['prob_score']*100)}% QE {round(1/p['prob_score'],1) if p['prob_score']>0 else 99})"
                    odds[f"scorer_h_{pid}"] = _inp(f"scorer_h_{pid}", label)
            with c2:
                st.caption(f"⚽ Marcatori {away}")
                for p in _sc_a:
                    pid = str(p['id'])
                    label = f"{p['name']} ({int(p['prob_score']*100)}% QE {round(1/p['prob_score'],1) if p['prob_score']>0 else 99})"
                    odds[f"scorer_a_{pid}"] = _inp(f"scorer_a_{pid}", label)
        except Exception as _e:
            st.info(f"Dati marcatori non disponibili: {_e}")

    # ── TAB 9: Ammoniti top 5 ────────────────────────────────────────────────
    with tab9:
        try:
            from models.cards_model import CardsModel
            _pb9 = load_player_stats_builder()
            _opp_h9 = _pb9.get_team_context(away)
            _opp_a9 = _pb9.get_team_context(home)
            try:
                _cp9 = CardsModel(df_raw).predict(home, away, referee if referee else None)
                _exp_h9 = _cp9.get('home_expected', 1.8)
                _exp_a9 = _cp9.get('away_expected', 1.8)
            except Exception:
                _exp_h9, _exp_a9 = 1.8, 1.8
            _bk_h = _pb9.booking_probability(home, _exp_h9, opp_context=_opp_h9, limit=5)
            _bk_a = _pb9.booking_probability(away, _exp_a9, opp_context=_opp_a9, limit=5)
            c1, c2 = st.columns(2)
            with c1:
                st.caption(f"🟨 Ammoniti {home}")
                for p in _bk_h:
                    pid = str(p['id'])
                    label = f"{p['name']} ({p.get('position','?')}) — {int(p['prob_booking']*100)}% QE {round(1/p['prob_booking'],1) if p['prob_booking']>0 else 99}"
                    odds[f"booking_h_{pid}"] = _inp(f"booking_h_{pid}", label)
            with c2:
                st.caption(f"🟨 Ammoniti {away}")
                for p in _bk_a:
                    pid = str(p['id'])
                    label = f"{p['name']} ({p.get('position','?')}) — {int(p['prob_booking']*100)}% QE {round(1/p['prob_booking'],1) if p['prob_booking']>0 else 99}"
                    odds[f"booking_a_{pid}"] = _inp(f"booking_a_{pid}", label)
        except Exception as _e:
            st.info(f"Dati ammoniti non disponibili: {_e}")

    odds_clean = {k: v for k, v in odds.items() if v > 1.0}
    st.divider()

    st.session_state[match_key] = {k: v for k, v in odds.items() if v > 0}
    _all_odds[match_key] = st.session_state[match_key]
    _odds_file.write_text(_json.dumps(_all_odds, ensure_ascii=False, indent=2))
    if st.button("⚡ Analizza partita", type="primary", use_container_width=True):
        from models.markets import compute_all_markets, find_value_bets_extended, UNRELIABLE_MARKETS
        from models.features import get_feature_columns

        import datetime
        match_date = datetime.datetime.now().strftime('%Y-%m-%d')
        match_time = None
        try:
            upcoming = df[(df['HomeTeam']==home) & (df['AwayTeam']==away)].sort_values('Date').tail(1)
            if not upcoming.empty:
                match_date = str(upcoming['Date'].values[0])[:10]
                if 'Time' in upcoming.columns:
                    match_time = str(upcoming['Time'].values[0])
        except:
            pass

        # Scarica quote Pinnacle automaticamente
        # Infortuni dal backend
        if BACKEND_CLIENT_OK and is_backend_available():
            infortuni_home = api_get_injuries(home)
            infortuni_away = api_get_injuries(away)
            if infortuni_home or infortuni_away:
                st.warning(f"🏥 Infortuni: " +
                    (f"{home}: {', '.join(i['player'] + ' (' + i['status'] + ')' for i in infortuni_home)}" if infortuni_home else "") +
                    (" | " if infortuni_home and infortuni_away else "") +
                    (f"{away}: {', '.join(i['player'] + ' (' + i['status'] + ')' for i in infortuni_away)}" if infortuni_away else ""))

        pinnacle_odds = {}
        try:
            from data.odds_tracker import get_odds_for_match
            pinnacle_odds = get_odds_for_match(home, away)
        except Exception as e:
            st.caption(f"⚠️ Errore quote Pinnacle: {e}")
        if pinnacle_odds:
            st.caption(f"📡 Quote Pinnacle: H={pinnacle_odds.get('H','—')} D={pinnacle_odds.get('D','—')} A={pinnacle_odds.get('A','—')}")
        else:
            st.caption("📡 Quote Pinnacle non disponibili per questa partita")

        preds = compute_all_markets(model.poisson, home, away,
                                    match_date=match_date,
                                    match_time=match_time,
                                    odds_h=pinnacle_odds.get('H'),
                                    odds_d=pinnacle_odds.get('D'),
                                    odds_a=pinnacle_odds.get('A'))
        st.session_state[f"preds_{home}_{away}"] = preds
        if not preds:
            st.error("Squadre non trovate nel modello. Controlla i nomi.")
        else:
            cards_ema = float(df[df["HomeTeam"]==home]["f_cards_ema_total"].dropna().tail(1).values[0]) if "f_cards_ema_total" in df.columns and len(df[df["HomeTeam"]==home]) > 0 else 3.8

            # Aggiustamento arbitro Serie A
            try:
                from models.referee import get_referee_adjustments
                ref_adj = get_referee_adjustments(referee)
                if ref_adj["affidabile"]:
                    cards_mult = ref_adj["cards_mult"]
                    gialli_att = ref_adj["gialli_attesi"]
                    st.caption(f"👨‍⚖️ {referee}: {gialli_att:.1f} gialli/p "
                              f"(media Serie A: 4.18) — moltiplicatore: {cards_mult:.2f}x")
                    # Aggiusta probabilità cartellini
                    for key in ["prob_cards_over25","prob_cards_over35",
                                "prob_cards_over45","prob_cards_over55"]:
                        if key in preds:
                            preds[key] = min(0.99, preds[key] * cards_mult)
                    for key in ["prob_cards_under25","prob_cards_under35",
                                "prob_cards_under45","prob_cards_under55"]:
                        if key in preds:
                            preds[key] = max(0.01, preds[key] * (2 - cards_mult))
            except Exception as e:
                pass
            cards_preds = compute_cards_markets(
                preds["prob_D"], preds["prob_over25"], {}, cards_ema_total=cards_ema)
            preds.update(cards_preds)

            # ── Probabilità principali ────────────────────────────────────────
            st.subheader(f"📊 {home} vs {away}")
            c1,c2,c3 = st.columns(3)
            c1.metric(f"1 — {home}",   f"{preds['prob_H']*100:.1f}%")
            c2.metric("X — Pareggio",  f"{preds['prob_D']*100:.1f}%")
            c3.metric(f"2 — {away}",   f"{preds['prob_A']*100:.1f}%")
            c4,c5,c6 = st.columns(3)
            c4.metric("Over 2.5",   f"{preds['prob_over25']*100:.1f}%")
            c5.metric("Goal/Goal",  f"{preds['prob_gg']*100:.1f}%")
            c6.metric("Gialli attesi", f"{cards_preds['cards_lambda']:.1f}")

            # ── Riepilogo compatto: migliori value bet a colpo d'occhio ─────────
            _summary_odds = odds_clean if odds_clean else {k: v for k, v in {
                "H": pinnacle_odds.get("H"), "D": pinnacle_odds.get("D"), "A": pinnacle_odds.get("A"),
            }.items() if v and v > 1.0} if pinnacle_odds else {}
            if _summary_odds:
                try:
                    _vbs_preview = find_value_bets_extended(preds, _summary_odds, min_edge=min_edge, bankroll=bankroll)
                    _top_vbs = sorted([v for v in _vbs_preview if v.get("edge_%", 0) > 0],
                                       key=lambda v: -v.get("edge_%", 0))[:3]
                    if _top_vbs:
                        st.markdown("**🎯 Migliori value bet**")
                        _vb_cols = st.columns(len(_top_vbs))
                        for _col, v in zip(_vb_cols, _top_vbs):
                            _badge = "🟢" if v.get("affidabile") else "🟡"
                            _col.metric(f"{_badge} {v.get('mercato','?')}", f"@ {v.get('quota','?')}",
                                        f"+{v.get('edge_%',0):.0f}% edge")
                    else:
                        st.caption("🎯 Nessuna value bet sopra la soglia minima con le quote inserite finora")
                except Exception:
                    pass
            else:
                st.caption("🎯 Inserisci le quote nei tab qui sotto per vedere le migliori value bet a colpo d'occhio")

            # ── Tabella quote eque ────────────────────────────────────────────
            st.subheader("📐 Quote eque del modello")
            st.caption("Se la quota del bookmaker è **più alta** della quota equa → potenziale valore")

            prob_to_odds_key = {
                "prob_H":"H","prob_D":"D","prob_A":"A",
                "prob_1X":"1X","prob_X2":"X2","prob_12":"12",
                "prob_over15":"over15","prob_under15":"under15",
                "prob_over25":"over25","prob_under25":"under25",
                "prob_over35":"over35","prob_under35":"under35",
                "prob_over45":"over45","prob_under45":"under45",
                "prob_over55":"over55","prob_under55":"under55",
                "prob_gg":"gg","prob_ng":"ng",
                "prob_cards_over25":"cards_over25",
                "prob_cards_under25":"cards_under25",
                "prob_cards_over35":"cards_over35",
                "prob_cards_under35":"cards_under35",
                "prob_cards_over45":"cards_over45",
                "prob_cards_under45":"cards_under45",
                "prob_cards_over55":"cards_over55",
                "prob_cards_under55":"cards_under55",
                "ht_prob_H":"ht_H",
                "ht_prob_D":"ht_D",
                "ht_prob_A":"ht_A",
                "ht_prob_1X":"ht_1X",
                "ht_prob_X2":"ht_X2",
                "ht_prob_over05":"ht_over05",
                "ht_prob_under05":"ht_under05",
                "ht_prob_over15":"ht_over15",
                "ht_prob_under15":"ht_under15",
                "ht_prob_gg":"ht_gg",
                "ht_prob_ng":"ht_ng",
            }
            labels_fair = {
                "prob_H":"1 — Vittoria Casa","prob_D":"X — Pareggio",
                "prob_A":"2 — Vittoria Ospite","prob_1X":"1X","prob_X2":"X2","prob_12":"12",
                "prob_over15":"Over 1.5","prob_under15":"Under 1.5",
                "prob_over25":"Over 2.5","prob_under25":"Under 2.5",
                "prob_over35":"Over 3.5","prob_under35":"Under 3.5",
                "prob_over45":"Over 4.5","prob_under45":"Under 4.5",
                "prob_over55":"Over 5.5","prob_under55":"Under 5.5",
                "prob_gg":"Goal/Goal","prob_ng":"No Goal",
                "prob_cards_over25":"🟨 Cart. Over 2.5",
                "prob_cards_under25":"🟨 Cart. Under 2.5",
                "prob_cards_over35":"🟨 Cart. Over 3.5",
                "prob_cards_under35":"🟨 Cart. Under 3.5",
                "prob_cards_over45":"🟨 Cart. Over 4.5",
                "prob_cards_under45":"🟨 Cart. Under 4.5",
                "prob_cards_over55":"🟨 Cart. Over 5.5",
                "prob_cards_under55":"🟨 Cart. Under 5.5",
                "ht_prob_H":"⏱ HT 1 Casa",
                "ht_prob_D":"⏱ HT X Pareggio",
                "ht_prob_A":"⏱ HT 2 Ospite",
                "ht_prob_1X":"⏱ HT 1X",
                "ht_prob_X2":"⏱ HT X2",
                "ht_prob_over05":"⏱ HT Over 0.5",
                "ht_prob_under05":"⏱ HT Under 0.5",
                "ht_prob_over15":"⏱ HT Over 1.5",
                "ht_prob_under15":"⏱ HT Under 1.5",
                "ht_prob_gg":"⏱ HT GG",
                "ht_prob_ng":"⏱ HT NG",
            }

            fair_rows = []
            for k, label in labels_fair.items():
                prob = preds.get(k)
                if prob is None or prob <= 0:
                    continue
                quota_equa = round(1 / prob, 2)
                odds_key   = prob_to_odds_key.get(k, "")
                quota_bk   = odds_clean.get(odds_key)
                warning    = "⚠️" if label in UNRELIABLE_MARKETS else ""
                if quota_bk and quota_bk > 1:
                    edge = round((prob * quota_bk - 1) * 100, 1)
                    if edge >= min_edge * 100:
                        edge_str = f"✅ +{edge}%"
                    elif edge > 0:
                        edge_str = f"+{edge}%"
                    else:
                        edge_str = f"{edge}%"
                    bk_str = str(quota_bk)
                else:
                    edge_str = "—"
                    bk_str   = "—"
                fair_rows.append({
                    "Mercato":          f"{warning} {label}".strip(),
                    "Prob. modello":    f"{prob*100:.1f}%",
                    "Quota equa":       quota_equa,
                    "Quota bookmaker":  bk_str,
                    "Edge":             edge_str,
                })

            st.dataframe(pd.DataFrame(fair_rows), hide_index=True,
                         use_container_width=True)

            if any(r["Mercato"].startswith("⚠️") for r in fair_rows
                   if r["Edge"] not in ("—","") and r["Edge"].startswith("✅")):
                st.warning("⚠️ **Vittoria Casa**: ROI storico negativo (−12%). "
                           "Gioca con cautela o evita questo mercato.")

            # ── Value bets ────────────────────────────────────────────────────
            st.subheader("💰 Value Bets consigliate")
            st.caption(f"Edge minimo: {min_edge*100:.0f}% · Kelly 1/8 · "
                       f"Bankroll: €{bankroll:,}")
            if not odds_clean and pinnacle_odds:
                odds_clean = {k: v for k, v in {
                    "H": pinnacle_odds.get("H"),
                    "D": pinnacle_odds.get("D"),
                    "A": pinnacle_odds.get("A"),
                }.items() if v and v > 1.0}
                if odds_clean:
                    st.info("📡 Analisi automatica con quote Pinnacle — inserisci le quote Marathonbet nei tab per un confronto personalizzato")

            if not odds_clean:
                st.info("Inserisci almeno una quota per vedere le value bet.")
            else:
                vbs = find_value_bets_extended(
                    preds, odds_clean, min_edge=min_edge, bankroll=bankroll)
                # Salva automaticamente le stelline nel bet tracker
                try:
                    from utils.bet_tracker import add_bets
                    star_vbs = [v for v in vbs if v.get("affidabile") and v.get("edge_%", 0) >= 15]
                    if star_vbs:
                        import datetime
                        match_date = str(datetime.date.today())
                        # Calcola giornata dal calendario
                        _round = None
                        try:
                            import pandas as _pd2
                            _cal = _pd2.read_csv("cache/calendario_2627.csv")
                            _match = _cal[(_cal["home"].str.lower() == home.lower()) & (_cal["away"].str.lower() == away.lower())]
                            if not _match.empty:
                                _round = int(_match.iloc[0]["giornata"])
                        except:
                            pass
                        n_saved = add_bets(home, away, match_date, star_vbs, round_num=_round)
                        if n_saved > 0:
                            st.toast(f"💾 {n_saved} giocate salvate nel tracker", icon="✅")
                            # Invia su Telegram solo se ci sono nuove giocate
                            try:
                                from utils.telegram_bot import send_value_bet_alert
                                send_value_bet_alert(home, away, match_date, star_vbs, {})
                            except Exception as _tge:
                                pass
                            pass
                except Exception as _te:
                    pass
                # ── Nuovi modelli: corner, cartellini, multigoal, giocatori ──────
                try:
                    from models.corner_model import CornerModel
                    from models.cards_model import CardsModel
                    from models.multigoal import multigoal_probs as mg_probs, score_matrix
                    from components.match_analysis import render_compact_analysis

                    _corner_model = CornerModel(df_raw)
                    _cards_model = CardsModel(df_raw, ref_stats if ref_adj.get('affidabile') else None)
                    _player_builder = load_player_stats_builder()
                    _opp_ctx_home = _player_builder.get_team_context(away)
                    _opp_ctx_away = _player_builder.get_team_context(home)

                    _corner_pred = _corner_model.predict(home, away)
                    _cards_pred = _cards_model.predict(home, away, referee if referee else None)

                    # Lambda Poisson per multigoal e marcatori
                    _lam_h = _lam_a = None
                    try:
                        _mat = model.poisson.predict_score_matrix(home, away)
                        if _mat is not None:
                            import numpy as _np
                            if home in model.poisson.attack and away in model.poisson.attack:
                                _lam_h = float(model.poisson.attack[home] / model.poisson.defense[away] * model.poisson.avg_goals * _np.exp(model.poisson.home_adv))
                                _lam_a = float(model.poisson.attack[away] / model.poisson.defense[home] * model.poisson.avg_goals)
                    except Exception:
                        pass

                    _mg = mg_probs(_lam_h or 1.4, _lam_a or 1.2) if _lam_h else {}
                    _score_data = score_matrix(_lam_h or 1.4, _lam_a or 1.2) if _lam_h else {}

                    _h_exp_goals = _lam_h or 1.4
                    _a_exp_goals = _lam_a or 1.2
                    _h_exp_cards = _cards_pred.get('home_expected', 1.5)
                    _a_exp_cards = _cards_pred.get('away_expected', 1.5)

                    _home_scorers = _player_builder.scorer_probability(home, _h_exp_goals)
                    _away_scorers = _player_builder.scorer_probability(away, _a_exp_goals)
                    _home_bookings = _player_builder.booking_probability(home, _h_exp_cards)
                    _away_bookings = _player_builder.booking_probability(away, _a_exp_cards)

                    _probs_dict = {
                        'home': preds.get('prob_H', 0),
                        'draw': preds.get('prob_D', 0),
                        'away': preds.get('prob_A', 0),
                    }

                    st.divider()
                    st.session_state["_new_analysis_active"] = True
                    render_compact_analysis(
                        home=home, away=away,
                        preds=preds,
                        corner_pred=_corner_pred,
                        cards_pred=_cards_pred,
                        lam_h=_lam_h,
                        lam_a=_lam_a,
                        vbs=vbs,
                        referee=referee if referee else None,
                        home_scorers=_home_scorers,
                        away_scorers=_away_scorers,
                        home_bookings=_home_bookings,
                        away_bookings=_away_bookings,
                        opp_ctx_home=_opp_ctx_home,
                        opp_ctx_away=_opp_ctx_away,
                    )
                except Exception as _upgrade_err:
                    pass  # Fall back silenzioso se i nuovi modelli non caricano

                # ── Nuovi modelli: corner, cartellini, multigoal, giocatori ──────
# ═══ CALENDARIO ═══
    # Calcolatore cluster
    with st.expander("🎯 Calcolatore cluster risultati esatti"):
        st.caption("Seleziona i risultati da combinare e confronta con la quota del bookmaker")
        try:
            import numpy as _np
            _preds_key = f"preds_{home}_{away}"
            _mat = model.poisson.predict_score_matrix(home, away)
            if _mat is not None:
                _flat = [(_mat[ii,jj]*100, ii, jj) for ii in range(_mat.shape[0]) for jj in range(_mat.shape[1])]
                _top = sorted(_flat, reverse=True)[:15]
                st.markdown("**Seleziona i risultati:**")
                _sel = []
                _cols = st.columns(5)
                for _idx, (_p, _h, _a) in enumerate(_top):
                    _qe = round(100/_p, 2) if _p > 0 else 99.0
                    _lbl = f"{_h}-{_a} ({_p:.1f}% QE:{_qe})"
                    if _cols[_idx % 5].checkbox(_lbl, key=f"cl_{home}_{away}_{_h}_{_a}"):
                        _sel.append((_p, _h, _a))
                if _sel:
                    _tot = sum(_p for _p,_,_ in _sel)
                    _qe_tot = round(100/_tot, 3) if _tot > 0 else 99.0
                    _ris = " + ".join(f"{_h}-{_a}" for _,_h,_a in _sel)
                    st.markdown(
                        f'<div style="background:#e8f0fe;border-left:4px solid #3d84f7;border-radius:8px;padding:14px 18px;margin:8px 0">'
                        f'<b>Cluster:</b> {_ris}<br>'
                        f'<b>Probabilità totale:</b> {_tot:.1f}%<br>'
                        f'<b>Quota equa:</b> <span style="font-size:1.2rem;font-weight:800;color:#1565c0">{_qe_tot}</span>'
                        f'</div>', unsafe_allow_html=True)
                    _qbk = st.number_input("Quota bookmaker per questo cluster",
                        min_value=1.0, value=float(_qe_tot), step=0.05, format="%.2f",
                        key=f"clq_{home}_{away}")
                    _edge = (_tot/100 * _qbk - 1) * 100
                    if _edge >= 15:
                        st.success(f"✅ VALUE BET! Edge: +{_edge:.1f}% — Stake: €{bankroll*0.01:.2f}")
                    elif _edge >= 7:
                        st.warning(f"⚠️ Edge marginale: +{_edge:.1f}%")
                    else:
                        st.error(f"❌ No value: Edge {_edge:.1f}%")
        except:
            st.caption("Analizza prima la partita")

elif page == "📅 Calendario 26/27":
    import pickle, numpy as np
    from pathlib import Path
    st.title("📅 Calendario Serie A 2026/27")
    st.info("🏆 Inizio 22 agosto 2026 · Finale 25 maggio 2027")
    cal_path = Path("cache/calendario_2627.csv")
    if not cal_path.exists():
        st.warning("Calendario non disponibile.")
    else:
        cal = pd.read_csv(cal_path, parse_dates=["data"])
        giornate = sorted(cal["giornata"].unique())
        # Calcola giornata corrente in base alla data
        from datetime import date
        oggi = pd.Timestamp(date.today())
        import datetime as _dt
        adesso = pd.Timestamp(_dt.datetime.now())
        cal["datetime"] = pd.to_datetime(cal["data"].astype(str) + " " + cal["ora"].astype(str), errors="coerce")
        future = cal[cal["datetime"] >= adesso]
        if not future.empty:
            current_giornata = int(future.iloc[0]["giornata"])
        else:
            current_giornata = int(giornate[-1])
        default_idx = giornate.index(current_giornata) if current_giornata in giornate else 0
        giornata_sel = st.selectbox("Giornata", giornate,
            format_func=lambda x: f"Giornata {int(x)}", index=default_idx)
        gdf = cal[cal["giornata"] == giornata_sel].copy()
        gdf = gdf.sort_values(["data","ora"])
        model_ok = False
        try:
            with open("model_cache.pkl","rb") as f:
                model_cal = pickle.load(f)
            from models.markets import compute_all_markets
            model_ok = True
        except:
            pass
        GIORNI_IT = {
            "Monday":"Lunedi", "Tuesday":"Martedi", "Wednesday":"Mercoledi",
            "Thursday":"Giovedi", "Friday":"Venerdi", "Saturday":"Sabato", "Sunday":"Domenica"
        }
        MESI_IT = {
            "January":"gennaio","February":"febbraio","March":"marzo","April":"aprile",
            "May":"maggio","June":"giugno","July":"luglio","August":"agosto",
            "September":"settembre","October":"ottobre","November":"novembre","December":"dicembre"
        }
        for data_partita, gruppo in gdf.groupby("data"):
            ts = pd.Timestamp(data_partita)
            giorno_en = ts.strftime("%A")
            mese_en = ts.strftime("%B")
            data_it = f"{GIORNI_IT.get(giorno_en,giorno_en)} {ts.day} {MESI_IT.get(mese_en,mese_en)} {ts.year}"
            st.subheader(f"📅 {data_it}")
            for _, row in gruppo.iterrows():
                hc = row["home"]
                ac = row["away"]
                ora_raw = str(row.get("ora",""))
                try:
                    h, m = int(ora_raw[:2]), int(ora_raw[3:5])
                    h_it = (h + 2) % 24
                    ora_str = f"{h_it:02d}:{m:02d}"
                except:
                    ora_str = ""
                with st.expander(f"**{hc}** vs **{ac}**  ⏰ {ora_str}", expanded=False):
                    if model_ok:
                        try:
                            preds = compute_all_markets(
                                model_cal.poisson, hc, ac,
                                match_date=str(data_partita)[:10])
                            if preds:
                                ph = preds["prob_H"]*100
                                pdx = preds["prob_D"]*100
                                pa = preds["prob_A"]*100
                                c1,c2,c3 = st.columns(3)
                                c1.metric(f"1 {hc}", f"{ph:.1f}%")
                                c2.metric("X Pareggio", f"{pdx:.1f}%")
                                c3.metric(f"2 {ac}", f"{pa:.1f}%")
                                mat = model_cal.poisson.predict_score_matrix(hc, ac)
                                if mat is not None:
                                    flat = [(mat[i,j]*100, i, j) for i in range(mat.shape[0]) for j in range(mat.shape[1])]
                                    top10 = sorted(flat, reverse=True)[:10]
                                    st.caption("**Risultati più probabili:**")
                                    cols = st.columns(5)
                                    for idx_r, (prob, i, j) in enumerate(top10):
                                        qe = round(100/prob, 2) if prob > 0 else 99.99
                                        cols[idx_r % 5].metric(f"{i}-{j}", f"{prob:.1f}%", delta=f"QE: {qe}")
                        except Exception as _ec:
                            st.caption("-")
elif page == "👨‍⚖️ Arbitri":
    st.title("👨‍⚖️ Statistiche Arbitri")
    st.caption("Medie calcolate sul dataset storico (Premier League, Bundesliga, La Liga, Ligue 1)")
    st.info("ℹ️ Gli arbitri Serie A non sono disponibili in football-data.co.uk. Inserisci l'arbitro manualmente nella sezione Predizione usando il menu a tendina o il pulsante Cerca AIA.")
    if ref_stats.empty:
        st.warning("Colonna Referee non trovata nel dataset.")
    else:
        c1,c2,c3 = st.columns(3)
        c1.metric("Arbitri nel dataset", len(ref_stats))
        c2.metric("Media gialli Serie A", f"{ref_stats['gialli_media'].mean():.1f}")
        c3.metric("Media rossi Serie A",  f"{ref_stats['rossi_media'].mean():.2f}")
        st.divider()
        col1,col2 = st.columns(2)
        with col1:
            st.subheader("🔴 Più severi")
            top = ref_stats.nlargest(10,"gialli_media")[
                ["Referee","partite","gialli_media","rossi_media","severita_pct"]].copy()
            top.columns = ["Arbitro","Partite","Gialli/g","Rossi/g","Severità%"]
            for c in ["Gialli/g","Rossi/g","Severità%"]: top[c] = top[c].round(2)
            st.dataframe(top, hide_index=True, use_container_width=True)
        with col2:
            st.subheader("🟢 Più permissivi")
            bot = ref_stats.nsmallest(10,"gialli_media")[
                ["Referee","partite","gialli_media","rossi_media","severita_pct"]].copy()
            bot.columns = ["Arbitro","Partite","Gialli/g","Rossi/g","Severità%"]
            for c in ["Gialli/g","Rossi/g","Severità%"]: bot[c] = bot[c].round(2)
            st.dataframe(bot, hide_index=True, use_container_width=True)
        st.divider()
        st.subheader("⚖️ Bias casa/ospite")
        st.caption("Positivo = fischia più gialli all'ospite")
        bd = ref_stats[ref_stats["partite"]>=10].sort_values(
            "bias_casa", ascending=False)[
            ["Referee","partite","bias_casa","gialli_casa_avg","gialli_ospite_avg"]].copy()
        bd.columns=["Arbitro","Partite","Bias","Gialli casa","Gialli ospite"]
        for c in ["Bias","Gialli casa","Gialli ospite"]: bd[c] = bd[c].round(2)
        st.dataframe(bd, hide_index=True, use_container_width=True)
        st.divider()
        st.subheader("🔍 Cerca arbitro")
        search = st.text_input("Nome o cognome")
        if search:
            from models.referee import get_referee_features
            feat = get_referee_features(ref_stats, search)
            if feat.get("ref_found"):
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Gialli/partita",   f"{feat['ref_gialli_media']:.1f}")
                c2.metric("Rossi/partita",    f"{feat['ref_rossi_media']:.2f}")
                c3.metric("Percentile sev.",  f"{feat['ref_severita_pct']:.0f}°")
                c4.metric("Bias casa",        f"{feat['ref_bias_casa']:+.2f}")
                if st.button(f"Usa {search} nella prossima analisi"):
                    st.session_state["cal_ref"] = search
                    st.success("Impostato! Vai su 🔮 Predizione.")
            else:
                st.warning(f"'{search}' non trovato nel dataset.")

# ═══ BACKTEST ═══
elif page == "📊 Classifica":
    import json
    from pathlib import Path
    st.title("📊 Classifica Serie A 2026/27")
    standings_path = Path("cache/standings_detailed.json")
    if not standings_path.exists():
        st.warning("Classifica non disponibile. Lancia utils/weekly_update.py")
    else:
        data = json.loads(standings_path.read_text(encoding="utf-8"))
        api_table = data["api_table"]
        stats = data["stats"]
        name_map = {
            "AS Roma":"Roma","FC Internazionale Milano":"Inter","AC Milan":"Milan",
            "Como 1907":"Como","SSC Napoli":"Napoli","Atalanta BC":"Atalanta",
            "Juventus FC":"Juventus","ACF Fiorentina":"Fiorentina","SS Lazio":"Lazio",
            "Bologna FC 1909":"Bologna","US Sassuolo Calcio":"Sassuolo",
            "Parma Calcio 1913":"Parma","Genoa CFC":"Genoa","Udinese Calcio":"Udinese",
            "Torino FC":"Torino","Cagliari Calcio":"Cagliari","Venezia FC":"Venezia",
            "Frosinone Calcio":"Frosinone","US Lecce":"Lecce","AC Monza":"Monza",
        }
        def form_icon(results, n=5):
            last = results[-n:] if len(results) >= n else results
            c = {'W':'#16a34a','D':'#f59e0b','L':'#dc2626'}
            padding = '<span style="color:#e5e7eb">&#9679;</span>' * (n - len(last))
            dots = ''.join('<span style="color:' + c.get(r,'#e5e7eb') + ';font-size:14px">&#9679;</span>' for r in last)
            return padding + dots
        def trend_icon(results):
            last3 = results[-3:]
            pts = sum(3 if r=="W" else 1 if r=="D" else 0 for r in last3)
            if pts >= 7: return "🔥"
            elif pts >= 5: return "📈"
            elif pts >= 3: return "➡️"
            elif pts >= 1: return "📉"
            else: return "❌"
        # Vista selezionabile
        vista = st.radio("Vista", ["Totale", "Casa", "Trasferta"], horizontal=True)
        rows = []
        for entry in api_table:
            team_api = entry["team"]["name"]
            team = name_map.get(team_api, team_api)
            s = stats.get(team, {})
            form = s.get("form", [])
            if vista == "Totale":
                w = entry["won"]; d = entry["draw"]; l = entry["lost"]
                gf = entry["goalsFor"]; ga = entry["goalsAgainst"]
                pts = entry["points"]; pg = entry["playedGames"]
            elif vista == "Casa":
                w = s.get("home_w",0); d = s.get("home_d",0); l = s.get("home_l",0)
                gf = s.get("home_gf",0); ga = s.get("home_ga",0)
                pts = w*3+d; pg = w+d+l
            else:
                w = s.get("away_w",0); d = s.get("away_d",0); l = s.get("away_l",0)
                gf = s.get("away_gf",0); ga = s.get("away_ga",0)
                pts = w*3+d; pg = w+d+l
            rows.append({
                "Pos": entry["position"],
                "Trend": trend_icon(form),
                "Squadra": team,
                "PG": pg, "V": w, "P": d, "S": l,
                "GF": gf, "GS": ga, "DR": gf-ga,
                "Pts": pts,
                "Form": form_icon(form),
            })
        if vista != "Totale":
            rows = sorted(rows, key=lambda x: (-x["Pts"], -x["DR"], -x["GF"]))
            for i, r in enumerate(rows): r["Pos"] = i+1
        df_class = pd.DataFrame(rows)
        # Colora zone
        def color_zone(row):
            pos = row["Pos"]
            if pos <= 4: return ["background-color:#e8f0fe"]*len(row)
            elif pos <= 6: return ["background-color:#fff3e6"]*len(row)
            elif pos == 7: return ["background-color:#fffbe6"]*len(row)
            elif pos >= 18: return ["background-color:#ffe8e8"]*len(row)
            return [""]*len(row)
        st.markdown('<style>.cr2{display:flex;align-items:center;padding:6px 8px;border-bottom:1px solid #f1f5f9;font-size:13px;gap:6px}.cp2{min-width:30px;color:#9ca3af;font-size:12px}.ct2{flex:1;font-weight:500}.cg2{min-width:25px;text-align:center;color:#6b7280;font-size:11px}.cw2{min-width:52px;color:#6b7280;font-size:11px}.cgol2{min-width:38px;color:#6b7280;font-size:11px;text-align:center}.cdr2{min-width:28px;text-align:center;font-size:11px}.cpts2{min-width:34px;text-align:center;font-size:18px;font-weight:800;color:#111827}.cform2{letter-spacing:2px;min-width:80px;text-align:right}</style>', unsafe_allow_html=True)
        zone = {1:'#e8f0fe',2:'#e8f0fe',3:'#e8f0fe',4:'#e8f0fe',5:'#fff3e6',6:'#fff3e6',7:'#fffbe6',18:'#ffe8e8',19:'#ffe8e8',20:'#ffe8e8'}
        html_c = ''
        for r in rows:
            bg = zone.get(r['Pos'],'')
            bg_s = ('background:' + bg + ';') if bg else ''
            dr = r['DR']
            dr_col = '#16a34a' if dr>0 else ('#dc2626' if dr<0 else '#6b7280')
            dr_s = ('+' + str(dr)) if dr>0 else str(dr)
            html_c += '<div class="cr2" style="' + bg_s + '">'
            html_c += '<span class="cp2">' + str(r['Trend']) + ' ' + str(r['Pos']) + '</span>'
            html_c += '<span class="ct2">' + str(r['Squadra']) + '</span>'
            html_c += '<span class="cg2">' + str(r['PG']) + 'G</span>'
            html_c += '<span class="cw2">' + str(r['V']) + 'V ' + str(r['P']) + 'P ' + str(r['S']) + 'S</span>'
            html_c += '<span class="cgol2">' + str(r['GF']) + ':' + str(r['GS']) + '</span>'
            html_c += '<span class="cdr2" style="color:' + dr_col + '">' + dr_s + '</span>'
            html_c += '<span class="cpts2">' + str(r['Pts']) + '</span>'
            html_c += '<span class="cform2">' + str(r['Form']) + '</span>'
            html_c += '</div>'
        st.markdown(html_c, unsafe_allow_html=True)
        st.caption("🔵 Champions League · 🟠 Europa League · 🟡 Conference League · 🔴 Retrocessione")
        st.caption("Trend: 🔥 in forma · 📈 positivo · ➡️ stabile · 📉 calo · ❌ crisi")
elif page == "🎯 Tracker":
    stats = get_stats()
    bets_all = stats.get("bets", [])
    closed = [b for b in bets_all if b["status"] != "pending"]
    pending = [b for b in bets_all if b["status"] == "pending"]
    dash_closed = [b for b in closed if b.get("source","dashboard") == "dashboard"]
    tg_closed = [b for b in closed if b.get("source") == "telegram"]
    dash_pend = [b for b in pending if b.get("source","dashboard") == "dashboard"]
    tg_pend = [b for b in pending if b.get("source") == "telegram"]

    tot_profit = sum(b.get("profitto",0) or 0 for b in closed)
    tot_stake = sum(b.get("stake",4) for b in closed)
    tot_roi = tot_profit/tot_stake*100 if tot_stake > 0 else 0
    tot_won = sum(1 for b in closed if b["status"] == "won")
    wr = tot_won/len(closed)*100 if closed else 0

    # CSS del redesign
    st.markdown("""<style>
    .pnl-big { font-size: 52px; font-weight: 500; line-height: 1; margin-bottom: 4px; }
    .pnl-pos { color: #16a34a; }
    .pnl-neg { color: #dc2626; }
    .pnl-neu { color: #6b7280; }
    .pnl-sub { font-size: 13px; color: #6b7280; margin-bottom: 12px; }
    .progress-label { font-size: 12px; color: #9ca3af; margin-top: 3px; }

    /* Card partita — bordo e ombra leggera per separare */
    .match-group { 
        background: #ffffff; 
        border: 1px solid #e5e7eb; 
        border-radius: 12px; 
        overflow: hidden; 
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    /* Header partita — sfondo colorato scuro per distinguersi */
    .match-hdr { 
        padding: 12px 16px; 
        display: flex; 
        justify-content: space-between; 
        align-items: center; 
        background: #e2e8f0;
        border-bottom: none;
    }
    .match-name { font-size: 15px; font-weight: 500; color: #1e293b; }
    .match-date { font-size: 12px; color: #64748b; margin-right: 10px; }

    /* Badge fonte */
    .badge-dash { font-size: 10px; padding: 2px 8px; border-radius: 4px; background: #0f766e; color: #fff; margin-left: 8px; vertical-align: middle; font-weight: 600; letter-spacing: 0.03em; }
    .badge-tg { font-size: 10px; padding: 2px 8px; border-radius: 4px; background: #b45309; color: #fff; margin-left: 8px; vertical-align: middle; font-weight: 600; letter-spacing: 0.03em; }

    /* Profitto nella header */
    .match-profit-pos { font-size: 15px; font-weight: 600; color: #16a34a; }
    .match-profit-neg { font-size: 15px; font-weight: 600; color: #dc2626; }

    /* Riga singola bet dentro la card */
    .bet-line { 
        padding: 9px 16px; 
        border-top: 1px solid #f1f5f9; 
        font-size: 13px; 
        display: flex; 
        align-items: center; 
        gap: 10px;
        background: #fff;
    }
    .bet-line:hover { background: #f8fafc; }

    /* Icona vinta/persa */
    .bet-icon-w { color: #16a34a; font-weight: 700; font-size: 13px; min-width: 14px; }
    .bet-icon-l { color: #dc2626; font-weight: 700; font-size: 13px; min-width: 14px; }

    /* Testo mercato — grande e leggibile */
    .bet-market-name { flex: 1; font-size: 14px; font-weight: 500; color: #111827; }

    /* Dati secondari — più piccoli e grigi */
    .bet-quota-val { font-size: 13px; color: #374151; font-weight: 500; min-width: 52px; }
    .bet-prob-val { font-size: 12px; color: #6b7280; min-width: 60px; background: #f3f4f6; padding: 2px 7px; border-radius: 100px; }
    .bet-edge-val { font-size: 12px; color: #2563eb; min-width: 52px; }

    /* Profitto bet */
    .bet-profit-pos { font-weight: 600; font-size: 14px; color: #16a34a; min-width: 68px; text-align: right; }
    .bet-profit-neg { font-weight: 600; font-size: 14px; color: #dc2626; min-width: 68px; text-align: right; }

    .pend-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #ea580c; margin-right: 5px; vertical-align: middle; }
    .source-card { border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px 16px; }
    .src-title { font-size: 11px; color: #9ca3af; margin-bottom: 6px; }
    .src-profit { font-size: 22px; font-weight: 500; }
    .src-detail { font-size: 12px; color: #6b7280; margin-top: 2px; }
    .sec-label { font-size: 12px; color: #9ca3af; margin: 16px 0 6px; letter-spacing: 0.05em; text-transform: uppercase; }
    </style>""", unsafe_allow_html=True)

    # ── P&L Prominente ──────────────────────────────────────────────────────────
    prof_class = "pnl-pos" if tot_profit > 0 else ("pnl-neg" if tot_profit < 0 else "pnl-neu")
    st.markdown(f'<div class="pnl-big {prof_class}">{tot_profit:+.2f}€</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pnl-sub">Profitto netto · stagione 2026/27</div>', unsafe_allow_html=True)

    # Sparkline P&L cumulativo
    if closed:
        import pandas as _pd_sp
        _df_sp = _pd_sp.DataFrame(closed).sort_values("closed_at", na_position="last")
        _df_sp["cum"] = _df_sp["profitto"].fillna(0).cumsum()
        _vals = _df_sp["cum"].tolist()
        if len(_vals) > 1:
            _mn, _mx = min(_vals), max(_vals)
            _rng = _mx - _mn if _mx != _mn else 1
            _pts = " ".join([f"{int(i*400/max(len(_vals)-1,1))},{int(38 - (_v-_mn)/_rng*34)}" for i, _v in enumerate(_vals)])
            _area = _pts + f" {int((len(_vals)-1)*400/max(len(_vals)-1,1))},40 0,40"
            _spline_color = "#16a34a" if tot_profit >= 0 else "#dc2626"
            _area_color = "#bbf7d0" if tot_profit >= 0 else "#fee2e2"
            st.markdown(f'''<svg width="100%" height="42" viewBox="0 0 400 42" preserveAspectRatio="none" style="margin:6px 0 2px">
                <polygon points="{_area}" fill="{_area_color}" opacity="0.6"/>
                <polyline points="{_pts}" fill="none" stroke="{_spline_color}" stroke-width="1.5"/>
            </svg>''', unsafe_allow_html=True)

    # Progress bar stagione
    from utils.bet_tracker import load_tracker as _lt
    _all_rounds = set(b.get("round") for b in bets_all if b.get("round"))
    _cur_round = max(_all_rounds) if _all_rounds else 1
    _target = 127.0
    _pct = min(tot_profit / _target * 100, 100) if _target > 0 else 0
    st.progress(max(0, int(_pct)) / 100)
    st.markdown(f'<div class="progress-label">Giornata {_cur_round} di 38 · €{tot_profit:.0f} di €{_target:.0f} target stagionale</div>', unsafe_allow_html=True)

    # ── Stats Row ────────────────────────────────────────────────────────────────
    _s1, _s2, _s3, _s4 = st.columns(4)
    _s1.metric("Giocate", len(bets_all))
    _s2.metric("Vinte/Perse", f"{tot_won}/{len(closed)-tot_won}")
    _s3.metric("Win rate", f"{wr:.0f}%")
    _s4.metric("ROI", f"{tot_roi:+.1f}%")

    st.divider()

    # ── Source Cards ─────────────────────────────────────────────────────────────
    _c_d, _c_t = st.columns(2)
    with _c_d:
        _d_prof = sum(b.get("profitto",0) or 0 for b in dash_closed)
        _d_stake = sum(b.get("stake",4) for b in dash_closed)
        _d_roi = _d_prof/_d_stake*100 if _d_stake > 0 else 0
        _d_won = sum(1 for b in dash_closed if b["status"] == "won")
        _d_col = "#16a34a" if _d_prof > 0 else ("#dc2626" if _d_prof < 0 else "#6b7280")
        st.markdown(f'''<div class="source-card">
            <div class="src-title">📊 Dashboard</div>
            <div class="src-profit" style="color:{_d_col}">{_d_prof:+.2f}€</div>
            <div class="src-detail">{len(dash_closed)} chiuse · {_d_won} vinte · ROI {_d_roi:+.1f}%</div>
        </div>''', unsafe_allow_html=True)
    with _c_t:
        _t_prof = sum(b.get("profitto",0) or 0 for b in tg_closed)
        _t_stake = sum(b.get("stake",4) for b in tg_closed)
        _t_roi = _t_prof/_t_stake*100 if _t_stake > 0 else 0
        _t_won = sum(1 for b in tg_closed if b["status"] == "won")
        _t_col = "#16a34a" if _t_prof > 0 else ("#dc2626" if _t_prof < 0 else "#6b7280")
        st.markdown(f'''<div class="source-card">
            <div class="src-title">📱 Telegram</div>
            <div class="src-profit" style="color:{_t_col}">{_t_prof:+.2f}€</div>
            <div class="src-detail">{len(tg_closed)} chiuse · {_t_won} vinte · {len(tg_pend)} pendenti</div>
        </div>''', unsafe_allow_html=True)

    # ── Filtro per mercato ────────────────────────────────────────────────────
    if closed:
        st.markdown('<div class="sec-label">Analisi per mercato</div>', unsafe_allow_html=True)
        _all_markets = sorted(set(b["mercato"] for b in closed))
        _mkt_sel = st.selectbox("Seleziona mercato", ["Tutti"] + _all_markets, key="mkt_filter")

        _filtered = closed if _mkt_sel == "Tutti" else [b for b in closed if b["mercato"] == _mkt_sel]

        if _filtered:
            _f_won = [b for b in _filtered if b["status"] == "won"]
            _f_lost = [b for b in _filtered if b["status"] == "lost"]
            _f_profit = sum(b.get("profitto",0) or 0 for b in _filtered)
            _f_stake = sum(b.get("stake",4) for b in _filtered)
            _f_roi = _f_profit/_f_stake*100 if _f_stake > 0 else 0
            _f_wr = len(_f_won)/len(_filtered)*100 if _filtered else 0
            _avg_odds = sum(b.get("quota",0) for b in _filtered)/len(_filtered) if _filtered else 0
            _avg_edge = sum(b.get("edge_pct",0) for b in _filtered)/len(_filtered) if _filtered else 0

            _col1, _col2, _col3, _col4 = st.columns(4)
            _col1.metric("Giocate", len(_filtered))
            _col2.metric("Win Rate", f"{_f_wr:.0f}%")
            _col3.metric("ROI", f"{_f_roi:+.1f}%")
            _col4.metric("Profitto", f"€{_f_profit:+.2f}")

            _c1, _c2 = st.columns(2)
            _c1.metric("Quota media", f"{_avg_odds:.2f}")
            _c2.metric("Edge medio", f"+{_avg_edge:.1f}%")

            # Mini grafico P&L per mercato
            if len(_filtered) > 1:
                import pandas as _pd_mkt
                _df_mkt = _pd_mkt.DataFrame(_filtered).sort_values("closed_at", na_position="last")
                _df_mkt["cum"] = _df_mkt["profitto"].fillna(0).cumsum()
                _vals = _df_mkt["cum"].tolist()
                _mn, _mx = min(_vals), max(_vals)
                _rng = _mx - _mn if _mx != _mn else 1
                _pts = " ".join([f"{int(i*300/max(len(_vals)-1,1))},{int(38-(_v-_mn)/_rng*34)}" for i, _v in enumerate(_vals)])
                _area = _pts + f" {int((len(_vals)-1)*300/max(len(_vals)-1,1))},40 0,40"
                _spline_color = "#16a34a" if _f_profit >= 0 else "#dc2626"
                _area_color = "#bbf7d0" if _f_profit >= 0 else "#fee2e2"
                st.markdown(f'''<div style="margin:8px 0 4px;font-size:11px;color:#9ca3af">P&L cumulativo — {_mkt_sel}</div>
                <svg width="100%" height="42" viewBox="0 0 300 42" preserveAspectRatio="none">
                    <polygon points="{_area}" fill="{_area_color}" opacity="0.6"/>
                    <polyline points="{_pts}" fill="none" stroke="{_spline_color}" stroke-width="1.5"/>
                </svg>''', unsafe_allow_html=True)

    st.divider()

    # ── Chiudi giornata ──────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Chiudi giornata</div>', unsafe_allow_html=True)
    _col_rnd, _col_btn = st.columns([2, 1])
    _round_close = _col_rnd.number_input("Giornata", min_value=1, max_value=38, value=_cur_round, label_visibility="collapsed")
    if _col_btn.button("🔄 Aggiorna risultati", use_container_width=True):
        from utils.bet_tracker import close_bets_from_sofascore
        _n = close_bets_from_sofascore(int(_round_close))
        st.success(f"{_n} giocate chiuse")
        st.rerun()

    # ── Pendenti raggruppate per partita ─────────────────────────────────────────
    if pending:
        st.markdown(f'<div class="sec-label">Pendenti ({len(pending)})</div>', unsafe_allow_html=True)
        from collections import defaultdict as _dd
        _pend_groups = _dd(list)
        for b in pending:
            _key = f"{b['home']} vs {b['away']}"
            _pend_groups[_key].append(b)

        for _match_key, _bets in _pend_groups.items():
            _src = _bets[0].get("source","dashboard")
            _badge = '<span class="badge-dash">Dashboard</span>' if _src == "dashboard" else '<span class="badge-tg">Telegram</span>'
            _mdate = _bets[0].get("match_date","")[:10] if _bets[0].get("match_date") else ""
            st.markdown(f'''<div class="match-group">
            <div class="match-hdr">
                <div><span class="match-name">{_match_key}</span>{_badge}</div>
                <div><span class="match-date">{_mdate}</span><span style="color:#ea580c;font-size:12px;font-weight:500"><span class="pend-dot"></span>{len(_bets)} pending</span></div>
            </div></div>''', unsafe_allow_html=True)

            for _b in _bets:
                try:
                    _prob = round(((_b.get("edge_pct",0)/100)+1) / _b.get("quota",2) * 100, 1)
                except:
                    _prob = 0
                _c1, _c2, _c3, _c4, _c5 = st.columns([3, 1.2, 1.2, 1.2, 1.8])
                _c1.markdown(f'**{_b["mercato"]}**')
                _c2.markdown(f'@ **{_b["quota"]:.2f}**')
                _c3.markdown(f'<span style="color:#2563eb;font-size:12px">Prob {_prob:.0f}%</span>', unsafe_allow_html=True)
                _c4.markdown(f'<span style="color:#2563eb;font-size:12px">+{_b.get("edge_pct",0):.0f}%</span>', unsafe_allow_html=True)
                _cb1, _cb2, _cb3 = _c5.columns(3)
                if _cb1.button("✅", key=f"w_{_b['id']}", help="Vinta"):
                    from utils.bet_tracker import load_tracker, save_tracker
                    import datetime as _dtt
                    _dd2 = load_tracker()
                    for _bx in _dd2["bets"]:
                        if _bx["id"] == _b["id"]:
                            _bx["status"] = "won"
                            _bx["profitto"] = round(_bx["stake"]*(_bx["quota"]-1),2)
                            _bx["closed_at"] = _dtt.datetime.now().isoformat()[:19]
                            _bx["result"] = "manuale"
                            break
                    save_tracker(_dd2); st.rerun()
                if _cb2.button("❌", key=f"l_{_b['id']}", help="Persa"):
                    from utils.bet_tracker import load_tracker, save_tracker
                    import datetime as _dtt
                    _dd2 = load_tracker()
                    for _bx in _dd2["bets"]:
                        if _bx["id"] == _b["id"]:
                            _bx["status"] = "lost"
                            _bx["profitto"] = -_bx["stake"]
                            _bx["closed_at"] = _dtt.datetime.now().isoformat()[:19]
                            _bx["result"] = "manuale"
                            break
                    save_tracker(_dd2); st.rerun()
                if _cb3.button("🗑", key=f"d_{_b['id']}", help="Elimina"):
                    from utils.bet_tracker import load_tracker, save_tracker
                    _trkr = load_tracker()
                    _trkr["bets"] = [_bx for _bx in _trkr["bets"] if _bx["id"] != _b["id"]]
                    save_tracker(_trkr); st.rerun()

    # ── Storico raggruppato per partita ──────────────────────────────────────────
    if closed:
        st.markdown('<div class="sec-label">Storico</div>', unsafe_allow_html=True)
        from collections import defaultdict as _ddc
        _hist_groups = _ddc(list)
        for b in sorted(closed, key=lambda x: x.get("closed_at",""), reverse=True):
            _key = f"{b['home']} vs {b['away']}"
            _hist_groups[_key].append(b)

        for _match_key, _bets in _hist_groups.items():
            _src = _bets[0].get("source","dashboard")
            _badge = '<span class="badge-dash">Dashboard</span>' if _src == "dashboard" else '<span class="badge-tg">Telegram</span>'
            _match_profit = sum(b.get("profitto",0) or 0 for b in _bets)
            _mdate = _bets[0].get("match_date","")[:10] if _bets[0].get("match_date") else ""
            _prof_col = "#16a34a" if _match_profit > 0 else "#dc2626"
            st.markdown(f'''<div class="match-group">
            <div class="match-hdr">
                <div><span class="match-name">{_match_key}</span>{_badge}</div>
                <div><span class="match-date">{_mdate}</span>
                <span style="font-size:14px;font-weight:500;color:{_prof_col}">{_match_profit:+.2f}€</span></div>
            </div>''', unsafe_allow_html=True)

            for _b in _bets:
                _won = _b["status"] == "won"
                _icon = '✓' if _won else '✗'
                _icon_col = "#16a34a" if _won else "#dc2626"
                _prof = _b.get("profitto",0) or 0
                _prof_col2 = "#16a34a" if _prof > 0 else "#dc2626"
                try:
                    _prob2 = round(((_b.get("edge_pct",0)/100)+1) / _b.get("quota",2) * 100, 1)
                except:
                    _prob2 = 0
                st.markdown(f'''<div class="bet-line">
                    <span style="color:{_icon_col};font-weight:700;font-size:12px;min-width:14px">{_icon}</span>
                    <span class="bet-market-name">{_b["mercato"]}</span>
                    <span class="bet-quota-val">@ {_b["quota"]:.2f}</span>
                    <span class="bet-prob-val">Prob {_prob2:.0f}%</span>
                    <span class="bet-edge-val">+{_b.get("edge_pct",0):.0f}%</span>
                    <span style="font-weight:500;color:{_prof_col2};min-width:65px;text-align:right">{_prof:+.2f}€</span>
                </div></div>''', unsafe_allow_html=True)

elif page == "📈 Backtest":
    st.caption("Il modello si allena solo sul passato e prevede solo il futuro — nessun lookahead bias")

    st.info("""
    **Walk-forward**: per ogni stagione testata, il modello viene riallenato da zero
    usando solo le stagioni precedenti. È il modo corretto per simulare le performance reali.
    **Arbitro rolling**: le statistiche di ogni arbitro vengono aggiornate partita per partita,
    usando solo le partite già giocate fino a quel momento.
    """)

    col1, col2 = st.columns(2)
    min_edge_bt  = col1.slider("Edge minimo (%)", 1, 30, 7) / 100
    bankroll_bt  = col2.number_input("Bankroll simulato (€)", min_value=100,
                                      max_value=100000, value=1000, step=100)

    col3, col4 = st.columns(2)
    show_referee = col3.checkbox("Mostra analisi per arbitro", value=True)

    st.warning("⏱️ Il walk-forward è più lento del backtest semplice — "
               "allena un modello per ogni stagione. Su 8 stagioni: ~5-8 minuti.")

    if st.button("▶️ Avvia Walk-Forward Backtest", type="primary",
                 use_container_width=True):
        from models.backtest import run_walkforward_backtest, summarize_walkforward, summarize_by_referee

        with st.spinner("Walk-forward in corso — alleno il modello stagione per stagione..."):
            sa_only = df_raw[df_raw["league"] == "Serie A"] if "league" in df_raw.columns else df_raw
        bt = run_walkforward_backtest(
                sa_only,
                min_edge=min_edge_bt,
                bankroll=bankroll_bt,
                verbose=False,
            )

        if bt.empty:
            st.warning("Nessuna value bet trovata. Prova a ridurre l'edge minimo.")
        else:
            # ── Metriche globali ──────────────────────────────────────────────
            tot_stake  = bt["stake_€"].sum()
            tot_profit = bt["profitto_€"].sum()
            roi        = tot_profit / tot_stake * 100 if tot_stake > 0 else 0
            wr         = (bt["esito"] == "✓ Vinta").mean() * 100
            ref_cov    = bt["ref_found"].mean() * 100 if "ref_found" in bt.columns else 0

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Giocate",       f"{len(bt)}")
            c2.metric("Win Rate",      f"{wr:.1f}%")
            c3.metric("Profitto",      f"€{tot_profit:+,.2f}",
                      delta=f"{roi:+.1f}% ROI")
            c4.metric("Stake totale",  f"€{tot_stake:,.2f}")
            c5.metric("Copertura arb.", f"{ref_cov:.0f}%",
                      help="% di partite con dati arbitro disponibili")

            # ── P&L cumulativo ────────────────────────────────────────────────
            st.subheader("📈 P&L Cumulativo (walk-forward)")
            bt_s = bt.sort_values("data").copy()
            bt_s["pnl_cum"] = bt_s["profitto_€"].cumsum()
            st.line_chart(bt_s.set_index("data")["pnl_cum"])

            # ── Per stagione ──────────────────────────────────────────────────
            st.subheader("📅 Performance per stagione")
            summary = summarize_walkforward(bt)
            summary = summary[["stagione","giocate","profitto_€","win_rate","roi_%"]]
            summary.columns = ["Stagione","Giocate","Profitto €","Win Rate %","ROI %"]
            st.dataframe(summary, hide_index=True, use_container_width=True)

            # ── Per mercato ───────────────────────────────────────────────────
            st.subheader("📋 Performance per mercato")
            mp = bt.groupby("mercato").agg(
                giocate=("stake_€","count"),
                stake  =("stake_€","sum"),
                profit =("profitto_€","sum"),
            ).reset_index()
            mp["ROI%"] = (mp["profit"]/mp["stake"]*100).round(1)
            mp["profit"] = mp["profit"].round(2)
            mp = mp.sort_values("ROI%", ascending=False)
            mp.columns = ["Mercato","Giocate","Stake €","Profitto €","ROI %"]
            st.dataframe(mp, hide_index=True, use_container_width=True)

            # ── Per arbitro ───────────────────────────────────────────────────
            if show_referee:
                ref_summary = summarize_by_referee(bt)
                if not ref_summary.empty:
                    st.subheader("👨‍⚖️ Performance per arbitro")
                    st.caption("Solo arbitri con almeno 10 giocate nel backtest")
                    ref_summary.columns = ["Arbitro","Giocate","Stake €",
                                           "Profitto €","Gialli medi","ROI %"]
                    st.dataframe(ref_summary, hide_index=True,
                                 use_container_width=True)
                else:
                    st.caption("Dati arbitro non disponibili — "
                               "colonna Referee assente nel dataset")

            # ── Download ──────────────────────────────────────────────────────
            with st.expander("📜 Log completo"):
                cols_show = ["stagione","data","partita","arbitro","mercato",
                             "quota","prob_%","edge_%","stake_€","esito","profitto_€"]
                cols_show = [c for c in cols_show if c in bt.columns]
                st.dataframe(bt[cols_show], hide_index=True,
                             use_container_width=True)
                st.download_button(
                    "⬇️ Scarica CSV",
                    bt[cols_show].to_csv(index=False).encode(),
                    "backtest_walkforward.csv", "text/csv")


# ═══ INFO ═══
elif page == "ℹ️ Info modello":
    st.title("ℹ️ Info Modello")
    c1,c2,c3 = st.columns(3)
    c1.metric("Partite totali", f"{len(df):,}")
    c2.metric("Stagioni", df["season"].nunique() if "season" in df.columns else "—")
    c3.metric("Squadre",  df["HomeTeam"].nunique())
    if "Date" in df.columns:
        c4,c5 = st.columns(2)
        c4.metric("Prima partita",  str(df["Date"].min().date()))
        c5.metric("Ultima partita", str(df["Date"].max().date()))

    st.subheader("🔧 Fix v3 attivi")
    st.markdown("""
| Fix | Dettaglio | Impatto |
|---|---|---|
| Home correction | −10pp prob. vittoria casa, redistribuita su X e 2 | ROI da −1.8% a +4.2% |
| Kelly 1/8 | Dimezza stake rispetto a Kelly 1/4 | Dimezza drawdown massimo |
| Edge minimo 7% | Ottimizzato su 8 stagioni di backtest | Filtra giocate borderline |
| Warning vittoria casa | Segnala mercato con ROI storico −12% | Evita perdite sistematiche |
    """)

    st.subheader("🔬 Come funziona il modello")
    st.markdown("""
**1. Poisson bivariato** — stima forza offensiva/difensiva → probabilità tutti i mercati.  
**2. Gradient Boosting** — feature: EMA forma, xG, ELO, fixture density, H2H.  
**3. Ensemble** — 70% Poisson + 30% GBM.  
**4. Cartellini con arbitro** — Poisson calibrato su storico arbitro designato.  
**5. Kelly 1/8** — sizing ottimale con rischio controllato.  
**Value bet**: `prob × quota − 1 > edge_minimo`
    """)
    st.warning("Il modello non conosce infortuni o squalifiche. Gioca responsabilmente.")
