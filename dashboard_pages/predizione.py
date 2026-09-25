"""
Pagina "Predizione" — analisi partita: quote, arbitro, value bet.
Estratta da dashboard.py nella separazione in pagine (roadmap #2).
"""
import streamlit as st
import pandas as pd


def render(teams, model, df, df_raw, ref_stats, min_edge, bankroll,
           BACKEND_CLIENT_OK, is_backend_available,
           _cached_injuries, _cached_pinnacle_odds, load_player_stats_builder):
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
        # Bug pre-esistente: qui c'era anche un ramo "else:" sul try, che in
        # Python scatta ogni volta che il try NON solleva eccezioni — quindi
        # veniva eseguito SEMPRE dopo un try riuscito, anche quando la card
        # dell'arbitro sopra era stata mostrata correttamente, stampando un
        # fuorviante "Nessun arbitro" subito sotto. Rimosso: i due casi
        # (arbitro trovato / non specificato) sono già gestiti dall'if/else
        # dentro il try.
        st.info("ℹ️ Nessun arbitro — usata media Serie A per i cartellini")

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

                # 1X2 + DC Primo Tempo (id=341)
                try:
                    _r7 = _req.get("http://localhost:8000/marathonbet/serie-a-bet/pre-match/eventi?id_aggregata=341",timeout=10).json()
                    _p7 = _find_match(_r7.get("avs",[]), _home_mb, _away_mb)
                    if _p7:
                        for _sc in _p7.get("scs",[]):
                            _d = str(_sc.get("d",""))
                            _eqs = _sc.get("eqs",[])
                            if _d.startswith("1X2") and len(_eqs)==3:
                                for _eq in _eqs:
                                    _ce=_eq.get("ce"); _q=round(_eq.get("q",0)/100,2)
                                    if _q>1:
                                        if _ce==1: _new_odds["ht_H"]=_q
                                        elif _ce==2: _new_odds["ht_D"]=_q
                                        elif _ce==3: _new_odds["ht_A"]=_q
                            elif _d.startswith("1X ") and _eqs:
                                _q = round(_eqs[0].get("q",0)/100,2)
                                if _q>1: _new_odds["ht_1X"]=_q
                            elif _d.startswith("X2") and _eqs:
                                _q = round(_eqs[0].get("q",0)/100,2)
                                if _q>1: _new_odds["ht_X2"]=_q
                        if "ht_H" in _new_odds: _loaded.append("1X2+DC 1T")
                except Exception:
                    pass

                # 1X2 + DC Secondo Tempo (id=362)
                try:
                    _r8 = _req.get("http://localhost:8000/marathonbet/serie-a-bet/pre-match/eventi?id_aggregata=362",timeout=10).json()
                    _p8 = _find_match(_r8.get("avs",[]), _home_mb, _away_mb)
                    if _p8:
                        for _sc in _p8.get("scs",[]):
                            _d = str(_sc.get("d",""))
                            _eqs = _sc.get("eqs",[])
                            if _d.startswith("1X2") and len(_eqs)==3:
                                for _eq in _eqs:
                                    _ce=_eq.get("ce"); _q=round(_eq.get("q",0)/100,2)
                                    if _q>1:
                                        if _ce==1: _new_odds["st_H"]=_q
                                        elif _ce==2: _new_odds["st_D"]=_q
                                        elif _ce==3: _new_odds["st_A"]=_q
                            elif _d.startswith("1X ") and _eqs:
                                _q = round(_eqs[0].get("q",0)/100,2)
                                if _q>1: _new_odds["st_1X"]=_q
                            elif _d.startswith("X2") and _eqs:
                                _q = round(_eqs[0].get("q",0)/100,2)
                                if _q>1: _new_odds["st_X2"]=_q
                        if "st_H" in _new_odds: _loaded.append("1X2+DC 2T")
                except Exception:
                    pass

                # U/O Secondo Tempo (id=2903)
                try:
                    _r6 = _req.get("http://localhost:8000/marathonbet/serie-a-bet/pre-match/eventi?id_aggregata=2903",timeout=10).json()
                    _p6 = _find_match(_r6.get("avs",[]), _home_mb, _away_mb)
                    if _p6:
                        for _sc in _p6.get("scs",[]):
                            _eqs = _sc.get("eqs",[])
                            if len(_eqs)==2:
                                for _eq in _eqs:
                                    _ce=_eq.get("ce"); _q=round(_eq.get("q",0)/100,2)
                                    _csn=str(_eq.get("csn",""))
                                    if _q>1:
                                        for _s,_k1,_k2 in [("05","st_over05","st_under05"),("15","st_over15","st_under15"),("25","st_over25","st_under25")]:
                                            if _s in _csn:
                                                if _ce==1: _new_odds[_k1]=_q
                                                elif _ce==2: _new_odds[_k2]=_q
                        if any(k.startswith("st_") for k in _new_odds): _loaded.append("2HT O/U")
                except Exception:
                    pass

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
            infortuni_home = _cached_injuries(home)
            infortuni_away = _cached_injuries(away)
            if infortuni_home or infortuni_away:
                st.warning(f"🏥 Infortuni: " +
                    (f"{home}: {', '.join(i['player'] + ' (' + i['status'] + ')' for i in infortuni_home)}" if infortuni_home else "") +
                    (" | " if infortuni_home and infortuni_away else "") +
                    (f"{away}: {', '.join(i['player'] + ' (' + i['status'] + ')' for i in infortuni_away)}" if infortuni_away else ""))

        pinnacle_odds = {}
        try:
            pinnacle_odds = _cached_pinnacle_odds(home, away)
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

