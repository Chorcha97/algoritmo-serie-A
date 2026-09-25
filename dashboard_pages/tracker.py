"""
Pagina "Tracker" — storico e P&L delle giocate (dashboard + Telegram).
Estratta da dashboard.py nella separazione in pagine (roadmap #2).
"""
import streamlit as st


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


def render():
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

