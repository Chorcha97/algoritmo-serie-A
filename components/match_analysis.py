"""
components/match_analysis.py v3
- Tab corretti (9 tab, indici giusti)
- Tiri: shots/shots_ot da team_context
- QE marcatori/ammoniti: usa lambda reale passato dall'esterno
- Value Bet: solo value bet, non mischiato con analisi
"""
import streamlit as st
import numpy as np
from scipy.stats import poisson

CSS = """<style>
.cr{display:flex;align-items:center;padding:5px 0;border-bottom:1px solid #f1f5f9;font-size:13px;gap:8px}
.cr-l{flex:1;color:#374151}.cr-p{min-width:38px;text-align:right;font-weight:600;color:#111827}
.cr-q{min-width:62px;text-align:right;color:#9ca3af;font-size:11px}.cr-e{min-width:62px;text-align:right}
.ep{background:#d1fae5;color:#065f46;padding:1px 7px;border-radius:100px;font-size:11px;font-weight:600}
.en{background:#fee2e2;color:#991b1b;padding:1px 7px;border-radius:100px;font-size:11px}
.ou-grid{display:grid;grid-template-columns:1fr 1fr;gap:2px 12px}
.ou-row{display:flex;align-items:center;padding:4px 0;border-bottom:1px solid #f8fafc;font-size:13px;gap:6px}
.ou-l{flex:1;color:#374151;font-size:12px}.ou-p{min-width:36px;text-align:right;font-weight:600;color:#111827}
.ou-q{min-width:58px;text-align:right;color:#9ca3af;font-size:11px}
.pr{display:flex;align-items:center;padding:5px 0;border-bottom:1px solid #f1f5f9;font-size:13px;gap:8px}
.pr-n{flex:1;color:#111827;font-weight:500}.pr-s{color:#6b7280;font-size:11px;min-width:28px;text-align:center}
.pr-p{color:#2563eb;font-weight:600;min-width:38px;text-align:right;font-size:14px}
.pr-q{color:#9ca3af;font-size:11px;min-width:52px;text-align:right}
.pr-t{color:#9ca3af;font-size:10px;margin-left:2px}
.th{display:flex;justify-content:space-between;align-items:center;padding:8px 14px;border-radius:8px;margin-bottom:6px}
.th-home{border-left:4px solid #2563eb;background:#f8fafc}
.th-away{border-left:4px solid #dc2626;background:#f8fafc}
.th-name{font-size:14px;font-weight:600;color:#1e293b}.th-stat{font-size:13px;color:#64748b}
.sl{font-size:10px;color:#9ca3af;letter-spacing:0.08em;text-transform:uppercase;margin:10px 0 4px}
.lega-row{display:flex;align-items:center;padding:4px 0;font-size:13px;gap:8px;border-bottom:1px solid #f3f4f6}
.lega-l{flex:1;color:#6b7280;font-size:12px}.lega-h{min-width:48px;text-align:right;font-weight:500;color:#111827}
.lega-a{min-width:48px;text-align:right;font-weight:500;color:#111827}
.lega-avg{min-width:48px;text-align:right;color:#9ca3af;font-size:11px}
</style>"""


def _vb_badge(label, vbs_map):
    lbl = label.lower()
    for k, vb in vbs_map.items():
        if lbl in k or k in lbl:
            edge = vb.get('edge_%', 0)
            if edge > 0:
                cls = 'ep' if edge >= 10 else 'en'
                return f'<span class="{cls}">+{edge:.0f}%</span>'
    return ''


def _row(label, prob, vbs_map=None):
    qe = round(1/prob, 2) if prob > 0.01 else 99.0
    badge = _vb_badge(label, vbs_map or {})
    return (f'<div class="cr"><span class="cr-l">{label}</span>'
            f'<span class="cr-p">{int(prob*100)}%</span>'
            f'<span class="cr-q">QE {qe}</span>'
            f'<div class="cr-e">{badge}</div></div>')


def _ou_pair(label_o, label_u, p_o, p_u, vbs_map=None):
    qe_o = round(1/p_o, 2) if p_o > 0.01 else 99.0
    qe_u = round(1/p_u, 2) if p_u > 0.01 else 99.0
    b_o = _vb_badge(label_o, vbs_map or {})
    b_u = _vb_badge(label_u, vbs_map or {})
    return (f'<div class="ou-row"><span class="ou-l">{label_o}</span>'
            f'<span class="ou-p">{int(p_o*100)}%</span>'
            f'<span class="ou-q">QE {qe_o}</span>{b_o}</div>'
            f'<div class="ou-row"><span class="ou-l">{label_u}</span>'
            f'<span class="ou-p">{int(p_u*100)}%</span>'
            f'<span class="ou-q">QE {qe_u}</span>{b_u}</div>')


def render_compact_analysis(home, away, preds, corner_pred, cards_pred,
                             lam_h, lam_a, vbs, home_scorers, away_scorers,
                             home_bookings, away_bookings, referee=None,
                             opp_ctx_home=None, opp_ctx_away=None):
    st.markdown(CSS, unsafe_allow_html=True)

    prob_h = preds.get('prob_H', 0)
    prob_d = preds.get('prob_D', 0)
    prob_a = preds.get('prob_A', 0)
    vbs_map = {v.get('mercato', '').lower(): v for v in (vbs or [])}
    lh = lam_h or 1.4
    la = lam_a or 1.2
    lam_t = lh + la

    # Dati squadre: opp_ctx_away = stats home, opp_ctx_home = stats away
    home_stats = opp_ctx_away or {}
    away_stats = opp_ctx_home or {}

    # 9 tab con indici corretti
    tab_gol, tab_mg, tab_corn, tab_tiri, tab_cart, tab_ht, tab_marc, tab_amm, tab_vb = st.tabs([
        "⚽ Gol", "🎯 MultiGoal", "📐 Corner",
        "🎯 Tiri", "🟨 Cartellini", "⏱ 1° Tempo",
        "👤 Marcatori", "⚠️ Ammoniti", "💰 Value Bet"
    ])

    # ── GOL (tab 0) ──────────────────────────────────────────────────────────
    with tab_gol:
        if lh and la:
            st.caption(f"xG attesi — {home}: {lh:.2f}  {away}: {la:.2f}")
        p_gg = (1-poisson.pmf(0,lh))*(1-poisson.pmf(0,la))
        col_l, col_r = st.columns([3, 2])
        with col_l:
            html = '<div class="sl">1X2</div>'
            html += _row(f"1 — {home}", prob_h, vbs_map)
            html += _row("X — Pareggio", prob_d, vbs_map)
            html += _row(f"2 — {away}", prob_a, vbs_map)
            html += '<div class="sl">Goal / No Goal</div>'
            html += _row("Goal/Goal", p_gg, vbs_map)
            html += _row("No Goal", 1-p_gg, vbs_map)
            html += '<div class="sl">Over / Under Gol</div>'
            html += '<div class="ou-grid">'
            for t in [0.5, 1.5, 2.5, 3.5, 4.5]:
                p_o = 1 - poisson.cdf(int(t), lam_t)
                html += _ou_pair(f"Over {t}", f"Under {t}", p_o, 1-p_o, vbs_map)
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)
        with col_r:
            st.markdown('<div class="sl">Risultati più probabili</div>', unsafe_allow_html=True)
            scores = {}
            for h in range(6):
                for a in range(6):
                    scores[f"{h}-{a}"] = poisson.pmf(h, lh) * poisson.pmf(a, la)
            html = ''
            for sc, prob in sorted(scores.items(), key=lambda x: -x[1])[:8]:
                qe = round(1/prob, 1) if prob > 0.001 else 99
                html += f'<div class="cr"><span class="cr-l">{sc}</span><span class="cr-p">{int(prob*100)}%</span><span class="cr-q">QE {qe}</span></div>'
            st.markdown(html, unsafe_allow_html=True)

    # ── MULTIGOAL (tab 1) ────────────────────────────────────────────────────
    with tab_mg:
        bands = [("1-2",1,2),("2-3",2,3),("1-3",1,3),("2-4",2,4),
                 ("3-4",3,4),("1-4",1,4),("2-5",2,5),("3-5",3,5),
                 ("0-1",0,1),("0-2",0,2)]
        html = '<div class="sl">Fasce MultiGoal</div><div class="ou-grid">'
        for label, lo, hi in bands:
            p = sum(poisson.pmf(k, lam_t) for k in range(lo, hi+1))
            qe = round(1/p, 2) if p > 0.01 else 99
            html += f'<div class="ou-row"><span class="ou-l">MG {label}</span><span class="ou-p">{int(p*100)}%</span><span class="ou-q">QE {qe}</span></div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    # ── CORNER (tab 2) ───────────────────────────────────────────────────────
    with tab_corn:
        c_h = corner_pred.get('home_expected', '—') if corner_pred else '—'
        c_a = corner_pred.get('away_expected', '—') if corner_pred else '—'
        c_t = corner_pred.get('total_expected', '—') if corner_pred else '—'
        real_h = round(home_stats.get('corners_avg', 0), 1)
        real_a = round(away_stats.get('corners_avg', 0), 1)
        lega_c = 4.65

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="th th-home"><span class="th-name">{home}</span><span class="th-stat">{c_h} corner attesi</span></div>', unsafe_allow_html=True)
            if real_h > 0:
                st.markdown(f'<div class="cr"><span class="cr-l">Media stagionale</span><span class="cr-p">{real_h}</span><span class="cr-q">⌀ {lega_c}</span></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="th th-away"><span class="th-name">{away}</span><span class="th-stat">{c_a} corner attesi</span></div>', unsafe_allow_html=True)
            if real_a > 0:
                st.markdown(f'<div class="cr"><span class="cr-l">Media stagionale</span><span class="cr-p">{real_a}</span><span class="cr-q">⌀ {lega_c}</span></div>', unsafe_allow_html=True)

        html = f'<div class="sl">Totale previsto: {c_t} · Media lega: 9.3</div><div class="ou-grid">'
        if corner_pred:
            cp = corner_pred.get('probabilities', {})
            for t in [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]:
                p_o = cp.get(f'over_{t}', 0)
                html += _ou_pair(f"Over {t}", f"Under {t}", p_o, 1-p_o, vbs_map)
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    # ── TIRI (tab 3) ─────────────────────────────────────────────────────────
    with tab_tiri:
        sh_h    = round(home_stats.get('shots_avg', 0), 1)
        sh_a    = round(away_stats.get('shots_avg', 0), 1)
        sh_ot_h = round(home_stats.get('shots_ot_avg', 0), 1)
        sh_ot_a = round(away_stats.get('shots_ot_avg', 0), 1)
        cr_h    = round(home_stats.get('crosses_avg', 0), 1)
        cr_a    = round(away_stats.get('crosses_avg', 0), 1)
        sp_h    = int(home_stats.get('sprints_avg', 0))
        sp_a    = int(away_stats.get('sprints_avg', 0))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="th th-home"><span class="th-name">{home}</span><span class="th-stat">{sh_h} tiri · {sh_ot_h} in porta</span></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="th th-away"><span class="th-name">{away}</span><span class="th-stat">{sh_a} tiri · {sh_ot_a} in porta</span></div>', unsafe_allow_html=True)

        html = '<div class="sl">Media stagionale per partita</div>'
        html += f'<div class="lega-row" style="font-weight:600;color:#6b7280;font-size:11px"><span class="lega-l">Statistica</span><span class="lega-h">{home[:8]}</span><span class="lega-a">{away[:8]}</span><span class="lega-avg">⌀ Lega</span></div>'
        for label, vh, va, avg in [
            ("Tiri totali", sh_h, sh_a, "11.0"),
            ("Tiri in porta", sh_ot_h, sh_ot_a, "4.0"),
            ("Cross", cr_h, cr_a, "12.0"),
            ("Sprint", sp_h, sp_a, "170"),
        ]:
            html += f'<div class="lega-row"><span class="lega-l">{label}</span><span class="lega-h">{vh}</span><span class="lega-a">{va}</span><span class="lega-avg">{avg}</span></div>'
        st.markdown(html, unsafe_allow_html=True)

    # ── CARTELLINI (tab 4) ───────────────────────────────────────────────────
    with tab_cart:
        if cards_pred:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="th th-home"><span class="th-name">{home}</span><span class="th-stat">{cards_pred.get("home_expected","—")} cart.</span></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="th th-away"><span class="th-name">{away}</span><span class="th-stat">{cards_pred.get("away_expected","—")} cart.</span></div>', unsafe_allow_html=True)
            ref_f = cards_pred.get('referee_factor', 1.0)
            ref_icon = '🟡' if ref_f > 1.1 else ('🟢' if ref_f < 0.9 else '⚪')
            st.caption(f"Arbitro: fattore {ref_f:.2f}x {ref_icon} — Totale previsto: {cards_pred.get('total_expected','—')}")
            cp = cards_pred.get('probabilities', {})
            html = '<div class="ou-grid">'
            for t in [1.5, 2.5, 3.5, 4.5, 5.5]:
                p_o = cp.get(f'over_{t}', 0)
                html += _ou_pair(f"Over {t} Cart.", f"Under {t} Cart.", p_o, 1-p_o, vbs_map)
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("Dati cartellini non disponibili")

    # ── PRIMO TEMPO (tab 5) ──────────────────────────────────────────────────
    with tab_ht:
        lht_h, lht_a = lh*0.441, la*0.441
        lht = lht_h + lht_a
        p_ht_h = sum(poisson.pmf(h, lht_h)*sum(poisson.pmf(a, lht_a) for a in range(h)) for h in range(1,7))
        p_ht_d = sum(poisson.pmf(k, lht_h)*poisson.pmf(k, lht_a) for k in range(7))
        p_ht_a = max(0, 1-p_ht_h-p_ht_d)
        html = '<div class="sl">HT 1X2</div>'
        html += _row(f"HT 1 — {home}", p_ht_h, vbs_map)
        html += _row("HT X — Pareggio", p_ht_d, vbs_map)
        html += _row(f"HT 2 — {away}", p_ht_a, vbs_map)
        html += '<div class="sl">HT Over/Under</div><div class="ou-grid">'
        for t in [0.5, 1.5, 2.5]:
            p_o = 1-poisson.cdf(int(t), lht)
            html += _ou_pair(f"HT Over {t}", f"HT Under {t}", p_o, 1-p_o, vbs_map)
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    # ── MARCATORI (tab 6) ────────────────────────────────────────────────────
    with tab_marc:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="th th-home"><span class="th-name">{home}</span></div>', unsafe_allow_html=True)
            if home_scorers:
                html = ''
                for p in home_scorers:
                    prob = p.get('prob_score', 0)
                    qe = round(1/prob, 1) if prob > 0.01 else 99
                    tags = f'<span class="pr-t">({p["tags"]})</span>' if p.get("tags") else ''
                    html += (f'<div class="pr"><span class="pr-n">{p["name"]}{tags}</span>'
                             f'<span class="pr-s">⚽{p["goals"]}</span>'
                             f'<span class="pr-p">{int(prob*100)}%</span>'
                             f'<span class="pr-q">QE {qe}</span></div>')
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.caption("Dati non disponibili")
        with c2:
            st.markdown(f'<div class="th th-away"><span class="th-name">{away}</span></div>', unsafe_allow_html=True)
            if away_scorers:
                html = ''
                for p in away_scorers:
                    prob = p.get('prob_score', 0)
                    qe = round(1/prob, 1) if prob > 0.01 else 99
                    tags = f'<span class="pr-t">({p["tags"]})</span>' if p.get("tags") else ''
                    html += (f'<div class="pr"><span class="pr-n">{p["name"]}{tags}</span>'
                             f'<span class="pr-s">⚽{p["goals"]}</span>'
                             f'<span class="pr-p">{int(prob*100)}%</span>'
                             f'<span class="pr-q">QE {qe}</span></div>')
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.caption("Dati non disponibili")

    # ── AMMONITI (tab 7) ─────────────────────────────────────────────────────
    with tab_amm:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="th th-home"><span class="th-name">{home}</span></div>', unsafe_allow_html=True)
            if home_bookings:
                html = ''
                for p in home_bookings:
                    prob = p.get('prob_booking', 0)
                    qe = round(1/prob, 1) if prob > 0.01 else 99
                    pos = p.get('position', '?')
                    html += (f'<div class="pr"><span class="pr-n">{p["name"]} <span class="pr-t">({pos})</span></span>'
                             f'<span class="pr-s">🟨{p["yellow_cards"]}</span>'
                             f'<span class="pr-p">{int(prob*100)}%</span>'
                             f'<span class="pr-q">QE {qe}</span></div>')
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.caption("Dati non disponibili")
        with c2:
            st.markdown(f'<div class="th th-away"><span class="th-name">{away}</span></div>', unsafe_allow_html=True)
            if away_bookings:
                html = ''
                for p in away_bookings:
                    prob = p.get('prob_booking', 0)
                    qe = round(1/prob, 1) if prob > 0.01 else 99
                    pos = p.get('position', '?')
                    html += (f'<div class="pr"><span class="pr-n">{p["name"]} <span class="pr-t">({pos})</span></span>'
                             f'<span class="pr-s">🟨{p["yellow_cards"]}</span>'
                             f'<span class="pr-p">{int(prob*100)}%</span>'
                             f'<span class="pr-q">QE {qe}</span></div>')
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.caption("Dati non disponibili")

    # ── VALUE BET (tab 8) ────────────────────────────────────────────────────
    with tab_vb:
        st.caption("⚡ Value bet vs quote Pinnacle (blend 50/50). Inserisci le quote Marathonbet sopra per confronto personalizzato.")
        if vbs:
            html = '<div class="sl">Ordinate per edge</div>'
            for vb in sorted(vbs, key=lambda x: -x.get('edge_%', 0)):
                edge = vb.get('edge_%', 0)
                quota = vb.get('quota', 0)
                prob = vb.get('prob_modello_%', 0)
                stake = vb.get('stake_€', 4)
                mercato = vb.get('mercato', '')
                star = '⭐ ' if edge >= 15 else '✅ '
                badge = f'<span class="ep">+{edge:.0f}%</span>'
                html += (f'<div class="cr"><span class="cr-l">{star}{mercato}</span>'
                         f'<span class="cr-p">@ {quota:.2f}</span>'
                         f'<span class="cr-q">Prob {prob:.0f}%</span>'
                         f'<div class="cr-e">{badge}</div>'
                         f'<span style="font-size:12px;color:#374151;min-width:50px;text-align:right">€{stake:.2f}</span></div>')
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("Nessuna value bet con le quote inserite")

# Script di installazione (eseguire da football_predictor/)
if __name__ == '__main__':
    import shutil
    from pathlib import Path
    BASE = Path('/Users/davideforcato/Desktop/football_predictor')
    
    # 1. Installa nuovo componente
    shutil.copy(__file__, BASE / 'components/match_analysis.py')
    print('OK componente installato')
    
    # 2. Aggiungi shots/shots_ot a player_stats.py
    ps = (BASE / 'models/player_stats.py').read_text()
    old = "                            elif n == 'Total tackles':\n                                team_ctx[home]['tackles'].append(hv)\n                                team_ctx[away]['tackles'].append(av)"
    new = """                            elif n == 'Total tackles':
                                team_ctx[home]['tackles'].append(hv)
                                team_ctx[away]['tackles'].append(av)
                            elif n == 'Total shots':
                                team_ctx[home]['shots'].append(hv)
                                team_ctx[away]['shots'].append(av)
                            elif n == 'Shots on target':
                                team_ctx[home]['shots_ot'].append(hv)
                                team_ctx[away]['shots_ot'].append(av)"""
    if old in ps:
        ps = ps.replace(old, new)
        print('OK shots aggiunto a player_stats')
    # Aggiungi shots al return di get_team_context
    old2 = "            'tackles_avg':  ctx.get('tackles',  15.0),\n        }"
    new2 = "            'tackles_avg':  ctx.get('tackles',  15.0),\n            'shots_avg':    ctx.get('shots',    11.0),\n            'shots_ot_avg': ctx.get('shots_ot', 4.0),\n        }"
    if old2 in ps and 'shots_avg' not in ps:
        ps = ps.replace(old2, new2)
        print('OK shots_avg nel return di get_team_context')
    (BASE / 'models/player_stats.py').write_text(ps)
    
    # 3. Fix QE nel tab Marcatori input: usa lam_h reale invece di 1.4
    dash = (BASE / 'dashboard.py').read_text()
    old_sc = "_sc_h = _pb8.scorer_probability(home, 1.4, opp_context=_opp_h, limit=5)"
    new_sc = "_xg_h8 = _lam_h or 1.4\n            _xg_a8 = _lam_a or 1.2\n            _sc_h = _pb8.scorer_probability(home, _xg_h8, opp_context=_opp_h, limit=5)"
    old_sc2 = "_sc_a = _pb8.scorer_probability(away, 1.2, opp_context=_opp_a, limit=5)"
    new_sc2 = "_sc_a = _pb8.scorer_probability(away, _xg_a8, opp_context=_opp_a, limit=5)"
    if old_sc in dash:
        dash = dash.replace(old_sc, new_sc).replace(old_sc2, new_sc2)
        print('OK QE marcatori usa lambda reale')
    # Stessa cosa per ammoniti tab 9
    old_bk = "_bk_h = _pb9.booking_probability(home, 1.8, opp_context=_opp_h9, limit=5)"
    new_bk = "_exp_h9 = _cards_pred.get('home_expected', 1.8) if '_cards_pred' in dir() else 1.8\n            _exp_a9 = _cards_pred.get('away_expected', 1.8) if '_cards_pred' in dir() else 1.8\n            _bk_h = _pb9.booking_probability(home, _exp_h9, opp_context=_opp_h9, limit=5)"
    old_bk2 = "_bk_a = _pb9.booking_probability(away, 1.8, opp_context=_opp_a9, limit=5)"
    new_bk2 = "_bk_a = _pb9.booking_probability(away, _exp_a9, opp_context=_opp_a9, limit=5)"
    if old_bk in dash:
        dash = dash.replace(old_bk, new_bk).replace(old_bk2, new_bk2)
        print('OK QE ammoniti usa expected cards reale')
    (BASE / 'dashboard.py').write_text(dash)
    
    # 4. Verifica sintassi
    import py_compile, sys
    sys.path.insert(0, str(BASE))
    errors = 0
    for f, label in [(str(BASE/'components/match_analysis.py'), 'match_analysis.py'),
                      (str(BASE/'models/player_stats.py'), 'player_stats.py'),
                      (str(BASE/'dashboard.py'), 'dashboard.py')]:
        try:
            py_compile.compile(f, doraise=True)
            print(f'✅ {label}')
        except Exception as e:
            print(f'❌ {label}: {e}')
            errors += 1
    
    if errors == 0:
        print('\n✅ Tutto OK — avvia: streamlit run dashboard.py')
