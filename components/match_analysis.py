"""
components/match_analysis.py v2
Visual redesign:
- Header leggibili (accent colorato su sfondo chiaro)
- Gol tab: grid compatto 2 colonne, tutto visibile senza scroll
- Corner: aggiunge tiri/tiri in porta per squadra
- Roster: solo giocatori 2026/27
- Nessun duplicato
"""
import streamlit as st
import numpy as np
from scipy.stats import poisson

CSS = """<style>
/* Reset spaziature eccessive */
.block-container { padding-top: 0.5rem !important; }

/* Header squadra: accent colorato su sfondo chiaro */
.th { 
    display:flex; justify-content:space-between; align-items:center;
    padding:8px 14px; border-radius:8px; margin-bottom:6px;
    border-left:4px solid var(--th-color, #2563eb);
    background:#f8fafc;
}
.th-home { --th-color: #2563eb; }
.th-away { --th-color: #dc2626; }
.th-name { font-size:14px; font-weight:600; color:#1e293b; }
.th-stat { font-size:13px; color:#64748b; }

/* Sezione label */
.sl { font-size:10px; color:#9ca3af; letter-spacing:0.08em; 
      text-transform:uppercase; margin:10px 0 4px; }

/* Riga compatta */
.cr { display:flex; align-items:center; padding:4px 0; 
      border-bottom:1px solid #f1f5f9; font-size:13px; gap:8px; }
.cr-l { flex:1; color:#374151; }
.cr-p { min-width:38px; text-align:right; font-weight:600; color:#111827; }
.cr-q { min-width:62px; text-align:right; color:#9ca3af; font-size:11px; }
.cr-e { min-width:62px; text-align:right; }
.ep { background:#d1fae5; color:#065f46; padding:1px 7px; 
       border-radius:100px; font-size:11px; font-weight:600; }
.en { background:#fee2e2; color:#991b1b; padding:1px 7px; 
       border-radius:100px; font-size:11px; }

/* Grid O/U a 2 colonne */
.ou-grid { display:grid; grid-template-columns:1fr 1fr; gap:2px 12px; }
.ou-row { display:flex; align-items:center; padding:4px 0; 
           border-bottom:1px solid #f8fafc; font-size:13px; gap:6px; }
.ou-l { flex:1; color:#374151; font-size:12px; }
.ou-p { min-width:36px; text-align:right; font-weight:600; color:#111827; }
.ou-q { min-width:58px; text-align:right; color:#9ca3af; font-size:11px; }

/* Player row */
.pr { display:flex; align-items:center; padding:5px 0; 
       border-bottom:1px solid #f1f5f9; font-size:13px; gap:8px; }
.pr-n { flex:1; color:#111827; font-weight:500; }
.pr-s { color:#6b7280; font-size:11px; min-width:28px; text-align:center; }
.pr-p { color:#2563eb; font-weight:600; min-width:38px; text-align:right; font-size:14px; }
.pr-t { color:#9ca3af; font-size:10px; margin-left:2px; }
</style>"""


def _vb_badge(label, vbs_map):
    """Cerca una value bet per il mercato dato."""
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


def _ou_pair(label_o, label_u, p_over, p_under, vbs_map=None):
    qe_o = round(1/p_over, 2) if p_over > 0.01 else 99.0
    qe_u = round(1/p_under, 2) if p_under > 0.01 else 99.0
    b_o = _vb_badge(label_o, vbs_map or {})
    b_u = _vb_badge(label_u, vbs_map or {})
    return (f'<div class="ou-row"><span class="ou-l">{label_o}</span>'
            f'<span class="ou-p">{int(p_over*100)}%</span>'
            f'<span class="ou-q">QE {qe_o}</span>{b_o}</div>'
            f'<div class="ou-row"><span class="ou-l">{label_u}</span>'
            f'<span class="ou-p">{int(p_under*100)}%</span>'
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

    tabs = st.tabs(["⚽ Gol", "🎯 MultiGoal", "📐 Corner & Tiri",
                    "🟨 Cartellini", "⏱ 1° Tempo",
                    "👤 Marcatori", "⚠️ Ammoniti", "💰 Value Bet"])

    # ── GOL ──────────────────────────────────────────────────────────────────
    with tabs[0]:
        if lh and la:
            st.caption(f"xG attesi — {home}: {lh:.2f}  {away}: {la:.2f}")

        p_gg = (1-poisson.pmf(0,lh))*(1-poisson.pmf(0,la))

        # Layout 2 colonne: sinistra=1X2+GG, destra=risultati esatti
        col_l, col_r = st.columns([3, 2])

        with col_l:
            html = '<div class="sl">1X2</div>'
            html += _row(f"1 — {home}", prob_h, vbs_map)
            html += _row("X — Pareggio", prob_d, vbs_map)
            html += _row(f"2 — {away}", prob_a, vbs_map)
            html += '<div class="sl">Goal / No Goal</div>'
            html += _row("Goal/Goal", p_gg, vbs_map)
            html += _row("No Goal", 1-p_gg, vbs_map)
            html += '<div class="sl" style="margin-top:10px">Over / Under Gol</div>'
            html += '<div class="ou-grid">'
            for t in [0.5, 1.5, 2.5, 3.5, 4.5]:
                p_o = 1 - poisson.cdf(int(t), lam_t)
                p_u = poisson.cdf(int(t), lam_t)
                html += _ou_pair(f"Over {t}", f"Under {t}", p_o, p_u, vbs_map)
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

    # ── MULTIGOAL ────────────────────────────────────────────────────────────
    with tabs[1]:
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

    # ── CORNER & TIRI ────────────────────────────────────────────────────────
    with tabs[2]:
        # Header squadre con statistiche
        c_h = corner_pred.get('home_expected', '—') if corner_pred else '—'
        c_a = corner_pred.get('away_expected', '—') if corner_pred else '—'
        c_t = corner_pred.get('total_expected', '—') if corner_pred else '—'

        # Statistiche tiri dai context
        shots_h = round(opp_ctx_away.get('shots_conceded_avg', 0), 1) if opp_ctx_away else '—'
        shots_a = round(opp_ctx_home.get('shots_conceded_avg', 0), 1) if opp_ctx_home else '—'

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="th th-home"><span class="th-name">{home}</span>'
                        f'<span class="th-stat">{c_h} corner attesi</span></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="th th-away"><span class="th-name">{away}</span>'
                        f'<span class="th-stat">{c_a} corner attesi</span></div>', unsafe_allow_html=True)

        html = f'<div class="sl">Totale previsto: {c_t} corner</div><div class="ou-grid">'
        if corner_pred:
            cp = corner_pred.get('probabilities', {})
            for t in [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]:
                p_o = cp.get(f'over_{t}', 0)
                p_u = cp.get(f'under_{t}', 0)
                html += _ou_pair(f"Over {t}", f"Under {t}", p_o, p_u, vbs_map)
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

        # Tiri da context avversario
        if opp_ctx_home or opp_ctx_away:
            st.markdown('<div class="sl" style="margin-top:8px">Stile di gioco (media stagione)</div>', unsafe_allow_html=True)
            ctx_h = opp_ctx_home or {}
            ctx_a = opp_ctx_away or {}
            rows_data = [
                ("Crosses", ctx_h.get('crosses_avg',0), ctx_a.get('crosses_avg',0)),
                ("Sprints", ctx_h.get('sprints_avg',0), ctx_a.get('sprints_avg',0)),
                ("Falli commessi", ctx_h.get('fouls_avg',0), ctx_a.get('fouls_avg',0)),
            ]
            html2 = f'<div class="cr" style="font-weight:600;color:#6b7280;font-size:11px"><span class="cr-l">Stat</span><span class="cr-p">{home[:10]}</span><span class="cr-q">{away[:10]}</span></div>'
            for label, vh, va in rows_data:
                if vh or va:
                    html2 += f'<div class="cr"><span class="cr-l">{label}</span><span class="cr-p">{vh:.1f}</span><span class="cr-q">{va:.1f}</span></div>'
            st.markdown(html2, unsafe_allow_html=True)

    # ── CARTELLINI ───────────────────────────────────────────────────────────
    with tabs[3]:
        if cards_pred:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="th th-home"><span class="th-name">{home}</span>'
                            f'<span class="th-stat">{cards_pred.get("home_expected","—")} cart.</span></div>',
                            unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="th th-away"><span class="th-name">{away}</span>'
                            f'<span class="th-stat">{cards_pred.get("away_expected","—")} cart.</span></div>',
                            unsafe_allow_html=True)
            ref_f = cards_pred.get('referee_factor', 1.0)
            ref_icon = '🟡' if ref_f > 1.1 else ('🟢' if ref_f < 0.9 else '⚪')
            st.caption(f"Arbitro: fattore {ref_f:.2f}x {ref_icon} — Totale previsto: {cards_pred.get('total_expected','—')}")
            cp = cards_pred.get('probabilities', {})
            html = '<div class="ou-grid">'
            for t in [1.5, 2.5, 3.5, 4.5, 5.5]:
                p_o = cp.get(f'over_{t}', 0)
                p_u = cp.get(f'under_{t}', 0)
                html += _ou_pair(f"Over {t} Cart.", f"Under {t} Cart.", p_o, p_u, vbs_map)
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)

    # ── PRIMO TEMPO ──────────────────────────────────────────────────────────
    with tabs[4]:
        lht_h, lht_a = lh*0.47, la*0.47
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
            p_u = poisson.cdf(int(t), lht)
            html += _ou_pair(f"HT Over {t}", f"HT Under {t}", p_o, p_u, vbs_map)
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    # ── MARCATORI ────────────────────────────────────────────────────────────
    with tabs[5]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="th th-home"><span class="th-name">{home}</span></div>',
                        unsafe_allow_html=True)
            if home_scorers:
                html = ''
                for p in home_scorers:
                    tags = f'<span class="pr-t">({p["tags"]})</span>' if p.get("tags") else ''
                    pos_h = f'{p.get("position","?")}/{p.get("height",0)}cm'
                    html += (f'<div class="pr"><span class="pr-n">{p["name"]}{tags}</span>'
                             f'<span class="pr-s" title="{pos_h}">⚽{p["goals"]}</span>'
                             f'<span class="pr-p">{int(p["prob_score"]*100)}%</span></div>')
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.caption("Dati non disponibili")
        with c2:
            st.markdown(f'<div class="th th-away"><span class="th-name">{away}</span></div>',
                        unsafe_allow_html=True)
            if away_scorers:
                html = ''
                for p in away_scorers:
                    tags = f'<span class="pr-t">({p["tags"]})</span>' if p.get("tags") else ''
                    html += (f'<div class="pr"><span class="pr-n">{p["name"]}{tags}</span>'
                             f'<span class="pr-s">⚽{p["goals"]}</span>'
                             f'<span class="pr-p">{int(p["prob_score"]*100)}%</span></div>')
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.caption("Dati non disponibili")

    # ── AMMONITI ─────────────────────────────────────────────────────────────
    with tabs[6]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="th th-home"><span class="th-name">{home}</span></div>',
                        unsafe_allow_html=True)
            if home_bookings:
                html = ''
                for p in home_bookings:
                    html += (f'<div class="pr"><span class="pr-n">{p["name"]}</span>'
                             f'<span class="pr-s">🟨{p["yellow_cards"]}</span>'
                             f'<span class="pr-p">{int(p["prob_booking"]*100)}%</span></div>')
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.caption("Dati non disponibili")
        with c2:
            st.markdown(f'<div class="th th-away"><span class="th-name">{away}</span></div>',
                        unsafe_allow_html=True)
            if away_bookings:
                html = ''
                for p in away_bookings:
                    html += (f'<div class="pr"><span class="pr-n">{p["name"]}</span>'
                             f'<span class="pr-s">🟨{p["yellow_cards"]}</span>'
                             f'<span class="pr-p">{int(p["prob_booking"]*100)}%</span></div>')
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.caption("Dati non disponibili")

    # ── VALUE BET ────────────────────────────────────────────────────────────
    with tabs[7]:
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
