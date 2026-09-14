
import streamlit as st
import numpy as np
from scipy.stats import poisson

COMPACT_CSS = """<style>
.cr { display:flex; align-items:center; padding:5px 0; border-bottom:1px solid #f3f4f6; font-size:13px; gap:12px; }
.cr-label { flex:1; color:#374151; font-weight:500; }
.cr-prob { min-width:42px; text-align:right; font-weight:600; color:#111827; }
.cr-qe { min-width:60px; text-align:right; color:#9ca3af; font-size:12px; }
.cr-edge { min-width:65px; text-align:right; }
.cr-edge-pos { background:#d1fae5; color:#065f46; padding:1px 7px; border-radius:100px; font-size:11px; font-weight:600; }
.cr-edge-neg { background:#fee2e2; color:#991b1b; padding:1px 7px; border-radius:100px; font-size:11px; }
.cr-section { font-size:11px; color:#9ca3af; text-transform:uppercase; letter-spacing:0.06em; margin:12px 0 4px; }
.cr-player { display:flex; align-items:center; padding:5px 0; border-bottom:1px solid #f3f4f6; font-size:13px; gap:8px; }
.cr-pl-name { flex:1; color:#111827; font-weight:500; }
.cr-pl-stat { color:#6b7280; font-size:12px; min-width:32px; text-align:center; }
.cr-pl-prob { color:#2563eb; font-weight:600; min-width:40px; text-align:right; }
.cr-hdr { background:#1e293b; color:#f1f5f9; padding:8px 14px; border-radius:8px; font-size:14px; font-weight:600; margin-bottom:8px; display:flex; justify-content:space-between; }
.cr-hdr-stat { font-size:12px; color:#94a3b8; }
</style>"""

def _row(label, prob, market_odds=None, vbs_map=None):
    """Riga compatta: label | prob% | QE | [edge]"""
    qe = round(1/prob, 2) if prob > 0.01 else 99.0
    qe_str = f"QE {qe}"
    edge_html = ""
    if vbs_map:
        for k, vb in vbs_map.items():
            if label.lower() in k.lower() or k.lower() in label.lower():
                edge = vb.get("edge_%", 0)
                if edge >= 15:
                    mkt_odds = vb.get("quota", qe)
                    edge_html = f'<span class="cr-edge-pos">⭐ +{edge:.0f}%</span>'
                elif edge > 0:
                    edge_html = f'<span class="cr-edge-neg">+{edge:.0f}%</span>'
                break
    return f'<div class="cr"><span class="cr-label">{label}</span><span class="cr-prob">{int(prob*100)}%</span><span class="cr-qe">{qe_str}</span><div class="cr-edge">{edge_html}</div></div>'

def render_compact_analysis(home, away, preds, corner_pred, cards_pred, lam_h, lam_a, vbs, home_scorers, away_scorers, home_bookings, away_bookings, referee=None):
    st.markdown(COMPACT_CSS, unsafe_allow_html=True)
    
    # Header match
    prob_h = preds.get("prob_H", 0)
    prob_d = preds.get("prob_D", 0)
    prob_a = preds.get("prob_A", 0)
    
    # Mappa vbs per label rapido
    vbs_map = {vb.get("mercato","").lower(): vb for vb in (vbs or [])}
    
    tabs = st.tabs(["⚽ Gol", "🎯 MultiGoal", "📐 Corner", "🟨 Cartellini", "⏱ 1° Tempo", "👤 Marcatori", "⚠️ Ammoniti", "💰 Value Bet"])

    # ── GOL ──────────────────────────────────────────────────────────────────
    with tabs[0]:
        if lam_h and lam_a:
            st.caption(f"xG: {home} {lam_h:.2f} — {away} {lam_a:.2f}")
        lam_t = (lam_h or 1.4) + (lam_a or 1.2)
        html = ""
        # 1X2
        html += '<div class="cr-section">1X2</div>'
        html += _row(f"1 — {home}", prob_h, vbs_map=vbs_map)
        html += _row("X — Pareggio", prob_d, vbs_map=vbs_map)
        html += _row(f"2 — {away}", prob_a, vbs_map=vbs_map)
        # GG/NG
        p_gg = (1-poisson.pmf(0,lam_h or 1.4))*(1-poisson.pmf(0,lam_a or 1.2))
        html += '<div class="cr-section">Goal / No Goal</div>'
        html += _row("Goal/Goal", p_gg, vbs_map=vbs_map)
        html += _row("No Goal", 1-p_gg, vbs_map=vbs_map)
        # Over/Under
        html += '<div class="cr-section">Over / Under Gol</div>'
        for t in [0.5, 1.5, 2.5, 3.5, 4.5]:
            p_o = 1 - poisson.cdf(int(t), lam_t)
            p_u = poisson.cdf(int(t), lam_t)
            html += _row(f"Over {t}", p_o, vbs_map=vbs_map)
            html += _row(f"Under {t}", p_u, vbs_map=vbs_map)
        # Risultati esatti top 6
        html += '<div class="cr-section">Risultati più probabili</div>'
        scores = {}
        for h in range(6):
            for a in range(6):
                p = poisson.pmf(h, lam_h or 1.4) * poisson.pmf(a, lam_a or 1.2)
                scores[f"{h}-{a}"] = p
        for sc, prob in sorted(scores.items(), key=lambda x: -x[1])[:6]:
            html += _row(sc, prob)
        st.markdown(html, unsafe_allow_html=True)

    # ── MULTIGOAL ─────────────────────────────────────────────────────────────
    with tabs[1]:
        lam_t = (lam_h or 1.4) + (lam_a or 1.2)
        html = '<div class="cr-section">Fasce MultiGoal</div>'
        bands = [("1-2",1,2),("1-3",1,3),("1-4",1,4),("2-3",2,3),("2-4",2,4),("2-5",2,5),("3-4",3,4),("3-5",3,5),("0-1",0,1),("0-2",0,2)]
        for label, lo, hi in bands:
            p = sum(poisson.pmf(k, lam_t) for k in range(lo, hi+1))
            html += _row(f"MultiGoal {label}", p, vbs_map=vbs_map)
        st.markdown(html, unsafe_allow_html=True)

    # ── CORNER ────────────────────────────────────────────────────────────────
    with tabs[2]:
        if corner_pred:
            he = corner_pred.get("home_expected", "—")
            ae = corner_pred.get("away_expected", "—")
            te = corner_pred.get("total_expected", "—")
            st.markdown(f'<div class="cr-hdr">{home} <span class="cr-hdr-stat">{he} corner</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="cr-hdr">{away} <span class="cr-hdr-stat">{ae} corner</span></div>', unsafe_allow_html=True)
            html = f'<div class="cr-section">Totale previsto: {te} corner</div>'
            cp = corner_pred.get("probabilities", {})
            for t in [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]:
                p_o = cp.get(f"over_{t}", 0)
                p_u = cp.get(f"under_{t}", 0)
                html += _row(f"Over {t} Corner", p_o, vbs_map=vbs_map)
                html += _row(f"Under {t} Corner", p_u, vbs_map=vbs_map)
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("Dati corner non disponibili")

    # ── CARTELLINI ────────────────────────────────────────────────────────────
    with tabs[3]:
        if cards_pred:
            he = cards_pred.get("home_expected", "—")
            ae = cards_pred.get("away_expected", "—")
            te = cards_pred.get("total_expected", "—")
            ref_f = cards_pred.get("referee_factor", 1.0)
            st.markdown(f'<div class="cr-hdr">{home} <span class="cr-hdr-stat">{he} cart.</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="cr-hdr">{away} <span class="cr-hdr-stat">{ae} cart.</span></div>', unsafe_allow_html=True)
            ref_label = f"Arbitro: fattore {ref_f:.2f}x {'🟡' if ref_f>1.1 else '🟢' if ref_f<0.9 else '⚪'}"
            html = f'<div class="cr-section">{ref_label} — Totale previsto: {te}</div>'
            cp = cards_pred.get("probabilities", {})
            for t in [1.5, 2.5, 3.5, 4.5, 5.5]:
                p_o = cp.get(f"over_{t}", 0)
                p_u = cp.get(f"under_{t}", 0)
                html += _row(f"Over {t} Cartellini", p_o, vbs_map=vbs_map)
                html += _row(f"Under {t} Cartellini", p_u, vbs_map=vbs_map)
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("Dati cartellini non disponibili")

    # ── PRIMO TEMPO ──────────────────────────────────────────────────────────
    with tabs[4]:
        lam_ht_h = (lam_h or 1.4) * 0.47
        lam_ht_a = (lam_a or 1.2) * 0.47
        lam_ht = lam_ht_h + lam_ht_a
        p_ht_h = sum(poisson.pmf(h, lam_ht_h) * sum(poisson.pmf(a, lam_ht_a) for a in range(h)) for h in range(1,7))
        p_ht_d = sum(poisson.pmf(k, lam_ht_h) * poisson.pmf(k, lam_ht_a) for k in range(7))
        p_ht_a = max(0, 1 - p_ht_h - p_ht_d)
        html = '<div class="cr-section">HT 1X2</div>'
        html += _row(f"HT 1 — {home}", p_ht_h, vbs_map=vbs_map)
        html += _row("HT X — Pareggio", p_ht_d, vbs_map=vbs_map)
        html += _row(f"HT 2 — {away}", p_ht_a, vbs_map=vbs_map)
        html += '<div class="cr-section">HT Over/Under</div>'
        for t in [0.5, 1.5, 2.5]:
            p_o = 1 - poisson.cdf(int(t), lam_ht)
            p_u = poisson.cdf(int(t), lam_ht)
            html += _row(f"HT Over {t}", p_o, vbs_map=vbs_map)
            html += _row(f"HT Under {t}", p_u, vbs_map=vbs_map)
        st.markdown(html, unsafe_allow_html=True)

    # ── MARCATORI ────────────────────────────────────────────────────────────
    with tabs[5]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="cr-hdr">{home}</div>', unsafe_allow_html=True)
            if home_scorers:
                html = ""
                for p in home_scorers[:6]:
                    prob = p.get("prob_score", 0)
                    tags = p.get('tags',''); tag_html = f' <span style="font-size:10px;color:#9ca3af">({tags})</span>' if tags else ''
                html += f'<div class="cr-player"><span class="cr-pl-name">{p["name"]}{tag_html}</span><span class="cr-pl-stat">⚽{p["goals"]}</span><span class="cr-pl-prob">{int(prob*100)}%</span></div>'
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.caption("Dati non disponibili")
        with c2:
            st.markdown(f'<div class="cr-hdr">{away}</div>', unsafe_allow_html=True)
            if away_scorers:
                html = ""
                for p in away_scorers[:6]:
                    prob = p.get("prob_score", 0)
                    tags = p.get('tags',''); tag_html = f' <span style="font-size:10px;color:#9ca3af">({tags})</span>' if tags else ''
                html += f'<div class="cr-player"><span class="cr-pl-name">{p["name"]}{tag_html}</span><span class="cr-pl-stat">⚽{p["goals"]}</span><span class="cr-pl-prob">{int(prob*100)}%</span></div>'
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.caption("Dati non disponibili")

    # ── AMMONITI ─────────────────────────────────────────────────────────────
    with tabs[6]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="cr-hdr">{home}</div>', unsafe_allow_html=True)
            if home_bookings:
                html = ""
                for p in home_bookings[:6]:
                    prob = p.get("prob_booking", 0)
                    html += f'<div class="cr-player"><span class="cr-pl-name">{p["name"]}</span><span class="cr-pl-stat">🟨{p["yellow_cards"]}</span><span class="cr-pl-prob">{int(prob*100)}%</span></div>'
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.caption("Dati non disponibili")
        with c2:
            st.markdown(f'<div class="cr-hdr">{away}</div>', unsafe_allow_html=True)
            if away_bookings:
                html = ""
                for p in away_bookings[:6]:
                    prob = p.get("prob_booking", 0)
                    html += f'<div class="cr-player"><span class="cr-pl-name">{p["name"]}</span><span class="cr-pl-stat">🟨{p["yellow_cards"]}</span><span class="cr-pl-prob">{int(prob*100)}%</span></div>'
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.caption("Dati non disponibili")

    # ── VALUE BET ─────────────────────────────────────────────────────────────
    with tabs[7]:
        if vbs:
            html = ""
            for vb in sorted(vbs, key=lambda x: -x.get("edge_%", 0)):
                edge = vb.get("edge_%", 0)
                quota = vb.get("quota", 0)
                prob = vb.get("prob_modello_%", 0)
                stake = vb.get("stake_€", 4)
                mercato = vb.get("mercato", "")
                star = "⭐ " if edge >= 15 else "✅ " if edge >= 10 else ""
                badge = f'<span style="background:#d1fae5;color:#065f46;padding:1px 7px;border-radius:100px;font-size:11px;font-weight:600">+{edge:.0f}%</span>'
                html += f'<div class="cr"><span class="cr-label">{star}{mercato}</span><span class="cr-prob">@ {quota:.2f}</span><span class="cr-qe">Prob {prob:.0f}%</span><div class="cr-edge">{badge}</div><span style="font-size:12px;color:#374151;min-width:55px;text-align:right">€{stake:.2f}</span></div>'
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("Nessuna value bet trovata con le quote inserite")
