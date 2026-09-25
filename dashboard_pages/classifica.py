"""
Pagina "Classifica" — classifica Serie A con vista Totale/Casa/Trasferta.
Estratta da dashboard.py nella separazione in pagine (roadmap #2).
"""
import streamlit as st
import pandas as pd


def render():
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
