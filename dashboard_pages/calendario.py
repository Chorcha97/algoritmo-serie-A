"""
Pagina "Calendario 26/27" — calendario Serie A con anteprima predizioni.
Estratta da dashboard.py nella separazione in pagine (roadmap #2).
"""
import streamlit as st
import pandas as pd


def render():
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
