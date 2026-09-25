"""
Pagina "Arbitri" — statistiche arbitrali.
Estratta da dashboard.py nella separazione in pagine (roadmap #2).

Nota: questa pagina era presente nel codice ma NON raggiungibile dal menu
(la voce "Arbitri" non era più nella lista st.radio della sidebar) — vedi
il commit di separazione per i dettagli. Riportata nel menu perché il
codice funziona ed è comunque utile.
"""
import streamlit as st


def render(ref_stats):
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

