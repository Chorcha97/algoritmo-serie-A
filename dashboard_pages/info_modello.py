"""
Pagina "Info modello" — statistiche dataset e spiegazione del modello.
Estratta da dashboard.py nella separazione in pagine (roadmap #2).
"""
import streamlit as st


def render(df):
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
