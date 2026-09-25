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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

SERIE_A_2627_TEAMS = [
    "Atalanta", "Bologna", "Cagliari", "Como", "Fiorentina",
    "Frosinone", "Genoa", "Inter", "Juventus", "Lazio",
    "Lecce", "Milan", "Monza", "Napoli", "Parma",
    "Roma", "Sassuolo", "Torino", "Udinese", "Venezia",
]


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

@st.cache_data(ttl=300, show_spinner=False)
def _cached_injuries(team):
    return api_get_injuries(team)

@st.cache_data(ttl=300, show_spinner=False)
def _cached_pinnacle_odds(home, away):
    from data.odds_tracker import get_odds_for_match
    return get_odds_for_match(home, away)

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

    # Carica modello da disco se esiste, altrimenti allena e salva.
    # PRIMA: dopo aver caricato "loaded" dal pickle lo riallenava comunque da
    # zero subito dopo (le due righe "model = EnsembleModel(); model.fit(df)"
    # erano fuori posto, indentate come se servissero solo quando il file non
    # esiste, ma essendo dentro l'"if" partivano SEMPRE) — la cache su disco
    # non veniva quindi mai davvero usata, e il modello si riallenava a ogni
    # riavvio del server. In più, se model_cache.pkl non esisteva, "model"
    # non veniva mai assegnato e la funzione andava in UnboundLocalError.
    if Path(model_path).exists():
        with open(model_path, "rb") as f:
            loaded = pickle.load(f)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
    else:
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
        "📅 Calendario 26/27", "🔮 Predizione", "📊 Classifica", "👨‍⚖️ Arbitri",
        "🎯 Tracker", "📈 Backtest", "ℹ️ Info modello"
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

# Dashboard divisa in più pagine sotto dashboard_pages/ (una per sezione del
# menu laterale) per rendere il file principale gestibile e ogni pagina
# testabile/modificabile in isolamento.
import dashboard_pages.predizione as _page_predizione
import dashboard_pages.calendario as _page_calendario
import dashboard_pages.arbitri as _page_arbitri
import dashboard_pages.classifica as _page_classifica
import dashboard_pages.tracker as _page_tracker
import dashboard_pages.backtest as _page_backtest
import dashboard_pages.info_modello as _page_info_modello

if page == "🔮 Predizione":
    _page_predizione.render(
        teams=teams, model=model, df=df, df_raw=df_raw, ref_stats=ref_stats,
        min_edge=min_edge, bankroll=bankroll,
        BACKEND_CLIENT_OK=BACKEND_CLIENT_OK, is_backend_available=is_backend_available,
        _cached_injuries=_cached_injuries, _cached_pinnacle_odds=_cached_pinnacle_odds,
        load_player_stats_builder=load_player_stats_builder,
    )
elif page == "📅 Calendario 26/27":
    _page_calendario.render()
elif page == "👨‍⚖️ Arbitri":
    _page_arbitri.render(ref_stats)
elif page == "📊 Classifica":
    _page_classifica.render()
elif page == "🎯 Tracker":
    _page_tracker.render()
elif page == "📈 Backtest":
    _page_backtest.render(df_raw)
elif page == "ℹ️ Info modello":
    _page_info_modello.render(df)
