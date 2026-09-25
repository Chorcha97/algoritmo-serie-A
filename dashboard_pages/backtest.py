"""
Pagina "Backtest" — walk-forward backtest del modello.
Estratta da dashboard.py nella separazione in pagine (roadmap #2).
"""
import streamlit as st


def render(df_raw):
    st.caption("Il modello si allena solo sul passato e prevede solo il futuro — nessun lookahead bias")

    st.info("""
    **Walk-forward**: per ogni stagione testata, il modello viene riallenato da zero
    usando solo le stagioni precedenti. È il modo corretto per simulare le performance reali.
    **Arbitro rolling**: le statistiche di ogni arbitro vengono aggiornate partita per partita,
    usando solo le partite già giocate fino a quel momento.
    """)

    col1, col2 = st.columns(2)
    min_edge_bt  = col1.slider("Edge minimo (%)", 1, 30, 7) / 100
    bankroll_bt  = col2.number_input("Bankroll simulato (€)", min_value=100,
                                      max_value=100000, value=1000, step=100)

    col3, col4 = st.columns(2)
    show_referee = col3.checkbox("Mostra analisi per arbitro", value=True)

    st.warning("⏱️ Il walk-forward è più lento del backtest semplice — "
               "allena un modello per ogni stagione. Su 8 stagioni: ~5-8 minuti.")

    if st.button("▶️ Avvia Walk-Forward Backtest", type="primary",
                 use_container_width=True):
        from models.backtest import run_walkforward_backtest, summarize_walkforward, summarize_by_referee

        with st.spinner("Walk-forward in corso — alleno il modello stagione per stagione..."):
            sa_only = df_raw[df_raw["league"] == "Serie A"] if "league" in df_raw.columns else df_raw
        bt = run_walkforward_backtest(
                sa_only,
                min_edge=min_edge_bt,
                bankroll=bankroll_bt,
                verbose=False,
            )

        if bt.empty:
            st.warning("Nessuna value bet trovata. Prova a ridurre l'edge minimo.")
        else:
            # ── Metriche globali ──────────────────────────────────────────────
            tot_stake  = bt["stake_€"].sum()
            tot_profit = bt["profitto_€"].sum()
            roi        = tot_profit / tot_stake * 100 if tot_stake > 0 else 0
            wr         = (bt["esito"] == "✓ Vinta").mean() * 100
            ref_cov    = bt["ref_found"].mean() * 100 if "ref_found" in bt.columns else 0

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Giocate",       f"{len(bt)}")
            c2.metric("Win Rate",      f"{wr:.1f}%")
            c3.metric("Profitto",      f"€{tot_profit:+,.2f}",
                      delta=f"{roi:+.1f}% ROI")
            c4.metric("Stake totale",  f"€{tot_stake:,.2f}")
            c5.metric("Copertura arb.", f"{ref_cov:.0f}%",
                      help="% di partite con dati arbitro disponibili")

            # ── P&L cumulativo ────────────────────────────────────────────────
            st.subheader("📈 P&L Cumulativo (walk-forward)")
            bt_s = bt.sort_values("data").copy()
            bt_s["pnl_cum"] = bt_s["profitto_€"].cumsum()
            st.line_chart(bt_s.set_index("data")["pnl_cum"])

            # ── Per stagione ──────────────────────────────────────────────────
            st.subheader("📅 Performance per stagione")
            summary = summarize_walkforward(bt)
            summary = summary[["stagione","giocate","profitto_€","win_rate","roi_%"]]
            summary.columns = ["Stagione","Giocate","Profitto €","Win Rate %","ROI %"]
            st.dataframe(summary, hide_index=True, use_container_width=True)

            # ── Per mercato ───────────────────────────────────────────────────
            st.subheader("📋 Performance per mercato")
            mp = bt.groupby("mercato").agg(
                giocate=("stake_€","count"),
                stake  =("stake_€","sum"),
                profit =("profitto_€","sum"),
            ).reset_index()
            mp["ROI%"] = (mp["profit"]/mp["stake"]*100).round(1)
            mp["profit"] = mp["profit"].round(2)
            mp = mp.sort_values("ROI%", ascending=False)
            mp.columns = ["Mercato","Giocate","Stake €","Profitto €","ROI %"]
            st.dataframe(mp, hide_index=True, use_container_width=True)

            # ── Per arbitro ───────────────────────────────────────────────────
            if show_referee:
                ref_summary = summarize_by_referee(bt)
                if not ref_summary.empty:
                    st.subheader("👨‍⚖️ Performance per arbitro")
                    st.caption("Solo arbitri con almeno 10 giocate nel backtest")
                    ref_summary.columns = ["Arbitro","Giocate","Stake €",
                                           "Profitto €","Gialli medi","ROI %"]
                    st.dataframe(ref_summary, hide_index=True,
                                 use_container_width=True)
                else:
                    st.caption("Dati arbitro non disponibili — "
                               "colonna Referee assente nel dataset")

            # ── Download ──────────────────────────────────────────────────────
            with st.expander("📜 Log completo"):
                cols_show = ["stagione","data","partita","arbitro","mercato",
                             "quota","prob_%","edge_%","stake_€","esito","profitto_€"]
                cols_show = [c for c in cols_show if c in bt.columns]
                st.dataframe(bt[cols_show], hide_index=True,
                             use_container_width=True)
                st.download_button(
                    "⬇️ Scarica CSV",
                    bt[cols_show].to_csv(index=False).encode(),
                    "backtest_walkforward.csv", "text/csv")


