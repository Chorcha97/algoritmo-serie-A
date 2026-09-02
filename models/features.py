"""
Feature Engineering per Serie A
Genera le feature predittive a partire dal dataset grezzo.

Feature principali:
  - Forma recente pesata (EMA gol, xG, risultati)
  - Fixture congestion (stanchezza)
  - ELO differenziale
  - Quote implicite e no-vig probabilities
"""

import pandas as pd
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────
# 1. FORMA RECENTE (EMA)
# ─────────────────────────────────────────────

def compute_ema_form(df: pd.DataFrame, span: int = 5) -> pd.DataFrame:
    """
    Calcola la forma recente per ogni squadra usando EMA (media mobile esponenziale).
    Considera ogni squadra sia come casa che come ospite.

    Aggiunge colonne:
        ema_gol_fatti_home/away, ema_gol_subiti_home/away
        ema_xg_fatti_home/away, ema_xg_subiti_home/away
        ema_punti_home/away  (3/1/0)
        ema_tiri_porta_home/away
    """
    df = df.sort_values("Date").copy()

    teams = pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique()
    team_stats: dict = {t: [] for t in teams}

    # Dizionari per EMA rolling (aggiornati partita per partita)
    ema_state: dict = {}  # team → {metric: valore_corrente}

    alpha = 2 / (span + 1)

    def ema_update(state: dict, team: str, new_vals: dict):
        if team not in state:
            state[team] = new_vals.copy()
        else:
            for k, v in new_vals.items():
                if not np.isnan(v):
                    state[team][k] = alpha * v + (1 - alpha) * state[team].get(k, v)

    def ema_get(state: dict, team: str, metric: str) -> float:
        return state.get(team, {}).get(metric, np.nan)

    # Mappa risultato → punti
    def points(result: str, side: str) -> float:
        if result == side:
            return 3.0
        elif result == "D":
            return 1.0
        else:
            return 0.0

    rows_out = []

    for _, row in df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]

        # Leggi EMA PRIMA di aggiornare (feature = stato precedente alla partita)
        feature_row = {
            # Casa
            "f_ema_gol_fatti_h":    ema_get(ema_state, h, "gol_fatti"),
            "f_ema_gol_subiti_h":   ema_get(ema_state, h, "gol_subiti"),
            "f_ema_xg_fatti_h":     ema_get(ema_state, h, "xg_fatti"),
            "f_ema_xg_subiti_h":    ema_get(ema_state, h, "xg_subiti"),
            "f_ema_punti_h":        ema_get(ema_state, h, "punti"),
            "f_ema_tiri_porta_h":   ema_get(ema_state, h, "tiri_porta"),
            # Ospite
            "f_ema_gol_fatti_a":    ema_get(ema_state, a, "gol_fatti"),
            "f_ema_gol_subiti_a":   ema_get(ema_state, a, "gol_subiti"),
            "f_ema_xg_fatti_a":     ema_get(ema_state, a, "xg_fatti"),
            "f_ema_xg_subiti_a":    ema_get(ema_state, a, "xg_subiti"),
            "f_ema_punti_a":        ema_get(ema_state, a, "punti"),
            "f_ema_tiri_porta_a":   ema_get(ema_state, a, "tiri_porta"),
            # Differenziali (utili per l'ELO implicito)
            "f_ema_xg_diff":        ema_get(ema_state, h, "xg_fatti") - ema_get(ema_state, a, "xg_fatti"),
            "f_ema_gol_diff":       ema_get(ema_state, h, "gol_fatti") - ema_get(ema_state, a, "gol_fatti"),
        }
        rows_out.append(feature_row)

        # Aggiorna EMA con i risultati di questa partita
        fthg = row.get("FTHG", np.nan)
        ftag = row.get("FTAG", np.nan)
        ftr  = row.get("FTR", "")
        xgh  = row.get("xg_home", np.nan)
        xga  = row.get("xg_away", np.nan)
        hstp = row.get("HST", np.nan)
        astp = row.get("AST", np.nan)

        ema_update(ema_state, h, {
            "gol_fatti": fthg, "gol_subiti": ftag,
            "xg_fatti": xgh, "xg_subiti": xga,
            "punti": points(ftr, "H"), "tiri_porta": hstp,
        })
        ema_update(ema_state, a, {
            "gol_fatti": ftag, "gol_subiti": fthg,
            "xg_fatti": xga, "xg_subiti": xgh,
            "punti": points(ftr, "A"), "tiri_porta": astp,
        })

    feature_df = pd.DataFrame(rows_out, index=df.index)
    return pd.concat([df, feature_df], axis=1)


# ─────────────────────────────────────────────
# 2. FIXTURE CONGESTION (stanchezza)
# ─────────────────────────────────────────────

def compute_fixture_density(df: pd.DataFrame, window_days: int = 10) -> pd.DataFrame:
    """
    Conta quante partite ha giocato ogni squadra nei `window_days` giorni
    precedenti la partita corrente. Proxy per stanchezza e rotazioni.
    """
    df = df.sort_values("Date").copy()
    dates = df["Date"].values
    home_teams = df["HomeTeam"].values
    away_teams = df["AwayTeam"].values

    n = len(df)
    density_home = np.zeros(n)
    density_away = np.zeros(n)

    for i in range(n):
        d = pd.Timestamp(dates[i])
        h, a = home_teams[i], away_teams[i]
        window_start = d - pd.Timedelta(days=window_days)

        count_h = count_a = 0
        for j in range(i):
            dj = pd.Timestamp(dates[j])
            if dj < window_start:
                continue
            if home_teams[j] == h or away_teams[j] == h:
                count_h += 1
            if home_teams[j] == a or away_teams[j] == a:
                count_a += 1

        density_home[i] = count_h
        density_away[i] = count_a

    df["f_fixture_density_h"] = density_home
    df["f_fixture_density_a"] = density_away
    df["f_fixture_density_diff"] = density_home - density_away
    return df


# ─────────────────────────────────────────────
# 3. QUOTE → PROBABILITÀ NO-VIG
# ─────────────────────────────────────────────

def odds_to_prob_novig(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte le quote Pinnacle (le più efficienti) in probabilità no-vig
    (rimuove il margine del bookmaker per avere prob. pulite da usare
    come baseline o come confronto con le nostre stime).

    Aggiunge: prob_h_bk, prob_d_bk, prob_a_bk (probabilità implicite no-vig)
              overround (margine totale del bookmaker)
    """
    df = df.copy()

    # Usa Pinnacle se disponibile, altrimenti Bet365
    for pfx in [("PSH", "PSD", "PSA"), ("B365H", "B365D", "B365A")]:
        oh, od, oa = pfx
        if all(c in df.columns for c in [oh, od, oa]):
            mask = df[oh].notna() & df[od].notna() & df[oa].notna()
            raw_h = 1 / df.loc[mask, oh]
            raw_d = 1 / df.loc[mask, od]
            raw_a = 1 / df.loc[mask, oa]
            total = raw_h + raw_d + raw_a
            df.loc[mask, "bk_prob_h"] = raw_h / total
            df.loc[mask, "bk_prob_d"] = raw_d / total
            df.loc[mask, "bk_prob_a"] = raw_a / total
            df.loc[mask, "bk_overround"] = total - 1
            df.loc[mask, "bk_source"] = pfx[0][:2]
            break

    return df


# ─────────────────────────────────────────────
# 4. HEAD TO HEAD
# ─────────────────────────────────────────────

def compute_h2h(df: pd.DataFrame, n_matches: int = 5) -> pd.DataFrame:
    """
    Aggiunge statistiche head-to-head sulle ultime N sfide tra le due squadre:
    - win rate casa/ospite/pareggio
    - media gol nelle sfide dirette
    """
    df = df.sort_values("Date").copy()
    h2h_win_h = []
    h2h_win_a = []
    h2h_draw = []
    h2h_avg_goals = []

    for i, row in df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        past = df.loc[:i].iloc[:-1]  # escludi la partita corrente
        h2h = past[
            ((past["HomeTeam"] == h) & (past["AwayTeam"] == a)) |
            ((past["HomeTeam"] == a) & (past["AwayTeam"] == h))
        ].tail(n_matches)

        if h2h.empty:
            h2h_win_h.append(np.nan)
            h2h_win_a.append(np.nan)
            h2h_draw.append(np.nan)
            h2h_avg_goals.append(np.nan)
            continue

        # Normalizza: chi è "home" in ogni sfida passata?
        wins_h = wins_a = draws = 0
        total_goals = 0
        for _, pr in h2h.iterrows():
            if pr["HomeTeam"] == h:
                ftr = pr.get("FTR", "")
                wins_h += (ftr == "H")
                wins_a += (ftr == "A")
                draws  += (ftr == "D")
            else:
                ftr = pr.get("FTR", "")
                wins_h += (ftr == "A")
                wins_a += (ftr == "H")
                draws  += (ftr == "D")
            total_goals += pr.get("FTHG", 0) + pr.get("FTAG", 0)

        n = len(h2h)
        h2h_win_h.append(wins_h / n)
        h2h_win_a.append(wins_a / n)
        h2h_draw.append(draws / n)
        h2h_avg_goals.append(total_goals / n)

    df["f_h2h_winrate_h"] = h2h_win_h
    df["f_h2h_winrate_a"] = h2h_win_a
    df["f_h2h_draw_rate"] = h2h_draw
    df["f_h2h_avg_goals"] = h2h_avg_goals
    return df


# ─────────────────────────────────────────────
# 5. COLONNE TARGET
# ─────────────────────────────────────────────

def add_cards_tendency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola la tendenza storica ai cartellini per ogni squadra
    usando EMA — più preciso dell arbitro generico.
    """
    df = df.sort_values("Date").copy()
    team_cards = {}
    alpha = 2 / (5 + 1)

    home_cards_ema = []
    away_cards_ema = []

    for _, row in df.iterrows():
        h, a = row.get("HomeTeam"), row.get("AwayTeam")

        hc = team_cards.get(h, 2.0)
        ac = team_cards.get(a, 2.0)
        home_cards_ema.append(hc)
        away_cards_ema.append(ac)

        hy = float(row.get("HY", 0) or 0)
        ay = float(row.get("AY", 0) or 0)
        if not (hy == 0 and ay == 0):
            team_cards[h] = alpha * hy + (1 - alpha) * team_cards.get(h, hy)
            team_cards[a] = alpha * ay + (1 - alpha) * team_cards.get(a, ay)

    df["f_cards_ema_home"] = home_cards_ema
    df["f_cards_ema_away"] = away_cards_ema
    df["f_cards_ema_total"] = df["f_cards_ema_home"] + df["f_cards_ema_away"]
    return df


def add_motivation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge feature di motivazione: classifica, coppe europee, Coppa Italia.
    """
    try:
        from models.motivation import compute_motivation_features, fetch_standings
        standings = fetch_standings()
        if not standings:
            df["f_motivation_home"]  = 1.0
            df["f_motivation_away"]  = 1.0
            df["f_motivation_diff"]  = 0.0
            df["f_fatigue_home"]     = 1.0
            df["f_fatigue_away"]     = 1.0
            df["f_fatigue_diff"]     = 0.0
            df["f_position_home"]    = 10
            df["f_position_away"]    = 10
            df["f_coppa_europa_home"]= 0
            df["f_coppa_europa_away"]= 0
            return df

        rows = []
        for _, row in df.iterrows():
            feat = compute_motivation_features(
                row["HomeTeam"], row["AwayTeam"],
                match_date=str(row.get("Date", "")),
                standings=standings,
            )
            rows.append(feat)

        feat_df = pd.DataFrame(rows, index=df.index)
        df = pd.concat([df, feat_df], axis=1)
    except Exception as e:
        print(f"  [warning] Motivazione non disponibile: {e}")
    return df


def add_dynamic_elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge ELO dinamico calcolato dai risultati reali
    invece dell ELO statico di Club Elo.
    """
    try:
        from models.elo_dynamic import compute_dynamic_elo
        df = compute_dynamic_elo(df)
        # Aggiorna anche elo_diff con il valore dinamico
        if "elo_diff_dyn" in df.columns:
            df["elo_diff"] = df["elo_diff_dyn"]
    except Exception as e:
        print(f"  [warning] ELO dinamico non disponibile: {e}")
    return df


def add_opta_team_stats(df):
    """
    Aggiunge statistiche Opta per squadra come feature.
    xG for/against, gol, tiri — aggiornati ogni settimana.
    """
    import json
    from pathlib import Path
    path = Path('cache/opta_team_stats.json')
    if not path.exists():
        for col in ['f_opta_xg_for_h','f_opta_xg_for_a','f_opta_xg_ag_h',
                    'f_opta_xg_ag_a','f_opta_xg_ratio_h','f_opta_xg_ratio_a','f_opta_xg_diff']:
            df[col] = 0.0
        return df

    stats = json.load(open(path))
    avg_xgf = sum(s.get('xg_for',0) for s in stats.values()) / max(len(stats),1)
    avg_xga = sum(s.get('xg_against',0) for s in stats.values()) / max(len(stats),1)

    def get_xgf(team): return stats.get(team, {}).get('xg_for', avg_xgf)
    def get_xga(team): return stats.get(team, {}).get('xg_against', avg_xga)

    df = df.copy()
    df['f_opta_xg_for_h']   = df['HomeTeam'].apply(get_xgf)
    df['f_opta_xg_for_a']   = df['AwayTeam'].apply(get_xgf)
    df['f_opta_xg_ag_h']    = df['HomeTeam'].apply(get_xga)
    df['f_opta_xg_ag_a']    = df['AwayTeam'].apply(get_xga)
    df['f_opta_xg_ratio_h'] = df['f_opta_xg_for_h'] - df['f_opta_xg_ag_h']
    df['f_opta_xg_ratio_a'] = df['f_opta_xg_for_a'] - df['f_opta_xg_ag_a']
    df['f_opta_xg_diff']    = df['f_opta_xg_ratio_h'] - df['f_opta_xg_ratio_a']
    return df


def add_sofascore_team_stats(df):
    """
    Aggiunge statistiche Sofascore per squadra come feature.
    Corner, xG, falli, cartellini, tiri, big chances — aggiornati ogni settimana.
    """
    import json
    from pathlib import Path
    path_sc = Path('cache/sofascore_team_stats_clean.json')
    if not path_sc.exists():
        cols = [
            'f_sc_corners_h','f_sc_corners_a','f_sc_corners_diff',
            'f_sc_xg_h','f_sc_xg_a',
            'f_sc_fouls_h','f_sc_fouls_a','f_sc_fouls_total',
            'f_sc_yellow_h','f_sc_yellow_a','f_sc_yellow_total',
            'f_sc_shots_h','f_sc_shots_a',
            'f_sc_bigch_h','f_sc_bigch_a',
            'f_sc_possession_h','f_sc_possession_a',
        ]
        for c in cols: df[c] = 0.0
        return df

    stats = json.loads(path_sc.read_text())

    # Medie di default
    avg = {k: sum(s.get(k,0) for s in stats.values())/max(len(stats),1)
           for k in ['corners_per_match','expectedGoals_per_match','fouls_per_match',
                     'yellowCards_per_match','shots_per_match','bigChances_per_match',
                     'averageBallPossession_per_match']}

    def get(team, key): return stats.get(team, {}).get(key, avg.get(key, 0))

    df = df.copy()
    df['f_sc_corners_h']     = df['HomeTeam'].apply(lambda t: get(t,'corners_per_match'))
    df['f_sc_corners_a']     = df['AwayTeam'].apply(lambda t: get(t,'corners_per_match'))
    df['f_sc_corners_diff']  = df['f_sc_corners_h'] - df['f_sc_corners_a']
    df['f_sc_xg_h']          = df['HomeTeam'].apply(lambda t: get(t,'expectedGoals_per_match'))
    df['f_sc_xg_a']          = df['AwayTeam'].apply(lambda t: get(t,'expectedGoals_per_match'))
    df['f_sc_fouls_h']       = df['HomeTeam'].apply(lambda t: get(t,'fouls_per_match'))
    df['f_sc_fouls_a']       = df['AwayTeam'].apply(lambda t: get(t,'fouls_per_match'))
    df['f_sc_fouls_total']   = df['f_sc_fouls_h'] + df['f_sc_fouls_a']
    df['f_sc_yellow_h']      = df['HomeTeam'].apply(lambda t: get(t,'yellowCards_per_match'))
    df['f_sc_yellow_a']      = df['AwayTeam'].apply(lambda t: get(t,'yellowCards_per_match'))
    df['f_sc_yellow_total']  = df['f_sc_yellow_h'] + df['f_sc_yellow_a']
    df['f_sc_shots_h']       = df['HomeTeam'].apply(lambda t: get(t,'shots_per_match'))
    df['f_sc_shots_a']       = df['AwayTeam'].apply(lambda t: get(t,'shots_per_match'))
    df['f_sc_bigch_h']       = df['HomeTeam'].apply(lambda t: get(t,'bigChances_per_match'))
    df['f_sc_bigch_a']       = df['AwayTeam'].apply(lambda t: get(t,'bigChances_per_match'))
    df['f_sc_possession_h']  = df['HomeTeam'].apply(lambda t: get(t,'averageBallPossession_per_match'))
    df['f_sc_possession_a']  = df['AwayTeam'].apply(lambda t: get(t,'averageBallPossession_per_match'))
    return df


def add_opta_powerranking(df):
    import json
    from pathlib import Path
    pr_path = Path('cache/opta_powerranking.json')
    if not pr_path.exists():
        df['f_pr_home'] = 0.5
        df['f_pr_away'] = 0.5
        df['f_pr_diff'] = 0.0
        return df
    data = json.load(open(pr_path))
    ranking = data['division'][0]['ranking']
    pr = {r['contestantShortName']: float(r['currentRating']) for r in ranking}
    # Mappa nomi alternativi
    pr['Internazionale'] = pr.get('Inter', 86.3)
    min_r = min(pr.values()); max_r = max(pr.values())
    def norm(team):
        raw = pr.get(team, (min_r+max_r)/2)
        return (raw - min_r) / (max_r - min_r) if max_r > min_r else 0.5
    df = df.copy()
    df['f_pr_home'] = df['HomeTeam'].apply(norm)
    df['f_pr_away'] = df['AwayTeam'].apply(norm)
    df['f_pr_diff'] = df['f_pr_home'] - df['f_pr_away']
    return df


def add_team_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge qualita rosa per reparto dalle quotazioni fantacalcio.
    Piu preciso di Transfermarkt perche aggiornato e specifico per ruolo.
    """
    import json
    from pathlib import Path

    fanta_path = Path('cache/fanta_quality_2627.json')
    val_path   = Path('cache/team_values.json')

    if fanta_path.exists():
        with open(fanta_path) as f:
            fanta = json.load(f)

        avg_att = sum(d.get('Offence',   {}).get('avg_qt', 10) for d in fanta.values()) / len(fanta)
        avg_mid = sum(d.get('Midfield',  {}).get('avg_qt', 7)  for d in fanta.values()) / len(fanta)
        avg_def = sum(d.get('Defence',   {}).get('avg_qt', 6)  for d in fanta.values()) / len(fanta)
        avg_gk  = sum(d.get('Goalkeeper',{}).get('avg_qt', 4)  for d in fanta.values()) / len(fanta)

        def get_qt(team, pos, default):
            return fanta.get(team, {}).get(pos, {}).get('avg_qt', default)

        df = df.copy()
        df['f_att_quality_h']  = df['HomeTeam'].apply(lambda t: get_qt(t, 'Offence',    avg_att))
        df['f_att_quality_a']  = df['AwayTeam'].apply(lambda t: get_qt(t, 'Offence',    avg_att))
        df['f_mid_quality_h']  = df['HomeTeam'].apply(lambda t: get_qt(t, 'Midfield',   avg_mid))
        df['f_mid_quality_a']  = df['AwayTeam'].apply(lambda t: get_qt(t, 'Midfield',   avg_mid))
        df['f_def_quality_h']  = df['HomeTeam'].apply(lambda t: get_qt(t, 'Defence',    avg_def))
        df['f_def_quality_a']  = df['AwayTeam'].apply(lambda t: get_qt(t, 'Defence',    avg_def))
        df['f_gk_quality_h']   = df['HomeTeam'].apply(lambda t: get_qt(t, 'Goalkeeper', avg_gk))
        df['f_gk_quality_a']   = df['AwayTeam'].apply(lambda t: get_qt(t, 'Goalkeeper', avg_gk))
        df['f_att_vs_def']     = df['f_att_quality_h'] - df['f_def_quality_a']
        df['f_att_vs_def_away']= df['f_att_quality_a'] - df['f_def_quality_h']

        # Mantieni anche valore totale da Transfermarkt per compatibilita
        if val_path.exists():
            with open(val_path) as f:
                values = json.load(f)
            avg_val = sum(values.values()) / len(values)
            df['f_value_home']  = df['HomeTeam'].map(values).fillna(avg_val)
            df['f_value_away']  = df['AwayTeam'].map(values).fillna(avg_val)
            df['f_value_diff']  = df['f_value_home'] - df['f_value_away']
            df['f_value_ratio'] = df['f_value_home'] / df['f_value_away'].replace(0, avg_val)
        return df

    elif val_path.exists():
        with open(val_path) as f:
            values = json.load(f)
        avg_value = sum(values.values()) / len(values) if values else 300.0
        df = df.copy()
        df['f_value_home']  = df['HomeTeam'].map(values).fillna(avg_value)
        df['f_value_away']  = df['AwayTeam'].map(values).fillna(avg_value)
        df['f_value_diff']  = df['f_value_home'] - df['f_value_away']
        df['f_value_ratio'] = df['f_value_home'] / df['f_value_away'].replace(0, avg_value)
        return df

    df['f_value_home'] = 0.0
    df['f_value_away'] = 0.0
    df['f_value_diff'] = 0.0
    df['f_value_ratio'] = 1.0
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge le colonne target per i diversi mercati:
        target_1x2:   0=H, 1=D, 2=A
        target_ou25:  0=under 2.5, 1=over 2.5
        target_gg:    1=entrambe segnano, 0=no
        target_result_exact: stringa "2-1" ecc.
    """
    df = df.copy()
    df["target_1x2"] = df["FTR"].map({"H": 0, "D": 1, "A": 2})
    total_goals = df["FTHG"] + df["FTAG"]
    df["target_ou25"] = (total_goals > 2.5).astype(int)
    df["target_gg"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
    df["target_exact"] = df["FTHG"].astype(str) + "-" + df["FTAG"].astype(str)
    df["total_goals"] = total_goals
    return df


# ─────────────────────────────────────────────
# 6. PIPELINE COMPLETA
# ─────────────────────────────────────────────

def build_features(df_raw: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Applica tutto il feature engineering in sequenza.
    Restituisce il dataset arricchito pronto per il training.
    """
    if verbose:
        print("[feature engineering] Avvio pipeline ...")

    df = df_raw.copy()

    if verbose: print("  → EMA forma recente ...")
    df = compute_ema_form(df)

    if verbose: print("  → Fixture congestion ...")
    df = compute_fixture_density(df)

    if verbose: print("  → Quote no-vig ...")
    df = odds_to_prob_novig(df)

    if verbose: print("  → Head to head ...")
    df = compute_h2h(df)

    if verbose: print("  → Tendenza cartellini ...")
    df = add_cards_tendency(df)

    if verbose: print("  → ELO dinamico ...")
    df = add_dynamic_elo(df)

    if verbose: print("  → Motivazione e coppe ...")
    df = add_motivation_features(df)

    if verbose: print("  → Valori di mercato ...")
    df = add_team_values(df)
    df = add_opta_powerranking(df)
    df = add_opta_team_stats(df)
    df = add_sofascore_team_stats(df)

    if verbose: print("  → Colonne target ...")
    df = add_targets(df)

    # Aggiungi ELO diff se presente
    if "elo_diff" not in df.columns:
        df["elo_diff"] = np.nan

    feature_cols = [c for c in df.columns if c.startswith("f_") or c.startswith("elo_")]
    if verbose:
        print(f"  ✓ {len(feature_cols)} feature generate")
        print(f"  ✓ Dataset finale: {len(df)} righe × {len(df.columns)} colonne")

    return df


def get_feature_columns() -> list:
    """Restituisce la lista canonica delle feature per il modello."""
    return [
        # Forma EMA
        "f_ema_gol_fatti_h", "f_ema_gol_subiti_h",
        "f_ema_xg_fatti_h",  "f_ema_xg_subiti_h",
        "f_ema_punti_h",     "f_ema_tiri_porta_h",
        "f_ema_gol_fatti_a", "f_ema_gol_subiti_a",
        "f_ema_xg_fatti_a",  "f_ema_xg_subiti_a",
        "f_ema_punti_a",     "f_ema_tiri_porta_a",
        "f_ema_xg_diff",     "f_ema_gol_diff",
        # Stanchezza
        "f_fixture_density_h", "f_fixture_density_a", "f_fixture_density_diff",
        # ELO
        "elo_home", "elo_away", "elo_diff",
        # Head to head
        "f_h2h_winrate_h", "f_h2h_winrate_a", "f_h2h_draw_rate", "f_h2h_avg_goals",
        # ELO dinamico
        "elo_home_dyn", "elo_away_dyn", "elo_diff_dyn",
        # Motivazione e coppe
        "f_motivation_home", "f_motivation_away", "f_motivation_diff",
        "f_fatigue_home", "f_fatigue_away", "f_fatigue_diff",
        "f_position_home", "f_position_away",
        "f_coppa_europa_home", "f_coppa_europa_away",
        # Valore di mercato Transfermarkt
        "f_value_home", "f_value_away", "f_value_diff", "f_value_ratio",
        "f_pr_home", "f_pr_away", "f_pr_diff",
        "f_opta_xg_for_h", "f_opta_xg_for_a",
        "f_opta_xg_ag_h", "f_opta_xg_ag_a",
        "f_opta_xg_ratio_h", "f_opta_xg_ratio_a", "f_opta_xg_diff",
        "f_sc_corners_h", "f_sc_corners_a", "f_sc_corners_diff",
        "f_sc_xg_h", "f_sc_xg_a",
        "f_sc_fouls_h", "f_sc_fouls_a", "f_sc_fouls_total",
        "f_sc_yellow_h", "f_sc_yellow_a", "f_sc_yellow_total",
        "f_sc_shots_h", "f_sc_shots_a",
        "f_sc_bigch_h", "f_sc_bigch_a",
        "f_sc_possession_h", "f_sc_possession_a",
        # Qualita per reparto (quotazioni fantacalcio)
        "f_att_quality_h", "f_att_quality_a",
        "f_mid_quality_h", "f_mid_quality_a",
        "f_def_quality_h", "f_def_quality_a",
        "f_gk_quality_h",  "f_gk_quality_a",
        "f_att_vs_def",    "f_att_vs_def_away",
        # Tendenza cartellini
        "f_cards_ema_home", "f_cards_ema_away", "f_cards_ema_total",
    ]


if __name__ == "__main__":
    # Test rapido con dati sintetici
    import random
    random.seed(42)
    teams = ["Juventus","Inter","Milan","Napoli","Roma","Lazio","Atalanta","Fiorentina"]
    rows = []
    for i in range(200):
        h, a = random.sample(teams, 2)
        gh, ga = random.randint(0,4), random.randint(0,4)
        ftr = "H" if gh > ga else ("A" if ga > gh else "D")
        rows.append({
            "Date": pd.Timestamp("2022-08-01") + pd.Timedelta(days=i*3),
            "HomeTeam": h, "AwayTeam": a,
            "FTHG": gh, "FTAG": ga, "FTR": ftr,
            "HTHG": 0, "HTAG": 0, "HTR": "D",
            "HST": random.randint(2,8), "AST": random.randint(1,6),
            "HS": random.randint(5,18), "AS": random.randint(3,14),
            "HC": random.randint(2,10), "AC": random.randint(1,8),
            "xg_home": round(random.uniform(0.5, 2.5), 2),
            "xg_away": round(random.uniform(0.3, 2.0), 2),
            "elo_home": random.randint(1500, 1900),
            "elo_away": random.randint(1500, 1900),
            "elo_diff": 0,
            "PSH": round(random.uniform(1.5, 4.0), 2),
            "PSD": round(random.uniform(2.8, 4.5), 2),
            "PSA": round(random.uniform(1.8, 5.0), 2),
            "season": "2022-23",
        })
    raw = pd.DataFrame(rows)
    df_feat = build_features(raw)
    print("\nFeature columns:")
    feat_cols = [c for c in df_feat.columns if c.startswith("f_") or c.startswith("elo_")]
    for c in feat_cols:
        nn = df_feat[c].notna().sum()
        print(f"  {c:35s}  non-null: {nn}/{len(df_feat)}")
