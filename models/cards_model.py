"""models/cards_model.py — Modello per cartellini"""
import numpy as np
import pandas as pd
from scipy.stats import poisson

LEAGUE_AVG_CARDS = 4.7   # media storica (2021-2023)
LEAGUE_AVG_CARDS_2627 = 3.5  # stima nuove direttive 2026/27

class CardsModel:
    def __init__(self, df: pd.DataFrame = None, ref_stats: pd.DataFrame = None):
        self.df = df
        self.ref_stats = ref_stats

    def _team_cards_avg(self, team: str, role: str, before_date=None, window: int = 7) -> float:
        if self.df is None:
            return LEAGUE_AVG_CARDS_2627 / 2
        col = 'HY' if role == 'home' else 'AY'
        team_col = 'HomeTeam' if role == 'home' else 'AwayTeam'
        mask = self.df[team_col] == team
        if before_date is not None:
            mask &= self.df['Date'] < before_date
        sub = self.df[mask].tail(window)
        if len(sub) >= 3 and col in sub.columns:
            val = pd.to_numeric(sub[col], errors='coerce').dropna().mean()
            if not np.isnan(val):
                return val
        return LEAGUE_AVG_CARDS_2627 / 2

    def _referee_factor(self, referee: str) -> float:
        """Fattore arbitro (1.0 = media lega). Usa dati 2026/27 se disponibili."""
        if self.ref_stats is None or not referee or referee == 'Non specificato':
            return 1.0
        try:
            ref_data = self.ref_stats.get(referee, {})
            if ref_data.get('partite', 0) >= 3:
                ref_cards_avg = ref_data.get('media_cartellini', LEAGUE_AVG_CARDS_2627)
                return ref_cards_avg / LEAGUE_AVG_CARDS_2627
        except Exception:
            pass
        return 1.0

    def predict(self, home: str, away: str, referee: str = None, before_date=None) -> dict:
        cards_h = self._team_cards_avg(home, 'home', before_date)
        cards_a = self._team_cards_avg(away, 'away', before_date)
        ref_factor = self._referee_factor(referee)
        lam_total = (cards_h + cards_a) * ref_factor

        result = {
            'home_expected': round(cards_h, 1),
            'away_expected': round(cards_a, 1),
            'total_expected': round(lam_total, 1),
            'referee_factor': round(ref_factor, 2),
            'probabilities': {}
        }
        for threshold in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]:
            k = int(threshold)
            result['probabilities'][f'over_{threshold}'] = round(1 - poisson.cdf(k, lam_total), 3)
            result['probabilities'][f'under_{threshold}'] = round(poisson.cdf(k, lam_total), 3)
        return result
