"""models/corner_model.py — Modello Poisson per corner"""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from pathlib import Path

LEAGUE_AVG_HOME = 4.8   # media storica corner casa
LEAGUE_AVG_AWAY = 4.7   # media storica corner ospite

class CornerModel:
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self._cache = {}

    def _team_avg(self, team: str, role: str, before_date=None, window: int = 7) -> float:
        col_h = 'home_corners_avg' if 'home_corners_avg' in (self.df.columns if self.df is not None else []) else None
        col_a = 'away_corners_avg' if 'away_corners_avg' in (self.df.columns if self.df is not None else []) else None

        if self.df is not None and col_h and col_a:
            if role == 'home':
                mask = self.df['HomeTeam'] == team
                if before_date is not None:
                    mask &= self.df['Date'] < before_date
                sub = self.df[mask].tail(window)
                col = 'HC'
            else:
                mask = self.df['AwayTeam'] == team
                if before_date is not None:
                    mask &= self.df['Date'] < before_date
                sub = self.df[mask].tail(window)
                col = 'AC'
            if len(sub) >= 3 and col in sub.columns:
                val = pd.to_numeric(sub[col], errors='coerce').dropna().mean()
                if not np.isnan(val):
                    return val
        return LEAGUE_AVG_HOME if role == 'home' else LEAGUE_AVG_AWAY

    def predict(self, home: str, away: str, before_date=None) -> dict:
        lam_h = self._team_avg(home, 'home', before_date)
        lam_a = self._team_avg(away, 'away', before_date)
        lam_total = lam_h + lam_a

        result = {
            'home_expected': round(lam_h, 1),
            'away_expected': round(lam_a, 1),
            'total_expected': round(lam_total, 1),
            'probabilities': {}
        }
        for threshold in [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]:
            k = int(threshold)
            result['probabilities'][f'over_{threshold}'] = round(1 - poisson.cdf(k, lam_total), 3)
            result['probabilities'][f'under_{threshold}'] = round(poisson.cdf(k, lam_total), 3)
        return result
