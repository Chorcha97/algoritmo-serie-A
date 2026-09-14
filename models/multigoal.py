"""models/multigoal.py — MultiGoal e distribuzione risultati"""
import numpy as np
from scipy.stats import poisson

MULTIGOAL_BANDS = [
    ('1-2', 1, 2), ('1-3', 1, 3), ('1-4', 1, 4),
    ('2-3', 2, 3), ('2-4', 2, 4), ('2-5', 2, 5),
    ('3-4', 3, 4), ('3-5', 3, 5),
]

def multigoal_probs(lam_home: float, lam_away: float) -> dict:
    """Calcola probabilità MultiGoal da lambda Poisson."""
    lam_total = lam_home + lam_away
    probs = {}
    for label, low, high in MULTIGOAL_BANDS:
        p = sum(poisson.pmf(k, lam_total) for k in range(low, high + 1))
        probs[label] = round(p, 3)
    return probs

def score_matrix(lam_home: float, lam_away: float, max_goals: int = 6) -> dict:
    """Matrice probabilità risultato esatto e cluster."""
    matrix = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson.pmf(h, lam_home) * poisson.pmf(a, lam_away)
            matrix[f'{h}-{a}'] = round(p, 4)

    # Cluster risultati
    clusters = {
        '0-0': matrix.get('0-0', 0),
        '1-0': matrix.get('1-0', 0),
        '0-1': matrix.get('0-1', 0),
        '1-1': matrix.get('1-1', 0),
        '2-0': matrix.get('2-0', 0),
        '0-2': matrix.get('0-2', 0),
        '2-1': matrix.get('2-1', 0),
        '1-2': matrix.get('1-2', 0),
        '2-2': matrix.get('2-2', 0),
        '3-0': matrix.get('3-0', 0),
        '0-3': matrix.get('0-3', 0),
        '3-1': matrix.get('3-1', 0),
        '1-3': matrix.get('1-3', 0),
        '3-2': matrix.get('3-2', 0),
        '2-3': matrix.get('2-3', 0),
    }
    return {
        'full_matrix': matrix,
        'top_scores': sorted(clusters.items(), key=lambda x: -x[1])[:8]
    }

def expected_goals_breakdown(lam_home: float, lam_away: float) -> dict:
    """Statistiche attese dalla distribuzione Poisson."""
    lam_total = lam_home + lam_away
    return {
        'expected_home': round(lam_home, 2),
        'expected_away': round(lam_away, 2),
        'expected_total': round(lam_total, 2),
        'prob_gg': round((1 - poisson.pmf(0, lam_home)) * (1 - poisson.pmf(0, lam_away)), 3),
        'prob_ng': round(1 - (1 - poisson.pmf(0, lam_home)) * (1 - poisson.pmf(0, lam_away)), 3),
        'prob_btts_no': round(poisson.cdf(0, lam_home) + poisson.cdf(0, lam_away) - poisson.cdf(0, lam_home) * poisson.cdf(0, lam_away), 3),
    }
