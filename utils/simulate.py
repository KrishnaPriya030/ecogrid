# ─────────────────────────────────────────────────────────────
#  EcoGrid — utils/simulate.py
#  Phase 09: What-if scenario simulation using saved Pipeline
# ─────────────────────────────────────────────────────────────

import pandas as pd
import joblib

from .pipeline import FEATURES, NUMERIC, CATEGORICAL, FOSSIL

RISK_LABEL = {0: 'Low risk',    1: 'Medium risk', 2: 'High risk'}
RISK_EMOJI = {0: '🟢',          1: '🟡',           2: '🔴'}
RISK_COLOR = {0: '#22c55e',     1: '#f59e0b',      2: '#ef4444'}

FUEL_OPTIONS  = ['COAL', 'GAS', 'OIL', 'NUCLEAR', 'WIND', 'BIOMASS', 'HYDRO', 'GEOTHERMAL']


def predict_risk(model_pipeline,
                 capfac: float,
                 namepcap: float,
                 plngenan: float,
                 plhtian: float,
                 plfuelct: str,
                 pstatabb: str = 'TX',
                 subrgn: str   = 'SRSO') -> dict:
    """
    Run the full Pipeline on a single plant's parameters.
    The Pipeline handles all scaling and encoding internally —
    raw original values are passed in, exactly as in the training data.
    """
    row = pd.DataFrame([{
        'CAPFAC':   capfac,
        'NAMEPCAP': namepcap,
        'PLNGENAN': plngenan,
        'PLHTIAN':  plhtian,
        'PLFUELCT': plfuelct.upper(),
        'PSTATABB': pstatabb.upper(),
        'SUBRGN':   subrgn.upper(),
    }])

    pred = int(model_pipeline.predict(row)[0])
    prob = None
    if hasattr(model_pipeline, 'predict_proba'):
        try:
            prob = round(float(model_pipeline.predict_proba(row)[0][pred]) * 100, 1)
        except Exception:
            pass

    return {
        'class_id':    pred,
        'label':       RISK_LABEL[pred],
        'emoji':       RISK_EMOJI[pred],
        'color':       RISK_COLOR[pred],
        'probability': prob,
    }


def check_retirement(capfac: float, risk_class: int, fuel: str) -> dict:
    """
    Apply the three retirement conditions to a single plant.
    Uses CAPFAC in its original 0-1 form — same as the fixed notebook.
    """
    cond_a = capfac < 0.15
    cond_b = risk_class == 2
    cond_c = fuel.upper() in FOSSIL

    score   = int(cond_a) + int(cond_b) + int(cond_c)
    flagged = (score == 3)

    return {
        'cond_a':   cond_a,
        'cond_b':   cond_b,
        'cond_c':   cond_c,
        'score':    score,
        'flagged':  flagged,
    }


def run_scenario(model_pipeline,
                 before: dict,
                 after: dict) -> dict:
    """
    Compare two plant configurations.
    Each dict must have keys matching predict_risk() parameters.
    """
    r_before = predict_risk(model_pipeline, **before)
    r_after  = predict_risk(model_pipeline, **after)

    if r_after['class_id'] < r_before['class_id']:
        direction = 'improved'
    elif r_after['class_id'] > r_before['class_id']:
        direction = 'worsened'
    else:
        direction = 'unchanged'

    return {
        'before':    r_before,
        'after':     r_after,
        'direction': direction,
        'changed':   r_before['class_id'] != r_after['class_id'],
    }
