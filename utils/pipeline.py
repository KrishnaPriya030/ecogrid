
# ─────────────────────────────────────────────────────────────
# EcoGrid — utils/pipeline.py
# Phases 02-06: Load, clean, emissions score, retirement, anomaly
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

SEED = 42

EGRID_COLS = [
    'PNAME', 'PSTATABB', 'ORISPL', 'PLPRMFL', 'PLFUELCT',
    'CAPFAC', 'NAMEPCAP', 'PLNGENAN', 'PLHTIAN',
    'BACODE', 'SUBRGN', 'PLCO2AN', 'PLSO2AN', 'PLNOXAN'
]

EMISSION_COLS = ['PLCO2AN', 'PLSO2AN', 'PLNOXAN']
NUMERIC = ['CAPFAC', 'NAMEPCAP', 'PLNGENAN', 'PLHTIAN']
CATEGORICAL = ['PLFUELCT', 'PSTATABB', 'SUBRGN']
FEATURES = NUMERIC + CATEGORICAL
FOSSIL = ['COAL', 'OIL', 'GAS']


# ── Phase 02: Load ────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    df_raw = pd.read_excel(
        filepath,
        sheet_name='PLNT23',
        skiprows=1
    )

    available = [c for c in EGRID_COLS if c in df_raw.columns]
    df = df_raw[available].copy()

    required = EMISSION_COLS + NUMERIC + CATEGORICAL
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Required columns missing from eGRID worksheet: {missing}"
        )

    print(
        f"[load] {len(df):,} rows × "
        f"{len(df.columns)} columns"
    )

    return df


# ── Phase 03: Clean ───────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean invalid records without imputing feature values.

    Missing feature values are deliberately preserved here.
    SimpleImputer in train.py will learn imputation values
    from training data only.
    """
    df = df.copy()

    # Drop rows without emissions needed to construct the target.
    n0 = len(df)
    df = df.dropna(subset=EMISSION_COLS).copy()

    print(
        f"[clean] Dropped "
        f"{n0 - len(df):,} rows with missing emissions"
    )

    # Remove physically invalid values.
    # Preserve missing CAPFAC and NAMEPCAP values for imputation.
    valid_capfac = (
        df['CAPFAC'].isna()
        | df['CAPFAC'].between(0, 1)
    )

    valid_capacity = (
        df['NAMEPCAP'].isna()
        | (df['NAMEPCAP'] > 0)
    )

    valid_emissions = (df[EMISSION_COLS] >= 0).all(axis=1)

    valid_rows = (
        valid_capfac
        & valid_capacity
        & valid_emissions
    )

    before = len(df)
    df = df.loc[valid_rows].reset_index(drop=True)

    print(
        f"[clean] Removed {before - len(df):,} invalid rows "
        f"→ {len(df):,} remaining"
    )

    # Validate only observed values; missing values are allowed.
    assert df['CAPFAC'].dropna().between(0, 1).all(), (
        "CAPFAC must contain original values between 0 and 1"
    )

    assert (df[EMISSION_COLS] >= 0).all().all(), (
        "Emission values must be non-negative"
    )

    return df


# ── Phase 04: Split, emissions score + risk labels ────────────
def create_labels(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, float, float]:
    """
    1. Split the cleaned data into training and test sets.
    2. Learn each gas's percentile reference distribution
       from training data only.
    3. Convert each gas to a percentile score from 0 to 1.
    4. Average the three percentile scores.
    5. Learn Low/Medium/High thresholds from training scores.
    6. Apply the same reference distributions and thresholds
       to both training and test records.

    Internal columns:
      _is_train
      _co2_reference
      _so2_reference
      _nox_reference

    The reference columns are retained so the exact
    training-derived transformation is reproducible.
    """

    df = df.copy()

    # The emission columns must be numeric.
    for col in EMISSION_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Do not silently discard records at this stage.
    # clean_data() should already have removed missing emissions.
    if df[EMISSION_COLS].isna().any().any():
        raise ValueError(
            "Emission columns contain missing or non-numeric values. "
            "Run clean_data() and inspect the source data."
        )

    if (df[EMISSION_COLS] < 0).any().any():
        raise ValueError(
            "Emission columns contain negative values."
        )

    # Split BEFORE learning percentile distributions or thresholds.
    train_indices, test_indices = train_test_split(
        df.index.to_numpy(),
        test_size=0.2,
        random_state=SEED
    )

    df['_is_train'] = df.index.isin(train_indices)

    # Fit separate empirical reference distributions on training data.
    training_emissions = df.loc[train_indices, EMISSION_COLS]

    reference_values = {
        col: np.sort(training_emissions[col].to_numpy(dtype=float))
        for col in EMISSION_COLS
    }

    def percentile_against_training(
        values: pd.Series,
        reference: np.ndarray
    ) -> np.ndarray:
        """
        Calculate empirical percentile scores using training data.

        Average the proportions strictly below and at-or-below
        each value. This handles ties without using test-set ranks.
        """
        values_array = values.to_numpy(dtype=float)

        left = np.searchsorted(
            reference,
            values_array,
            side='left'
        )

        right = np.searchsorted(
            reference,
            values_array,
            side='right'
        )

        return (
            (left + right)
            / (2.0 * len(reference))
        )

    # Convert each gas to its own percentile score.
    df['_co2_reference'] = percentile_against_training(
        df['PLCO2AN'],
        reference_values['PLCO2AN']
    )

    df['_so2_reference'] = percentile_against_training(
        df['PLSO2AN'],
        reference_values['PLSO2AN']
    )

    df['_nox_reference'] = percentile_against_training(
        df['PLNOXAN'],
        reference_values['PLNOXAN']
    )

    # Combine the three equally weighted percentile scores.
    df['emission_score'] = (
        df['_co2_reference']
        + df['_so2_reference']
        + df['_nox_reference']
    ) / 3.0

    # Learn risk thresholds from training records only.
    training_scores = df.loc[
        df['_is_train'],
        'emission_score'
    ]

    low_t = float(training_scores.quantile(0.33))
    high_t = float(training_scores.quantile(0.66))

    if low_t >= high_t:
        raise ValueError(
            "The Low and High threshold boundaries are not distinct. "
            "Inspect the emissions score distribution and ties."
        )

    # Apply the same thresholds to training and test records.
    # Labels are integers: 0=Low, 1=Medium, 2=High.
    df['risk_label'] = np.select(
        [
            df['emission_score'] <= low_t,
            df['emission_score'] <= high_t
        ],
        [
            0,
            1
        ],
        default=2
    ).astype(int)

    counts = df['risk_label'].value_counts().sort_index()

    print(
        "[label] Percentile-based emissions score calculated "
        "using three gases"
    )
    print(f"[label] Training Low threshold: {low_t:.6f}")
    print(f"[label] Training High threshold: {high_t:.6f}")
    print(
        f"[label] Low={counts.get(0, 0):,} "
        f"Medium={counts.get(1, 0):,} "
        f"High={counts.get(2, 0):,}"
    )

    print(
        f"[label] Training rows={len(train_indices):,}; "
        f"Test rows={len(test_indices):,}"
    )

    return df, low_t, high_t


# ── Phase 05: Retirement detection (rule-based) ──────────────
def detect_retirement(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    assert (
        df['CAPFAC'].dropna().between(0, 1).all()
    ), "CAPFAC must contain original values between 0 and 1"

    # Missing CAPFAC values cannot satisfy the low-capacity rule.
    cond_a = df['CAPFAC'] < 0.15
    cond_b = df['risk_label'] == 2

    cond_c = (
        df['PLFUELCT']
        .astype('string')
        .str.upper()
        .isin(FOSSIL)
        .fillna(False)
    )

    df['retirement_risk'] = (
        cond_a & cond_b & cond_c
    ).astype(int)

    df['retirement_score'] = (
        cond_a.astype(int)
        + cond_b.astype(int)
        + cond_c.astype(int)
    )

    print(
        f"[retire] Flagged: {df['retirement_risk'].sum():,} "
        f"| Score-3: "
        f"{(df['retirement_score'] == 3).sum():,}"
    )

    return df


# ── Phase 06: Anomaly detection (4 methods) ───────────────────
def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Method 1: Global Z-score.
    global_z = stats.zscore(
        df['emission_score'].to_numpy(dtype=float),
        nan_policy='omit'
    )

    df['anom_global'] = (
        np.abs(np.nan_to_num(global_z, nan=0.0)) > 2.5
    ).astype(int)

    # Method 2: Group Z-score within fuel category.
    group_z = (
        df.groupby('PLFUELCT')['emission_score']
        .transform(
            lambda x: stats.zscore(
                x.astype(float),
                nan_policy='omit'
            )
        )
    )

    df['anom_group'] = (
        group_z.abs() > 2.5
    ).fillna(False).astype(int)

    # Method 3: IQR.
    q1, q3 = df['emission_score'].quantile([0.25, 0.75])
    iqr = q3 - q1

    df['anom_iqr'] = (
        (df['emission_score'] < q1 - 1.5 * iqr)
        | (df['emission_score'] > q3 + 1.5 * iqr)
    ).astype(int)

    # Method 4: Isolation Forest.
    # This is an exploratory, full-dataset anomaly analysis.
    # It is not part of the predictive model's training pipeline.
    iso_cols = [
        'CAPFAC',
        'NAMEPCAP',
        'PLCO2AN',
        'PLSO2AN',
        'PLNOXAN',
        'PLHTIAN'
    ]

    X_iso = df[iso_cols].copy()

    # Median imputation here is only for this separate
    # exploratory anomaly detector, not the predictive model.
    for col in iso_cols:
        X_iso[col] = X_iso[col].fillna(X_iso[col].median())

    X_iso = X_iso.fillna(0)

    iso = IsolationForest(
        contamination=0.05,
        random_state=SEED
    )

    df['anom_iso'] = (
        iso.fit_predict(X_iso) == -1
    ).astype(int)

    # Voting: confirmed anomaly if at least two methods agree.
    df['anomaly_votes'] = (
        df['anom_global']
        + df['anom_group']
        + df['anom_iqr']
        + df['anom_iso']
    )

    df['final_anomaly'] = (
        df['anomaly_votes'] >= 2
    ).astype(int)

    print(
        f"[anomaly] Confirmed (2+ votes): "
        f"{df['final_anomaly'].sum():,}"
    )

    return df


# ── Master pipeline ───────────────────────────────────────────
def run_pipeline(filepath: str) -> pd.DataFrame:
    print("\n=== EcoGrid Data Pipeline ===")

    df = load_data(filepath)
    df = clean_data(df)

    # Split and derive targets using training-only references.
    df, low_t, high_t = create_labels(df)

    # These analyses use the processed dataset and are intended
    # for descriptive dashboard insights, not model features.
    df = detect_retirement(df)
    df = detect_anomalies(df)

    print(
        f"[pipeline] Training-derived thresholds: "
        f"Low={low_t:.6f}, High={high_t:.6f}"
    )

    print("=== Pipeline complete ===\n")

    return df
