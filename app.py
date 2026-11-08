# ─────────────────────────────────────────────────────────────
#  EcoGrid — app.py
#  Phase 10: Complete 4-page Streamlit dashboard
#
#  Run:  streamlit run app.py
#  Requires: python run_pipeline.py --data egrid2023_data_rev2.xlsx
# ─────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import joblib, os, sys, re
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                              accuracy_score, classification_report)
from sklearn.inspection import permutation_importance

sys.path.insert(0, os.path.dirname(__file__))
from utils.simulate import predict_risk, check_retirement, run_scenario, FUEL_OPTIONS
from utils.pipeline import FEATURES, NUMERIC, CATEGORICAL

# ─────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='EcoGrid — Emission Risk Dashboard',
    page_icon='🌿',
    layout='wide',
)

RISK_COLOR = {0: '#22c55e', 1: '#f59e0b', 2: '#ef4444'}
RISK_BG    = {0: '#f0fdf4', 1: '#fffbeb', 2: '#fef2f2'}
RISK_LABEL = {0: '🟢 Low risk', 1: '🟡 Medium risk', 2: '🔴 High risk'}
MODEL_NAMES = ['Logistic Regression', 'Decision Tree', 'KNN', 'Random Forest']
MODEL_SLUGS = ['logistic_regression', 'decision_tree', 'knn', 'random_forest']
MODEL_COLORS= ['#3b82f6', '#8b5cf6', '#f59e0b', '#22c55e']


def get_best_model_name(results_df, loaded_models):
    """Choose the highest-CV-accuracy model, with safe fallbacks."""
    if results_df is not None and not results_df.empty and {'model', 'cv_accuracy'}.issubset(results_df.columns):
        candidate = str(results_df.loc[results_df['cv_accuracy'].idxmax(), 'model'])
        if candidate in loaded_models:
            return candidate
    if 'Random Forest' in loaded_models:
        return 'Random Forest'
    return next(iter(loaded_models), None)

# ─────────────────────────────────────────────────────────────
#  Load shared resources (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_df():
    p = 'models/processed_data.csv'
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_results():
    """Load model metrics and normalize column names for the dashboard."""
    p = 'models/results.csv'
    if not os.path.exists(p):
        return None

    data = pd.read_csv(p)
    # Support both the current snake_case CSV and older title-case exports.
    data.columns = [
        re.sub(r'_+', '_', re.sub(r'[^a-z0-9]+', '_', str(col).strip().lower())).strip('_')
        for col in data.columns
    ]
    aliases = {
        'f1': 'f1_score',
        'cv_score': 'cv_accuracy',
        'cross_validation_accuracy': 'cv_accuracy',
        'best_parameters': 'best_params',
    }
    data = data.rename(columns={old: new for old, new in aliases.items() if old in data.columns and new not in data.columns})

    # The current training script does not save these derived columns.
    if 'gap_pp' not in data.columns and {'train_accuracy', 'test_accuracy'}.issubset(data.columns):
        data['gap_pp'] = (data['train_accuracy'] - data['test_accuracy']) * 100

    return data

@st.cache_resource
def load_models():
    out = {}
    for name, slug in zip(MODEL_NAMES, MODEL_SLUGS):
        p = f'models/{slug}.pkl'
        if os.path.exists(p):
            out[name] = joblib.load(p)
    return out

df      = load_df()
results = load_results()
models  = load_models()
ready   = (df is not None) and (len(models) == 4)

# ─────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## 🌿 EcoGrid')
    st.caption('Power Plant Emission Risk Assessment')
    st.divider()

    page = st.radio('Navigation', [
        '📊 Live predictor',
        '🏆 Model performance',
        '🔍 Analytical insights',
        '🔬 Scenario simulator',
    ], label_visibility='collapsed')

    st.divider()
    if ready:
        st.markdown('**Dataset**')
        st.caption(f"Plants: {len(df):,}")
        st.caption(f"Features: {len(FEATURES)}")
        if 'retirement_risk' in df.columns:
            st.caption(f"Retirement flagged: {int(df['retirement_risk'].sum()):,}")
        if 'final_anomaly' in df.columns:
            st.caption(f"Anomalies: {int(df['final_anomaly'].sum()):,}")
    else:
        st.error('Run `python run_pipeline.py --data egrid2023_data_rev2.xlsx` first.')


# ─────────────────────────────────────────────────────────────
#  Helper: coloured risk result card
# ─────────────────────────────────────────────────────────────
def risk_card(class_id: int, prob: float = None):
    label = RISK_LABEL[class_id]
    color = RISK_COLOR[class_id]
    bg    = RISK_BG[class_id]
    prob_str = f'  ({prob}% confidence)' if prob else ''
    st.markdown(
        f'<div style="background:{bg};border:2px solid {color};border-radius:12px;'
        f'padding:14px 20px;font-size:20px;font-weight:600;color:{color};'
        f'text-align:center">{label}{prob_str}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
#  Helper: confusion matrix figure
# ─────────────────────────────────────────────────────────────
def cm_fig(name, y_true, y_pred, color):
    cm   = confusion_matrix(y_true, y_pred)
    cmap = LinearSegmentedColormap.from_list('c', ['#f8fafc', color], N=256)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('white')
    ax.imshow(cm, cmap=cmap)
    classes = ['Low', 'Medium', 'High']
    ax.set_xticks(range(3)); ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticks(range(3)); ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel('Predicted', fontsize=10, fontweight='bold')
    ax.set_ylabel('Actual',    fontsize=10, fontweight='bold')
    thresh = cm.max() / 2.0
    for i in range(3):
        for j in range(3):
            val = cm[i, j]
            pct = val / cm[i].sum() * 100
            tc  = 'white' if val > thresh else '#1e293b'
            ax.text(j, i, f'{val:,}\n({pct:.1f}%)',
                    ha='center', va='center', color=tc,
                    fontsize=9, fontweight='bold')
    acc = accuracy_score(y_true, y_pred)
    ax.set_title(f'{name}\nAccuracy: {acc*100:.2f}%',
                 fontsize=11, fontweight='bold', color=color, pad=10)
    plt.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════
#  PAGE 1 — LIVE PREDICTOR
# ═════════════════════════════════════════════════════════════
if page == '📊 Live predictor':
    st.title('Live emission risk predictor')
    st.caption('Enter plant parameters to get an instant risk classification from the best model.')

    if not ready:
        st.error('Pipeline not ready — run the training script first.')
        st.stop()

    best_name = get_best_model_name(results, models)
    best_model = models.get(best_name)
    if best_model is None:
        st.error('No trained model could be loaded. Check the files in the models/ folder.')
        st.stop()

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.subheader('Plant parameters')
        fuel    = st.selectbox('Fuel category', FUEL_OPTIONS, index=0)
        capfac  = st.slider('Capacity factor (0–1)', 0.0, 1.0, 0.45, step=0.01,
                             help='Fraction of max capacity actually generated per year')
        mw      = st.number_input('Nameplate capacity (MW)', 1.0, 10000.0, 600.0, step=10.0)
        netgen  = st.number_input('Annual net generation (MWh)', 0.0, 5e7, 2_000_000.0, step=10_000.0)
        heat    = st.number_input('Annual heat input (MMBtu)', 0.0, 5e8, 20_000_000.0, step=100_000.0)
        state   = st.text_input('Plant state (abbreviation)', 'TX').upper()
        subrgn  = st.text_input('eGRID subregion', 'SRSO').upper()

        st.divider()
        if st.button('Predict emission risk', type='primary', use_container_width=True):
            result = predict_risk(
                best_model,
                capfac=capfac, namepcap=mw,
                plngenan=netgen, plhtian=heat,
                plfuelct=fuel, pstatabb=state, subrgn=subrgn,
            )
            retire = check_retirement(capfac, result['class_id'], fuel)
            st.session_state['pred']   = result
            st.session_state['retire'] = retire
            st.session_state['capfac'] = capfac
            st.session_state['fuel']   = fuel

    with col2:
        st.subheader('Result')
        if 'pred' in st.session_state:
            r  = st.session_state['pred']
            rc = st.session_state['retire']

            risk_card(r['class_id'], r['probability'])
            st.caption(f'Model used: {best_name}')
            st.divider()

            st.markdown('**Retirement conditions (on the original 0–1 CAPFAC)**')
            cf_disp = st.session_state['capfac']
            fl_disp = st.session_state['fuel']
            st.markdown(
                ('✅' if rc['cond_a'] else '❌') +
                f' Capacity factor < 15%  `({cf_disp*100:.1f}%)`'
            )
            st.markdown(('✅' if rc['cond_b'] else '❌') + ' High risk classification')
            st.markdown(
                ('✅' if rc['cond_c'] else '❌') +
                f' Fossil fuel type  `({fl_disp})`'
            )

            score = rc['score']
            colour = '#ef4444' if score == 3 else ('#f59e0b' if score == 2 else '#22c55e')
            st.markdown(
                f'<div style="border-left:4px solid {colour};padding:8px 12px;'
                f'margin-top:10px;font-weight:500;color:{colour}">'
                f'Retirement score: {score} / 3</div>',
                unsafe_allow_html=True
            )
            if rc['flagged']:
                st.error('⚠️ This plant meets ALL three retirement conditions.')
            elif score == 2:
                st.warning('This plant meets 2 of 3 retirement conditions — worth monitoring.')
            else:
                st.success('This plant is not flagged for retirement.')
        else:
            st.info('Fill in parameters on the left and click **Predict**.')
            st.markdown('''
**Risk classes**
- 🟢 **Low** — bottom 33% of emission scores
- 🟡 **Medium** — middle 33%
- 🔴 **High** — top 33%

**Retirement flag requires all 3:**
- Capacity factor < 15%
- Classified as High risk
- Burning fossil fuel (coal / oil / gas)
''')


# ═════════════════════════════════════════════════════════════
#  PAGE 2 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════
elif page == '🏆 Model performance':
    st.title('Model performance')
    st.caption('All 4 models trained with Pipeline + GridSearchCV. '
               'Scaler and encoder fit on training data only — no preprocessing leakage.')

    if not ready:
        st.error('Pipeline not ready.')
        st.stop()

    # ── Summary metrics ───────────────────────────────────────
    if results is not None and not results.empty:
        best_row = results.loc[results['cv_accuracy'].idxmax()]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Best model', str(best_row.get('model', 'N/A')))
        m2.metric('Test accuracy', f"{best_row.get('test_accuracy', float('nan'))*100:.2f}%")
        m3.metric('CV accuracy', f"{best_row.get('cv_accuracy', float('nan'))*100:.2f}%")
        m4.metric('Macro F1-score', f"{best_row.get('f1_score', float('nan'))*100:.2f}%")
    else:
        st.warning('Model results are unavailable. Check models/results.csv.')

    st.divider()

    # ── Comparison table ──────────────────────────────────────
    st.subheader('Algorithm comparison')
    if results is not None and not results.empty:
        display_columns = [
            col for col in ['model', 'train_accuracy', 'cv_accuracy', 'test_accuracy',
                            'precision', 'recall', 'f1_score', 'gap_pp', 'best_params']
            if col in results.columns
        ]
        disp = results[display_columns].copy()
        metric_columns = ['train_accuracy', 'cv_accuracy', 'test_accuracy',
                          'precision', 'recall', 'f1_score']
        for col in metric_columns:
            if col in disp.columns:
                disp[col] = disp[col].map(lambda value: f'{value:.2%}' if pd.notna(value) else 'N/A')
        if 'gap_pp' in disp.columns:
            disp['gap_pp'] = disp['gap_pp'].map(lambda value: f'{value:.2f} pp' if pd.notna(value) else 'N/A')
        disp = disp.rename(columns={
            'model': 'Model', 'train_accuracy': 'Train accuracy',
            'cv_accuracy': 'CV accuracy', 'test_accuracy': 'Test accuracy',
            'precision': 'Macro precision', 'recall': 'Macro recall',
            'f1_score': 'Macro F1-score', 'gap_pp': 'Train-test gap',
            'best_params': 'Best parameters',
        })
        st.dataframe(disp, use_container_width=True, hide_index=True)

    st.caption('CV accuracy = 5-fold cross-validation on training data. '
               'Train-test gap = training accuracy − test accuracy. '
               'A large gap can indicate overfitting. Precision, recall and F1 are macro-averaged.')

    st.divider()

    # ── Confusion matrices — 4 side by side ──────────────────
    st.subheader('Confusion matrices')

    # Need test predictions — regenerate from saved models
    @st.cache_data
    def get_test_preds():
        """Recreate the deterministic test split with the internal split flag.

        models/processed_data.csv intentionally omits _is_train, so do not use
        that export for split_data(). Re-run the data pipeline on the original
        workbook; the fixed random seed recreates the split used for training.
        """
        import warnings
        warnings.filterwarnings("ignore")

        from utils.pipeline import run_pipeline
        from utils.train import split_data

        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "egrid2023_data_rev2.xlsx")
        if not os.path.exists(data_path):
            st.error(
                "Source dataset not found: egrid2023_data_rev2.xlsx. "
                "Place it in the same folder as app.py."
            )
            return None, None, None

        # run_pipeline() calls create_labels(), which creates _is_train.
        pipeline_df = run_pipeline(data_path)
        _, X_test, _, y_test = split_data(pipeline_df)

        preds = {}
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        for name, slug in zip(MODEL_NAMES, MODEL_SLUGS):
            model_path = os.path.join(model_dir, f"{slug}.pkl")
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    preds[name] = model.predict(X_test)
                except Exception as exc:
                    st.warning(f"Could not generate predictions for {name}: {exc}")

        return preds, X_test, y_test

    preds, X_test, y_test = get_test_preds()

    if preds and y_test is not None:
        cols = st.columns(2)
        for idx, (name, color) in enumerate(zip(MODEL_NAMES, MODEL_COLORS)):
            with cols[idx % 2]:
                if name in preds:
                    fig = cm_fig(name, y_test, preds[name], color)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

        st.divider()

        # ── Model comparison bar chart ─────────────────────────
        st.subheader('Test accuracy vs Macro F1-score vs Macro recall')
        if results is not None and not results.empty:
            accs = results['test_accuracy'].tolist()
            f1s = results['f1_score'].tolist()
            recalls = results['recall'].tolist()
            x = np.arange(len(results))
            w = 0.25
            chart_colors = (MODEL_COLORS * ((len(results) // len(MODEL_COLORS)) + 1))[:len(results)]

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('white')
            b1 = ax.bar(x - w, accs, w, label='Test accuracy', color=chart_colors, alpha=0.95)
            b2 = ax.bar(x, f1s, w, label='Macro F1-score', color=chart_colors, alpha=0.60)
            b3 = ax.bar(x + w, recalls, w, label='Macro recall', color=chart_colors, alpha=0.30,
                        edgecolor=chart_colors, linewidth=1.5)
            for bars in [b1, b2, b3]:
                for bar in bars:
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                            f'{h*100:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([str(name).replace(' ', '\n', 1) for name in results['model']], fontsize=10)
            ax.set_ylim(0, 1.15)
            ax.set_ylabel('Score')
            ax.set_title('Model comparison', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='lower right')
            ax.spines[['top','right']].set_visible(False)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v*100:.0f}%'))
            ax.axhline(0.85, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.text(len(results) - 0.45, 0.855, 'Target\n(85%)', color='gray', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.divider()

        # ── Feature importance ─────────────────────────────────
        st.subheader('Feature importance — Random Forest')
        feat_labels = {
            'CAPFAC':   'Capacity factor',
            'NAMEPCAP': 'Nameplate capacity (MW)',
            'PLNGENAN': 'Annual net generation (MWh)',
            'PLHTIAN':  'Annual heat input (MMBtu)',
            'PLFUELCT': 'Fuel category',
            'PSTATABB': 'Plant state',
            'SUBRGN':   'eGRID subregion',
        }
        rf_pipe = models.get('Random Forest')

        if rf_pipe is None:
            st.warning('Random Forest model was not found in models/.')
        else:
            col_a, col_b = st.columns(2)

            # Built-in tree importance is optional. In this project, the final
            # classifier may be wrapped in OneVsRestClassifier, so read each
            # fitted underlying estimator instead of assuming the wrapper has
            # feature_importances_.
            with col_a:
                st.caption('Built-in importance (fitted Random Forest)')
                try:
                    named_steps = getattr(rf_pipe, 'named_steps', {})
                    preprocessor = named_steps.get('preprocessor') or named_steps.get('prep')
                    if preprocessor is None:
                        preprocessor = next(
                            (step for step in named_steps.values()
                             if hasattr(step, 'get_feature_names_out')),
                            None,
                        )

                    classifier = named_steps.get('classifier') or named_steps.get('clf')
                    if classifier is None and named_steps:
                        classifier = list(named_steps.values())[-1]

                    transformed_names = (
                        list(preprocessor.get_feature_names_out())
                        if preprocessor is not None and hasattr(preprocessor, 'get_feature_names_out')
                        else []
                    )

                    if classifier is None:
                        raise ValueError('Could not locate the fitted classifier step.')

                    if hasattr(classifier, 'feature_importances_'):
                        raw_importances = np.asarray(classifier.feature_importances_, dtype=float)
                    elif hasattr(classifier, 'estimators_'):
                        fitted_estimators = [
                            estimator for estimator in classifier.estimators_
                            if hasattr(estimator, 'feature_importances_')
                        ]
                        if not fitted_estimators:
                            raise ValueError('The wrapped estimators do not expose feature_importances_.')
                        raw_importances = np.mean(
                            [np.asarray(estimator.feature_importances_, dtype=float)
                             for estimator in fitted_estimators],
                            axis=0,
                        )
                    else:
                        raise ValueError('This classifier does not expose built-in tree importances.')

                    if not transformed_names or len(transformed_names) != len(raw_importances):
                        raise ValueError(
                            'Transformed feature names do not match the model importance vector.'
                        )

                    aggregated = {feature: 0.0 for feature in FEATURES}
                    for transformed_name, value in zip(transformed_names, raw_importances):
                        # ColumnTransformer output commonly looks like
                        # "num__CAPFAC" or "cat__PLFUELCT_COAL".
                        stripped = str(transformed_name).split('__', 1)[-1]
                        matched_feature = next(
                            (feature for feature in FEATURES
                             if stripped == feature or stripped.startswith(feature + '_')),
                            None,
                        )
                        if matched_feature is not None:
                            aggregated[matched_feature] += float(value)

                    built_in = pd.Series(aggregated, dtype=float).sort_values()
                    display_values = built_in[built_in > 0]
                    if display_values.empty:
                        raise ValueError('No raw input features could be matched to transformed features.')

                    display_labels = [feat_labels.get(feature, feature) for feature in display_values.index]
                    colors = [
                        '#22c55e' if value >= display_values.median() else '#86efac'
                        for value in display_values.values
                    ]
                    fig, ax = plt.subplots(figsize=(6, 4))
                    fig.patch.set_facecolor('white')
                    ax.barh(display_labels, display_values.values, color=colors,
                            edgecolor='white', height=0.55)
                    ax.set_title('Built-in feature importance', fontsize=11, fontweight='bold')
                    ax.set_xlabel('Mean importance across fitted estimators')
                    ax.spines[['top', 'right']].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                except Exception as exc:
                    st.info(
                        'Built-in feature importance is unavailable for this saved model. '
                        f'Permutation importance below is used as the reliable cross-check. Details: {exc}'
                    )

            # Permutation importance works with the complete fitted pipeline
            # and the original, untransformed test columns. It does not require
            # get_feature_names_out() or feature_importances_ on the classifier.
            with col_b:
                st.caption('Permutation importance (held-out test set)')
                if X_test is None or y_test is None:
                    st.warning('Test data is unavailable, so permutation importance cannot be calculated.')
                else:
                    try:
                        perm = permutation_importance(
                            rf_pipe,
                            X_test,
                            y_test,
                            n_repeats=5,
                            random_state=42,
                            scoring='accuracy',
                            n_jobs=1,
                        )
                        p_imp = pd.Series(
                            perm.importances_mean,
                            index=list(X_test.columns),
                            dtype=float,
                        ).sort_values()
                        p_labels = [feat_labels.get(feature, feature) for feature in p_imp.index]

                        fig, ax = plt.subplots(figsize=(6, 4))
                        fig.patch.set_facecolor('white')
                        bar_colors = ['#ef4444' if value < 0 else '#3b82f6'
                                      for value in p_imp.values]
                        ax.barh(p_labels, p_imp.values, color=bar_colors,
                                edgecolor='white', height=0.55)
                        ax.axvline(0, color='black', linewidth=0.8)
                        ax.set_xlabel('Decrease in test accuracy after shuffling')
                        ax.set_title('Permutation importance', fontsize=11, fontweight='bold')
                        ax.spines[['top', 'right']].set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                        st.caption(
                            'Positive values indicate that shuffling a feature reduced accuracy. '
                            'Values near zero indicate little measured effect; negative values can occur '
                            'because of sampling variability. Permutation importance is model- and test-set-dependent.'
                        )
                    except Exception as exc:
                        st.error(f'Could not calculate permutation importance: {exc}')


# ═════════════════════════════════════════════════════════════
#  PAGE 3 — ANALYTICAL INSIGHTS
# ═════════════════════════════════════════════════════════════
elif page == '🔍 Analytical insights':
    st.title('Analytical insights')
    st.caption('Emission distribution, retirement-prone plants, and confirmed anomalies.')

    if not ready:
        st.error('Pipeline not ready.')
        st.stop()

    # ── Summary metrics ───────────────────────────────────────
    total   = len(df)
    high    = int((df['risk_label'] == 2).sum())
    retire  = int(df['retirement_risk'].sum()) if 'retirement_risk' in df.columns else 0
    anomaly = int(df['final_anomaly'].sum())   if 'final_anomaly'   in df.columns else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Total plants',       f'{total:,}')
    m2.metric('High risk',          f'{high:,}')
    m3.metric('Retirement flagged', f'{retire:,}')
    m4.metric('Anomalies',          f'{anomaly:,}')

    st.divider()

    tab1, tab2, tab3 = st.tabs([
        '📈 Emission distribution',
        '🔴 Retirement-prone plants',
        '🟡 Anomalous plants',
    ])

    # ── TAB 1: Emission distribution ──────────────────────────
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            if 'PLFUELCT' in df.columns and 'emission_score' in df.columns:
                fig = px.box(
                    df, x='PLFUELCT', y='emission_score',
                    color='PLFUELCT',
                    title='Emission score by fuel type',
                    labels={'PLFUELCT':'Fuel type','emission_score':'Emission score'},
                )
                fig.update_layout(showlegend=False, plot_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            risk_counts = df['risk_label'].value_counts().sort_index()
            fig = px.pie(
                values=risk_counts.values,
                names=['Low', 'Medium', 'High'],
                color_discrete_map={'Low':'#22c55e','Medium':'#f59e0b','High':'#ef4444'},
                title='Risk class distribution',
            )
            st.plotly_chart(fig, use_container_width=True)

        if 'CAPFAC' in df.columns:
            fig = px.scatter(
                df.sample(min(2000, len(df)), random_state=42),
                x='CAPFAC', y='emission_score',
                color=df.loc[
                    df.sample(min(2000, len(df)), random_state=42).index,
                    'risk_label'
                ].map({0:'Low', 1:'Medium', 2:'High'}),
                color_discrete_map={'Low':'#22c55e','Medium':'#f59e0b','High':'#ef4444'},
                title='Capacity factor vs emission score',
                labels={'CAPFAC':'Capacity factor (original 0–1)',
                        'emission_score':'Emission score','color':'Risk'},
                opacity=0.5,
            )
            fig.update_layout(plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 2: Retirement-prone plants ─────────────────────────
    with tab2:
        if 'retirement_risk' not in df.columns:
            st.info('Retirement flags not found — re-run pipeline.')
        else:
            st.markdown(
                'Plants flagged using **original CAPFAC** (0–1 range) — '
                'the fixed notebook bug where scaled CAPFAC was compared to 0.15 is corrected here.'
            )

            score_filter = st.selectbox(
                'Minimum retirement score to show',
                [3, 2, 1],
                format_func=lambda x: f'Score {x}/3'
            )
            retire_df = df[df['retirement_score'] >= score_filter].copy()
            retire_df = retire_df.sort_values('emission_score', ascending=False)

            show_cols = [c for c in ['PNAME','PSTATABB','PLFUELCT','CAPFAC',
                                     'emission_score','retirement_score','risk_label']
                         if c in retire_df.columns]
            st.metric('Plants shown', f'{len(retire_df):,}')
            st.dataframe(
                retire_df[show_cols].rename(columns={
                    'PNAME':'Plant name','PSTATABB':'State','PLFUELCT':'Fuel',
                    'CAPFAC':'Capacity factor','emission_score':'Emission score',
                    'retirement_score':'Retirement score','risk_label':'Risk class',
                }),
                use_container_width=True, height=380,
            )

            score_counts  = df['retirement_score'].value_counts().sort_index()
            fig = px.bar(
                x=score_counts.index.astype(str),
                y=score_counts.values,
                labels={'x':'Retirement score (0–3)','y':'Number of plants'},
                color=score_counts.index.astype(str),
                color_discrete_map={'0':'#86efac','1':'#fde68a','2':'#fdba74','3':'#ef4444'},
                title='Retirement score distribution',
            )
            fig.update_layout(showlegend=False, plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 3: Anomalous plants ────────────────────────────────
    with tab3:
        if 'final_anomaly' not in df.columns:
            st.info('Anomaly flags not found — re-run pipeline.')
        else:
            st.markdown(
                'Flagged by **2 or more** of 4 detection methods: '
                'global Z-score, group Z-score (within fuel type), IQR, and Isolation Forest.'
            )
            votes_filter = st.selectbox(
                'Minimum votes to show',
                [4, 3, 2],
                index=1,
                format_func=lambda x: f'{x} / 4 methods agree'
            )
            anom_df = df[df['anomaly_votes'] >= votes_filter].copy()
            anom_df = anom_df.sort_values('anomaly_votes', ascending=False)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Total anomalies',  f"{int(df['final_anomaly'].sum()):,}")
            c2.metric('4/4 votes',        f"{int((df['anomaly_votes']==4).sum()):,}")
            c3.metric('3/4 votes',        f"{int((df['anomaly_votes']==3).sum()):,}")
            c4.metric('2/4 votes',        f"{int((df['anomaly_votes']==2).sum()):,}")

            show_cols = [c for c in ['PNAME','PSTATABB','PLFUELCT',
                                     'emission_score','anomaly_votes',
                                     'anom_global','anom_group','anom_iqr','anom_iso']
                         if c in anom_df.columns]
            st.dataframe(
                anom_df[show_cols].rename(columns={
                    'PNAME':'Plant name','PSTATABB':'State','PLFUELCT':'Fuel',
                    'emission_score':'Emission score','anomaly_votes':'Total votes',
                    'anom_global':'Global Z','anom_group':'Group Z',
                    'anom_iqr':'IQR','anom_iso':'Isolation Forest',
                }),
                use_container_width=True, height=380,
            )

            vote_counts = df['anomaly_votes'].value_counts().sort_index()
            fig = px.bar(
                x=vote_counts.index.astype(str),
                y=vote_counts.values,
                labels={'x':'Anomaly votes (0–4)','y':'Number of plants'},
                color=vote_counts.index.astype(str),
                color_discrete_map={'0':'#d1fae5','1':'#fef3c7',
                                    '2':'#fed7aa','3':'#fca5a5','4':'#ef4444'},
                title='Anomaly vote distribution',
            )
            fig.update_layout(showlegend=False, plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════
#  PAGE 4 — SCENARIO SIMULATOR
# ═════════════════════════════════════════════════════════════
elif page == '🔬 Scenario simulator':
    st.title('Scenario simulator')
    st.caption(
        'Change plant parameters and see how emission risk shifts. '
        'The Pipeline handles all scaling and encoding internally — '
        'raw original values are entered here.'
    )

    if not ready:
        st.error('Pipeline not ready.')
        st.stop()

    best_name = get_best_model_name(results, models)
    best_model = models.get(best_name)
    if best_model is None:
        st.error('No trained model could be loaded. Check the files in the models/ folder.')
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Before — current plant')
        fuel_b   = st.selectbox('Fuel type', FUEL_OPTIONS, index=0, key='fb')
        capfac_b = st.slider('Capacity factor', 0.0, 1.0, 0.12, 0.01, key='cfb')
        mw_b     = st.number_input('Nameplate MW', 1.0, 10000.0, 800.0, step=10.0, key='mwb')
        ng_b     = st.number_input('Net gen (MWh)', 0.0, 5e7, 1_500_000.0, step=10_000.0, key='ngb')
        ht_b     = st.number_input('Heat input (MMBtu)', 0.0, 5e8, 18_000_000.0, step=100_000.0, key='htb')

    with col2:
        st.subheader('After — proposed change')
        fuel_a   = st.selectbox('Fuel type', FUEL_OPTIONS, index=1, key='fa')
        capfac_a = st.slider('Capacity factor', 0.0, 1.0, 0.55, 0.01, key='cfa')
        mw_a     = st.number_input('Nameplate MW', 1.0, 10000.0, 800.0, step=10.0, key='mwa')
        ng_a     = st.number_input('Net gen (MWh)', 0.0, 5e7, 2_800_000.0, step=10_000.0, key='nga')
        ht_a     = st.number_input('Heat input (MMBtu)', 0.0, 5e8, 14_000_000.0, step=100_000.0, key='hta')

    st.divider()
    if st.button('Run simulation', type='primary', use_container_width=True):
        before_p = dict(capfac=capfac_b, namepcap=mw_b,
                        plngenan=ng_b,   plhtian=ht_b, plfuelct=fuel_b)
        after_p  = dict(capfac=capfac_a, namepcap=mw_a,
                        plngenan=ng_a,   plhtian=ht_a, plfuelct=fuel_a)
        sim      = run_scenario(best_model, before_p, after_p)
        rb, ra   = sim['before'], sim['after']

        # ── Risk shift display ────────────────────────────────
        rc1, rc2, rc3 = st.columns([2, 1, 2])
        with rc1:
            st.markdown('**Before**')
            risk_card(rb['class_id'], rb['probability'])
        with rc2:
            icon = {'improved':'✅','worsened':'⚠️','unchanged':'ℹ️'}[sim['direction']]
            st.markdown(
                f'<div style="text-align:center;padding-top:18px;font-size:26px">{icon}'
                f'<br><span style="font-size:12px;color:gray">{sim["direction"]}</span></div>',
                unsafe_allow_html=True
            )
        with rc3:
            st.markdown('**After**')
            risk_card(ra['class_id'], ra['probability'])

        st.divider()

        # ── Retirement check before and after ─────────────────
        st.subheader('Retirement conditions — before vs after')
        t1, t2 = st.columns(2)
        for col, cf, risk, fuel, label in [
            (t1, capfac_b, rb['class_id'], fuel_b, 'Before'),
            (t2, capfac_a, ra['class_id'], fuel_a, 'After'),
        ]:
            rc = check_retirement(cf, risk, fuel)
            with col:
                st.markdown(f'**{label}**')
                st.markdown(f"{'✅' if rc['cond_a'] else '❌'} CAPFAC < 15%  `({cf*100:.1f}%)`")
                st.markdown(f"{'✅' if rc['cond_b'] else '❌'} High risk class")
                st.markdown(f"{'✅' if rc['cond_c'] else '❌'} Fossil fuel  `({fuel})`")
                st.markdown(f"**Retirement score: {rc['score']} / 3**")

        st.divider()

        # ── Interpretation ────────────────────────────────────
        if sim['direction'] == 'improved':
            st.success(
                f"✅ Switching from **{fuel_b}** → **{fuel_a}** and changing "
                f"capacity factor from **{capfac_b*100:.0f}%** → **{capfac_a*100:.0f}%** "
                f"reduces risk from **{rb['label']}** → **{ra['label']}**."
            )
        elif sim['direction'] == 'worsened':
            st.error(
                f"⚠️ These changes worsen risk from **{rb['label']}** → **{ra['label']}**. "
                "Consider a different fuel switch or higher capacity factor."
            )
        else:
            st.info('These parameter changes do not move the plant to a different risk class.')

    else:
        st.info('Adjust parameters above and click **Run simulation**.')
        st.subheader('Scenario ideas to try')
        for icon, idea in [
            ('⚡', 'Coal → Natural gas — does the risk class drop?'),
            ('🌱', 'Coal → Wind/Solar — is it now Low risk?'),
            ('📈', 'Keep fuel, raise capacity factor from 12% to 65%'),
            ('🔴', 'Set capfac = 0.08, fuel = COAL — check retirement score'),
        ]:
            st.markdown(f'{icon} {idea}')
