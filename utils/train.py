# utils/train.py

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from .pipeline import (
    NUMERIC,
    CATEGORICAL,
    FEATURES,
    EMISSION_COLS,
    SEED,
)


# ============================================================
# 1. LEAKAGE GUARD
# ============================================================

def _check_leakage():
    """
    Prevent target-derived information and raw emission values
    from being used as predictive features.
    """

    forbidden_columns = set(EMISSION_COLS) | {
        "risk_label",
        "emission_score",
        "_is_train",
        "_co2_reference",
        "_so2_reference",
        "_nox_reference",
    }

    overlapping_columns = forbidden_columns.intersection(FEATURES)

    if overlapping_columns:
        raise ValueError(
            "Data leakage detected! The following columns must "
            f"not be used as predictive features: {overlapping_columns}"
        )

    print("[train] Leakage guard passed ✅")


# ============================================================
# 2. PREPROCESSING PIPELINE
# ============================================================

def make_preprocessor():
    """
    Build preprocessing pipelines.

    Numeric features:
        1. Replace missing values with the median.
        2. Standardize the features.

    Categorical features:
        1. Replace missing values with the most frequent category.
        2. Apply one-hot encoding.

    IMPORTANT:
    The imputers and scaler are fitted only on the training
    portion of each cross-validation fold.
    """

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                NUMERIC,
            ),
            (
                "cat",
                categorical_pipeline,
                CATEGORICAL,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ============================================================
# 3. TRAIN/TEST SPLIT
# ============================================================

def split_data(df):
    """
    Use the train/test split already created by the data pipeline.

    The _is_train column must be created by create_labels()
    in utils/pipeline.py.

    Do not perform another random split here because the labels
    and thresholds were created using the existing split.
    """

    _check_leakage()

    if "_is_train" not in df.columns:
        raise ValueError(
            "The '_is_train' column is missing. "
            "Ensure create_labels() creates the training/test "
            "split before calling train_all()."
        )

    if "risk_label" not in df.columns:
        raise ValueError(
            "The 'risk_label' column is missing from the dataset."
        )

    train_df = df.loc[df["_is_train"]].copy()
    test_df = df.loc[~df["_is_train"]].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "The training or test dataset is empty. "
            "Check the _is_train column."
        )

    X_train = train_df[FEATURES].copy()
    X_test = test_df[FEATURES].copy()

    y_train = train_df["risk_label"].copy()
    y_test = test_df["risk_label"].copy()

    # Ensure that the training set contains all three classes.
    if y_train.nunique() < 3:
        raise ValueError(
            "The training dataset must contain all three "
            "risk classes: Low, Medium, and High."
        )

    # Check that the training and test sets contain no missing
    # target labels.
    if y_train.isna().any() or y_test.isna().any():
        raise ValueError(
            "Missing target labels detected. "
            "Check create_labels() in utils/pipeline.py."
        )

    print(
        f"[train] Train={len(X_train):,}  "
        f"Test={len(X_test):,}"
    )

    print("[train] Training class distribution:")
    print(y_train.value_counts().sort_index())

    return X_train, X_test, y_train, y_test


# ============================================================
# 4. MODEL DEFINITIONS AND HYPERPARAMETER GRIDS
# ============================================================

PARAM_GRIDS = {

    "Logistic Regression": {
        "classifier__estimator__C": [
            0.1,
            1.0,
            10.0,
        ],
    },

    "Decision Tree": {
        "classifier__estimator__max_depth": [
            None,
            5,
            10,
            20,
        ],
        "classifier__estimator__min_samples_split": [
            2,
            5,
            10,
        ],
    },

    "KNN": {
        "classifier__estimator__n_neighbors": [
            3,
            5,
            7,
        ],
        "classifier__estimator__weights": [
            "uniform",
            "distance",
        ],
    },

    "Random Forest": {
        "classifier__estimator__n_estimators": [
            100,
            200,
        ],
        "classifier__estimator__max_depth": [
            None,
            10,
            20,
        ],
        "classifier__estimator__min_samples_split": [
            2,
            5,
        ],
    },

}


def make_classifier(name):
    """
    Return the requested base classifier.

    OneVsRestClassifier is used to preserve the project's
    existing multiclass classification approach.
    """

    if name == "Logistic Regression":

        estimator = LogisticRegression(
            max_iter=2000,
            random_state=SEED,
        )

    elif name == "Decision Tree":

        estimator = DecisionTreeClassifier(
            random_state=SEED,
        )

    elif name == "KNN":

        estimator = KNeighborsClassifier()

    elif name == "Random Forest":

        estimator = RandomForestClassifier(
            random_state=SEED,
            n_jobs=1,
        )

    else:
        raise ValueError(f"Unknown model: {name}")

    return OneVsRestClassifier(estimator)


# ============================================================
# 5. MODEL TRAINING AND EVALUATION
# ============================================================

def fit_model(name, X_train, y_train, X_test, y_test, cv):
    """
    Train a model using GridSearchCV.

    Hyperparameter tuning is performed using training data only.
    The test data is reserved for final evaluation.
    """

    print(f"\n[train] Fitting {name}...")

    # Preprocessing and classification are combined into one
    # pipeline. This is important for preventing preprocessing
    # leakage during cross-validation.
    model_pipeline = Pipeline([
        (
            "preprocessor",
            make_preprocessor(),
        ),
        (
            "classifier",
            make_classifier(name),
        ),
    ])

    search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=PARAM_GRIDS[name],
        scoring="accuracy",
        cv=cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
        return_train_score=True,
    )

    # Imputers, scaler, encoder and classifier are fitted inside
    # each training fold during cross-validation.
    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # Evaluate the best model on the untouched test dataset.
    y_pred = best_model.predict(X_test)

    # Training accuracy is useful for checking possible
    # overfitting.
    y_train_pred = best_model.predict(X_train)

    train_accuracy = accuracy_score(
        y_train,
        y_train_pred,
    )

    test_accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # Display the evaluation results.
    print(f"[train] Model: {name}")
    print(f"[train] Best CV accuracy: {search.best_score_:.4f}")
    print(f"[train] Training accuracy: {train_accuracy:.4f}")
    print(f"[train] Test accuracy: {test_accuracy:.4f}")
    print(f"[train] Macro precision: {precision:.4f}")
    print(f"[train] Macro recall: {recall:.4f}")
    print(f"[train] Macro F1-score: {f1:.4f}")

    print("[train] Best parameters:")
    print(search.best_params_)

    print("\n[train] Classification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            target_names=["Low", "Medium", "High"],
            zero_division=0,
        )
    )

    print("[train] Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1, 2],
        )
    )

    return {
        "model": best_model,
        "best_params": search.best_params_,
        "cv_accuracy": search.best_score_,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "y_test": y_test,
        "y_pred": y_pred,
    }


# ============================================================
# 6. TRAIN ALL MODELS AND SELECT THE BEST
# ============================================================

def train_all(X_train, y_train, X_test, y_test):
    """
    Train all four models and select the model with the highest
    cross-validation accuracy.

    The test set is not used to select hyperparameters or choose
    the winning model.
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    trained = {}

    model_names = [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Random Forest",
    ]

    for name in model_names:

        trained[name] = fit_model(
            name,
            X_train,
            y_train,
            X_test,
            y_test,
            cv,
        )

    best_name = max(
        trained,
        key=lambda name: trained[name]["cv_accuracy"],
    )

    print("\n" + "=" * 55)
    print(f"[train] Best model: {best_name}")
    print(
        "[train] Best CV accuracy: "
        f"{trained[best_name]['cv_accuracy']:.4f}"
    )
    print(
        "[train] Corresponding test accuracy: "
        f"{trained[best_name]['test_accuracy']:.4f}"
    )
    print("=" * 55)

    return trained, best_name


# ============================================================
# 7. SAVE MODELS AND ARTEFACTS
# ============================================================

def save_artefacts(trained, df, path="models"):
    """
    Save the trained models and evaluation results.

    The processed dataset excludes internal columns used for
    creating the train/test split and emission reference scores.
    """

    os.makedirs(path, exist_ok=True)

    results = []

    # Save each trained model and collect evaluation metrics.
    for name, result in trained.items():

        filename = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            + ".pkl"
        )

        model_path = os.path.join(
            path,
            filename,
        )

        joblib.dump(
            result["model"],
            model_path,
        )

        results.append({
            "model": name,
            "cv_accuracy": result["cv_accuracy"],
            "train_accuracy": result["train_accuracy"],
            "test_accuracy": result["test_accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1_score": result["f1_score"],
            "best_params": str(result["best_params"]),
        })

        print(f"[save] Model saved: {model_path}")

    # Save the model comparison results.
    results_df = pd.DataFrame(results)

    results_path = os.path.join(
        path,
        "results.csv",
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print(f"[save] Results saved: {results_path}")

    # Remove internal pipeline columns from the exported dataset.
    internal_columns = [
        "_is_train",
        "_co2_reference",
        "_so2_reference",
        "_nox_reference",
    ]

    processed_df = df.drop(
        columns=internal_columns,
        errors="ignore",
    )

    processed_path = os.path.join(
        path,
        "processed_data.csv",
    )

    processed_df.to_csv(
        processed_path,
        index=False,
    )

    print(f"[save] Processed data saved: {processed_path}")