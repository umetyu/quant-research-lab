from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def make_logistic_model(random_state: int = 42) -> Pipeline:
    """Simple linear baseline for direction classification."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    multi_class="auto",
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_random_forest_model(random_state: int = 42) -> RandomForestClassifier:
    """Nonlinear baseline with conservative complexity."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=25,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def walk_forward_predict(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    splitter,
) -> pd.Series:
    """Fit model on each walk-forward window and collect out-of-sample predictions."""
    preds = pd.Series(index=X.index, dtype=float, name="prediction")

    for train_idx, test_idx in splitter.split(len(X)):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds.iloc[test_idx] = model.predict(X.iloc[test_idx])

    return preds.dropna().astype(int)
