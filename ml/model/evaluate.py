from typing import cast
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, log_loss

from ml.model.train import REVERSE_MAPPING

logger = logging.getLogger(__name__)


def evaluate_predictions(y_true, y_pred, title: str = "Baseline", probabilities=None):
    labels = [-1, 0, 1]
    names = ["DOWN", "FLAT", "UP"]
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    report = cast(dict, classification_report(
        y_true, y_pred, labels=labels, target_names=names, zero_division=0, output_dict=True
    ))
    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "labels": labels,
        "per_class": {name: report[name] for name in names},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "log_loss": float(log_loss(y_true, probabilities, labels=labels)) if probabilities is not None else None,
    }
    print(f"\n================ {title} Evaluation ================")
    print(f"Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Macro F1 : {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")
    print(classification_report(y_true, y_pred, labels=labels, target_names=names, zero_division=0))
    return metrics


def evaluate_xgboost(model, X_test, y_test):
    encoded_predictions = model.predict(X_test)
    print("\nXGBoost Prediction Distribution:")
    unique, counts = np.unique(encoded_predictions, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Class {label}: {count}")

    y_pred = [REVERSE_MAPPING[int(label)] for label in encoded_predictions]
    y_true = [REVERSE_MAPPING[int(label)] for label in y_test]
    probabilities = model.predict_proba(X_test)
    column_order = [list(model.classes_).index(label) for label in (0, 1, 2)]
    return evaluate_predictions(
        y_true, y_pred, title="XGBoost Model", probabilities=probabilities[:, column_order]
    )


def run_baselines(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    history_df: pd.DataFrame,
    lookback_days: int = 3,
):
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    y_true = eval_df["target"].reset_index(drop=True)
    keys = ["ticker", "date"]
    prices = pd.concat(
        [history_df[keys + ["close"]], eval_df[keys + ["close"]]], ignore_index=True
    ).drop_duplicates(keys, keep="last").sort_values(keys)
    previous_close = prices.groupby("ticker")["close"].shift(lookback_days)
    prices["past_return"] = prices["close"] / previous_close - 1
    past_return = prices.set_index(keys)["past_return"].reindex(pd.MultiIndex.from_frame(eval_df[keys]))
    if not np.isfinite(past_return.to_numpy()).all():
        raise ValueError("Insufficient contiguous price history for momentum baseline")

    y_pred_momentum = np.select(
        [past_return > 0.0075, past_return < -0.0075], [1, -1], default=0
    )
    majority_class = train_df["target"].mode()[0]
    y_pred_majority = np.full(len(y_true), majority_class)
    return {
        "majority": evaluate_predictions(y_true, y_pred_majority, title="Majority Class Baseline"),
        "momentum": evaluate_predictions(y_true, y_pred_momentum, title=f"{lookback_days}-Day Momentum Baseline"),
    }
