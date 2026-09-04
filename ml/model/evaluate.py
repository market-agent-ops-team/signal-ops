import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score
import logging

logger = logging.getLogger(__name__)


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series, title: str = "Baseline"):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n================ {title} Evaluation ================")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print("\nDetailed Classification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["DOWN (-1)", "FLAT (0)", "UP (1)"],
            zero_division=0,
        )
    )

    return {"accuracy": acc, "macro_f1": macro_f1}


def run_baselines(val_df: pd.DataFrame, lookback_days: int = 3):
    """
    Computes two baselines on the validation set:
    1. Majority Class: Always predicts the most common class in validation.
    2. Momentum / Persistence: Assumes past N-day direction continues for next N days.
    """
    y_true = val_df["target"]

    # 1. Majority Class Baseline
    majority_class = y_true.mode()[0]
    y_pred_majority = pd.Series(majority_class, index=val_df.index)
    evaluate_predictions(y_true, y_pred_majority, title="Majority Class Baseline")

    # 2. Momentum / Persistence Baseline
    past_return = (val_df["close"] - val_df["close"].shift(lookback_days)) / (
        val_df["close"].shift(lookback_days) + 1e-9
    )
    flat_threshold = 0.0075

    conditions = [
        past_return > flat_threshold,
        past_return < -flat_threshold,
    ]
    choices = [1, -1]
    y_pred_momentum = pd.Series(
        np.select(conditions, choices, default=0), index=val_df.index
    )

    evaluate_predictions(
        y_true, y_pred_momentum, title=f"{lookback_days}-Day Momentum Baseline"
    )