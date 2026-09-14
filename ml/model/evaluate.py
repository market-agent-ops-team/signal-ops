import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score
import logging

logger = logging.getLogger(__name__)


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series, title: str = "Baseline"):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n================ {title} Evaluation ================")
    print(f"Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Macro F1 : {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")
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

def evaluate_xgboost(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\nXGBoost Prediction Distribution:")
    unique, counts = np.unique(y_pred, return_counts=True)

    for label, count in zip(unique, counts):
        print(f"Class {label}: {count}")

    return evaluate_predictions(
        y_test,
        y_pred,
        title="XGBoost Model"
    )

def run_baselines(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    history_df: pd.DataFrame,
    lookback_days: int = 3
):
    """
    Evaluates majority-class and momentum baselines.

    train_df:
        Used to determine the majority class.

    eval_df:
        Dataset on which baseline performance is measured.

    history_df:
        Previous chronological split used to provide price history
        for the first few rows of eval_df.
    """

    y_true = eval_df["target"].reset_index(drop=True)

    # 1. Majority Class Baseline
    majority_class = train_df["target"].mode()[0]

    y_pred_majority = pd.Series(
        majority_class,
        index=y_true.index
    )

    majority_results = evaluate_predictions(
        y_true,
        y_pred_majority,
        title="Majority Class Baseline"
    )

    # 2. Momentum Baseline

    # Take enough previous data to calculate momentum for
    # the first rows of each ticker in eval_df
    history_tail = (
        history_df
        .sort_values(["ticker", "date"])
        .groupby("ticker", group_keys=False)
        .tail(lookback_days)
        [["ticker", "date", "close"]]
        .copy()
    )

    history_tail["_is_eval"] = False
    history_tail["_row_id"] = -1

    eval_prices = eval_df[
        ["ticker", "date", "close"]
    ].copy()

    eval_prices["_is_eval"] = True
    eval_prices["_row_id"] = np.arange(len(eval_prices))

    combined = pd.concat(
        [history_tail, eval_prices],
        ignore_index=True
    )

    combined = combined.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    previous_close = (
        combined
        .groupby("ticker")["close"]
        .shift(lookback_days)
    )

    past_return = (
        combined["close"] - previous_close
    ) / (previous_close + 1e-9)

    flat_threshold = 0.0075

    conditions = [
        past_return > flat_threshold,
        past_return < -flat_threshold,
    ]

    choices = [1, -1]

    combined["momentum_prediction"] = np.select(
        conditions,
        choices,
        default=0
    )

    eval_predictions = (
        combined[combined["_is_eval"]]
        .sort_values("_row_id")
    )

    y_pred_momentum = pd.Series(
        eval_predictions["momentum_prediction"].to_numpy(),
        index=y_true.index
    )

    momentum_results = evaluate_predictions(
        y_true,
        y_pred_momentum,
        title=f"{lookback_days}-Day Momentum Baseline"
    )

    return {
        "majority": majority_results,
        "momentum": momentum_results,
    }

    