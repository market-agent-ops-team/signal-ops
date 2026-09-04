# ml/model/train.py
import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Canonical feature list from Week 2 indicators
FEATURE_COLS = [
    "dist_ema_21",
    "dist_sma_50",
    "macd_hist",
    "bb_width",
    "bb_pct",
    "vol_ratio",
    "obv_divergence",
]

TARGET_COL = "target"
LABEL_MAPPING = {-1: 0, 0: 1, 1: 2}  # XGBoost expects classes [0, 1, 2]
REVERSE_MAPPING = {0: -1, 1: 0, 2: 1}


def prepare_tensors(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Fits scaler ONLY on train data to prevent lookahead bias.
    Maps targets {-1, 0, 1} to {0, 1, 2}.
    """
    scaler = StandardScaler()
    
    # Fit scaler strictly on training split
    X_train = scaler.fit_transform(train_df[FEATURE_COLS])
    X_val = scaler.transform(val_df[FEATURE_COLS])
    X_test = scaler.transform(test_df[FEATURE_COLS])

    y_train = train_df[TARGET_COL].map(LABEL_MAPPING).to_numpy()
    y_val = val_df[TARGET_COL].map(LABEL_MAPPING).to_numpy()
    y_test = test_df[TARGET_COL].map(LABEL_MAPPING).to_numpy()

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> XGBClassifier:
    """Trains an XGBoost multi-class classifier with early stopping."""
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    return model


def save_artifacts(model: XGBClassifier, scaler: StandardScaler, output_dir: str = "ml/saved_models"):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "xgb_model.joblib"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    joblib.dump(FEATURE_COLS, os.path.join(output_dir, "feature_names.joblib"))
    print(f"Artifacts successfully saved to {output_dir}/")