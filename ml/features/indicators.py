# ml/features/indicators.py
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

V1_HISTORY_START = "2020-01-01"


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's Exponential Smoothing (standard for RSI)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (num_std * std)
    lower = sma - (num_std * std)
    return upper, sma, lower


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = close.diff().fillna(0).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    direction[direction == 0] = 0
    return (direction * volume).cumsum()

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates technical features only.
    Does NOT create target labels.

    Safe for both training and inference.
    """
    df = df.sort_values("date").copy()

    # Moving averages
    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()

    # Relative trend features
    df["dist_ema_21"] = (
        (df["close"] - df["ema_21"]) / df["ema_21"]
    )
    df["dist_sma_50"] = (
        (df["close"] - df["sma_50"]) / df["sma_50"]
    )

    # RSI and MACD
    df["rsi_14"] = compute_rsi(df["close"], period=14)

    (
        df["macd"],
        df["macd_signal"],
        df["macd_hist"],
    ) = compute_macd(df["close"])

    # Bollinger bands
    bb_upper, bb_mid, bb_lower = compute_bollinger_bands(df["close"])

    df["bb_width"] = (
        (bb_upper - bb_lower) / (bb_mid + 1e-9)
    )

    df["bb_pct"] = (
        (df["close"] - bb_lower)
        / (bb_upper - bb_lower + 1e-9)
    )

    # Volume features
    df["vol_sma_20"] = df["volume"].rolling(window=20).mean()

    df["vol_ratio"] = (
        df["volume"] / (df["vol_sma_20"] + 1e-9)
    )

    df["obv"] = compute_obv(
        df["close"],
        df["volume"],
    )

    df["obv_ema_20"] = (
        df["obv"]
        .ewm(span=20, adjust=False)
        .mean()
    )

    df["obv_divergence"] = (
        (df["obv"] - df["obv_ema_20"])
        / (df["obv_ema_20"].abs() + 1e-9)
    )

     # Price returns
    df["return_1d"] = df["close"].pct_change(1)
    df["return_3d"] = df["close"].pct_change(3)
    df["return_5d"] = df["close"].pct_change(5)

    # Short-term volatility
    df["volatility_5d"] = (
        df["return_1d"]
        .rolling(window=5)
        .std()
    )
    
    return df


def build_indicators_for_ticker(
    df: pd.DataFrame,
    horizon_days: int = 3,
    flat_threshold_pct: float = 0.0075,
) -> pd.DataFrame:
    """
    Computes technical indicators and labels for a single ticker.
    Target logic:
      forward_return = (Close[t + horizon] - Close[t]) / Close[t]
      1 (UP)    if return > +flat_threshold_pct
     -1 (DOWN)  if return < -flat_threshold_pct
      0 (FLAT)  otherwise
    """
    df = add_technical_indicators(df)

    # Target Definition: Direction over the next N days
    forward_return = (df["close"].shift(-horizon_days) - df["close"]) / df["close"]
    df["target_return"] = forward_return

    conditions = (
        (forward_return > flat_threshold_pct),
        (forward_return < -flat_threshold_pct),
    )
    choices = [1, -1]  # 1: UP, -1: DOWN
    df["target"] = np.select(conditions, choices, default=0)  # 0: FLAT

    # Drop target lookup tail rows (last N days have no forward target)
    df = df.head(-horizon_days).copy()

    # Drop early rows that don't have enough history for 50 SMA
    df = df.dropna().reset_index(drop=True)
    return df


def generate_feature_dataset(
    df: pd.DataFrame,
    horizon_days: int = 3,
    flat_threshold_pct: float = 0.0075,
) -> pd.DataFrame:
    processed = []
    for ticker, group in df.groupby("ticker"):
        if len(group) < 60:
            logger.warning(f"Skipping {ticker}: Insufficient rows for feature calculation.")
            continue
        feat_group = build_indicators_for_ticker(
            group, horizon_days=horizon_days, flat_threshold_pct=flat_threshold_pct
        )
        processed.append(feat_group)

    if not processed:
        raise ValueError("No feature frames generated.")

    out = pd.concat(processed, ignore_index=True)
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)

def generate_inference_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds features without requiring future prices.
    Keeps the latest valid row for inference.
    """
    if len(df) < 60:
        raise ValueError(
            "At least 60 historical rows are required "
            "to compute technical indicators."
        )

    df = add_technical_indicators(df)

    return df.dropna().reset_index(drop=True)
