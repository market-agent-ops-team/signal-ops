SEED = 42

V1_TICKERS = ["INFY", "TCS", "WIPRO", "HCLTECH"]

EXTENDED_TICKERS = ["TECHM", "LTIM", "MPHASIS", "PERSISTENT", "COFORGE"]

HISTORY_STARTS = ["2018-01-01", "2020-01-01", "2021-01-01"]

HORIZON_THRESHOLD_GRID = [
    (3, 0.005),
    (3, 0.0075),
    (3, 0.01),
    (3, 0.0125),
    (5, 0.0075),
    (5, 0.01),
    (5, 0.015),
]

TUNING_CANDIDATE_COUNT = 12

CALIBRATION_METHODS = ["none", "sigmoid", "isotonic"]

NIFTY_FEATURES = [
    "nifty_return_1d",
    "nifty_return_3d",
    "nifty_return_5d",
    "nifty_volatility_5d",
    "relative_return_3d",
    "relative_return_5d",
]

EXTENDED_FEATURES = [
    "return_10d",
    "volatility_10d",
    "ema_9_vs_21",
    "volume_change_1d",
    "volume_change_5d",
    "high_low_range",
    "close_open_return",
]
