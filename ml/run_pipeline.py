from ml.dataProcessing.data_fetch import fetch_grp
from ml.dataProcessing.data_preprocessing import drop_badtickers, small_gap_skip, chronological_split
from ml.features.indicators import generate_feature_dataset
from ml.model.evaluate import run_baselines

if __name__ == "__main__":
    tickers = ["INFY", "TCS", "WIPRO", "HCLTECH"]

    # 1. Fetch & Quality Checks
    raw_data, reports = fetch_grp(tickers, "2020-01-01", "2024-01-01")
    clean = drop_badtickers(raw_data, reports)
    clean = small_gap_skip(clean)

    # 2. Generate Features & Targets BEFORE the split
    features_df = generate_feature_dataset(
        clean, horizon_days=3, flat_threshold_pct=0.0075
    )

    # 3. Chronological Split
    train_df, val_df, test_df = chronological_split(
        features_df, train_frac=0.70, val_frac=0.20
    )

    print(f"Features created. Total samples: {len(features_df)}")
    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")
    print("\nTarget Distribution in Validation Set:")
    print(val_df["target"].value_counts(normalize=True).round(3))

    # 4. Evaluate Naive Baselines
    run_baselines(val_df, lookback_days=3)