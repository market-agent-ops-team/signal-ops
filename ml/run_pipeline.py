from ml.dataProcessing.data_fetch import fetch_grp
from ml.dataProcessing.data_preprocessing import drop_badtickers, small_gap_skip, chronological_split ,purge_split_boundary
from ml.features.indicators import generate_feature_dataset
from ml.model.evaluate import run_baselines , evaluate_xgboost
from ml.model.train import prepare_tensors, train_xgboost ,FEATURE_COLS ,save_artifacts

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

    train_df = purge_split_boundary(train_df,horizon_days=3)
    val_df = purge_split_boundary(val_df,horizon_days=3)

    (X_train , y_train) , (X_val , y_val) , (X_test , y_test) , scaler = prepare_tensors(train_df,val_df,test_df)


    print(f"Features created. Total samples: {len(features_df)}")
    print(
        f"Train samples: {len(train_df)} | "
        f"Val samples: {len(val_df)} | "
        f"Test samples: {len(test_df)}"
    )
    print("\nTarget Distribution in Validation Set:")
    print(val_df["target"].value_counts(normalize=True).round(3))

    model = train_xgboost(X_train,y_train,X_val,y_val)

    # 4. Evaluate Naive Baselines
    baseline_results = run_baselines(
        train_df=train_df,
        eval_df=test_df,
        history_df=val_df,
        lookback_days=3
    )

    xgb_results = evaluate_xgboost(
        model,
        X_test,
        y_test
    )

    print("\nTraining Target Distribution:")
    print(train_df["target"].value_counts(normalize=True).sort_index().round(3))

    print("\nValidation Target Distribution:")
    print(val_df["target"].value_counts(normalize=True).sort_index().round(3))

    print("\nTest Target Distribution:")
    print(test_df["target"].value_counts(normalize=True).sort_index().round(3))

    model = train_xgboost(
        X_train,
        y_train,
        X_val,
        y_val
    )

    print("\nXGBoost Best Iteration:", model.best_iteration)
    print("XGBoost Best Score:", model.best_score)

    print("\nFeature Importances:")

    importance_pairs = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )

    for feature, importance in importance_pairs:
        print(f"{feature:20s} : {importance:.4f}")

    print("\n================ Saving Model Artifacts ================")

    save_artifacts(
        model,
        scaler
    )
