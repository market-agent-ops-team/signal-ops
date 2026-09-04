import pandas as pd
import logging 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def drop_badtickers(combined:pd.DataFrame,reports:list[dict])->pd.DataFrame:
  tickers_good=[r["ticker"] for r in reports if r.get("ok",False)]
  tickers_bad=[r["ticker"] for r in reports if not r.get("ok",False)]
  if tickers_bad:
        logger.warning(f"Dropping tickers that failed quality checks: {tickers_bad}")
 
  if not tickers_good:
     raise RuntimeError("No tickers passed quality checks - nothing to preprocess.")
  good_formatted = [t if t.endswith(".NS") else t.upper() + ".NS" for t in tickers_good]
 
  filtered = combined[combined["ticker"].isin(good_formatted)].copy()
  return filtered

def small_gap_skip(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()

    value_cols = ["open", "high", "low", "close", "volume"]
    df[value_cols] = df.groupby("ticker")[value_cols].ffill()

    remaining_na = df.isnull().sum().sum()
    if remaining_na > 0:
        logger.warning(f"{remaining_na} values still NaN after ffill (likely leading gaps) - dropping those rows")
        df = df.dropna()

    return df.reset_index(drop=True)

def chronological_split(df: pd.DataFrame,train_frac: float = 0.7,val_frac: float = 0.20,) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  
    train_parts, val_parts, test_parts = [], [], []
 
    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True)
        n = len(group)
 
        train_end = int(n * train_frac)
        val_end = train_end + int(n * val_frac)
 
        train_parts.append(group.iloc[:train_end])
        val_parts.append(group.iloc[train_end:val_end])
        test_parts.append(group.iloc[val_end:])
 
    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
 
    return train_df, val_df, test_df

def preprocess(combined:pd.DataFrame,reports:list[dict],training_frac:float=0.7,validation_frac:float=0.20)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    clean = drop_badtickers(combined, reports)
    clean = small_gap_skip(clean)
    train_df, val_df, test_df = chronological_split(clean, training_frac, validation_frac)
 
    logger.info(
        f"Preprocessing done: {clean['ticker'].nunique()} tickers kept, "
        f"train={len(train_df)} rows, val={len(val_df)} rows, test={len(test_df)} rows"
    )
 
    return train_df, val_df, test_df




if __name__ == "__main__":
    from data_fetch import fetch_grp
 
    tickers = ["INFY", "TCS", "WIPRO", "HCLTECH"]
    combined, reports = fetch_grp(tickers, "2020-01-01", "2024-01-01")
 
    train_df, val_df, test_df = preprocess(combined, reports)
 
    print("Train:", train_df.shape, train_df["date"].min(), "to", train_df["date"].max())
    print("Val:  ", val_df.shape, val_df["date"].min(), "to", val_df["date"].max())
    print("Test: ", test_df.shape, test_df["date"].min(), "to", test_df["date"].max())
 