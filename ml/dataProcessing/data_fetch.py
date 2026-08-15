import yfinance as yf
import pandas as pd
import logging
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
 
def format_tickers(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if not ticker.endswith(".NS"):
        ticker += ".NS"
    return ticker
 
 
def pick_OHLCV(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    index = format_tickers(ticker)
    df = yf.download(index, start=start, end=end, interval=interval,
                      auto_adjust=True, progress=False)
 
    if df.empty:
        raise ValueError(f"No data returned for {index}")
 
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
 
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
 
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df["ticker"] = index
 
    return df
 
 
def ohlcv_validation(
    df: pd.DataFrame,
    ticker: str,
    max_row_missing_pct: float = 2.0,
    max_consecutive_gap: int = 3,
) -> dict:
    row_missing_pct = round(df.isnull().any(axis=1).mean() * 100, 2)
    col_missing_pct = (df.isnull().mean() * 100).round(2).to_dict()
 
    
    is_null_close = df["close"].isnull()
    if is_null_close.any():
        max_gap = int(is_null_close.groupby((~is_null_close).cumsum()).sum().max())
    else:
        max_gap = 0
 
    report = {
        "ticker": ticker,
        "rows": len(df),
        "row_missing_pct": row_missing_pct,
        "col_missing_pct": col_missing_pct,
        "max_consecutive_gap_days": max_gap,
        "zero_volume_days": int((df["volume"] == 0).sum()),
        "duplicate_dates": int(df["date"].duplicated().sum()),
        "sorted": bool(df["date"].is_monotonic_increasing),
    }
 
    report["ok"] = (
        row_missing_pct <= max_row_missing_pct
        and max_gap <= max_consecutive_gap
        and report["duplicate_dates"] == 0
    )
 
    return report
 
 
def fetch_grp(
    tickers: list[str],
    start: str,
    end: str,
    min_success_rate: float = 0.5,
) -> tuple[pd.DataFrame, list[dict]]:
    frames = []
    reports = []
 
    for t in tickers:
        try:
            df = pick_OHLCV(t, start, end)
            report = ohlcv_validation(df, t)
            reports.append(report)
 
            if not report["ok"]:
                logger.warning(f"{t}: quality issue -> {report}")
            frames.append(df)
 
        except ValueError as e:
            logger.error(f"{t}: fetch failed -> {e}")
            reports.append({"ticker": t, "ok": False, "error": str(e)})
 
    
    if not frames:
        failed = [r["ticker"] for r in reports]
        raise RuntimeError(
            f"All {len(tickers)} tickers failed to fetch: {failed}. "
            f"Check date range, ticker symbols, and network access."
        )
 
    success_rate = len(frames) / len(tickers)
    if success_rate < min_success_rate:
        raise RuntimeError(
            f"Only {len(frames)}/{len(tickers)} tickers fetched successfully "
            f"({success_rate:.0%} < required {min_success_rate:.0%}). "
            f"Reports: {reports}"
        )
 
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
 
    return combined, reports
 
 
if __name__ == "__main__":
    tickers = ["INFY", "TCS", "WIPRO", "HCLTECH"]  
    data, reports = fetch_grp(tickers, "2020-01-01", "2024-01-01")
 
    print(data.shape)
    print(data.head())
 
    for r in reports:
        print(r)
 