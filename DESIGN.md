# Market-Agent-Ops Design

## Chronological discipline

Random splits leak future market outcomes into training. All splits partition sorted unique dates globally so every ticker on a date shares a split, then purge the final `horizon_days` feature rows per ticker so retained label endpoints precede the cutoff. Scaling fits on train only; validation/test use `transform`. Momentum baselines receive complete unpurged price history and shift within ticker so early evaluation rows have valid lookbacks.

## Model

Twelve scale-invariant features: `dist_ema_21`, `dist_sma_50`, `rsi_14`, `macd_hist`, `bb_width`, `bb_pct`, `vol_ratio`, `obv_divergence`, `return_1d`, `return_3d`, `return_5d`, `volatility_5d`. Target is 3-day forward direction with ±0.75% flat band, mapped DOWN=-1, FLAT=0, UP=1 and XGBoost-encoded 0/1/2. XGBoost was selected as a strong tabular baseline with native multi-class probabilities, early stopping, and balanced weights; exact price prediction is out of scope because direction with confidence is the contracted agent signal.

Artifacts (`xgb_model.joblib`, `scaler.joblib`, `feature_names.joblib`) load once via `lru_cache`. Training and inference share `add_technical_indicators` and the V1 origin `2020-01-01` so cumulative OBV/EMA initialization matches. Candidates save under `ml/candidate_models/` with `metrics.json`; serving reads only `ml/saved_models/`.

## Agents

- Researcher (`agents/researcher_agent.py`): ticker to company name, NewsAPI 7-day window, relevance filter, VADER classification, majority aggregation. Returns `news_items`, `news_sentiments` (null when not observed), `news_status` (`available`/`empty`/`unavailable`). Network, schema, and key failures return `unavailable` without secrets in logs; empty relevant results return `empty`.
- Analyst (`agents/analyst_agent.py`): normalizes aliases (`positive`->`bullish`, `negative`->`bearish`, `flat`->`neutral`, etc.), diverges only on bullish vs bearish when news is available, formats finite 0-1 confidence, explains unavailable/empty without claiming observed sentiment.
- Editor (`agents/editor_agent.py`): builds Markdown with technical signal, news status/sentiment, headlines, divergence/aligned/no-directional-alignment sections, and disclaimer. Neutral combinations never claim directional alignment.

Separation keeps research, judgment, and presentation independently testable instead of one monolith.

## Graph

`agents/graph.py` defines `MarketState` (total=False) with ticker, target_date, ml outputs, news outputs plus status, divergence, reasoning, report. `technical_research` (XGBoost) and `news_research` run in parallel from START, join into `analyst`, then `editor`, then END. One static typing warning remains on `StateGraph(MarketState)` under pyrefly 1.3.0; runtime invocation is covered by integration tests.

## News pipeline

Ticker -> normalized symbol -> company name -> NewsAPI `everything` (7 days, pageSize 50, English) -> company-mention filter -> VADER per-article positive/negative/neutral -> majority bullish/bearish/neutral. VADER is lexicon-based and general-purpose, not finance-tuned.

## Backend

`api/routes.py` validates requests with `normalize_ticker`/`validate_target_date`, checks Redis, runs `graph_app.invoke` via `asyncio.to_thread`, validates graph output (canonical trends, finite confidence, status/sentiment consistency, exact divergence, ticker/date match, non-empty reasoning/report), caches the singular-`news_sentiment` response, and returns it. `GET /health` does no work.

## Redis

Key `analysis:{TICKER}:{DATE}`, JSON final response, TTL 900 default. Validation rejects non-dict, missing/extra keys, type, range, and consistency errors. Failures log generic warnings and fall back to live analysis.

## Docker

`Dockerfile` uses `python:3.14-slim`, installs `requirements.txt`, copies the app including `ml/saved_models/`, exposes 8000, runs uvicorn. Compose adds `redis:7-alpine`, wires `REDIS_URL`, loads `.env` at runtime, and health-checks both services. `.dockerignore` excludes venv, caches, `.env`, notebooks, reports, and candidates but not serving artifacts.

## Failure boundaries

Invalid input 422, missing artifacts 503, other failures generic 500. News unavailable/empty never masquerades as neutral. Cache corruption expires into recomputation. No raw tracebacks or secrets reach clients.

## Interview Q&A

- Why not random split? It leaks future outcomes; time order must hold across all tickers.
- Target? 3-day forward return with ±0.75% flat band.
- Why XGBoost? Tabular multi-class baseline with calibrated probabilities and early stopping.
- Baseline? Majority and 3-day momentum on identical evaluation rows; XGBoost leads on macro F1 (37.86% vs 32.38% test).
- Leakage controls? Shared date cutoffs, boundary purge, train-only scaler, causal indicators, latest-row inference.
- `predict_trend` parity? Shared indicator builder plus fixed V1 history origin.
- Confidence? Max class probability in [0,1]; finite and validated.
- News disagreement? Analyst flags bullish-vs-bearish divergence; otherwise no directional claim.
- Why not exact price? Contract is direction plus confidence for downstream reasoning.
- Weaknesses? ~39% accuracy, UP strongest (0.48 F1), regime/drift risk, VADER generality.
- API flow? Validate -> cache -> thread-offloaded graph -> validate -> cache -> respond.
- Why FastAPI holds no ML logic? It orchestrates; `ml/` owns features and inference.
- Cache hit? Validated stored response returned; graph skipped.
- Compose networking? API reaches Redis via service name `redis`, not localhost.
- LangGraph? Parallel research branches joined before analysis.
- State? Ticker/date, ML outputs, news items/sentiment/status, divergence/reasoning/report.
- NewsAPI down? Status `unavailable`, null sentiment, valid report without comparison.
- Why Redis? Avoid recomputation for identical reports with TTL expiry.
- Chronological ML importance? Prevents training on the future being predicted.
