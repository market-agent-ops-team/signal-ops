# Market-Agent-Ops

Local stock-analysis application combining XGBoost direction prediction with news sentiment in a LangGraph pipeline served by FastAPI with Redis caching.

The **Signal Ops** frontend is a monochrome, server-rendered research workspace. It uses Jinja2, compiled Tailwind CSS, and self-hosted fonts, with no browser JavaScript.

## Architecture

```
POST /analyze -> FastAPI -> Redis lookup -> LangGraph -> response -> Redis store
                                    miss -> technical (XGBoost) + news (NewsAPI+VADER) -> Analyst -> Editor
```

## Requirements

- Python 3.14
- Node.js 24 for rebuilding CSS and font assets (not needed by the running Python app)
- Redis 7 for caching (optional at runtime; API falls back without it)
- NewsAPI key for live news (`NEWS_API_KEY`); without it news is reported unavailable
- Docker for containerized runs (optional)

## Setup

```
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m pip install -r requirements-dev.txt
copy .env.example .env
```

Set `NEWS_API_KEY` in `.env`. For local runs `REDIS_URL=redis://localhost:6379/0`. Compose overrides to `redis://redis:6379/0`.

Build the frontend assets and start the application:

```powershell
npm ci
npm run build:css
venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Open **http://127.0.0.1:8000/**. Generated CSS and font files are included in the working tree; rebuild after editing templates or `static/css/source.css`. The build copies Latin variable WOFF2 files and their OFL licenses from pinned Fontsource packages. Node is used only for this asset build.

## Frontend

| Route | Purpose |
|---|---|
| `GET /` | Workspace with ticker/date inputs and documented V1 development metrics |
| `GET /report?ticker=INFY&target_date=2026-09-11` | Run or retrieve an analysis, render the report, and offer the same form to rerun |
| `GET /examples` | Three explicitly labeled offline fixtures |
| `GET /examples/bullish-alignment`, `/examples/bearish-alignment`, `/examples/divergence` | Allowlisted fixture reports; no market or news calls |
| `GET /methodology` | Model, news pipeline, evaluation, assumptions, and limitations |
| `POST /theme` | `theme=light`, `dark`, or `system`; optional safe local `return_to`; 303 redirect |

The report page calls the existing async `api.routes.analyze()` orchestration directly. HTML and JSON share input normalization, Redis entries, graph execution, output validation, and error codes. Invalid form values are retained with inline errors and a focused error summary. Requests navigate normally while research runs; the form explains the wait.

Light/Dark/System controls store only an appearance preference in an HttpOnly, SameSite=Lax cookie for up to one year (Secure over HTTPS). System follows CSS `prefers-color-scheme`; explicit preferences are rendered on the HTML element. Reports support browser Print/Save as PDF, and disclosures use native `<details>` controls.

Report Markdown disables raw HTML, images, and unsafe link protocols. Section headings are nested below the page heading; line breaks preserve the existing editor’s report layout. Templates escape all other dynamic text. Frontend responses disallow scripts with CSP and use `Cache-Control: no-store` so a shared browser/proxy cache does not mix theme variants; the analysis Redis cache still applies.

## Training and evaluation

```
venv\Scripts\python.exe -m ml.run_pipeline
```

- Dataset: INFY, TCS, WIPRO, HCLTECH from 2020-01-01 to 2024-01-01 via yfinance.
- Features computed before splitting; chronological split by shared dates (70/20/10) with 3-row boundary purge; scaler fit on train only.
- Target: 3-day forward return; UP above +0.75%, DOWN below -0.75%, else FLAT.
- Model: XGBoost multi-class (`n_estimators=300`, `learning_rate=0.03`, `max_depth=4`, `subsample=0.8`, `colsample_bytree=0.8`, early stopping 30, balanced weights).
- Pipeline saves evaluated candidates to `ml/candidate_models/` with `metrics.json`; promotion to `ml/saved_models/` is manual.

Measured 2026-09-14 (376 test rows):

| Split | Model | Accuracy | Macro F1 |
|---|---|---:|---:|
| Test | Majority | 39.36% | 18.83% |
| Test | 3-day momentum | 32.98% | 32.38% |
| Test | XGBoost | 38.83% | 37.86% |
| Validation | XGBoost | 39.59% | 39.43% |

Per-class test F1: DOWN 0.32, FLAT 0.34, UP 0.48. Best iteration 31, best log loss 1.095.

## Inference

`predict_trend(ticker, target_date)` in `ml/model/predict.py` returns `{"ml_trend": "bullish"|"neutral"|"bearish", "ml_confidence": 0.0-1.0}`. It validates strict `YYYY-MM-DD`, fetches from the documented V1 origin through the target day, rebuilds the shared indicator path, scales with the saved scaler, and maps classes 0/1/2 to bearish/neutral/bullish. Missing artifacts raise `FileNotFoundError`; bad inputs raise `ValueError`.

## API

- `GET /health` returns `{"status": "ok"}` without invoking research.
- `POST /analyze` with `{"ticker": "INFY", "target_date": "2026-09-11"}` returns ticker, target_date, ml_trend, ml_confidence, news_sentiment (`bullish`/`bearish`/`neutral` or null), news_status (`available`/`empty`/`unavailable`), divergence_flag, analyst_reasoning, final_report.
- Tickers normalize to stripped uppercase without `.NS`; other exchange suffixes are rejected with 422. Dates require valid `YYYY-MM-DD`.
- Graph runs in a worker thread. `ValueError` maps to 422, missing artifacts to 503, other failures to 500 with generic messages. Invalid pipeline output is rejected before caching.

## Redis

- Key: `analysis:{TICKER}:{YYYY-MM-DD}`, e.g. `analysis:INFY:2026-09-11`.
- Default TTL 900 seconds via `CACHE_TTL_SECONDS`.
- Final validated JSON response is cached. Hits skip the graph. Corrupt, mismatched, or extra-field payloads are treated as misses. Redis errors fall back to uncached analysis with a warning.

## Docker

```
docker compose config
docker compose build
docker compose up -d
```

API on `8000:8000`, Redis `redis:7-alpine` with health checks. Compose sets `REDIS_URL=redis://redis:6379/0` and loads `.env` at runtime; `.env` is never baked into the image. `ml/saved_models/` is included in the image; candidates are excluded.

The Dockerfile first builds CSS and fonts in a pinned Node stage using `npm ci`, then copies the generated assets into the Python runtime. Templates and fixture reports are included. `node_modules` is excluded from the image context.

## Testing

```
venv\Scripts\python.exe -m pytest -q
venv\Scripts\python.exe -m pyrefly check agents api ml tests --python-interpreter-path venv\Scripts\python.exe --output-format min-text
venv\Scripts\python.exe -m pip check
npm run build:css
```

Live checks are opt-in: `pytest -m live -q` (requires Redis and optionally a running API).

Frontend route/security/cache regression tests are in `tests/test_frontend.py`. No standalone linter is configured. The known pyrefly finding at `agents/graph.py:36` concerns LangGraph’s `MarketState` TypedDict bound; see `IMPLEMENTATION.md` for verification results and browser-check scope.

## Limitations

- Markets are noisy; direction accuracy near 39% is not profitability.
- Indicators use historical data only; regime changes and data revisions apply.
- VADER is general-purpose sentiment, not finance-specific.
- Test results were inspected during development; treat them as development results, not an untouched benchmark. A fresh holdout is reserved before future model selection.
- Predictions are not financial advice.

## Reports

`reports/` holds deterministic fixture examples: bullish alignment, bearish alignment, divergence. Each file states it was generated offline without live calls.
