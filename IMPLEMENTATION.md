# Market-Agent-Ops Implementation Plan

**Goal:** Complete the locally deployable analysis application, then execute bounded, reproducible model-improvement experiments.

**Architecture:** FastAPI validates requests and caches final responses in Redis. LangGraph joins XGBoost inference and NewsAPI/VADER research before Analyst and Editor produce a Markdown report. Model experiments remain separate from serving artifacts until validation supports promotion.

**Tech stack:** Python 3.14, pandas, scikit-learn, XGBoost, LangGraph, FastAPI, Redis 7, Docker Compose, pytest.

**Sources:** `../Person_A_Remaining_Work_Plan.md`, `../Person_B_Remaining_Work_Plan.md`, `../Market-Agent-Ops_Week5_Redis_Docker_Testing_Plan.md`, `../MODEL_IMPROVEMENT_PLAN.md`.

## Global constraints

- Finish application integration before model experiments; preserve the V1 model during Week 5.
- `predict_trend(ticker, target_date)` returns `ml_trend` in `bullish`, `neutral`, `bearish` and confidence in `[0, 1]`.
- Use chronological splits, purge forward-label boundaries, and fit scaling on training only.
- Compare baselines and model on identical evaluation rows. Never select on final holdout results.
- Keep normal tests deterministic and offline. Separate live service checks.
- Cache the final validated response as JSON with normalized `analysis:{ticker}:{target_date}` keys and default TTL `900` seconds.
- Redis failure must not prevent valid analysis. External news failure must not be represented as observed neutral sentiment.
- Keep credentials out of code, logs, cache keys, Git, and Docker images.
- Record evidence honestly; human interviews and unavailable service checks remain pending.
- No commits, pushes, deployment, destructive cleanup, or production artifact replacement without explicit authorization.

## Task 1: Reproducible baseline

Files: `requirements.txt`, `requirements-dev.txt`, `pytest.ini`, `.gitignore`, this document.

- [x] Convert requirements to UTF-8 and include actual runtime dependencies.
- [x] Install test/cache dependencies into the existing virtual environment.
- [x] Verify application imports and model/scaler/feature dimensions.
- [x] Establish pytest and available typecheck commands.

Verification: `venv\Scripts\python.exe -m pip check`; import `api.main:app`; call `_load_artifacts()` and compare feature counts.

## Task 2: ML correctness and regression coverage

Files: existing `ml/features/indicators.py`, `ml/dataProcessing/data_preprocessing.py`, `ml/model/{train,predict,evaluate}.py`, `ml/run_pipeline.py`; `tests/test_{features,preprocessing,model,evaluation}.py`.

- [x] Write regression tests first for causal features, train/inference equality, latest inference row, purge edge cases, train-only scaling, strict dates, class/confidence, missing artifacts, and fair baseline history.
- [x] Remove duplicated feature definitions; share training and inference indicator construction.
- [x] Train once, evaluate that model, and save that same model with structured metrics.
- [x] Fix boundary purging and momentum history; preserve all three labels in evaluation and include confusion matrix, per-class metrics and log loss.
- [x] Verify artifact compatibility and run focused offline tests.
- [x] Use shared unique-date split boundaries across tickers and verify purged label endpoints with unequal histories and missing sessions.
- [x] Initialize V1 training and inference from the same documented history origin and verify parity with a range-respecting fetch fixture.
- [x] Save evaluated pipeline candidates separately from serving artifacts; require deliberate promotion.

Verification: `venv\Scripts\python.exe -m pytest tests/test_features.py tests/test_preprocessing.py tests/test_model.py tests/test_evaluation.py -q`.

## Task 3: Agents, API and Redis

Files: `agents/{researcher_agent,analyst_agent,editor_agent,graph}.py`, `api/{main,routes,cache}.py`; `tests/test_{analyst,researcher,editor,api,cache,integration}.py`.

- [x] Write failing tests for six divergence cases, aliases, confidence formatting, malformed news, network failures, availability status and report semantics.
- [x] Normalize tickers and strict dates; preserve public `news_sentiment` mapping from internal `news_sentiments`.
- [x] Distinguish unavailable news, no relevant news, neutral news and aligned signals in API/report output.
- [x] Validate final response, return controlled errors, and keep synchronous graph work off the async event loop.
- [x] Implement Redis cache lookup/write with short timeouts, normalized keys, configurable TTL, validation of cached responses and outage fallback.
- [x] Test cache hit/miss, TTL, invalid JSON/payloads, outage behavior and exact cached/uncached equality.
- [x] Exercise the real LangGraph structure with external inputs mocked.

Verification: `venv\Scripts\python.exe -m pytest tests/test_analyst.py tests/test_researcher.py tests/test_editor.py tests/test_api.py tests/test_cache.py tests/test_integration.py -q`.

## Task 4: Containerization and live integration

Files: `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.env.example`, `tests/test_live_services.py`.

- [x] Package API, agents, inference and saved model artifacts using Python 3.14 slim.
- [x] Configure Redis 7, runtime environment, service DNS, and lightweight health checks.
- [x] Add opt-in Redis expiry/outage and live application checks.
- [x] Run full offline suite and dependency checks.
- [ ] Build image and verify artifacts, real inference, POST miss/hit and restart behavior inside Compose.

Verification: `docker compose config`; `docker compose build`; `docker compose up -d`; artifact loading inside API; health and repeated analysis requests. Do not remove existing volumes as a shortcut.

## Task 5: Bounded model experiments

Files: experiment modules under `ml/experiments/`, shared feature/training helpers where needed, `tests/test_experiments.py`, `experiments/` results.

- [x] Define dataset snapshots, seed 42, experiment records, and reserve a final date holdout before any selection.
- [ ] Run V1 validation baseline, NIFTY context, and seven extended stock features as separate factors.
- [ ] Run ablations for RSI, volume, Bollinger, returns and market context.
- [ ] Compare original four stocks with TECHM, LTIM, MPHASIS, PERSISTENT and COFORGE added.
- [ ] Compare 2018/2020/2021 history starts and rolling/expanding windows with chronological walk-forward folds.
- [ ] Test all seven documented horizon/threshold pairs and record class distributions.
- [ ] Limit randomized XGBoost tuning to 12 seeded candidates; compare Logistic Regression, Random Forest, LightGBM and CatBoost when installable.
- [ ] Compare sigmoid/isotonic calibration on separate chronological calibration rows.
- [ ] Freeze selection, evaluate final holdout once, record full metrics and preserve V1 unless promotion is justified.

Verification: offline leakage/selection tests; real-data experiment CLI results with every candidate's data/features/target/parameters/metrics/status. Failed downloads and unavailable optional estimators are recorded, never replaced with fabricated performance. LSTM remains conditional in the source roadmap and is not introduced without a justified sequence representation.

## Task 6: Documentation and final review

Files: `README.md`, `DESIGN.md`, `reports/`, this document.

- [x] Document data, target, features, baseline comparison, confidence meaning, limitations, endpoints, failure behavior, Redis and Docker.
- [x] Generate three clearly labeled deterministic example reports: bullish alignment, bearish alignment and divergence.
- [x] Provide interview questions/answers covering both ownership areas.
- [x] Review implementation against every source document; run full tests and configured static checks.
- [x] Record exact results and remaining blockers here.
- [ ] Human activity: both team members explain the full request path and complete mock interviews.

## Execution record

- Initial inspection: real inference, saved artifacts, graph and API exist; no tests, Redis, Docker, README or DESIGN were present.
- `requirements.txt` is UTF-16 and omits installed FastAPI/Uvicorn/dotenv/VADER packages.
- Python and project virtual environment report 3.14.7. Docker executable is unavailable on PATH.
- Ruling: work in the requested `signal-ops/` directory on the new `implement-market-ops` branch, rather than relocate the deliverable. Changes remain uncommitted and reviewable.
- Ruling: bounded experiments use 12 tuning candidates and finite feature/target/window comparisons. No guaranteed model improvement is assumed.

## Interface/dependency review

| Tasks | Shared contract | Resolution |
|---|---|---|
| 1 / 2 / 3 | Runtime versions and artifact compatibility | Verify installed versions before changing binary artifacts. |
| 2 / 3 | `predict_trend` | Keep two-field public prediction contract; errors reach API boundary. |
| 3 / 4 | Redis URL, TTL, API health | Localhost default; Compose overrides host to `redis`; health does not invoke research. |
| 2 / 5 | Features, labels and scaler | V1 feature defaults remain fixed; experiment options are explicit. |
| 3 / 6 | News/report semantics | Document unavailable versus neutral and preserve public singular sentiment field. |
| 4 / 5 | Phase gate | Build experiment tooling if Docker unavailable, but record gate as blocked and do not claim end-to-end completion. |
| 1–6 | Tests and status | Each task records actual evidence; no unchecked task is represented as completed. |

### Tasks 1–2 execution evidence — 2026-09-14

- Installed pytest 9.1.1, redis 8.1.0 and pyrefly 1.3.0 into the existing Python 3.14.7 venv. UTF-8 requirements now pin 14 direct runtime packages to queried installed versions; development requirements include pytest, pyrefly and HTTPX. `venv\Scripts\python.exe -m pip install -r requirements-dev.txt` succeeded; `venv\Scripts\python.exe -m pip check` reported `No broken requirements found.`
- Test-first baseline: **22 failed, 14 passed in 2.60s**. Additional zero/negative momentum-lookback regressions failed before validation was added. Train-only scaling already behaved correctly and is now regression-covered.
- Focused command from Task 2: **41 passed in 1.83s**. Final `venv\Scripts\python.exe -m pytest -q`: **41 passed in 2.16s**, no warnings. Normal tests block socket, Requests and curl-cffi network calls; `live` tests are excluded by default. Test artifacts are written under ignored `.pytest_cache/tmp`.
- Coverage includes causal/shared feature construction, chronological per-ticker purge endpoints and short/empty/zero boundaries, train-only scaler statistics, strict dates, all three trend mappings, missing artifacts, actual saved-model fixture inference, contiguous momentum history with overlap/future rows and exact evaluation-row alignment. The pipeline test trains real XGBoost once, verifies training/validation inputs and model identity at both evaluations and saving, reloads the saved model, and checks persisted metrics against causal returns.
- `venv\Scripts\pyrefly.exe check ml tests --python-interpreter-path venv\Scripts\python.exe --output-format min-text`: **0 errors**. Broadening the paths to `ml agents api tests` reports **1 existing error** at `agents/graph.py:34` (`bad-specialization`: `MarketState` does not satisfy LangGraph's `StateT` bound). No standalone linter is configured. `git diff --check` passes.
- `api.main:app` imports as FastAPI. `_load_artifacts()` loads model/scaler/feature counts **12/12/12**, classes **[0, 1, 2]**. All three original artifact SHA-256 hashes match the initial baseline; no serving binaries were replaced.
- Interfaces: `predict_trend` retains its two-field response and now rejects noncanonical or invalid `YYYY-MM-DD` dates before fetching. `evaluate_predictions` accepts optional probabilities and returns accuracy, fixed-three-class macro F1, label order, per-class precision/recall/F1/support, confusion matrix and log loss (`null` for hard-label baselines). `evaluate_xgboost` still accepts encoded targets, decoding them and ordering probability columns for reporting. `run_baselines` requires complete, unpurged trading-session price history, rejects nonpositive lookbacks or insufficient history, and preserves evaluation rows. `save_artifacts` accepts optional keyword-only `metrics`; the pipeline saves validation/test baseline and model metrics to `metrics.json` beside the evaluated model.
- Tasks 1–2 are complete. The broader agent typecheck finding remains for its owner. Docker is absent, so container/live-service checks remain pending under later tasks. Synthetic fixture results establish correctness, not investment performance or a new market benchmark. Changes were reviewed; no commits or branch changes were made.

### Task 2 review corrections — 2026-09-14

- Reproduced the review findings before implementation: `venv\Scripts\python.exe -m pytest tests/test_preprocessing.py tests/test_model.py -q --tb=short` produced **4 failed, 29 passed in 1.79s**. Unequal ticker histories allowed pooled training dates after validation began; inference requested only 180 days; the pipeline selected the serving output directory.
- `chronological_split` now partitions sorted unique dates once for the whole dataset, with the first `floor(date_count * train_frac)` dates in training and the next `floor(date_count * val_frac)` in validation. All ticker rows on a date stay in the same split. Regressions cover unequal start/end histories, missing sessions, pooled date ordering and actual forward-label endpoint dates.
- Purging remains per ticker and observation horizon. Removing the final `horizon_days` feature rows leaves at least that many later same-ticker observations before the shared cutoff for every retained row. Their forward-label endpoints therefore precede the cutoff even with missing sessions; dropped feature rows only make this purge more conservative. Tests check both boundaries and retained per-ticker counts.
- Existing V1 uses `V1_HISTORY_START = "2020-01-01"` in `ml/features/indicators.py`, shared by the current training pipeline and inference. Inference requests that origin through the exclusive day after the target, preserving cumulative OBV and EMA initialization rather than resetting them 180 days before each request. A multi-year fixture with a fetch double that honors the requested start/end verifies the requested range, all 12 training/inference inputs and real-artifact inference. This contract follows the documented V1 training window; the legacy artifacts do not embed origin metadata. Different-origin candidates must carry and consume an explicit history-origin contract before promotion. Historical data-provider revisions and adjusted-price changes can still prevent exact reproduction of an old dataset snapshot.
- `python -m ml.run_pipeline` now explicitly saves the evaluated model, scaler, feature names and `metrics.json` under project-relative **`ml/candidate_models/`**, resolved from the module location. It does not promote candidates. Promotion is a deliberate, separately authorized replacement of the complete compatible artifact bundle in `ml/saved_models/`, after validation review and history-origin verification; restart serving processes to clear cached artifacts. No candidate was promoted and the serving artifacts were not retrained.
- Verification: `venv\Scripts\python.exe -m pytest tests/test_features.py tests/test_preprocessing.py tests/test_model.py tests/test_evaluation.py -q --tb=short` → **44 passed in 1.92s**, no warnings. `venv\Scripts\pyrefly.exe check ml tests --python-interpreter-path venv\Scripts\python.exe --output-format min-text` → **0 errors**. No dependency changes were needed. The previously recorded out-of-scope agent typecheck issue and unavailable Docker checks remain pending.
- Final review: `git diff --check` passes; all three serving artifact SHA-256 hashes still match the recorded initial baseline, and loading confirms **12/12/12** model/scaler/feature dimensions. This review pass changed only the ML split/history/pipeline paths, their regression tests and this document.

### Tasks 3–6 execution evidence — 2026-09-15

- Task 3: `agents/graph.py` now uses `typing_extensions.TypedDict` with `total=False` and carries `news_status`; `api/routes.py` normalizes tickers, validates strict dates, serves singular `news_sentiment` plus `news_status`, validates graph output and cached payloads (canonical trends, finite 0-1 confidence with bool rejection, status/sentiment consistency, exact divergence, ticker/date match, non-empty reasoning/report, no extra keys), maps `ValueError` to 422, missing artifacts to 503, other failures to generic 500 without secret leakage, and runs the graph via `asyncio.to_thread`. `tests/test_api.py` **50 passed**, `tests/test_cache.py` + `tests/test_integration.py` **46 passed**. Real LangGraph structure is exercised with mocked externals only.
- Full suite: `venv\Scripts\python.exe -m pytest -q` → **213 passed, 3 deselected (live), 0 failed**. `pyrefly check agents api ml tests` → **1 error**, the pre-existing `agents/graph.py:36` `bad-specialization` on `StateGraph(MarketState)`; the six new route type errors were fixed. `pip check` clean.
- Task 4: added `Dockerfile` (python:3.14-slim, serving artifacts included), `docker-compose.yml` (redis:7-alpine, service DNS `redis://redis:6379/0`, health checks, runtime `.env`), `.dockerignore` (excludes candidates, keeps `ml/saved_models/`), `.env.example`, `tests/test_live_services.py` (opt-in `live` marker). Compose YAML parses and API imports offline. Docker executable is absent from PATH, so `docker compose build/up` and container inference/POST/restart checks are **blocked**, not claimed.
- Task 5: added `ml/experiments/{config,core}.py` (seed 42, V1/extended tickers, 3 history starts, 7 horizon/threshold pairs, 12 seeded tuning candidates, 3 calibration methods, NIFTY/extended feature builders, ablation sets, holdout reservation, walk-forward folds, record schema) with `tests/test_experiments.py` **7 passed**. Real `python -m ml.run_pipeline` reproduced V1: test majority 39.36%/18.83%, momentum 32.98%/32.38%, XGBoost 38.83%/37.86% (DOWN 0.32, FLAT 0.34, UP 0.48), validation XGBoost 39.59%/39.43%, best iteration 31, log loss 1.095; candidates saved under ignored `ml/candidate_models/`. Real-data NIFTY/ablation/window/threshold/tuning/calibration sweeps, extended tickers (TECHM, LTIM, MPHASIS, PERSISTENT, COFORGE unavailable offline), final-holdout selection, and promotion remain **pending**; LightGBM/CatBoost are not installed and are recorded unavailable rather than fabricated. LSTM was not introduced.
- Task 6: added `README.md`, `DESIGN.md` (with interview Q&A), and three deterministic fixture reports under `reports/` (`bullish_alignment`, `bearish_alignment`, `divergence`), each labeled as offline fixtures. Serving artifacts were not replaced. Remaining human activity: both members explain the full path and complete mock interviews. No commits, pushes, or destructive operations were performed.
