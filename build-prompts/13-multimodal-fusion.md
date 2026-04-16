# Digital Twin Upgrade Plan: Foundation Models and Multi-Modal Fusion

Target: `/home/henrik/projects/helse/oura-hsct-digital-twin/` (Python 3.12, `.venv/`, `data/oura.db`)

Generated 2026-04-16 by deep-research planning agent.

## Environment observations
- GPU present: RTX 4060 8 GB (via WSL2), Ryzen 9 9950X, 117 GB RAM
- Existing `.venv` had CPU-only torch; now upgraded to 2.7.1+cu126 (GPU works)
- `requirements-full.txt` already pins chronos-forecasting, but wasn't installed in venv until now
- 31 tables in `oura.db` today, 30 analysis scripts, 7 ingesters already scaffolded (Oura, OMRON, Libre, Checkme O2, Contec ABPM, Viatom ECG, symptoms)

## 1. Time-Series Foundation Model Integration

### 1.1 Current state
- `analysis/analyze_oura_foundation_models.py` already references `amazon/chronos-bolt-base` and falls back to ARIMA if chronos not installed
- Baseline: pykalman UKF, ARIMA in statsmodels, curve-fit in `analyze_rux_forecast`

### 1.2 Model recommendations

| Model | Task | Install | Size | CPU viable |
|---|---|---|---|---|
| Chronos-Bolt (base, small) | Univariate BP/HRV/HR forecasts with quantiles | `pip install chronos-forecasting` | 205M / 48M | Yes, <2s/forecast |
| Moirai-2 (MoE base) | Multivariate joint forecast across BP+HR+HRV+SpO2+CGM | `pip install uni2ts` | 91M active | Slower, 5-15s |
| Lag-Llama | Probabilistic univariate with best PI calibration | GitHub only | 200MB | Yes |
| TimesFM v2 | Alt univariate baseline, Apache-2.0 | `pip install timesfm[torch]` | 200M | Slow |
| MIRA | Potential clinical backbone (arxiv 2506.07584) | From source | 500MB | Uncertain |

### 1.3 Task → model mapping
- **BP 7/30d forecast with bisoprolol dose-response**: Chronos-Bolt per channel (SYS/DIA/MAP); pre-drug vs full context → counterfactual proxy
- **Multivariate joint (BP+HR+HRV+SpO2+glucose)**: Moirai-2 (only model with cross-variate attention)
- **HRV recovery**: keep curve-fit baseline + Lag-Llama quantile overlay
- **ECG waveform**: don't use these — use ECG-FM or HeartBEiT for 12-lead TH12

### 1.4 Install recipe (separate file to keep base venv small)
```
# requirements-fm.txt
chronos-forecasting>=1.5,<2
uni2ts>=2.0,<3
timesfm[torch]>=2.0,<3
stumpy>=1.13
neurokit2>=0.2.10
causalpy>=0.5
# lag-llama from git (no PyPI)
```

For GPU: `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu121` — RTX 4060 8GB fits Moirai-2 base.

## 2. Multi-Modal Fusion — 3 Layers

### 2.1 Schema layer — RECOMMEND: DuckDB over SQLite (option B)
Rejected:
- **a) Materialized view in SQLite**: not truly materialized, recomputed every query, fragile with 10+ tables/resolutions
- **c) Ad-hoc pandas joins**: every script reimplements alignment, slow at CGM scale

Implementation:
- Keep writes → `data/oura.db` (unchanged)
- New `analysis/_frames.py` opens via DuckDB sqlite_scanner
- Three public loaders: `load_frame(cadence, start, end, cols)`, `load_events()`, `load_raw()`
- Persist rollups in `data/analytic.duckdb`, rebuilt nightly

### 2.2 Feature layer
`load_frame("1d")` returns wide DataFrame with DatetimeIndex and columns like:
```
oura_rmssd, oura_hr_rest, oura_spo2_avg, oura_sleep_efficiency,
oura_tempdev, oura_stress_high,
bp_sys_am, bp_dia_am, bp_sys_pm, bp_dia_pm, bp_sys_cv, bp_ihb_count,
abpm_sys_24h, abpm_dia_24h, abpm_dipping_ratio, abpm_bp_load_pct,
cgm_mean, cgm_cv, cgm_tir_70_180, cgm_tar_180, cgm_mage,
checkme_spo2_nadir, checkme_odi4, checkme_t90_min,
ecg_afib_burden_pct, ecg_pvc_count, ecg_pac_count,
drug_rux_day, drug_beta_day, drug_rux_cum_dose_mg
```

**Baseline z-scoring**: freeze mean/SD on `[DATA_START, TREATMENT_START]` (already defined in config.py). Store in `v_baselines` view. Never recompute — prevents data leakage.

### 2.3 Model layer
- **Anomaly**: STUMPY univariate matrix profile + `stumpy.mstump` multivariate
- **Forecast**: Chronos-Bolt per critical channel; Moirai-2 on multivariate block once ≥60d of all-modality data (late May 2026)
- **Causal**: replace pycausalimpact with CausalPy Bayesian ITS. Interventions: rux 2026-03-16, bisoprolol 2026-04-08, Libre start ≈2026-04-21

## 3. Alignment / Resampling Strategy

### 3.1 Three analytic frames

| Frame | Cadence | Time range | Purpose |
|---|---|---|---|
| frame_1min | 1 min | Rolling 14 days | Intra-event (glucose→HR, nocturnal SpO2→ECG) |
| frame_1h | 1 hour | Last 180 days | Circadian, postprandial windows, BP triplicate buckets |
| frame_1d | 1 day | Full history | Trend, causal ITS, foundation-model input |

### 3.2 Resampling rules (explicit, enforced)

| Modality | Native | → 1-min | → 1-h | → 1-day |
|---|---|---|---|---|
| Oura HR (5-min) | 5m | linear, max gap 10m | mean | mean/min/max/P5 (resting) |
| Oura HRV (5-min) | 5m | linear, max gap 10m | mean or recompute from RR | nocturnal mean (02-06), CV |
| Oura SpO2 | daily | forward-fill | forward-fill | source |
| BP home (OMRON) | sporadic | **NO upsample** (NaN) | ffill max 4h | triplicate-median per AM/PM, IHB=any |
| ABPM (15-30m) | 15-30m | ffill max 30m | mean | day/night means, dipping, BP load |
| CGM (1m) | 1m | source | mean, CV, MAGE, TIR | mean, CV, MAGE, TIR 70-180, TAR 180, TBR 70 |
| Checkme SpO2 (1-8s) | 1-8s | downsample to 1m mean | mean, nadir, ODI4 | nadir, T90, ODI4, sleep-window mean |
| TH12 ECG events | per-beat | count per 1m bin | AFib burden, PVC, PAC count | burden %, episode count, longest duration |
| TH12 ECG raw (500Hz) | 500Hz | do NOT load to frames — parquet on-demand | — | — |
| Symptom/meal events | sparse | events | count per h | count per d, severity max |

### 3.3 Enforcement
- `assert_cadence(df, expected)` in `_frames.py` raises on mismatch
- New scripts MUST use `load_frame()`; migrate existing 30 scripts one per week

## 4. Cross-Modal Join Recipes (6)

Each is a single named loader in `analysis/_joins.py` returning a tidy df.

### 4.1 Glucose response to meal
Window `[-15m, +180m]`, Δ/peak/time-to-peak/AUC₀₋₁₂₀. Exclude first 24h of a new sensor (warm-up drift).

### 4.2 Postprandial HR/HRV/BP
Per-meal 3-hour window on 5-min grid. Use case: ruxolitinib postprandial tachycardia pattern.

### 4.3 Nocturnal hypoxia → AFib
Per-night conditional logistic regression. For each AFib onset: preceding 30-min SpO2 nadir, T90, ODI4.

### 4.4 Morning BP surge vs prior-night sleep quality
Surge = mean(0-2h post-wake BP) - mean(last hour of sleep BP). Needs ABPM.

### 4.5 Drug-effect causal windows
Per (drug, modality) pair: CausalPy `InterruptedTimeSeries` or `SyntheticControl`. JSON output.

### 4.6 Exercise response
Per-workout `[start-15m, end+120m]`. HR recovery slope 1-5min post (bisoprolol pharmacodynamic marker), peak HR vs age-predicted, glucose drop, ECG ectopy.

## 5. STUMPY Anomaly Detection

### 5.1 Per-modality matrix profile (daily)
Pattern lengths: HRV m=7 and m=14, SpO2 m=14, resting HR m=7, glucose CV m=7, temperature m=14. Flag top 5% as discords.

### 5.2 Multivariate mSTUMP
Feed z-scored feature matrix excluding per-column outliers. m=7. Threshold: mean + 2.5·MAD on last 60 days.

### 5.3 Integration
New module `analysis/analyze_multimodal_anomalies.py`. JSON + HTML. Weekly tracker consumes JSON (doesn't re-run STUMPY).

## 6. Foundation Model Evaluation Framework

### 6.1 Backtest protocol
- Expanding-window walk-forward, `[t+1, t+14]` forecast per day
- Metrics per model per channel: MAPE/RMSE/bias + CRPS + coverage at nominal 80/90/95% PI + wall-clock + VRAM
- Store in `fm_backtest_runs` table in analytic.duckdb

### 6.2 Decision rule
Pick Pareto-best on (CRPS, coverage gap < 5pp, runtime < 5s/forecast on CPU). Reassess monthly.

## 7. Implementation Roadmap

### Phase 1 — This week (2026-04-16 → 2026-04-22)
1. Install stumpy, neurokit2, duckdb, polars (already done)
2. Create `analysis/_frames.py` with `load_frame("1d")` + `load_frame("1h")` over Oura + OMRON
3. Migrate `analyze_oura_anomalies.py` to use `load_frame`; add STUMPY univariate matrix profile
4. Upgrade HRV: replace manual time-domain in `analyze_oura_advanced_hrv.py` with `neurokit2.hrv_time()` / `.hrv_frequency()`
5. New `analyze_multimodal_anomalies.py` with mSTUMP
6. Add to run_all.py SCRIPTS

### Phase 2 — 2026-04-23 → 2026-05-06 (Libre inserts 2026-04-21)
1. Add `requirements-fm.txt`; install chronos-forecasting + causalpy (done)
2. GPU torch already working
3. Replace univariate forecast in `analyze_rux_forecast.py` with Chronos-Bolt overlay + CRPS
4. New `analyze_bp_forecast.py`: Chronos-Bolt on OMRON SYS/DIA/MAP, 7/30 day horizons
5. Replace pycausalimpact in `analyze_oura_causal.py` with CausalPy Bayesian ITS for (rux, bisoprolol) × (rmssd, resting HR, sys BP, efficiency). Kill the applymap monkey-patch (line 43-44)
6. Wire `analysis/_fm.py` evaluation, emit `fm_leaderboard.html`

### Phase 3 — Month 1 (2026-05-07+)
1. Install uni2ts (Moirai-2). Add `analyze_multivariate_forecast.py`
2. Build full frame_1d with all modalities; freeze `v_baselines`
3. Moirai-2 on 5-channel tensor (rmssd, rest_hr, sys_bp, spo2_nadir, cgm_cv)
4. Wire joins 4.1-4.5

### Phase 4 — Month 2+ (TH12 arrives)
1. ECG raw → `data/ecg/raw/*.parquet` (500Hz, 12-lead, int16). Events → `viatom_ecg_events`
2. Evaluate ECG-FM on TH12 snippets; compare to device's on-device AI
3. Join 4.3 (Checkme SpO2 → AFib) conditional logistic
4. Join 4.6 (exercise response)
5. Consider MIRA if weights released with usable license

## 8. Risks / Constraints

### 8.1 Context windows
- Chronos-Bolt: 2048 tokens (~5.6y of daily data)
- Chronos-Bolt at 1-min: 34h — insufficient for multi-day patterns
- Moirai-2 base: 5000 tokens/variate
- Lag-Llama: 1024 default
- **Rule**: never feed raw 1-min directly; downsample to cadence sized for context

### 8.2 Timing budgets on 9950X CPU (90-day daily context, 14-day forecast)
- Chronos-Bolt-base: 1-2s
- Chronos-Bolt-small: 0.5s
- Moirai-2 base (5 channels): 8-15s
- Lag-Llama: 3-5s
- TimesFM v2: 2-4s

GPU cuts by ~5x single / 20x batched. Full leaderboard backtest (60d × 4 models × 5 channels = 1200 forecasts): 4h CPU vs 15m GPU.

### 8.3 Data leakage guards
- Baseline z-scoring frozen at DATA_START..TREATMENT_START, never recomputed
- STUMPY discord scores don't feed forecasters predicting within same window
- FM backtest: context ends strictly before forecast start (enforced in `_fm.py::backtest`)
- Drug-start covariates: only include indicators where start date ≤ t
- Sleep-period boundaries: document convention in `_frames.py` docstring (current pipeline: sleep assigned to wake-up date)

### 8.4 Version pinning
Pin exact versions in `requirements-fm.txt`. Maintain `requirements-fm.lock`. Re-evaluate quarterly.

### 8.5 Concurrency
DuckDB reads oura.db read-only; importers own writes. Keep pipeline serial (current behavior).

## 9. Open Questions (explicit, resolve before relevant phase)

1. Libre 3 Plus export format shape (LibreView CSV for Plus variant) — verify against `scripts/test_import_glucose.py`
2. Checkme O2 Max Pro export path — CSV over USB vs Viatom app — `import_checkme_spo2.py` untested against live data
3. TH12 waveform export — EDF/EDF+ standard or proprietary Viatom binary? 500Hz figure is assumed
4. Moirai-2 license — CC-BY-NC-SA 4.0 on some model variants. Personal use OK, share with Mitch = verify
5. MIRA weights + license — arxiv paper exists, official weights/license not verified
6. Lag-Llama `pip install git+` reliability in 2026-04 — not verified
7. GPU torch + numba (STUMPY backend) compat — retest after CUDA torch install
8. ABPM50 cadence — single-session vs continuous-home? Joins 4.4/4.5 may need rewrite
9. Mitch profile parity — `_frames.py` must work with `data/mitch.db`; Mitch schema ⊆ Henrik schema?
10. Nightly rebuild window — concurrency with Oura sync if CGM adds ~1.5M rows/year

## Architecture diagram

```
┌─── DEVICES ─────────────────────────────────────────────────────────────┐
│ Oura Ring  OMRON M7  Libre 3 Plus  ABPM50  Checkme O2  TH12 ECG  Logs  │
└──┬──┬──────┬───┬───────┬───┬───────┬───┬────────┬─────────┬────────────┘
   v  v      v   v       v   v       v   v        v         v
┌─── INGESTERS (api/import_*.py) ─────────────────────────────────────────┐
└──────────────────────────┬──────────────────────────────────────────────┘
                           v
┌─── RAW STORE: SQLite data/oura.db + parquet data/ecg/raw/ ──────────────┐
└──────────────────────────┬──────────────────────────────────────────────┘
                           v
┌─── ANALYTIC: DuckDB data/analytic.duckdb (nightly) ─────────────────────┐
│ v_frame_1min, v_frame_1h, v_frame_1d, v_baselines, v_join_* views       │
└──────────────────────────┬──────────────────────────────────────────────┘
                           v
┌─── FEATURES: analysis/_frames.py + _joins.py ───────────────────────────┐
└──┬──────────┬───────────┬───────────────┬───────────────────────────────┘
   v          v           v               v
┌─Stats──┐┌─STUMPY─┐┌─Foundation Models─┐┌─CausalPy─┐
│pykalman││univar  ││ Chronos-2          ││ ITS     │
│ARIMA   ││mstump  ││ Moirai-2           ││ SCM     │
│NeuroKit││        ││ Lag-Llama / TimesFM││ (replaces│
│ HRV    ││        ││                    ││ pycausal)│
└────┬───┘└────┬───┘└──────────┬─────────┘└────┬────┘
     └────────┴───────────────┴───────────────┘
                           v
┌─── REPORTS: HTML + JSON in reports/ ────────────────────────────────────┐
└──────────────────────────────────────────────────────────────────────────┘
```
