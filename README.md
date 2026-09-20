# Oura Digital Twin

Exploratory wearable-analysis toolkit for post-transplant monitoring with Oura Ring data.

Single-patient case study. The observation window is computed from `config.py` on every run: it starts at `DATA_START`, the intervention anchor is `TREATMENT_START`, and each generated page prints the window it actually modelled in its header. The known HEV confound (`HEV_DIAGNOSIS_DATE`) is carried into every report that discusses the intervention.

**Live reports:** [digital-twin.theeducationalequalityinstitute.org](https://digital-twin.theeducationalequalityinstitute.org)

## What This Does

Turns raw Oura Ring API data into clinically interpretable reports for post-HSCT (stem cell transplant) monitoring. The pipeline runs 36 scripts (28 analysis modules and their report generators) against a SQLite database and produces interactive HTML dashboards with machine-readable JSON metrics.

Every number a page prints is checked against the JSON its generator wrote, before anything is published. See [Gates](#gates).

| Script | Method | Output |
|--------|--------|--------|
| `analyze_oura_full.py` | Comprehensive Oura Ring Analysis | `oura_full_analysis.html` |
| `analyze_oura_advanced_hrv.py` | Advanced HRV & Autonomic Function Analysis | `advanced_hrv_analysis.html` |
| `analyze_oura_sleep_advanced.py` | Advanced Sleep Architecture Analysis | `advanced_sleep_analysis.html` |
| `analyze_oura_biomarkers.py` | Novel Composite Biomarker Scores for Post-HSCT Monitoring | `composite_biomarkers.html` |
| `analyze_oura_spo2_trend.py` | SpO2 Trend Analysis for BOS Early Detection in Post-HSCT Patient | `spo2_bos_screening.html` |
| `analyze_oura_anomalies.py` | ML-Powered Anomaly Detection Engine for Oura Ring Biometric Streams | `anomaly_detection_report.html` |
| `analyze_oura_foundation_models.py` | Foundation Model Time Series Analysis for Oura Ring Biometric Data | `foundation_model_report.html` |
| `analyze_oura_digital_twin.py` | Bayesian Cardiovascular Digital Twin from Oura Ring Data | `digital_twin_report.html` |
| `analyze_oura_causal.py` | Causal Inference Engine for Oura Ring Biometric Data | `causal_inference_report.html` |
| `analyze_oura_gvhd_predict.py` | GVHD Flare Prediction System from Oura Ring Wearable Data | `gvhd_prediction_report.html` |
| `generate_oura_3d_dashboard.py` | generate_oura_3d_dashboard.py - Unified CMO Dashboard | `oura_3d_dashboard.html` |
| `analyze_comparative_autonomic.py` | Module 1: Autonomic Recovery Trajectories | `comparative_autonomic_report.html` |
| `analyze_comparative_treatment.py` | Module 2: Treatment Response Detection | `comparative_treatment_response.html` |
| `analyze_comparative_sleep.py` | Module 3: Sleep Architecture as Health Signal | `comparative_sleep_analysis.html` |
| `analyze_comparative_coupling.py` | Module 4: Activity-Recovery Coupling | `comparative_activity_recovery_coupling.html` |
| `analyze_comparative_anomalies.py` | Module 5: Anomaly Pattern Comparison | `comparative_anomaly_report.html` |
| `analyze_comparative_breathing.py` | Module 6: Breathing Rate Analysis & BOS Screening | `comparative_breathing_analysis.html` |
| `analyze_comparative_temperature.py` | Comparative Temperature Deviation Analysis | `comparative_temperature_analysis.html` |
| `analyze_mitch_changepoints.py` | Patient 2 Changepoint Investigation | `mitch_changepoint_investigation.html` |
| `analyze_weekly_tracker.py` | Weekly Trend Tracker - This Week vs Last Week | `weekly_tracker.html` |
| `analyze_rux_forecast.py` | Ruxolitinib Dose-Response Forecast | `rux_forecast.html` |
| `analyze_piecewise_its.py` | Piecewise Interrupted Time Series (ITS) Regression with AR(1) Errors | `piecewise_regression.html` |
| `analyze_sequential_ci.py` | Sequential CausalImpact Analysis - Isolate Jakavi vs. Beta-Blocker Effects | `sequential_causal_impact.html` |
| `analyze_placebo_tests.py` | Placebo / Falsification Tests for Oura HSCT Digital Twin | `placebo_calibration.html` |
| `analyze_tau_u.py` | Tau-U Effect Size Analysis for Single-Case Experimental Design (SCED) | `tau_u_effects.html` |
| `analyze_omron_bp.py` | OMRON M7 Intelli IT AFib - blood pressure analysis | `omron_bp_report.html` |
| `analyze_multimodal_anomalies.py` | Multi-modal anomaly detection via STUMPY matrix profiles | `multimodal_anomaly_report.html` |
| `generate_roadmap.py` | Generate the What's Next page (reports/roadmap.html) | `roadmap.html` |
| `generate_research_synthesis.py` | Generate the Research Synthesis HTML report: Two-Hit Autonomic Recovery | `research_synthesis.html` |
| `generate_treatment_report.py` | Treatment Response Report -- primary clinical report for physicians | `treatment_response_report.html` |
| `generate_404.py` | Generate reports/404.html | `404.html` |
| `generate_how_built.py` | Generate reports/how_built.html: who did what, and how a number gets onto a page | `how_built.html` |
| `generate_anthropic_case.py` | Anthropic outreach case page: a bespoke, light, clinical-premium single page | `anthropic_case.html` |
| `generate_index.py` | Generate the Dashboard homepage (index.html) for the Oura Digital Twin | `index.html` |

## Gates

Nothing reaches the build directory unchecked. `run_all.py` runs all three after the analyses, and both `scripts/daily-pipeline-local.sh` and `scripts/deploy-local.sh` stop before the deploy step if any of them fails.

| Gate | What it enforces | Where |
|------|------------------|-------|
| Claim audit | Every p-value, correlation, R² and Cohen's d printed in a page matches the JSON its generator wrote. The JSON is the authority. | `analysis/statcheck_reports.py` → `reports/statcheck_audit.json`, `reports/claims.html` |
| Link crawl | Every internal `href`/`src` in the build resolves to a real file, and `404.html` exists so Cloudflare Pages stops answering unknown paths with `index.html` and HTTP 200. | `scripts/check-links.py` |
| Output smoke tests | Every registry page exists, is a complete render, contains no "module could not run" text, and its metrics JSON still carries the keys the page depends on. | `tests/analysis/test_report_outputs.py` |

`reports/claims.html` is published with the site: it lists every extracted claim with its JSON value and an OK / MISMATCH / UNMATCHED verdict.

`run_all.py` also writes `reports/run_summary.json` with per-script pass/fail and runtime.

## Key Findings

The pipeline re-computes all reported values from the current data on every run, so the findings live on the pages rather than in this file. See the [live reports](https://digital-twin.theeducationalequalityinstitute.org) for current outputs, the intervention window as modelled today, and the HEV caveat that applies to every treatment-effect claim.

## Structure

```
oura-digital-twin/
  config.example.py        Patient-specific constants (copy to config.py)
  config.py                Active config; DATABASE_PATH honours the env var
  profiles.py              Multi-patient profile registry
  .env.example             API credentials template (copy to .env)
  run_all.py               Analysis pipeline runner + claim-audit gate
  run_tests.py             Test suite runner (all tests/**/test_*.py)
  requirements.txt         Core dependencies
  requirements-full.txt    Full stack (optional backends included)
  pyproject.toml           Ruff linter configuration
  CHANGELOG.md             Dated record of changes
  api/
    oura_oauth2_setup.py   OAuth2 authorization flow
    import_oura.py         Oura API -> SQLite importer
    import_glucose.py      FreeStyle Libre / LibreView CSV importer
    import_viatom_ecg.py   Viatom TH12 12-lead ECG Holter importer
    import_contec_abpm.py  Contec ABPM50 24h ambulatory BP importer
    import_checkme_spo2.py Checkme O2 Max continuous SpO2 importer
    import_omron.py        OMRON M7 (HEM-7380T1) AFib BP importer
    import_symptom.py      Manual symptom-event CLI logger
    _ingest_common.py      Shared CSV/DB helpers for all importers
  analysis/
    _theme.py              Shared HTML/CSS design system + page registry
    _frames.py             Canonical analytic frames (DuckDB over the SQLite DB)
    _config.py             Backwards-compat config re-export
    _hardening.py          Numerical stability utilities
    statcheck_reports.py   Claim audit: every printed number vs its JSON
    analyze_oura_*.py      Oura-source analyses
    analyze_comparative_*.py  Cross-patient comparative analyses
    analyze_glucose_autonomic_coupling.py  CGM × Oura HRV coupling
    generate_*.py          Dashboard, roadmap and page generators
  docs/
    MULTI_DEVICE_INGEST.md Multi-device ingest guide
    checkme_o2_max_integration_plan.md  Vendor-format research notes
  tests/
    ingest/                Importer unit tests (79 tests, stdlib only)
    analysis/              Report-output smoke tests (reads reports/, never regenerates)
  scripts/
    daily-pipeline-local.sh  Daily import + analysis + gates + deploy (systemd timer)
    deploy-local.sh          Manual regenerate + gates + deploy
    check-links.py           Internal-link and 404.html gate for the build
    oura-reauth.py           One-time Oura OAuth re-authorization
    install_full_stack.sh    Full dependency installer (handles ssm/Cython)
  data/demo.db             Demo dataset (real Oura data, included)
  data/oura.db             Your own data (created by importer, gitignored)
  reports/                 Generated output (gitignored, see live site)
  reports/cgm_hypotheses_pre_registered.md  Pre-registered H1-H4 for CGM trial
```

## Quick Start (demo data included)

```bash
git clone https://github.com/The-Educational-Equality-Institute/oura-hsct-digital-twin.git
cd oura-hsct-digital-twin
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp config.example.py config.py
python run_all.py
```

No OAuth setup needed. The repo includes `data/demo.db` with real Oura Ring data (post-HSCT). The bundled snapshot covers 2026-01-08 to 2026-09-15 (240 nights with a scored long-sleep period). Reports appear in `reports/`.

With `requirements.txt` alone, the optional-backend modules degrade gracefully and say so on the page. Install `requirements-full.txt` for the full set.

## Setup (your own data)

1. Create and activate a Python 3.10+ virtual environment
2. Install dependencies (pick one):
   ```bash
   # Option A: Core only (recommended for most users)
   pip install -r requirements.txt

   # Option B: Full stack (all optional backends)
   pip install -r requirements-full.txt

   # Option C: Full stack including ssm rSLDS backend (needs C compiler)
   bash scripts/install_full_stack.sh
   ```
3. Copy `.env.example` to `.env` and add your Oura API credentials:
   ```bash
   cp .env.example .env
   # Edit .env - add your Personal Access Token or OAuth2 credentials
   ```
4. Copy and edit the config file for your patient:
   ```bash
   cp config.example.py config.py
   # Edit config.py - update dates and thresholds
   ```
5. Import your Oura data:
   ```bash
   python api/import_oura.py --days 90
   ```
6. Run the full pipeline:
   ```bash
   python run_all.py
   ```

For OAuth2 (multi-user / refresh token flow), run `python api/oura_oauth2_setup.py` before importing. If an existing refresh token is rejected, `python scripts/oura-reauth.py` prints an authorization URL to open in a browser and catches the callback.

## Running

Single analysis:
```bash
python analysis/analyze_oura_causal.py
```

Full pipeline (all 36 scripts):
```bash
python run_all.py
```

`DATABASE_PATH` selects the database; it defaults to `data/oura.db`:
```bash
DATABASE_PATH=data/demo.db python run_all.py
```

If data is missing or empty, `run_all.py` stops at precheck with an actionable import command instead of emitting long per-script tracebacks. It exits non-zero and skips the send bundle when the claim audit does not pass.

All output goes to `reports/`.

## Multi-Device Monitoring (optional)

Beyond Oura, the pipeline supports importing data from additional monitoring
devices into the same SQLite database, enabling cross-modal analyses (e.g. CGM
× HRV coupling, ECG events at glucose spikes, SpO2 desaturations × sleep phase).

| Device | Modality | Importer |
|---|---|---|
| FreeStyle Libre 3 Plus | Glucose (1/min) | `api/import_glucose.py` |
| Viatom TH12 | 12-lead ECG Holter + AI events | `api/import_viatom_ecg.py` |
| Contec ABPM50 | 24h ambulatory BP | `api/import_contec_abpm.py` |
| Checkme O2 Max | Continuous SpO2 | `api/import_checkme_spo2.py` |
| OMRON M7 (HEM-7380T1) | Home BP + AFib detection | `api/import_omron.py` |
| Manual symptom events | CLI logger | `api/import_symptom.py` |

Each importer supports `--init-only` (schema only), `--csv` (real import), and
`--profile` (multi-patient). See [`docs/MULTI_DEVICE_INGEST.md`](docs/MULTI_DEVICE_INGEST.md)
for per-device commands, schema details, and the format-validation workflow.

## Tests

```bash
python run_tests.py
```

Two suites, both stdlib-only:

- `tests/ingest/` - 79 importer unit tests.
- `tests/analysis/test_report_outputs.py` - smoke tests over the reports already in `reports/`. It reads outputs and never regenerates them, so run the pipeline first.

## Configuration

All patient-specific constants live in `config.py`. Start from the template:

```bash
cp config.example.py config.py
```

Key fields to set:

```python
TRANSPLANT_DATE = date(2023, 1, 1)       # Major clinical event / baseline anchor
TREATMENT_START = date(2026, 1, 15)      # Intervention start (for causal analysis)
KNOWN_EVENT_DATE = date(2026, 1, 10)     # Known acute episode (validation anchor)
DATA_START = date(2026, 1, 1)            # Analysis window start
PATIENT_AGE = 40
PATIENT_LABEL = "Patient"
```

`DATABASE_PATH` defaults to `data/oura.db` and is overridden by the environment
variable of the same name, which is how the pipeline scripts select a database.

Clinical reference thresholds (ESC, population norms) and the full visual identity (colors, Plotly layout) are also defined there and shared across all scripts.

## Dependencies

Core (data import, basic analyses):
```
pip install -r requirements.txt
```

Full stack (all optional backends - HRV entropy, Kalman, HMM, Chronos, STUMPY, tsfresh, DuckDB frames):
```
pip install -r requirements-full.txt
```

Full stack with `ssm` rSLDS backend (requires Cython build toolchain):
```bash
bash scripts/install_full_stack.sh          # includes ssm
bash scripts/install_full_stack.sh --no-ssm # skip ssm, use hmmlearn fallback
```

The install script handles the `ssm==0.0.1` Cython build isolation issue,
checks Python version, verifies imports after install, and provides clear
error messages at each step. Run `bash scripts/install_full_stack.sh --help`
for details.

What each optional backend unlocks:

- `nolds`, `antropy` - nonlinear dynamics (DFA, Lyapunov, entropy) for advanced HRV
- `filterpy`, `pykalman` - Kalman / UKF components for the digital twin
- `hmmlearn` - hidden Markov flare-state model for GvHD prediction (`ssm` is the optional rSLDS upgrade)
- `pycausalimpact` - Bayesian causal inference
- `chronos-forecasting` + `torch` - foundation-model forecasting
- `torchvision` - not used for analysis, but `transformers` performs import-time checks; install the build matching your `torch`
- `tigramite` - PCMCI+ causal discovery
- `stumpy` - Matrix Profile anomaly detection
- `duckdb` - canonical analytic frames (`analysis/_frames.py`), required by the multimodal anomaly module

Chronos runs on CPU by default. Set `CHRONOS_DEVICE=cuda` to opt in to the GPU when you know the VRAM is free.

## Reports

Running the pipeline generates 37 current self-contained HTML pages plus their JSON metrics in `reports/`. The assembled build additionally carries 64 dated snapshot pages kept from earlier runs, 101 HTML files in total. Each HTML file includes inline Plotly JS and opens directly in any browser.

**Live example:** [digital-twin.theeducationalequalityinstitute.org](https://digital-twin.theeducationalequalityinstitute.org)

The page registry - which pages exist and which JSON file is authoritative for each - is `HTML_TO_JSON` in `analysis/statcheck_reports.py`. The navigation registry is `REPORT_REGISTRY` in `analysis/_theme.py`; the smoke tests fail if the two disagree.

| Group | Page | What it holds |
|-------|------|---------------|
| Core | `index.html` | Current dashboard, report directory, and live status overview. |
| Core | `oura_full_analysis.html` | Heart rate, HRV, sleep, activity, SpO2, and readiness trends across the full observation window. |
| Core | `composite_biomarkers.html` | Composite biomarker indices combining multiple Oura signals into research-use summary scores. |
| Core | `advanced_sleep_analysis.html` | Sleep architecture, staging distribution, efficiency, and circadian rhythm analysis. |
| Core | `weekly_tracker.html` | One-page weekly tracker with watchpoints, week-over-week deltas, and clinician-style summary text. |
| Clinical | `causal_inference_report.html` | Bayesian causal impact and interrupted time-series analysis of ruxolitinib response. |
| Clinical | `gvhd_prediction_report.html` | Hidden Markov and state-space models predicting GvHD flare probability from wearable signals. |
| Clinical | `spo2_bos_screening.html` | SpO2 trend monitoring and bronchiolitis obliterans syndrome screening thresholds. |
| Clinical | `rux_forecast.html` | Near-term HRV and heart-rate recovery forecast from the current post-treatment trajectory. |
| Clinical | `research_synthesis.html` | Two-hit autonomic recovery hypothesis with live KPIs, timeline, and statistical evidence. |
| Clinical | `treatment_response_report.html` | Primary clinical report covering all systems and both medicines for specialist review. |
| Advanced | `advanced_hrv_analysis.html` | Frequency-domain HRV, Poincare plots, DFA, sample entropy, and autonomic balance metrics. |
| Advanced | `digital_twin_report.html` | Unscented Kalman Filter digital twin tracking latent inflammatory and autonomic states. |
| Advanced | `foundation_model_report.html` | Chronos foundation model forecasting with prediction intervals and anomaly scoring. |
| Advanced | `anomaly_detection_report.html` | Matrix Profile, Isolation Forest, and CUSUM anomaly detection across biometric channels. |
| Advanced | `oura_3d_dashboard.html` | Interactive 3D scatter of sleep, HRV, and activity with treatment phase coloring. |
| Statistical | `piecewise_regression.html` | Piecewise ITS regression with AR(1) errors: two-intervention model with date sensitivity analysis. |
| Statistical | `sequential_causal_impact.html` | Sequential Bayesian CausalImpact isolating Jakavi and beta-blocker effects in separate runs. |
| Statistical | `placebo_calibration.html` | Falsification tests at 20 random pre-treatment dates to calibrate false positive rates. |
| Statistical | `tau_u_effects.html` | Tau-U and NAP effect sizes for single-case experimental design with baseline trend correction. |
| Comparative | `comparative_autonomic_report.html` | HRV and resting HR recovery trajectories compared between post-HSCT and post-stroke patients. |
| Comparative | `comparative_treatment_response.html` | Changepoint detection and pre/post treatment response with Mann-Whitney U tests. |
| Comparative | `comparative_sleep_analysis.html` | Sleep architecture, efficiency, and timing compared against clinical benchmarks. |
| Comparative | `comparative_activity_recovery_coupling.html` | Activity-recovery coupling analysis: does day N activity predict day N+1 recovery? |
| Comparative | `comparative_anomaly_report.html` | Anomaly fingerprinting and clustering: how bad days manifest differently. |
| Comparative | `comparative_breathing_analysis.html` | Respiratory-rate trends, week-over-week shifts, and outlier nights against recent baseline. |
| Comparative | `comparative_temperature_analysis.html` | Temperature deviation tracking, excursion alerts, and post-treatment change patterns. |
| Comparative | `mitch_changepoint_investigation.html` | Patient 2 changepoint scan of HRV, sleep, and recovery markers around key timeline events. |
| Individual | `mitch_standalone_report.html` | Post-stroke patient P2: HRV, HR, sleep, and activity dashboard. |
| Individual | `wenche_standalone_report.html` | Healthy control P3: baseline HRV, HR, sleep, and activity reference. |
| Context | `roadmap.html#honest` | Methodology, limitations, and honest assessment of what this system can and cannot do. |
| Context | `roadmap.html#roadmap` | Planned analyses, validation targets, and next steps for the digital twin platform. |
| Context | `how_built.html` | Who did what, how a number gets onto a page, and how the pipeline checks itself. Measured, not claimed. |
| Context | `claims.html` | Every statistic printed on this site, cross-checked against the JSON the pipeline computed it from. |

## Methodology

All analyses are reproducible Python scripts running against a single SQLite database.

- **Causal inference**: CausalImpact (BSTS) with placebo falsification tests
- **Digital twin**: 5-state Kalman filter + UKF with EM-optimized parameters
- **Causal discovery**: PCMCI+ (tigramite) exploratory link discovery
- **GvHD prediction**: rSLDS with HMM fallback and retrospective alert validation
- **Anomaly detection**: Matrix Profile (STUMPY) + ensemble methods
- **Forecasting**: Amazon Chronos-bolt foundation model + ARIMA baseline
- **One row per date**: a night can carry more than one scored sleep period. Every daily matrix keeps the longest period per date, so no date is counted twice.

## Data Methodology

See [`reports/DATA_METHODOLOGY.md`](reports/DATA_METHODOLOGY.md) for detailed documentation of statistical methods, data completeness, known biases, and reproducibility notes.

## Built With

- [Oura Ring Gen 4](https://ouraring.com) - wearable biometric data
- [Claude Code](https://claude.ai/code) - AI-assisted development
- [Plotly](https://plotly.com/python/) - interactive visualizations
- [CausalImpact](https://pypi.org/project/pycausalimpact/) - Bayesian causal inference
- Python 3.10+ / pandas / scipy / statsmodels / scikit-learn (developed and run on Python 3.14.7)

## Disclaimer

Single-patient case study (N=1). Not validated for clinical decision-making. Not a medical device.

The HEV diagnosis is a potential confound that cannot be fully separated from the drug signal at this sample size. Every page that reports a treatment effect carries that caveat.

Not affiliated with, endorsed by, or sponsored by Oura Health Oy. Oura is a registered trademark of Oura Health Oy.

## License

MIT - see [LICENSE](LICENSE).
