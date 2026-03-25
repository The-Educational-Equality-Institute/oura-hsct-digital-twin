# Oura Digital Twin

Exploratory wearable-analysis toolkit for post-transplant monitoring with Oura Ring data.

Single-patient case study. Current window: 79 modeled days, 8 post-ruxolitinib days, known HEV confound diagnosed March 18, 2026.

**Live reports:** [digital-twin.theeducationalequalityinstitute.org](https://digital-twin.theeducationalequalityinstitute.org)

## What This Does

Turns raw Oura Ring API data into clinically interpretable reports for post-HSCT (stem cell transplant) monitoring. The pipeline runs 12 analysis scripts against a SQLite database and produces interactive HTML dashboards with machine-readable JSON metrics.

| Script | Method | Output |
|--------|--------|--------|
| `analyze_oura_full.py` | Comprehensive biometric overview | Interactive HTML dashboard |
| `analyze_oura_advanced_hrv.py` | Nonlinear HRV dynamics (MSE, RQA, DFA) | HTML + JSON |
| `analyze_oura_sleep_advanced.py` | Sleep architecture & HRV-sleep coupling | HTML + JSON |
| `analyze_oura_biomarkers.py` | Composite biomarker scores | HTML + JSON |
| `analyze_oura_spo2_trend.py` | SpO2 trend & BOS screening | HTML + JSON |
| `analyze_oura_anomalies.py` | 5-algorithm anomaly detection | HTML + JSON |
| `analyze_oura_foundation_models.py` | Amazon Chronos-bolt + ARIMA forecasting | HTML + JSON |
| `analyze_oura_digital_twin.py` | 5-state Kalman filter + UKF | HTML + JSON |
| `analyze_oura_causal.py` | CausalImpact, Granger, Transfer Entropy | HTML + JSON |
| `analyze_oura_gvhd_predict.py` | rSLDS flare-state modeling (HMM fallback) | HTML + JSON |
| `generate_oura_3d_dashboard.py` | 3D Plotly visualizations | Interactive HTML |
| `generate_roadmap.py` | Roadmap / appendix page | Interactive HTML |

## Key Findings

Current exploratory signals are visible in temperature deviation, sleep quality, HRV, and digital-twin summaries after ruxolitinib started on March 16, 2026, but the post-intervention window is still only 8 days and remains confounded by HEV diagnosed on March 18, 2026.

The pipeline re-computes all reported values from the current data on every run. See the [live reports](https://digital-twin.theeducationalequalityinstitute.org) for current outputs and caveats.

## Structure

```
oura-digital-twin/
  config.example.py        Patient-specific constants (copy to config.py)
  .env.example             API credentials template (copy to .env)
  run_all.py               Pipeline runner - all 12 scripts sequentially
  requirements.txt         Core dependencies
  requirements-full.txt    Full stack (optional backends included)
  pyproject.toml           Ruff linter configuration
  api/
    oura_oauth2_setup.py   OAuth2 authorization flow
    import_oura.py         Oura API -> SQLite importer
  analysis/
    _theme.py              Shared HTML/CSS design system
    _config.py             Backwards-compat config re-export
    _hardening.py          Numerical stability utilities
    statcheck_reports.py   QA - verify stats match between HTML and JSON
    analyze_oura_*.py      10 analysis scripts
    generate_*.py          2 dashboard/roadmap generators
  scripts/
    daily_pipeline.sh      Cron-ready daily import + analysis
    install_full_stack.sh  Full dependency installer (handles ssm/Cython)
  data/demo.db             Demo dataset (79 days of real Oura data, included)
  data/oura.db             Your own data (created by importer, gitignored)
  reports/                 Generated output (gitignored, see live site)
```

## Quick Start (demo data included)

```bash
git clone https://github.com/The-Educational-Equality-Institute/oura-hsct-digital-twin.git
cd oura-hsct-digital-twin
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.example.py config.py
python run_all.py
```

No OAuth setup needed. The repo includes `data/demo.db` with real Oura Ring data (79 days, post-HSCT). Reports appear in `reports/`.

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
   # Edit config.py - set DATABASE_PATH to data/oura.db, update dates and thresholds
   ```
5. Import your Oura data:
   ```bash
   python api/import_oura.py --days 90
   ```
6. Run the full pipeline:
   ```bash
   python run_all.py
   ```

For OAuth2 (multi-user / refresh token flow), run `python api/oura_oauth2_setup.py` before importing.

## Running

Single analysis:
```bash
python analysis/analyze_oura_causal.py
```

Full pipeline (all 12 scripts, ~35 seconds with core deps, ~2 minutes with full stack):
```bash
python run_all.py
```
If data is missing/empty, `run_all.py` now stops at precheck with an actionable import command
instead of emitting long per-script tracebacks.

All output goes to `reports/`.

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
PATIENT_AGE = 40
PATIENT_LABEL = "Patient"
```

Clinical reference thresholds (ESC, population norms) and the full visual identity (colors, Plotly layout) are also defined there and shared across all scripts.

## Dependencies

Core (data import, basic analyses):
```
pip install -r requirements.txt
```

Full stack (all optional backends - HRV entropy, Kalman, HMM, Chronos, STUMPY, tsfresh):
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

Optional extras (for advanced analyses beyond core):
```
pip install chronos-forecasting torch tigramite stumpy
```

- `nolds` — Nonlinear dynamics (DFA, Lyapunov) for advanced HRV
- `filterpy` — Kalman / UKF components for the digital twin
- `hmmlearn` — Hidden Markov Model for GVHD flare prediction
- `chronos-forecasting` + `torch` — Foundation model forecasting (GPU recommended)
- `tigramite` — PCMCI+ causal discovery
- `stumpy` — Matrix Profile anomaly detection

## Reports

Running the pipeline generates 12 self-contained HTML reports + JSON metrics in `reports/`. Each HTML file includes inline Plotly JS and opens directly in any browser.

**Live example:** [digital-twin.theeducationalequalityinstitute.org](https://digital-twin.theeducationalequalityinstitute.org)

| Report | Description |
|--------|-------------|
| `oura_full_analysis.html` | 12-section comprehensive dashboard |
| `advanced_hrv_analysis.html` | 80+ HRV metrics including Lomb-Scargle, multiscale entropy, RQA |
| `advanced_sleep_analysis.html` | Sleep architecture, Markov transitions, HRV-sleep coupling |
| `composite_biomarkers.html` | ADSI, GVHD risk, cardiac strain, immune recovery scores |
| `spo2_bos_screening.html` | SpO2 trends with BOS risk screening |
| `anomaly_detection_report.html` | Matrix Profile, Isolation Forest, LSTM, DBSCAN, statistical |
| `foundation_model_report.html` | Chronos-bolt + ARIMA forecasts |
| `digital_twin_report.html` | 5-state Kalman filter + UKF latent-state model |
| `causal_inference_report.html` | CausalImpact, Granger causality, Transfer Entropy |
| `gvhd_prediction_report.html` | 4-state flare-state model with retrospective alert validation |
| `oura_3d_dashboard.html` | Interactive 3D biometric visualizations |
| `roadmap.html` | What's next — family control, multi-user studies |

## Methodology

All analyses are reproducible Python scripts running against a single SQLite database.

- **Causal inference**: CausalImpact (BSTS) with placebo falsification tests
- **Digital twin**: 5-state Kalman filter + UKF with EM-optimized parameters
- **Causal discovery**: PCMCI+ (tigramite) exploratory link discovery
- **GvHD prediction**: rSLDS with HMM fallback and retrospective alert validation
- **Anomaly detection**: Matrix Profile (STUMPY) + ensemble methods
- **Forecasting**: Amazon Chronos-bolt foundation model + ARIMA baseline

## Data Methodology

See [`reports/DATA_METHODOLOGY.md`](reports/DATA_METHODOLOGY.md) for detailed documentation of statistical methods, data completeness, known biases, and reproducibility notes.

## Built With

- [Oura Ring Gen 4](https://ouraring.com) — wearable biometric data
- [Claude Code](https://claude.ai/code) — AI-assisted development
- [Plotly](https://plotly.com/python/) — interactive visualizations
- [CausalImpact](https://pypi.org/project/pycausalimpact/) — Bayesian causal inference
- Python 3.12 / pandas / scipy / statsmodels / scikit-learn

## Disclaimer

Single-patient case study (N=1). Not validated for clinical decision-making. Not a medical device.

HEV diagnosis (March 18) is a potential confound that cannot be fully separated from drug signal at current post-intervention sample size.

Not affiliated with, endorsed by, or sponsored by Oura Health Oy. Oura is a registered trademark of Oura Health Oy.

## License

MIT — see [LICENSE](LICENSE).
