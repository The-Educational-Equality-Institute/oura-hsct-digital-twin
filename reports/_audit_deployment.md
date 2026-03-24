# Deployment & Pipeline Audit - 2026-03-24

## 1. run_all.py Orchestrator

**Status: SOLID**

### Script inventory
Lists 12 scripts in execution order:
1. `analyze_oura_spo2_trend.py`
2. `analyze_oura_full.py`
3. `analyze_oura_advanced_hrv.py`
4. `analyze_oura_sleep_advanced.py`
5. `analyze_oura_biomarkers.py`
6. `analyze_oura_anomalies.py`
7. `analyze_oura_foundation_models.py`
8. `analyze_oura_digital_twin.py`
9. `analyze_oura_causal.py`
10. `analyze_oura_gvhd_predict.py`
11. `generate_oura_3d_dashboard.py`
12. `generate_roadmap.py`

All 12 scripts exist in `analysis/`. No missing files.

### Timeout handling
- 600s (10 min) per script via `subprocess.run(timeout=600)` -- GOOD
- `subprocess.TimeoutExpired` caught explicitly -- GOOD

### Error recovery
- **FAIL-FAST design**: On any failure (non-zero exit, timeout, exception), all remaining scripts are marked SKIPPED and the loop breaks
- Exit code 1 on any failure -- correct for CI usage
- This is a deliberate design choice (not a bug) -- ensures partial broken state doesn't compound

### Send bundle assembly
- After all scripts succeed, `assemble_send_bundle()` copies 12 HTML + 11 JSON files to `reports/send_bundle/`
- Validates all expected files exist before copying (raises `FileNotFoundError` if missing)
- Strips internal metadata: removes `progress_log` from GVHD JSON, removes `date_stamped_html` from full analysis JSON
- Cleans stale files from send_bundle that aren't in the expected set
- Writes `SEND_MANIFEST.md`

### Minor observations
- `SEND_BUNDLE_JSON` has 11 entries but `SEND_BUNDLE_HTML` has 12 (roadmap has no JSON counterpart) -- this is intentional, roadmap is presentation-only
- `capture_output=False` means stdout/stderr go to terminal, not captured -- correct for interactive use, logs are visible
- `PYTHONUNBUFFERED=1` set in subprocess env -- good for real-time output

---

## 2. Reports Directory

**Status: ALL 12 REPORTS PRESENT**

### HTML reports (15 files)
| File | Size | Status |
|------|------|--------|
| `oura_full_analysis.html` | 493 KB | OK |
| `advanced_hrv_analysis.html` | 3.0 MB | OK (large due to high-res Plotly traces) |
| `advanced_sleep_analysis.html` | 76 KB | OK |
| `composite_biomarkers.html` | 117 KB | OK |
| `spo2_bos_screening.html` | 123 KB | OK |
| `anomaly_detection_report.html` | 172 KB | OK |
| `foundation_model_report.html` | 131 KB | OK |
| `digital_twin_report.html` | 400 KB | OK |
| `causal_inference_report.html` | 696 KB | OK |
| `gvhd_prediction_report.html` | 166 KB | OK |
| `oura_3d_dashboard.html` | 668 KB | OK |
| `roadmap.html` | 55 KB | OK |
| `oura_full_analysis_20260323.html` | 361 KB | Historical snapshot |
| `oura_full_analysis_20260324.html` | 493 KB | Historical snapshot |
| `css_clinical_patterns.html` | 39 KB | Internal/demo artifact |

### JSON metrics (14 files)
All 11 expected bundle JSON files present, plus:
- `oura_full_analysis_20260323.json` (historical)
- `oura_full_analysis_20260324.json` (historical)
- `causal_timeseries.json` (internal generation input)

### Other files in reports/
- `DATA_METHODOLOGY.md` -- documentation
- `QA_JSON_HTML_CONSISTENCY_20260324.md` -- QA report
- `screenshots/` -- 8 PNG files for documentation
- `send_bundle/` -- curated release bundle (12 HTML + 11 JSON + manifest)
- Various `ss_*.png` screenshots -- internal development screenshots

### Send bundle
Complete. 12 HTML + 11 JSON + `SEND_MANIFEST.md`. All timestamps match latest pipeline run (2026-03-24 16:24-16:38).

---

## 3. Deployment Configuration

**Status: NO ACTIVE DEPLOYMENT**

### What exists
- `scripts/daily_pipeline.sh` -- cron-ready script (import -> analyze -> optional deploy)
- Deployment section is **commented out**: Astro + Cloudflare Pages via wrangler
- References `sync-reports.mjs`, `astro build`, `wrangler pages deploy`
- `.gitignore` includes `.wrangler/` and `index.html` suggesting past or planned Cloudflare Pages setup

### What doesn't exist
- No nginx config
- No GitHub Pages config (no `.github/workflows/`, no `CNAME`)
- No active Cloudflare deployment
- No Docker/container config

### Conclusion
Reports are generated locally. The daily pipeline cron is ready but the deployment step is commented out. Reports are shared via the `send_bundle/` directory (likely copied/emailed manually).

---

## 4. Data Symlinks

**Status: ALL VALID**

| Symlink | Target | Resolves to | Exists |
|---------|--------|-------------|--------|
| `data/oura.db` | `../../database/helseoversikt.db` | `/home/ovehe/projects/helseoversikt/database/helseoversikt.db` | YES |
| `data/helseoversikt.db` | `../../database/helseoversikt.db` | Same | YES |
| `data/investigation.db` | `../../database/investigation.db` | `/home/ovehe/projects/helseoversikt/database/investigation.db` | YES |

All three symlinks resolve correctly. `_config.py` includes runtime validation of symlink targets with clear error messages for dangling symlinks.

---

## 5. Dependencies (requirements.txt)

**Status: ALL IMPORTS COVERED, ONE NOTE**

### Core dependencies (always imported)
| Package | Required version | Used by |
|---------|-----------------|---------|
| plotly | >=5.18 | All 12 scripts |
| pandas | >=2.0,<3 | All 12 scripts |
| numpy | >=1.24 | All 12 scripts |
| scipy | >=1.11 | Multiple scripts |
| statsmodels | >=0.14 | Causal, biomarkers |
| scikit-learn | >=1.3 | Anomalies, GVHD |
| requests | >=2.31 | API import |
| python-dotenv | >=1.0 | API import |
| oura-ring | >=0.3 | API import |

### Optional dependencies (graceful fallback if missing)
| Package | Version | Script | Fallback |
|---------|---------|--------|----------|
| antropy | >=0.1 | HRV, biomarkers | Skips entropy metrics |
| nolds | >=0.5 | Advanced HRV | Skips nonlinear metrics |
| pycausalimpact | >=0.1.1 | Causal | Skips CausalImpact section |
| tigramite | 5.2.10.1 | Causal | Skips PCMCI+ section |
| stumpy | 1.14.1 | Anomalies | Skips Matrix Profile |
| tsfresh | 0.21.1 | Anomalies | Skips feature extraction |
| torch | 2.11.0 | Foundation, anomalies | Skips LSTM/Chronos |
| chronos-forecasting | 2.2.2 | Foundation models | Skips forecast section |
| filterpy | >=1.4 | Digital twin | Skips UKF analysis |
| pykalman | 0.11.2 | Digital twin | Required (no fallback) |
| ssm | 0.0.1 | GVHD prediction | Falls back to hmmlearn |
| hmmlearn | 0.3.3 | GVHD prediction | Fallback for ssm |
| cython | 3.2.4 | Build dependency for ssm | -- |

### Compatibility shim
`analyze_oura_causal.py` patches `DataFrame.applymap` for pandas>=3 compat with pycausalimpact (line 42-44). This is correct and necessary.

### Note on ssm install
`scripts/install_full_stack.sh` handles ssm's PEP517 build-isolation issue by installing it with `--no-build-isolation`. This is documented in requirements.txt comments.

---

## 6. API Key / Secret Exposure in Reports

**Status: NO LEAKS DETECTED**

### .env file
Contains real Oura API credentials (client ID, client secret, access tokens, refresh tokens for two users). This file is:
- Listed in `.gitignore` -- NOT committed to git
- File permissions: `-rw-------` (owner-only read/write) -- GOOD

### HTML reports scan
Searched all HTML reports for: `api_key`, `secret`, `token`, `password`, `credential`
- Only match: `roadmap.html` contains the words "OAuth token" in descriptive text (feature descriptions for the roadmap). NO actual credentials exposed.

### JSON metrics scan
No matches for credential-related strings in any JSON file.

### Analysis scripts
- No script reads from `.env` or `os.environ` for credential access
- `.env` is only used by `api/import_oura.py` (data import, not analysis)
- Analysis scripts only access the SQLite database, never the Oura API directly

### .env.example
Clean template with empty values. Safe to commit (already committed).

---

## Summary

| Check | Status | Notes |
|-------|--------|-------|
| run_all.py correctness | PASS | All 12 scripts listed, all exist |
| Timeout handling | PASS | 600s per script, explicit catch |
| Error recovery | PASS | Fail-fast with clear skip reporting |
| All reports present | PASS | 12/12 HTML + 11/11 JSON + roadmap |
| Send bundle complete | PASS | Curated, metadata stripped |
| Deployment config | INFO | No active deployment; cron-ready pipeline exists |
| Data symlinks valid | PASS | All 3 resolve correctly |
| Dependencies covered | PASS | All imports have matching requirements.txt entries |
| Graceful fallbacks | PASS | Optional deps handled with try/except |
| API key exposure | PASS | No credentials in reports |
| .env gitignored | PASS | Not committed, owner-only permissions |
| config.py gitignored | PASS | Contains patient dates, not committed |

### No action required
The pipeline and deployment setup are clean. The only "deployment" is local generation + manual sharing via send_bundle. All reports are current (2026-03-24). No security issues found in generated output.
