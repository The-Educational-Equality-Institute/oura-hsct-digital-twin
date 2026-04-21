---
title: Repo inventory, what already exists
status: Authoritative
date: 2026-04-21
survey-method: 5 parallel read-only agents (F1 through F5)
purpose: Single source of truth for existing functionality. Read this BEFORE building anything new.
---

# Repo inventory: oura-hsct-digital-twin

**This is the live-deployed digital twin.** Repo sits at
`The-Educational-Equality-Institute/oura-hsct-digital-twin` (public, MIT).
Live URL: `https://digital-twin.theeducationalequalityinstitute.org/`.

The helse-digital-twin monorepo under henrigkroine is a separate sandbox.
Do not confuse the two. This document is for the TEEI repo.

## Summary numbers

| Dimension | Count |
|---|---|
| Python analysis files (`analysis/*.py`) | 42 |
| Lines of Python analysis code | ~54,275 |
| HTML reports in `reports/` | 38 |
| JSON metrics files in `reports/` | 38 |
| DuckDB analysis_key rows | 38 |
| Active pasient profiles | 3 (henrik, mitch, wenche) |
| Device adapters | 7 |
| Devices currently live | 2 (Oura Ring, manual symptom CLI) |
| Devices with stub parser awaiting real data | 4 (Libre, Checkme, Viatom, Contec) |
| GitHub Actions workflows | 3 (ci, daily-pipeline, release) |
| Pre-registered hypothesis files | 1 (cgm_hypotheses_pre_registered.md) |
| Research synthesis drafts | 1 (two-hit autonomic recovery) |
| Integration plan docs | 3 |
| Build-prompt briefs | 16 |

## L1, instrumentation

### Active devices

| Device | Adapter | Status | Notes |
|---|---|---|---|
| Oura Ring Gen 4 (henrik, wenche) | `api/import_oura.py` | live, daily cron | Scope missing `heart_health` + `ring_configuration` |
| Oura Ring Gen 3 (mitch) | same | live, daily cron | Missing SpO2, resilience, cardiovascular age on Gen 3 |
| Manual symptoms | `api/import_symptom.py` | live, CLI | Coupled to CGM trial H3 hypothesis |

### Stub adapters (parsers built, real data pending)

| Device | Adapter | Expected live | Risk |
|---|---|---|---|
| FreeStyle Libre 3 Plus | `api/import_glucose.py` | 2026-04-21 | 15-day CGM trial, pre-registered |
| Checkme O2 Max Pro | `api/import_checkme_spo2.py` | 2026-04-28 | Format unvalidated |
| Viatom TH12 12-lead ECG | `api/import_viatom_ecg.py` | 2026-04-28 | Format unvalidated, AI events schema inferred |
| Contec ABPM50 24h BP | `api/import_contec_abpm.py` | 2026-04-28 | Format unvalidated |
| Omron M7 HEM-7380T1 | `api/import_omron.py` | real BP bridge C:\omron-bridge | Parser built from conventions, device data in `data/demo.db` |

### Patient profiles

| Profile | Condition | DB file | Data range | Env prefix |
|---|---|---|---|---|
| henrik | Post-HSCT (MDS-AML 2023-11-23) | `data/demo.db` (15 MB) | 2026-01-08 onwards | `OURA_*` (default) |
| mitch | Post-stroke + ALL survivor (2023-04-22) | `data/mitch.db` (71 MB) | 2021-02-01 onwards | `MITCH_OURA_*` |
| wenche | Healthy 61-year-old reference | `data/wenche.db` (5.9 MB) | 2026-03-24 onwards | `WENCHE_OURA_*` |

### Clinical constants (in `config.py`)

```
TRANSPLANT_DATE      = 2023-11-23
TREATMENT_START      = 2026-03-16   (Ruxolitinib 10mg BID)
BETA_BLOCKER_START   = 2026-04-08   (Bisoprolol 2.5mg)
HEV_DIAGNOSIS_DATE   = 2026-03-18   (viral confound)
KNOWN_EVENT_DATE     = 2026-02-09   (acute episode, validation anchor)
DATA_START           = 2026-01-08
PATIENT_AGE          = 36

ESC_RMSSD_DEFICIENCY     = 15 ms   (Kleiger 1987)
NOCTURNAL_HR_ELEVATED    = 80 bpm
SPO2_CONCERN_THRESHOLD   = 94.0 %
BOS_WEIGHTS              = spo2_slope 0.30, spo2_variability 0.20, desaturation_freq 0.20, bdi 0.15, hr_decoupling 0.15
DLCO_MEASUREMENTS        = (2024-01-01, 80.0), (2025-01-01, 75.0)   (placeholder)
```

## L2, inference (analyses)

### Analyze scripts, by clinical relevance

**Mayo-critical, top 10:**

1. `analyze_oura_causal.py` (4051 lines, largest): 4-method causal inference on ruxolitinib: CausalImpact, Synthetic Control, ITS, Granger
2. `analyze_oura_gvhd_predict.py`: rSLDS GvHD flare prediction, 6-stream composite
3. `analyze_oura_spo2_trend.py`: BOS screening (leading non-relapse mortality driver)
4. `analyze_sequential_ci.py`: dual-medicine attribution (rux vs bisoprolol staggered)
5. `generate_research_synthesis.py`: two-hit autonomic recovery narrative
6. `analyze_comparative_breathing.py`: BOS risk panel (includes `_bos_risk.py` helper)
7. `analyze_comparative_autonomic.py`: HRV + resting HR recovery trajectories (henrik vs mitch)
8. `analyze_comparative_treatment.py`: 4-method changepoint detection (PELT, CUSUM, BOCPD, Rolling)
9. `analyze_oura_advanced_hrv.py`: NeuroKit2 HRV suite (time, frequency, nonlinear)
10. `generate_treatment_report.py`: clinical summary of rux + bisoprolol response

**High-relevance, not Mayo-critical:**
- `analyze_oura_full.py`: comprehensive all-endpoint baseline
- `analyze_oura_biomarkers.py`: 6 composite indices (cardio age, resilience, recovery, inflammation, autonomic, integrated)
- `analyze_oura_sleep_advanced.py`: sleep architecture, fragmentation, ultradian
- `analyze_rux_forecast.py`: trajectory prediction, bootstrapped CIs
- `analyze_piecewise_its.py`: ITS with AR(1) errors, 3-phase design
- `analyze_weekly_tracker.py`: clinician-ready 7d vs prior-7d summary
- `analyze_omron_bp.py`: ESH-style BP analysis, AFib detection

**Medium / exploratory:**
- `analyze_oura_anomalies.py`: 5-method ensemble (STUMPY, Isolation Forest, LOF, z-score, ADTK)
- `analyze_multimodal_anomalies.py`: STUMPY univariate + mSTUMP
- `analyze_comparative_anomalies.py`: cross-patient anomaly patterns
- `analyze_comparative_sleep.py`: stage distribution comparison
- `analyze_comparative_coupling.py`: activity-recovery lagged correlation
- `analyze_comparative_temperature.py`: temperature drift, infection screening
- `analyze_mitch_changepoints.py`: retrospective event correlation
- `analyze_oura_foundation_models.py`: Chronos-2 forecasting
- `analyze_oura_digital_twin.py`: UKF state-space estimation
- `analyze_tau_u.py`: SCED effect sizes
- `analyze_placebo_tests.py`: falsification calibration
- `analyze_glucose_autonomic_coupling.py`: scaffolding, awaiting CGM 2026-04-21

### Generators

- `generate_index.py`: landing page, KPI aggregation
- `generate_oura_3d_dashboard.py`: CMO executive dashboard, interactive 3D
- `generate_research_synthesis.py`: two-hit narrative HTML
- `generate_treatment_report.py`: primary clinical report
- `generate_roadmap.py`: static What's Next page

### Helpers

- `_theme.py` (2355 lines): clinical-dark design system, used by 31 of 42 files. Exports `wrap_html`, `make_kpi_card`, `make_kpi_row`, `make_section`, `disclaimer_banner`, `format_p_value`, `METRIC_DESCRIPTIONS`, `STATUS_COLORS`, `COLORWAY`, Plotly template.
- `_hardening.py`: safe DB connections, DataFrame validation, section-wrapped exception handling
- `_comparative_utils.py`: shared multi-patient plumbing for 7 comparative scripts
- `_frames.py`: canonical DuckDB analytic frames (daily, hourly), z-scored against frozen pre-treatment baseline
- `_hrv_upgraded.py`: NeuroKit2 HRV computation wrapper
- `_bos_risk.py`: shared BOS scoring across breathing + SpO2 + gvhd_predict scripts
- `_config.py`: shared path resolution, backwards-compat
- `data_schemas.py`: Pandera DataFrame validation (min adoption)
- `statcheck_reports.py`: statistical integrity checker, cross-references HTML vs JSON

### Data flow

```
Raw Oura CSV / device exports
        |
        v
api/*.py adapters  ---writes-->  data/{demo,mitch,wenche}.db (SQLite)
                                         |
                                         v
                           analysis/_frames.py (DuckDB analytic cache)
                                         |
                                         v
                           analyze_*.py + generate_*.py (42 files)
                                         |
                  uses helpers: _theme, _hardening, _comparative_utils, _bos_risk
                                         |
                                         v
                           reports/*.html + reports/*.json
                                         |
                                         v
                           DuckDB analysis_runs (canonical metrics)
```

## L3, model

**Does not exist.** No parametric patient-specific twin model. `analyze_oura_digital_twin.py` has an UKF state-space estimator but it is post-hoc observational, not a simulator. Nothing can answer "what happens if ruxolitinib is increased to 15mg BID for 30 days".

## L4, interface (how output reaches users)

### Reports (live at digital-twin URL)

38 HTML reports grouped in `_theme.py` REPORT_REGISTRY:

- **Core** (5): home, full_analysis, biomarkers, sleep, weekly
- **Clinical** (6): causal, gvhd, spo2, forecast, synthesis, treatment_report
- **Advanced** (5): hrv, digital_twin, foundation, anomalies, 3d_dashboard
- **Comparative** (8): autonomic, treatment, sleep, coupling, anomalies, breathing, temperature, mitch_changepoints
- **Individual** (2): mitch_standalone, wenche_standalone
- **Statistical** (4): piecewise_its, sequential_ci, placebo, tau_u
- **Context** (2): about, roadmap

NAV_PRIMARY_IDS: home, weekly, full_analysis, comp_treatment, piecewise_its

### Design system

`_theme.py` is the design authority. Dark clinical palette, 45 CSS custom properties on `:root`:

- bg-primary `#0F1117`, bg-surface `#1A1D27`, bg-elevated `#242837`
- text-primary `#E8E8ED`, text-secondary `#9CA3AF`, text-tertiary `#6B7280`
- accent-blue `#3B82F6`, accent-green `#10B981`, accent-amber `#F59E0B`, accent-red `#EF4444`, plus purple, cyan, pink, orange, indigo
- type scale 2xl down to 3xs
- spacing xs through 3xl
- glass morphism, Inter font, spring-curve transitions

All 38 reports pass design consistency spot-check.

### Stale reports flagged

- `oura_full_analysis_20260331.html` (19 days old, snapshot)
- `compare_henrik_mitch.html` (19 days old, legacy)

## CI and deploy

### Workflows

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci.yml` | push, PR | Ruff lint, PHI regex scan, module smoke test, README link checker |
| `daily-pipeline.yml` | cron `0 6 * * *` UTC plus manual dispatch | Token refresh, ingest, run_all.py, commit demo.db, deploy to Cloudflare |
| `release.yml` | tag push | Release notes |

### Deploy path

```
Daily cron 06:00 UTC
        |
        v
refresh Oura token, rotate if new
        |
        v
python api/import_oura.py --start 2026-01-08 --end today
        |
        v
python run_all.py  (31 scripts, resilient mode)
        |
        v
copy data/oura.db -> data/demo.db, commit to main (bot)
        |
        v
wrangler pages deploy deploy --project-name=digital-twin --branch=main
        |
        v
digital-twin.theeducationalequalityinstitute.org/
```

### Secrets required

`OURA_CLIENT_ID`, `OURA_CLIENT_SECRET`, `OURA_REFRESH_TOKEN`, `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `ACTIONS_PAT`.

### Risk flags

1. CI does not test report generation. `run_all.py` is only run in daily-pipeline. Commits that break an analysis pass CI.
2. Resilient mode: `run_all.py` exits 0 if any script passes. Silent partial failures.
3. No PR preview or staging environment.
4. No alerting on pipeline failure (no email, no Slack).
5. Auto-backup commits of `data/demo.db` can conflict with manual edits.
6. Missing `CLOUDFLARE_API_TOKEN` silently skips deploy (no warn).
7. Token rotation via `gh secret set` requires `ACTIONS_PAT` with `repo:admin` scope.

## Methodology and documentation

### Pre-registered hypotheses

**1 file:** `reports/cgm_hypotheses_pre_registered.md` (dated 2026-04-16, before CGM activation 2026-04-21)

- H1 postprandial hyperglycemia >= 10 mmol/L
- H2 glycemic-autonomic coupling (glucose excursion causes HRV depression)
- H3 chest-pain coincidence with hyperglycemic windows
- H4 glycemic variability CV > 36% linked to lower nightly RMSSD

Analysis methods pre-specified. Decision rules explicit. Amendment policy stated.

### Research synthesis drafts

**1 file:** `reports/RESEARCH_SYNTHESIS_two_hit_autonomic_recovery.md` (2026-04-14, 2913 words)

Argues two-hit model: ruxolitinib alone shows flat HRV (-0.15 ms/day), bisoprolol addition accelerates recovery (+3.43 ms/day, p<0.001). 7-fold causal evidence convergence. Marked hypothesis-generating, not confirming, N=1.

### Integration plans (new device work)

| Plan | Device | Status | File |
|---|---|---|---|
| CGM | FreeStyle Libre 3 | In progress, activation 2026-04-21 | `docs/cgm_integration_plan.md` |
| Continuous SpO2 | Checkme O2 Max Pro | Planned, ~2026-04-28 | `docs/checkme_o2_max_integration_plan.md` |
| 12-lead ECG Holter | Viatom TH12 | Planned, ~2026-04-28 | `reports/TH12_INTEGRATION_PLAN.md` |
| 24h ABPM | Contec ABPM50 | Planned, ~2026-04-28 | `docs/build-prompts/12-abpm50-integration.md` |

### Documentation gaps

1. No `ARCHITECTURE.md` (README covers architecture narratively)
2. No `CONTRIBUTING.md`
3. No clinical-collaborator interpretation guide
4. No report-authoring README for adding new analysis
5. `DATA_METHODOLOGY.md` (2026-03-23) references 7-day post-intervention window, now 34 days, stale
6. No `CHANGELOG.md` tracking evolution
7. No formal data-provenance doc

## Duplication + consolidation candidates

Flagged by F1:

1. HRV computation appears in both `analyze_oura_advanced_hrv.py` (NeuroKit2) and `analyze_oura_full.py` (hand-rolled). Confirm relationship.
2. Three anomaly detectors: `analyze_oura_anomalies.py` (5-method), `analyze_multimodal_anomalies.py` (STUMPY), `analyze_comparative_anomalies.py` (cross-patient STUMPY). Complementary or consolidate.
3. Temperature analysis in `analyze_oura_gvhd_predict.py` and `analyze_comparative_temperature.py`. Can they share a helper.

## Validation questions flagged

1. `_bos_risk.py` component weights (spo2_slope 0.30, etc.) source and clinical validation.
2. GvHD 6-stream composite weighting source.
3. Biomarker index weighting: population-validated or Henrik-specific exploratory.

## Clinical context

This is an N=1 post-HSCT case study with comparator patients (mitch post-stroke, wenche healthy control). Research collaborators:

- Schoemans (GvHD, transplant, Ghent)
- Wolff (stroke, autonomic)
- Mayo Clinic (May 19 consultation)

The case argues a two-hit autonomic recovery model for ruxolitinib + bisoprolol combination therapy. Planned publication venue not yet declared.

## How to use this document

**Before building anything new:**

1. Grep this document for the feature name. If found, understand what already exists.
2. If adding a new analysis, check `analysis/*.py` for duplication risk.
3. If adding a new report, check REPORT_REGISTRY in `_theme.py` for naming conflicts.
4. If adding a new device, check `api/*.py` and integration plan docs.
5. If modifying clinical thresholds, update `config.py` plus memory file.

**Before the Mayo consultation (2026-05-19):**

- Top-10 Mayo-critical reports are listed above. Refresh `DATA_METHODOLOGY.md` post-intervention sample size. Regenerate `RESEARCH_SYNTHESIS_two_hit_autonomic_recovery.md` with 34+ days post-rux data.

## Maintenance

Refresh this document whenever:
- A new analysis script lands
- A new device adapter lands
- A new report publishes
- Clinical constants in `config.py` change
- Pre-registered hypothesis is amended

Dated successor: `docs/REPO-INVENTORY-2026-MM.md` preserving prior.
