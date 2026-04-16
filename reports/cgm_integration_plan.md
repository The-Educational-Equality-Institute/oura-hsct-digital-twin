# FreeStyle Libre 3 Plus Integration Plan

**Target project:** `/home/henrik/projects/teei/oura-hsct-digital-twin/`
**Patient:** Henrik (post-HSCT MDS-AML, ruxolitinib 10 mg BID + bisoprolol 2.5 mg daily)
**Sensor:** FreeStyle Libre 3 Plus, one sensor / 15-day trial, activation ~2026-04-21
**Plan committed:** 2026-04-16
**Existing scaffolding:** `api/import_glucose.py` (LibreView CSV parser, `glucose_readings` + `symptom_events` tables) and `analysis/analyze_glucose_autonomic_coupling.py` (stubbed hypothesis scaffolding) are already in place. This plan extends, not replaces, that work.

---

## 1. Data acquisition recommendation

### Landscape (verified April 2026)

| Path | Latency | Reliability | Legal / TOS | Setup friction | Stability |
|---|---|---|---|---|---|
| **Abbott FreeStyle Libre 3 app (official)** | Real-time on phone; no export beyond LibreView sync | High | Green (intended use) | None — required anyway for activation | Abbott-controlled; breaks with sensor redesigns, but official path is the reference |
| **LibreView CSV export (manual)** | 1–24 h lag (user triggers) | High once data is on LibreView | Green — user's own data, explicit export feature | Low — click-through from libreview.com | Very stable; CSV schema has been stable for ~5 years |
| **LibreLinkUp cloud via `pylibrelinkup`** | ~1–2 min (LLU app pushes latest reading + 12 h graph) | Medium — reverse-engineered, can break when Abbott changes auth headers | **Amber** — LLU is designed for caregiver read-only sharing; `pylibrelinkup` uses the same REST API the official LLU mobile app hits. Not explicitly blessed but not a TOS breach of the *patient's* account (LLU = share with family). Do not publish scraped data; do not share credentials. | Low — install pip package, use LLU email/password (separate from Libre app account, but auth-linked by sharer code) | Has broken historically on auth changes; maintained actively (last release v0.10.0 on 2026-02-28) |
| **Juggluco (Android app, BLE takeover)** | Real-time, 1-min resolution | High once takeover succeeds | **Amber** — GPL-3 open-source app that speaks the same BLE protocol as the official Abbott app. Requires the sensor to be *first* activated via Abbott's official Libre 3 app (same account email), then Juggluco "takes over" BLE. Abbott has sent DMCA notices against projects that *publish* decryption internals, but end-user use of Juggluco is not itself a TOS violation. | Medium — need Android phone, disable Abbott app notifications, takeover procedure. Libre 3 Plus is supported as of Juggluco 8.1.6 (15-day wear duration). | Medium — Abbott app updates have forced Juggluco updates several times; j-kaltes keeps up |
| **xDrip+ / AAPS / Loop** | Real-time via Juggluco xDrip broadcast | High (downstream of Juggluco) | Same as Juggluco | Adds a second Android app; useful only if we also want alarms / pump loop (we don't) | Medium |
| **Direct BLE from Python/Linux** | Theoretical real-time | Low — would need to re-implement Abbott's AES128-CCM + ECDH handshake with account-specific `blePIN` | **Red** — re-deriving the key flow risks DMCA; also brittle against Abbott firmware bumps | Very high — research project in itself | Unstable — Abbott has changed the protocol before |
| **NFC scan from Android** | Only useful on Libre 1/2; **Libre 3 Plus is BLE-only, NFC is activation-only** | — | — | — | — |

### Recommended path: **layered primary + backup**

**Primary (real-time, 1-minute resolution): Juggluco (Android) → HTTP server → WSL ingester**

Juggluco exposes a local HTTP endpoint that serves glucose history as TSV (this is documented at `https://www.juggluco.nl/Jugglucohelp/exchangehelp.html`; the exact endpoint path and port need to be read off the Juggluco app's Settings → Exchange data → Web server screen once installed — see open question Q1). On the phone, enable Juggluco's built-in web server on the LAN. On the WSL side, a new `api/import_libre.py` polls that endpoint every 5 minutes while the phone is on Wi-Fi, parses the TSV response, and upserts into the existing `glucose_readings` table.

This matches the Omron BLE bridge pattern (external device → host machine → inbox file → WSL ingester) in `api/import_omron.py`, but replaces file-drop with HTTP pull. No phone-side custom code needed — Juggluco provides the server.

**Backup #1 (daily sync-of-record): LibreView CSV**

The Abbott Libre 3 app uploads to LibreView automatically if logged in. Every 2–3 days during the trial, export a CSV from libreview.com and drop it in the inbox; the existing `api/import_glucose.py` (LibreView parser) already ingests it idempotently (UNIQUE constraint on `(timestamp, sensor_serial)`). This guards against:
- Juggluco takeover failure
- Phone/Wi-Fi outages during the trial
- Sensor dropouts where LibreView has interpolated values Juggluco missed

**Backup #2 (fall-through cloud path): `pylibrelinkup`**

If Juggluco takeover fails on the Libre 3 Plus form factor (unlikely per issue #191 on the Juggluco repo, but possible), install `pylibrelinkup` and pull every 15 minutes from LibreLinkUp. LLU requires a caregiver share, so set up Henrik's Libre 3 app to share with a secondary LLU account that the scraper uses.

### Setup steps (ordered by trial timeline)

**Pre-activation (2026-04-17 to 2026-04-20)**

1. Install Abbott's FreeStyle Libre 3 app on the patient's Android phone; sign in with the Abbott account (this is required to activate the sensor at all — Libre 3/3+ refuses to talk BLE without account-linked activation).
2. Install Juggluco (8.1.6 or newer, GPL-3, APK from `https://www.juggluco.nl/`) on the same phone.
3. In Juggluco: Settings → Exchange data → Libreview, enter the same Abbott account email, press "Get account ID". This primes the takeover.
4. In Juggluco: Settings → Exchange data → Web server, enable the local HTTP server. Note the port and the auth token (if any) that Juggluco displays.
5. Put the phone on the same Wi-Fi as the WSL host; firewall-allow the Juggluco port from the WSL subnet.
6. Register the LLU fallback: in Abbott Libre 3 app, add a LibreLinkUp sharer using a secondary email you control; install LibreLinkUp on any device and accept the share. This is the `pylibrelinkup` credential later.

**Activation day (2026-04-21)**

7. Activate the sensor via the Abbott Libre 3 app (NFC tap on arm).
8. Wait 60 minutes for Abbott's warm-up, confirm first reading in the Abbott app.
9. In Juggluco: scan the sensor once (Libre 3 uses a one-time "takeover" handshake). Juggluco should start logging at 1-min intervals.
10. On WSL: run the new `api/import_libre.py --once --init-only` to create the `libre_*` tables.
11. Enable the cron / systemd timer that polls Juggluco every 5 minutes.

**During trial (2026-04-21 to ~2026-05-06)**

12. Every 2–3 days: download LibreView CSV → drop in `~/libre-inbox/` → run `api/import_glucose.py` (this path already exists and is tested).
13. Daily pipeline (`scripts/daily_pipeline.sh`) runs both ingesters, dedupes on UNIQUE(timestamp, sensor_serial).

**Post-trial (post-2026-05-06)**

14. Final CSV pull from LibreView after sensor expiry is the authoritative record; Juggluco's TSV is the high-frequency live stream.

---

## 2. Python libraries to install

### CGM-specific libraries (ingestion + analytics)

| Library | Version (as of 2026-04) | Python | License | Status | Decision |
|---|---|---|---|---|---|
| **`pylibrelinkup`** | 0.10.0 (2026-02-28) | ≥3.11 | MIT | Actively maintained; Libre 3 / LLU caregiver data confirmed working | **Install** as backup ingester. `pip install pylibrelinkup` |
| **`cgmquantify`** | 0.4.0 (2020-11, last commit ~2020) | 3.7+ (tested 3.7–3.9) | MIT | **Abandoned** since 2020; authors moved to R/CRAN. 28 metrics; Dexcom G6-focused, but glucose-series agnostic once loaded. | **Skip** — use only the algorithms; do not take dependency. Re-implement the 3–4 metrics we need (MAGE, MODD, CONGA, J-Index) directly against our DataFrame. Python 3.12 compatibility untested and maintainer unresponsive. |
| **`glucose360`** | On PyPI as `glucose360` (source: github.com/vurhd2/Glucose360 per the Stanford Snyder Lab publication; the dhruv-aron/Glucose360 mirror has last commit 2024-07-08 and no tagged releases) | Not pinned; uses NumPy + pandas | **GPL-2.0** | Published 2026 in *Diabetes Technology & Therapeutics*; 84 functionalities including AGP percentile plots, event-aware MAGE, TIR stacked bar. Actively developed by Stanford group. | **Consider but verify** — GPL-2.0 is **incompatible with MIT/Apache preference**. If we use it, reports module must be isolated so GPL doesn't contaminate the rest. Recommendation: read the GPL-2.0 terms carefully before installing; if in doubt, use it only for offline metric cross-checks, not as a runtime dependency. Flag as open question Q3. |
| **`glucostats`** | On PyPI as `glucostats`; v1.x (2025 BMC Bioinformatics paper) | Not pinned | **BSD-2-Clause** | New in 2025; 59 statistics across 6 categories; sklearn-compatible; parallelizable; accepts pandas DataFrame with `time` + `glucose` columns. | **Install as primary metrics library.** BSD-2 is permissive. `pip install glucostats`. Verify Python 3.12 compatibility on install (BMC paper tested 3.10+). |
| **`iglu` (R)** | CRAN, mature | R-only | GPL-2 | Gold standard for AGP; iglu-py exists but is a thin wrapper | **Skip** unless we already have R in the pipeline (we don't). Glucose360 + Glucostats cover iglu's metric set between them. |

### Newer (2025–2026) Python CGM libraries surfaced in the research

- **Glucose360** (Stanford Snyder Lab, DOI 10.1177/15209156251374711, published 2026) — above.
- **GlucoStats** (URJC AI4Health, BMC Bioinformatics 2025, DOI 10.1186/s12859-025-06250-w) — above.
- No HuggingFace "continuous-glucose-monitor" topic yields usable libraries as of 2026-04; the HF CGM space is foundation-model papers, not ingestion libraries.
- GitHub `topic:continuous-glucose-monitor` returns the same cluster (cgmquantify, DiaBLE, OpenAPS ecosystem); nothing newer than Glucostats/Glucose360 with a BSD/MIT/Apache license.

### Final install list (to be added to `requirements.txt`)

```
pylibrelinkup>=0.10.0    # backup LLU ingester
glucostats>=1.0          # primary metric library (BSD-2)
# glucose360            # ONLY if GPL-2 is acceptable — see Q3
requests                # (already present) — for Juggluco HTTP poll
```

Do **not** add `cgmquantify` — abandoned, and re-implementing MAGE/MODD is ~30 lines of pandas each.

---

## 3. Schema proposal

Extends the existing `glucose_readings` + `symptom_events` (from `api/import_glucose.py`) with `libre_*` session/event tables that parallel the `omron_*` pattern. Existing tables are left unchanged; `glucose_readings` becomes the canonical point-in-time glucose store regardless of source (Juggluco, LibreView CSV, LLU).

```sql
-- EXISTING — do not modify, already populated by api/import_glucose.py
-- glucose_readings(id, timestamp, glucose_mmol_l, record_type, sensor_serial, source, imported_at)
-- symptom_events(id, timestamp, symptom_type, severity, context, note, logged_at)

-- NEW: sensor wear sessions (one row per sensor insertion)
CREATE TABLE IF NOT EXISTS cgm_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sensor_serial TEXT NOT NULL UNIQUE,
    sensor_model TEXT,                  -- 'libre_3_plus'
    activated_at TEXT NOT NULL,         -- Abbott warm-up end
    expected_end_at TEXT,               -- activated_at + 15 days
    first_valid_reading_at TEXT,
    last_valid_reading_at TEXT,
    reading_count INTEGER DEFAULT 0,
    notes TEXT,
    imported_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- NEW: manually entered meals, insulin, exercise, interventions
CREATE TABLE IF NOT EXISTS cgm_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,           -- 'meal' | 'exercise' | 'medication' | 'note'
    label TEXT,                         -- 'breakfast', 'ruxolitinib-morning', ...
    carbs_g REAL,                       -- only meaningful for meals
    intensity TEXT,                     -- 'light' | 'moderate' | 'vigorous' for exercise
    duration_min INTEGER,
    note TEXT,
    logged_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- NEW: raw Juggluco poll snapshots (debug / replay — small overhead, valuable if Abbott changes something)
CREATE TABLE IF NOT EXISTS cgm_juggluco_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    polled_at TEXT NOT NULL,
    source_host TEXT,
    http_status INTEGER,
    row_count INTEGER,
    first_ts TEXT,
    last_ts TEXT,
    raw_sample TEXT,                    -- first 500 chars for debugging
    imported_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cgm_events_ts ON cgm_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_cgm_events_type ON cgm_events(event_type);
CREATE INDEX IF NOT EXISTS idx_cgm_sessions_serial ON cgm_sessions(sensor_serial);
```

### Why keep `glucose_readings` (not rename to `cgm_glucose` / `libre_glucose`)

The existing scaffolding in `analysis/analyze_glucose_autonomic_coupling.py` and `reports/cgm_hypotheses_pre_registered.md` already references `glucose_readings`. Renaming forces rewriting the hypothesis scaffolding and the CSV path. Keep the existing table; add the `source` column value `"juggluco_http"` for the new ingester. All downstream analysis can filter by `source` if needed.

### `source` column enumeration

- `libreview_csv` — existing, Abbott LibreView CSV export
- `juggluco_http` — new, real-time Juggluco poll
- `pylibrelinkup` — new, LibreLinkUp cloud backup
- `manual` — reserved for any fingerstick the patient types in

---

## 4. Analysis module design

### New file: `analysis/analyze_cgm.py`

Standalone module following the existing pattern (`analyze_omron_bp.py` is the closest parallel — single source, single SQL load, theme helpers from `_theme.py`, emits HTML + JSON).

```
analysis/analyze_cgm.py
├── load_cgm_data(conn, start=None, end=None) -> pd.DataFrame
│     Loads glucose_readings + cgm_events + cgm_sessions, resamples to 5-min
│     grid, handles gaps (sensor warm-up, dropouts), returns tidy dataframe.
│
├── compute_agp_metrics(df) -> dict
│     TIR (70-180 mg/dL = 3.9-10.0 mmol/L), TAR (>180), TBR (<70), TBR-L2 (<54),
│     GMI (Glucose Management Indicator), CV (coefficient of variation),
│     MAGE (Mean Amplitude of Glycemic Excursions), MODD (Mean of Daily Differences),
│     mean, SD, percentiles, HBGI / LBGI.
│     Primary implementation: delegate to glucostats where possible;
│     MAGE / MODD backed up with reference implementation against peer-reviewed
│     definitions (Service 1970; Monnier 2006).
│
├── compute_agp_percentile_curve(df) -> pd.DataFrame
│     Aggregate over trial, bucketed by time-of-day (5-min bins), percentiles
│     5/25/50/75/95. This IS the AGP chart.
│
├── compute_daily_stats(df) -> pd.DataFrame
│     Per-day row: TIR, TBR, TAR, mean, CV, nocturnal min, largest post-meal
│     excursion, day-of-week, weekday/weekend flag.
│
├── detect_nocturnal_hypoglycemia(df) -> pd.DataFrame
│     Rows where a 00:00-06:00 window contains >=15 min of consecutive readings
│     <3.9 mmol/L. Returns (start, end, nadir, duration_min) per event.
│     CLINICAL JUSTIFICATION: Henrik is on bisoprolol — beta-blockers mask
│     adrenergic hypoglycemia symptoms (tachycardia, tremor), leaving only
│     sweating. A CGM-detected nocturnal <3.9 episode the patient didn't feel
│     is HIGH-value clinical information.
│
├── characterize_postprandial_excursions(df, events) -> pd.DataFrame
│     Joins glucose series with cgm_events WHERE event_type='meal'. For each
│     meal: pre-meal baseline (30-min mean), peak within 30-120 min, Δ, time-to-peak,
│     tail-return-to-baseline. Outputs per-meal table + aggregate stats.
│
├── join_glucose_with_bp(cgm_df, omron_df, tol_min=5) -> pd.DataFrame
│     Nearest-neighbor merge on timestamp, ±5 min tolerance. Joins with
│     omron_bp_readings. Computes rank correlation (Spearman) of glucose vs
│     SYS, DIA, MAP, pulse pressure. CAVEAT: Omron readings are sparse (a few
│     per day at best); this is a correlation probe, not a time-series model.
│
├── join_glucose_with_hrv(cgm_df, oura_hrv_df, lag_min=15) -> pd.DataFrame
│     Shifts HRV forward 15 min (ANS response lag); nearest-5-min merge on
│     Oura 5-min RMSSD series. Directly feeds H2 (glycemic-autonomic coupling
│     hypothesis from the pre-registered doc) — delegate the statistical test
│     to analyze_glucose_autonomic_coupling.py, this file only produces the
│     paired series.
│
├── join_glucose_with_sleep(cgm_df, oura_sleep_df) -> pd.DataFrame
│     For each sleep period, compute: mean glucose during sleep, dawn-phenomenon
│     slope (04:00-07:00 rise), time-below-3.9 overnight, time-above-10.0
│     overnight. Overlays sleep stages (deep/REM/light) where Oura sensor-hour
│     resolution allows.
│
├── join_glucose_with_stress(cgm_df, oura_stress_df) -> pd.DataFrame
│     Daytime coupling: Oura Daytime Stress score (day-level) vs that day's
│     glucose mean, CV, and max excursion. Low-power given 15-day n — report
│     as exploratory.
│
├── render_report(all_results, out_html, out_json) -> None
│     Uses _theme.make_kpi_row, make_section, wrap_html. Layout:
│     §1 Headline KPIs (TIR, GMI, CV, mean glucose, #nocturnal <3.9 events)
│     §2 AGP percentile chart (P5-P95 ribbon + median line)
│     §3 Daily TIR stacked bar chart (one bar per trial day, 5 TIR bands)
│     §4 Circadian glucose curve (mean ± SD by time-of-day)
│     §5 Day-of-week patterns (weekday vs weekend; work-day effect)
│     §6 Nocturnal hypoglycemia events table (timestamp, nadir, duration, HR/HRV during)
│     §7 Postprandial excursions table + histogram
│     §8 Cross-modal coupling (BP, HRV, sleep, stress) — small multiples
│     §9 Clinical interpretation narrative (template strings, no fabricated interpretation)
```

### Metric definitions pinned to consensus (2025 ADA Standards + 2019 international consensus)

| Metric | Definition | Target (general pop) | Henrik-specific note |
|---|---|---|---|
| TIR | % readings 70–180 mg/dL (3.9–10.0 mmol/L) | ≥70% | First-ever trial, no baseline |
| TBR | % readings <70 mg/dL (3.9) | <4% | Bisoprolol-masked; even 1 event is notable |
| TBR-L2 | % readings <54 mg/dL (3.0) | <1% | Clinically urgent if observed |
| TAR | % readings >180 mg/dL (10.0) | <25% | H1 hypothesis hinges on this |
| TAR-L2 | % readings >250 mg/dL (13.9) | <5% | Frank hyperglycemia flag |
| GMI | Glucose Management Indicator (%) = 3.31 + 0.02392 × mean mg/dL | ≈ HbA1c estimate | Anchors against Mayo OGTT |
| CV | 100 × SD / mean | <36% = stable | Ruxolitinib/GvHD confounds |
| MAGE | Mean amplitude of excursions >1 SD | n/a (descriptive) | Peak-vs-nadir amplitude |
| MODD | Day-to-day variability at same time | n/a | Night-to-night stability |

---

## 5. Clinical interpretation framework

### Ruxolitinib × dysglycemia signal

Literature is ambiguous — report interpretation must stay hypothesis-generating:

- A 72-week metabolic study of JAK1/2 inhibition in MPN patients (Ruxolitinib arm) found **no significant change** in fasting glucose (100 ± 40 vs 101 ± 30 mg/dL) and **no change** in hyperglycemia / diabetes prevalence (14.5% → 14.5%). (Nature Scientific Reports 2019, `s41598-019-53056-x`.)
- Conversely, JAK inhibition has been associated with **improved** glycemia in autoimmune diabetes (STAT1 gain-of-function case, NEJM 2020), suggesting bidirectional effects.
- WHO pharmacovigilance database (2022, `s41598-022-10777-w`) lists dyslipidemia as a signal for JAK inhibitors but not prominent dysglycemia.

**Conclusion for the report:** Do not pre-commit to ruxolitinib causing hyperglycemia. Report observed TIR/TAR/TBR against population norms; if Henrik shows >25% TAR, note it as worth investigating but not attributable to ruxolitinib without a withdrawal arm.

### Steroid-like effect question

The user's framing ("steroid-like effect of ruxolitinib") is common in the GvHD community because ruxolitinib is given precisely to **reduce** steroid dependency. True glucocorticoid therapy for aGVHD is itself strongly associated with dysglycemia — a PubMed 20637884 study showed minimum glucose 0–60 mg/dL predicted worsened overall survival and non-relapse mortality. But ruxolitinib alone is not a glucocorticoid and does not carry the same direct hepatic gluconeogenesis signal. If Henrik has been off steroids for some time, any dysglycemia observed should be coded as **post-HSCT malglycemia signal, rule out PTDM**, not as a ruxolitinib side effect.

### Beta-blocker masking of hypoglycemia symptoms

Well-established:
- Bisoprolol blocks β1-mediated adrenergic symptoms (tachycardia, tremor) of hypoglycemia; sweating (cholinergic) is preserved.
- Pharmacovigilance signal: bisoprolol ROR 1.42 (1.25–1.61) for hypoglycemia reports vs other beta-blockers (Carnovale 2021, BJCP `10.1111/bcp.14754`).
- Clinical implication for Henrik: **any CGM-detected glucose <3.9 mmol/L the patient did not feel is itself a clinically meaningful finding**, because his usual warning system is partially suppressed. The report should surface every such event, not just the severe ones.

### GvHD × dysglycemia

Broader "malglycemia after HSCT" literature (PubMed 30718242, `bmt201681`, `bmt201727`) consistently shows glucose variability is associated with:
- Increased infection risk
- Worsened aGVHD outcomes
- Delayed engraftment
- Increased non-relapse mortality

For Henrik (2+ years post-transplant, chronic GvHD on ruxolitinib), the acute-phase associations don't directly transfer, but the **signal-to-look-for** is high CV (>36%) and low TIR (<70%), either of which would reinforce a Mayo recommendation for HbA1c + OGTT.

### Narrative structure for §9 of the report

```
Population-normed reading:
  — TIR %, TBR %, TAR %, GMI vs ADA targets
  — CV vs stable/unstable threshold
  — Nocturnal hypo events: <count>, <severity>

Context flags (HENRIK-SPECIFIC, template strings, not fabricated):
  — "Bisoprolol may mask adrenergic warning symptoms — any TBR event is clinically
     non-trivial even if asymptomatic."
  — "Post-HSCT malglycemia is a recognized risk. CV > 36% or TIR < 70% would
     support a Mayo recommendation for OGTT + HbA1c formal screening."
  — "Ruxolitinib's own dysglycemia signal is weak in MPN literature; observed
     pattern is more likely post-HSCT / PTDM-related than drug-specific."

What this analysis cannot conclude:
  — Causality (no withdrawal, single sensor)
  — Comparison to pre-transplant baseline (never had CGM before)
  — Whether any nocturnal drop was symptomatic (needs patient diary correlation)
```

Every KPI gets a `status` of `normal` / `watch` / `critical` keyed off published thresholds, using existing `_theme.STATUS_COLORS`.

---

## 6. Dashboard

New report: `reports/cgm_report.html` + `reports/cgm_metrics.json`.

Assembled via the standard idiom from `analysis/_theme.py`:

```
make_kpi_row(
    make_kpi_card("TIR (70-180)", tir_pct, "%", status=...),
    make_kpi_card("Mean glucose", mean_mmol, "mmol/L", status=...),
    make_kpi_card("GMI", gmi_pct, "%", detail="Est. HbA1c"),
    make_kpi_card("CV", cv_pct, "%", status=...),
    make_kpi_card("Nocturnal <3.9", n_hypo_events, "events"),
)
make_section("Ambulatory Glucose Profile", agp_fig_html)
make_section("Daily TIR", daily_tir_fig_html)
make_section("Circadian glucose curve", circadian_fig_html)
make_section("Postprandial excursions", postprandial_html)
make_section("Nocturnal hypoglycemia events", nocturnal_table_html)
make_section("CGM × Oura HRV coupling", hrv_coupling_fig_html)
make_section("CGM × Omron BP coupling", bp_coupling_fig_html)
make_section("CGM × sleep architecture", sleep_coupling_fig_html)
make_section("Clinical interpretation", narrative_html)
wrap_html("CGM Analysis", body, report_id="cgm")
```

### Registry entry in `_theme.REPORT_REGISTRY`

Add:
```python
{"id": "cgm", "file": "cgm_report.html", "title": "CGM Analysis", "group": "Clinical"},
```

Place in `NAV_PRIMARY_IDS` between `spo2` and `hrv` so it's one click from the dashboard once real data exists.

### Data-availability guard

Copy the idiom from `analyze_glucose_autonomic_coupling.py` — if `glucose_readings` has <100 rows, render a "Data pending — sensor activation on 2026-04-21" placeholder page rather than crashing. This keeps `run_all.py` green between now and activation.

---

## 7. Integration into pipeline

### New files

- `api/import_libre.py` — Juggluco HTTP poller. Structure mirrors `api/import_omron.py`:
  - CLI: `--once` (single poll), `--init-only`, `--host`, `--port`, `--profile`, `--db`
  - Default loop mode: poll every 5 minutes, log to `~/logs/libre.log`
  - Resilience: HTTP timeout 10 s; on failure, log to `cgm_juggluco_snapshots` and continue; do not crash
  - Idempotency: `INSERT OR IGNORE` on `glucose_readings.UNIQUE(timestamp, sensor_serial)`
  - Creates/upserts `cgm_sessions` row on first reading with a new serial

- `api/import_libre_llu.py` — fallback `pylibrelinkup` poller. Same pattern, runs every 15 minutes, writes rows with `source='pylibrelinkup'`. Off by default (opt-in via env var `LLU_ENABLE=1` + `LLU_EMAIL` + `LLU_PASSWORD` from `~/.secrets.env`).

- `analysis/analyze_cgm.py` — the new analysis module (outlined in §4).

### Edits to existing files

- `scripts/daily_pipeline.sh`: insert between step [3/5] (OMRON) and [4/5] (analysis):
  ```
  # 3c. Ingest LibreView CSV drops (if any) + finalize with latest Juggluco/LLU state
  echo "[3c/5] Importing CGM data..."
  python "$DIGITAL_TWIN/api/import_glucose.py" --inbox ~/libre-inbox || echo "  LibreView CSV import non-fatal (continuing)"
  python "$DIGITAL_TWIN/api/import_libre.py" --once || echo "  Juggluco poll non-fatal (continuing)"
  if [ -n "${LLU_ENABLE:-}" ]; then
    python "$DIGITAL_TWIN/api/import_libre_llu.py" --once || echo "  LLU fallback poll non-fatal (continuing)"
  fi
  ```
  Renumber the pipeline comments accordingly. Pattern — "non-fatal on failure, continue" — matches the Omron step.

- `run_all.py` `SCRIPTS` list: append `"analyze_cgm.py"` before `"generate_roadmap.py"` so the new report shows up in the index.

- `run_all.py` `SEND_BUNDLE_HTML` list: add `"cgm_report.html"`.

- `analysis/_theme.py` `REPORT_REGISTRY` + `NAV_PRIMARY_IDS`: add the `cgm` entry as shown in §6.

- `requirements.txt`: add `pylibrelinkup>=0.10.0` and `glucostats>=1.0`.

### High-frequency polling (separate from daily pipeline)

`scripts/daily_pipeline.sh` only runs once per day. The Juggluco 5-min poll needs its own schedule. Two options:

**Option A (recommended — matches OMRON pattern):** add a systemd user timer:
- `~/.config/systemd/user/libre-poll.service`: runs `python $DIGITAL_TWIN/api/import_libre.py --once`
- `~/.config/systemd/user/libre-poll.timer`: `OnBootSec=2min`, `OnUnitActiveSec=5min`

**Option B:** crontab `*/5 * * * * python .../import_libre.py --once`.

Either way, the daily pipeline run continues to be the single source of truth for rebuilding reports; the timer just keeps the database fresh.

---

## 8. Open questions

**Q1. Juggluco HTTP server exact endpoint path and auth.** The Juggluco help page at `https://www.juggluco.nl/Jugglucohelp/exchangehelp.html` documents that a web server exists, but the specific URL path (e.g. `/cgi-bin/nightscout/api/v1/entries.json`, `/tsv`, or `/x/stream`), port default, and whether it uses bearer tokens must be read off the Juggluco app UI on the patient's phone during setup (step 4 in §1). **Do not hardcode a path in `api/import_libre.py` — read from config once verified on device.** Will update plan after 2026-04-17 phone setup.

**Q2. Libre 3 Plus + Juggluco takeover reliability on this specific phone.** Juggluco 8.1.6+ supports Libre 3 Plus (confirmed in Juggluco issue #191 on GitHub; 15-day wear duration specifically implemented). But reports say takeover fails if the sensor was activated by Abbott's *Unified* Libre app instead of the dedicated Libre 3 app, and on some Samsung/Xiaomi phones Juggluco's BLE needs battery-optimization exceptions. **Action:** verify on the test phone before activation day; pin Abbott Libre 3 app version (not Unified).

**Q3. Glucose360 license.** Glucose360 is GPL-2.0. Using it as a Python runtime dependency in an MIT/Apache/BSD-preferring project is acceptable (GPL-2 doesn't affect our code's license when distributed separately), but distributing a combined binary or Docker image with GPL code requires careful attention. **Action:** decide whether to (a) install Glucose360 and treat its use as internal/personal (no external distribution), (b) skip it entirely and rely on glucostats + home-grown MAGE/MODD/GRADE, or (c) run it only out-of-band for metric cross-checks. Recommendation: (b), because glucostats has a compatible BSD-2 license and covers 59 of the needed metrics.

**Q4. Mayo consultation date handoff.** Report targets 2026-05-19 Mayo consultation. Should `analyze_cgm.py` emit a PDF handoff doc (matplotlib → PDF) in addition to HTML so it can be attached to a MyChart message? Parallel to `generate_treatment_report.py`. Not in scope for MVP; flag for follow-up.

**Q5. Symptom diary integration.** `symptom_events` table already exists. To make the bisoprolol-masking narrative non-hypothetical, Henrik would need to log whenever he feels *any* hypoglycemia-like symptom (sweating, confusion, hunger, lightheadedness) — even if mild. Recommendation: add a quick-entry symptom logger (shell alias or Shortcut on phone) during the trial. Not strictly part of CGM ingestion but would make §5 narrative vastly more useful.

**Q6. Meal logging granularity.** Postprandial excursion characterization (§4) needs meal timestamps. If we can't trust manual logging (realistic for a 15-day trial with a sick patient), fall back to **detected-meal inference**: identify every >1.5 mmol/L rise within 60 min as a probable meal and characterize those excursions. Note this in the report as "detected" vs "logged". No code change implied for this plan — just a modeling choice to document.

**Q7. LLU credentials storage.** If we enable `pylibrelinkup` backup, credentials live in `~/.secrets.env` (already chmod 600 per project conventions). The user-local LLU sharer account must be distinct from the primary Abbott Libre account — LLU is designed for caregiver sharing, not the patient's own device. **Do not put the primary Abbott account credentials into a script.**

**Q8. Abbott / Juggluco cat-and-mouse risk.** Abbott has historically broken Juggluco with firmware updates and DMCA'd related projects. During a 15-day trial this is a low-probability event but the LibreView CSV backup (§1) is precisely the insurance against it. No action needed; noting for situational awareness.

---

## Summary

- **Primary acquisition:** Juggluco on Android (supports Libre 3 Plus per issue #191, v8.1.6+) → local HTTP server → new `api/import_libre.py` polls every 5 min. Real-time 1-min resolution.
- **Backup 1:** existing `api/import_glucose.py` LibreView CSV (already implemented and tested in this repo).
- **Backup 2:** `pylibrelinkup` (MIT, maintained, v0.10.0 2026-02-28) on a 15-min poll if Juggluco fails.
- **Analytics:** `glucostats` (BSD-2, 59 metrics, 2025 BMC Bioinformatics) as primary; skip `cgmquantify` (abandoned 2020); hold `glucose360` pending Q3 license decision.
- **New tables:** `cgm_sessions`, `cgm_events`, `cgm_juggluco_snapshots`. Existing `glucose_readings` stays canonical; `source` column distinguishes paths.
- **New analysis:** `analysis/analyze_cgm.py` — AGP, TIR/TAR/TBR, MAGE, MODD, nocturnal hypo detection, cross-modal joins with Oura HRV / Oura sleep / Omron BP, clinical narrative templated to Henrik's post-HSCT / ruxolitinib / bisoprolol context.
- **Pipeline integration:** daily pipeline gets a CGM ingest step (with the same non-fatal-on-failure pattern as OMRON); a separate 5-min systemd user timer handles the Juggluco poll.
- **Open questions tracked, no URLs or endpoints fabricated.**
