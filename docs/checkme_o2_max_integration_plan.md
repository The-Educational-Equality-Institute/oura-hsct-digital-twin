# Viatom Checkme O2 Max — Integration Plan

**Target project:** `/home/henrik/projects/helse/oura-hsct-digital-twin/`
**Author:** integration planning pass, 2026-04-16
**Status:** Plan only, no code changes in this document.

## 0. Naming note (read first)

The user referred to the device as **"Checkme O2 Max Pro"**. Web search across
manufacturer sites (viatomtech.com, viatomcare.com, getwellue.com, trendmedic.de,
stowood.com, medgrade.org) and the official user manual (Viatom "Checkme O2 Max
Smart Wrist Pulse Oximeter User Manual", model name "**Oxiband**") does **not**
surface a SKU called "Checkme O2 Max Pro" as of 2026-04-16. The separate
"Checkme O2 Pro" is a different (fingertip) device. The "Checkme O2 Ultra"
(B0DN115TGB) is a newer line.

This plan assumes the device is the **Checkme O2 Max (wrist, Oxiband, 72 h battery)**.
See section 11 for the flagged open question.

## Context: prior work already in tree

- `api/import_checkme_spo2.py` already exists (532 lines), was written
  **speculatively** against assumed Viatom CSV conventions, and explicitly states
  "Format status (as of 2026-04-16): Checkme O2 Max PC software CSV export has
  NOT been validated against a real file." It writes to `checkme_spo2_continuous`
  and `checkme_spo2_sessions` tables, parses CSV only, and assumes `perfusion_index`
  and `motion` columns exist.
- `api/test_import_checkme_spo2.py` exists.
- `api/import_viatom_ecg.py` (599 lines) follows a similar speculative pattern for TH12.
- `api/import_omron.py` has the established **Windows BLE bridge** pattern:
  `C:\Users\ovehe\omron-bridge\omron_pull.py` writes CSVs to `/mnt/c/Users/ovehe/omron-bridge/out/`, WSL-side importer moves to `ingested/` after import.

**Key correction surfaced by research (see Section 1):**
The device does **not** export Perfusion Index or a Motion column. The OSCAR
binary format (authoritative) carries `spo2, hr, oximetry_invalid, motion, vibration`
at 4 s (older) / 2 s (Checkme O2 Max) resolution, where `motion` is a boolean-ish
flag **not** a perfusion number. The Viatom user manual specification section
lists "Record parameters: SpO₂, pulse rate" only. The existing importer's
`perfusion_index` column should be renamed or nulled out for Checkme O2 Max.

## 1. Data acquisition

### Verified device facts (source: Viatom "Checkme O2 Max User Manual", Specifications section)

| Field | Value |
|---|---|
| Model name | Oxiband |
| Record parameters | **SpO₂, pulse rate** (only) |
| Record interval (manual) | **4 s** |
| Record interval (real) | 2 s for `sig==5` files, 4 s for `sig==3`; see OSCAR `viatom_loader.cpp` |
| Storage | 4 records × 10 h = 40 h max on device |
| Battery | 72 h runtime, 2 h charge |
| SpO₂ range | 0-100 % |
| SpO₂ accuracy | ±2 % (70-100 %), ±3 % (70-80 %), not defined <70 % |
| Pulse rate | 30-250 bpm, ±2 bpm or ±2 % whichever greater |
| Wireless | **Bluetooth 4.0 BLE only** — no USB data, no microSD |
| PC software | "O2 Insight Pro" (download: iwearpulse.com/pages/app-download) |
| Mobile app | **ViHealth** (iOS ≥9.0, Android ≥5.0) |
| Sensor | FP-10R |
| Vibration alarm | Yes (bits 0x40 / 0x80 in binary record) |

### Samples per night

At 2 s sampling, a 10 h recording is **18 000 samples**. At 4 s it is 9 000. The
requirement "must handle 30k+ samples per night" is covered headroom-wise for a
single 10 h session; a 72 h recording with the new 2 s rate yields ~130 000 samples.
SQLite on an indexed `(session_id, timestamp)` table handles this trivially.

### Export paths (three options, ranked)

**Option A — ViHealth app → CSV (RECOMMENDED primary path)**
- The ViHealth app exposes "Share as CSV / binary / image" per manual page 9 and
  trendmedic.de product page. This is a per-session export, user-triggered.
- Pros: text, documented by manufacturer, column order is stable.
- Cons: manual action per session; columns are **SpO₂ + pulse rate only**
  (no PI, no motion).

**Option B — O2 Insight Pro (Windows/Mac desktop) → binary + CSV + PDF**
- Reads files over USB (charging cable doubles as data, pairing to desktop via
  PC software).
- Pros: also writes the OSCAR-compatible **binary** format which carries the
  `motion` and `vibration` flags that the CSV drops.
- Cons: requires Windows/Mac PC; user already has a Windows bridge machine
  (`ovehe` host running OMRON bridge). Software is closed-source.

**Option C — Direct BLE ingest (future)**
- Third-party OSS proofs of concept exist: `ecostech/viatom-ble`, `MackeyStingray/o2r`.
  Both target the O2Ring, neither tested on Checkme O2 Max. Not recommended yet.

### Chosen path

**Dual-channel acquisition**:

1. **Primary (day-one workable):** ViHealth CSV export on phone → iCloud
   Drive / Google Drive sync → WSL-visible path. Works without touching the
   Windows bridge box.

2. **Preferred (best data fidelity):** O2 Insight Pro on the Windows bridge
   machine (same host as `omron_pull.py`) emits the raw **binary** file
   (`.bin`) into `C:\Users\ovehe\checkme-bridge\out\`. The WSL importer parses
   the binary directly using the OSCAR-verified 40-byte header + 5-byte record
   format. This gives us motion and vibration flags the CSV throws away, and
   is fully automatable (O2 Insight Pro's export-on-dock behaviour can be
   scripted via AutoHotkey / simple watcher, same pattern as OMRON bridge).

   After successful ingest, move the `.bin` file to `…\out\ingested\` matching
   the OMRON bridge idiom.

Either path feeds the same ingester (section 4); format is auto-detected.

### Windows bridge plumbing (match OMRON pattern)

```
C:\Users\ovehe\checkme-bridge\
├── checkme_pull.py           # new: watches O2 Insight Pro export dir + copies .bin
├── out\                      # WSL reads from /mnt/c/Users/ovehe/checkme-bridge/out/
└── out\ingested\             # moved after import (mirrors omron-bridge convention)
```

The WSL importer defaults to `DEFAULT_INBOX = Path("/mnt/c/Users/ovehe/checkme-bridge/out")`.
Note that global rule "projects must live on native Linux filesystem" applies to
source code only, not to third-party bridge drops on Windows side.

## 2. Python libraries

### Recommended additions (in priority order)

**pobm (PhysioZoo Oximetry Biomarkers Toolbox) — PRIMARY ANALYTIC LIBRARY**
- Install: `pip install pobm`
- PyPI version: **1.2.0** (released 2022-10-20, no newer release as of 2026-04-16)
- License: **GPL-3.0** (verified from `aim-lab/Oximetry_Toolbox/LICENSE.txt`).
  Note: PyPI metadata field says "proprietary" but this is a packaging metadata
  error; the upstream repo is GPL-3.0. Flag: GPL-3 is copyleft. The
  digital-twin project is MIT (`LICENSE` in repo root). **Using pobm as an
  import-time dependency triggers GPL-3 on distributed derivatives**. Mitigation
  options: (a) isolate pobm behind a subprocess CLI boundary so the analytic
  pipeline uses it as a tool, not a linked library; (b) re-implement the
  Levy-2020 biomarkers from the published paper (GPL applies to code, not to
  the algorithm); (c) keep project private/internal — no distribution = no
  copyleft trigger. Given this repo is internal patient data analysis, option
  (c) is the pragmatic default. Document the call-site.
- Dependencies: numpy>1.18, scikit-learn>0.22, scipy>1.4, lempel-ziv-complexity==0.2.2
- Wheel size: 31.9 kB; source tarball 21.7 MB (contains large example data).
- Maintenance: "Beta", last release 2022. Low active maintenance but feature-
  complete for the Levy-2020 biomarker set. Acceptable for the narrow use case.
- Exports the exact biomarker set this plan needs:
  - `pobm.obm.desat.DesaturationsMeasures(ODI_Threshold, hard_threshold=90, threshold_method, ...)`
    returns `ODI`, `DL_u/sd` (desat length), `DA100_u/sd` (area from 100 %),
    `DAmax_u/sd` (area from personal max), `DD100_u/sd` (depth from 100), `DDmax_u/sd`,
    `DS_u/sd` (slope), `TD_u/sd` (time between events).
  - `pobm.obm.burden.HypoxicBurdenMeasures` — CT90/CA90 time-under-threshold.
  - `pobm.obm.general.OverallGeneralMeasures` — mean/SD/min/max/zero-crossings.
  - `pobm.obm.complex` — ApEn, LZ complexity, sample entropy.
  - `pobm.obm.periodicity` — frequency-domain features.
- ODI_3 and ODI_4 are both obtainable by instantiating `DesaturationsMeasures`
  twice with `ODI_Threshold=3` and `ODI_Threshold=4`.

**NeuroKit2 — SpO2 smoothing / signal cleaning (already planned)**
- Install: `pip install neurokit2`
- Version: 0.2.13 (released 2026-03-02, actively maintained)
- License: MIT
- Python: ≥3.10 (compatible with project's 3.12)
- Use for: `nk.signal_filter`, `nk.signal_smooth`, `nk.ppg_clean` (for the
  pulse-rate series in the same record).

**STUMPY — anomaly detection via matrix profile (already in requirements-full)**
- Install: `pip install stumpy` (already pinned `stumpy>=1.12`)
- Version: 1.14.1 (2026-02-08 release)
- License: BSD-3-Clause (verified via GitHub API, upstream repo LICENSE.txt)
- Use for: within-night SpO2 anomaly patterns, discords in pulse-rate series.

**FLIRT — (planned; REEVALUATE BEFORE ADDING)**
- Version: **0.0.2, pre-alpha, last release 2021-03-19**. GitHub last push
  2024-03-28. Low activity.
- License: MIT. Size ~40 kB.
- FLIRT is oriented toward HRV/EDA/ACC from Empatica-style wearables. It has
  no direct SpO₂ feature extractors. For the SpO₂ pipeline, FLIRT adds little
  value beyond what NeuroKit2 already provides. **Recommendation: defer FLIRT
  until a concrete need surfaces** (e.g. if we add E4/Garmin).

**sleepecg — candidate for REM/NREM staging from ECG (future, not day-one)**
- Version: 0.5.9, BSD-3-Clause, Python ≥3.10. Relevant once the TH12 ECG
  pipeline (`analyze_viatom_ecg.py`) is live. Not part of this plan.

### Libraries deliberately NOT added

- `oxyjson`, `pycheckme` — do not exist on PyPI as of 2026-04-16.
- `viatom_pc60fw` (sza2) — PC60FW fingertip, not Checkme O2 Max. Reference only.
- `viatom-ble` (ecostech) — BLE live-stream, not file ingest. Reference only.
- `o2r` (MackeyStingray) — O2Ring BLE, not Checkme O2 Max. Reference only.

## 3. Schema (four tables, one cleanup)

Existing schema in `api/import_checkme_spo2.py`:
- `checkme_spo2_continuous` (sample-level)
- `checkme_spo2_sessions` (session metadata)

**Proposed additions / corrections:**

### 3.1 `checkme_spo2_sessions` (KEEP, add columns)

Add columns for analytic outputs:
```
device_model TEXT                  -- "Checkme O2 Max" vs other Viatom
sig_version INTEGER                -- 3 or 5 (from binary header)
sample_interval_s INTEGER          -- 2 or 4
vibration_event_count INTEGER      -- count of records with vibration flag
```

### 3.2 `checkme_spo2_samples` (RENAME from `checkme_spo2_continuous`)

Migration: `ALTER TABLE checkme_spo2_continuous RENAME TO checkme_spo2_samples`.
Drop or nullify `perfusion_index` (device does not export it). Keep `motion_flag`
(the binary file carries motion as a per-record byte — see OSCAR parser).
Add `oximetry_invalid INTEGER` (per-record invalid flag from the binary format,
distinct from our own `is_artifact` post-processing).

```
timestamp TEXT             -- ISO 8601 local time
session_id TEXT
spo2 REAL                  -- 0-100 %
pulse_rate INTEGER         -- bpm
motion_flag INTEGER        -- 0/1, from binary record byte 3
vibration_flag INTEGER     -- 0/0x40/0x80, from binary record byte 4
oximetry_invalid INTEGER   -- 0 or 0xFF, from binary record byte 2
is_artifact INTEGER        -- our post-processing call
artifact_reason TEXT
source_file TEXT
imported_at TEXT
UNIQUE(session_id, timestamp)
```

### 3.3 `checkme_spo2_metrics` (NEW, one row per session)

```
session_id TEXT PRIMARY KEY REFERENCES checkme_spo2_sessions(session_id),

-- Levy 2020 digital-oximetry-biomarker set, ODI family
odi_3 REAL                 -- events/hour, drop ≥3 %, recover within 120 s
odi_4 REAL                 -- events/hour, drop ≥4 %, recover within 120 s

-- Time-below-threshold
t88_pct REAL               -- % of recording SpO2 < 88
t90_pct REAL               -- % of recording SpO2 < 90
t95_pct REAL               -- % of recording SpO2 < 95
ct90_min REAL              -- cumulative minutes < 90

-- Hypoxic burden
hypoxic_burden_area REAL   -- %·min below 90 % reference, trapezoid integral
hb_azarbarzin REAL         -- Azarbarzin 2019 event-anchored hypoxic burden (optional)

-- Descriptive
mean_spo2 REAL
sd_spo2 REAL
min_spo2 REAL              -- absolute minimum (can be artifact)
nadir_mean REAL            -- mean of lowest 10 valid samples
di_events_per_hour REAL    -- classic desaturation index (any threshold)

-- Event-derived aggregates
event_count INTEGER
mean_event_depth REAL
mean_event_duration_s REAL
mean_recovery_slope REAL   -- %/s

-- Coverage / QC
valid_sample_pct REAL      -- (non-artifact samples / total) × 100
duration_hours REAL

computed_at TEXT DEFAULT CURRENT_TIMESTAMP
```

### 3.4 `checkme_spo2_events` (NEW, one row per desaturation event)

```
id INTEGER PRIMARY KEY AUTOINCREMENT
session_id TEXT REFERENCES checkme_spo2_sessions(session_id)
start_ts TEXT
nadir_ts TEXT
end_ts TEXT
duration_s REAL
baseline_spo2 REAL         -- pre-event baseline (max in last 100 s, Azarbarzin)
nadir_spo2 REAL
drop_pct REAL              -- baseline_spo2 - nadir_spo2
recovery_slope REAL        -- %/s during recovery limb
area_under_baseline REAL   -- %·s, trapezoid integral vs. baseline
threshold_triggered TEXT   -- "3pct" | "4pct" | "hard_90" | "relative"
associated_pulse_change REAL  -- ΔPR at nadir vs. 60 s prior (coupling signal)
```

### 3.5 Indexes

```
CREATE INDEX idx_ckm_samples_session_ts ON checkme_spo2_samples(session_id, timestamp);
CREATE INDEX idx_ckm_samples_ts         ON checkme_spo2_samples(timestamp);
CREATE INDEX idx_ckm_events_session     ON checkme_spo2_events(session_id);
CREATE INDEX idx_ckm_events_nadir_ts    ON checkme_spo2_events(nadir_ts);
```

## 4. Ingester design

`api/import_checkme_spo2.py` needs revision, not rewrite.

### 4.1 Format auto-detection

Peek at first 2 bytes:
- `0x0003`, `0x0005`, `0x0301` → **Viatom binary** path (OSCAR-compatible).
  Parse per `oscar/SleepLib/loader_plugins/viatom_loader.cpp` `ParseHeader()`
  (40-byte header) + `ReadData()` (5-byte records, little-endian).
- Otherwise, attempt CSV decoding (existing logic).

### 4.2 Binary parser (new)

```python
# Header (40 bytes, little-endian)
# Offsets:  0-1  sig (uint16)
#           2-3  year (uint16)
#           4    month, 5 day, 6 hour, 7 min, 8 sec
#           9-11 filesize (24-bit)  -- use as sanity check
#          12    reserved (must be 0)
#          13-14 duration_s (uint16; real rec may be wider)
#          15-16 reserved (0)
#          17    spo2_avg (device-reported — DO NOT trust, recompute)
#          18    spo2_min
#          19    spo2_3pct event count
#          20    spo2_4pct event count
#          22-23 seconds_under_90
#          24    events_under_90
#          25    o2_score (×0.1)
#          27-39 reserved (most must be 0)
#
# Record (5 bytes):
#   spo2, hr, oximetry_invalid, motion, vibration
# oximetry_invalid==0xFF → sample is off-finger/invalid; spo2==0xFF, hr==0xFF
# vibration ∈ {0, 0x40, 0x80}
# For sig==3: samples are double-written, dedupe pairwise (OSCAR does this).
# For sig==5 (Checkme O2 Max): true 2 s OR 4 s resolution — compute from
#   duration/record_count.
```

### 4.3 CSV parser (keep existing, fix assumptions)

Keep existing flexible column detection but:
- Stop treating absence of `perfusion_index` column as a warning. It is expected.
- Confirm ViHealth CSV column names against a real export before merging (flagged
  in Section 11).

### 4.4 Chunked insertion

For 30k-130k sample sessions, use `executemany` with 1 000-row chunks inside a
single transaction. Existing loop-per-row `upsert_spo2_readings` is correct but
will be 10-30× slower than chunked. Switch to:

```python
cur.executemany(
    "INSERT OR IGNORE INTO checkme_spo2_samples (...) VALUES (?, ..., ?)",
    rows_chunk_1000,
)
```

### 4.5 Artifact detection (expand existing `detect_spo2_artifact`)

Current rules: physiologic range, motion==1, invalid reading. Add:
- `oximetry_invalid == 0xFF` → artifact with reason `device_invalid_flag`.
- **Perfusion-based filter from spec is impossible** (no PI exported). Compensate:
  flag samples where pulse rate changes by >40 bpm between consecutive 2 s samples
  as probable off-finger transitions.
- Run a rolling median (window 5) over valid SpO₂. Samples differing from rolling
  median by >8 % → artifact `rolling_median_outlier`. POBM's preprocessing
  functions `pobm.prep.block_data` / `pobm.prep.median_spo2` do this.

### 4.6 Session bounds

Current code derives `session_id` from start timestamp. Retain that. Add
`device_model` from the binary `sig` field (3→"Checkme O2", 5→"Checkme O2 Max",
0x0301→"O2Ring S"). For CSV, set it to "Checkme (CSV)" and `sig_version=NULL`.

## 5. Clinical metrics (analyzer: `analysis/analyze_checkme_spo2.py`, NEW)

New script, follows `analyze_omron_bp.py` and `analyze_oura_spo2_trend.py` patterns.

### 5.1 Per-session computations (Levy 2020 set)

```
For each session in checkme_spo2_sessions (loop):
    samples = SELECT spo2, timestamp FROM checkme_spo2_samples
              WHERE session_id=? AND is_artifact=0 AND oximetry_invalid=0
              ORDER BY timestamp
    x = samples.spo2  (numpy array)
    fs = 1 / (samples.sample_interval_s)   # 0.5 Hz or 0.25 Hz

    # ---- POBM: all five categories in one pass ----
    from pobm.obm.desat import DesaturationsMeasures, DesatMethodEnum
    from pobm.obm.burden import HypoxicBurdenMeasures
    from pobm.obm.general import OverallGeneralMeasures

    # ODI_3: ≥3% drop, 120 s recovery window
    desat3 = DesaturationsMeasures(
        ODI_Threshold=3, hard_threshold=90,
        threshold_method=DesatMethodEnum.Relative,
        desat_max_length=120,
    ).compute(x)

    # ODI_4
    desat4 = DesaturationsMeasures(ODI_Threshold=4, ...).compute(x)

    # Hypoxic burden / CT90 family
    burden = HypoxicBurdenMeasures(CT_Threshold=90, CA_Threshold=90).compute(x)

    # Descriptive
    general = OverallGeneralMeasures().compute(x)

    metrics = {
        "odi_3": desat3.ODI,
        "odi_4": desat4.ODI,
        "ct90_min": burden.CT * (sample_interval_s / 60),
        "t90_pct": 100 * (x < 90).sum() / len(x),
        "t88_pct": 100 * (x < 88).sum() / len(x),
        "t95_pct": 100 * (x < 95).sum() / len(x),
        "hypoxic_burden_area": burden.CA * (sample_interval_s / 60),
        "mean_spo2": general.AV,
        "sd_spo2": general.MED,  # POBM names
        "min_spo2": x.min(),
        "nadir_mean": np.sort(x)[:10].mean(),
        "mean_event_depth": desat4.DD100_u,
        "mean_event_duration_s": desat4.DL_u * sample_interval_s,
        "mean_recovery_slope": desat4.DS_u,
    }
```

### 5.2 Event extraction

POBM's internal desaturation detector is not trivially exposed as a public
event-by-event list (it returns aggregate stats). Two approaches:

1. Re-implement a minimal detector for population `checkme_spo2_events`:
   slide a 5-sample window, find local maxima as candidate baselines, require
   a ≥3 % (or ≥4 %) drop within 90 s and a return to within 1 % of baseline
   within 120 s — this is the AASM 2012 / Levy-2020 standard and is exactly
   what POBM does internally.
2. Use POBM's private helper `_pobm.obm.desat._find_desat_events` (not part
   of the public API; relying on internals is fragile).

**Recommendation:** implement the AASM detector in `analysis/_spo2_events.py`
(~80 lines, numpy only). Write unit tests against a synthetic signal with
known events. This keeps event-table ownership inside our codebase.

### 5.3 Azarbarzin 2019 hypoxic burden (optional, second pass)

Definition (verified from Azarbarzin 2019 PMC6451769):
- For each detected event, the pre-event baseline is the **maximum SpO₂ in
  the 100 s window preceding event end**.
- The event's burden contribution is the **area under that baseline**, bounded
  above by the baseline, lower-bounded by the signal, over a
  "subject-specific search window" derived from the ensemble average of
  time-aligned SpO₂ curves around events.
- Total hypoxic burden = sum of per-event areas / total sleep time (minutes).
- Units: %·min / h.

Implement as `compute_azarbarzin_hb(events, signal, fs)` in `_spo2_events.py`.

### 5.4 Trend across sessions

Once ≥3 sessions exist, compute night-to-night:
- ODI slope (linear regression over time).
- T90 slope.
- Rolling 7-night ODI_4 mean.
- Change-point detection via `ruptures` (already in `analyze_mitch_changepoints.py`).

## 6. BOS risk extension

Current `_bos_risk.py` consumes daily Oura SpO₂ summaries via a JSON payload
produced by `analyze_oura_spo2_trend.py`. Components:
- `spo2_slope`, `spo2_variability`, `desaturation_freq`, `bdi`, `hr_decoupling`.

**Proposed v2 score using continuous data (overlay, not replacement):**

Generate a second payload `reports/spo2_bos_metrics_continuous.json` with the
same BOS component vocabulary but finer inputs:

| Component v2 | Input | Rationale |
|---|---|---|
| `within_night_spo2_sd` | SD of SpO₂ within a single night, independent of day mean | Oura's 1-per-night aggregate cannot see this; it is the most sensitive early BOS signal |
| `awake_sleep_spo2_delta` | SpO₂_awake_mean - SpO₂_sleep_mean | Requires join with `oura_sleep.sleep_phase_5_min` via timestamp overlap |
| `rem_nrem_spo2_delta` | SpO₂ in REM epochs - SpO₂ in NREM epochs | Same join; REM drops are common in early BOS/OSA |
| `exercise_desat` | Max SpO₂ drop during Oura-flagged workouts | Requires join with `oura_daily_activity.non_wear_time` or Oura workouts endpoint |
| `continuous_odi_4` | ODI_4 from current night | Replaces Oura's coarse "desaturation count" |

### Keeping v1 alive

`_bos_risk.py` should stay backward compatible. Add `load_bos_risk_continuous()`
alongside `load_bos_risk()`. Downstream reports can opt into v2 by reading the
new payload when present.

### Composite-score update

Current weights (from `analyze_oura_spo2_trend.py` line 82):
```
"desaturation_freq": 0.20
```
Suggested v2 reweighting (keep sum=1.0):
```
"within_night_spo2_sd":    0.25  (new; strongest early BOS signal)
"continuous_odi_4":        0.20  (replaces desaturation_freq)
"rem_nrem_spo2_delta":     0.15
"awake_sleep_spo2_delta":  0.10
"spo2_slope":              0.15  (kept)
"hr_decoupling":           0.15  (kept)
```

**Do not** ship weight changes until paired validation on ≥4 weeks of continuous
data. Start by emitting the metric under `bos_continuous_components` without
changing the composite.

## 7. OSA screening

### 7.1 Threshold-based severity lookup

Use the standard Levy-2020 / AASM cutoffs (verified from multiple sources,
including ERS publication PA2316 and Spring-Nature article s11325-023-02814-3):

| ODI_4 | Severity label |
|---|---|
| < 5 | Normal |
| 5 – 14 | Mild |
| 15 – 29 | Moderate |
| ≥ 30 | Severe |

Report these alongside the numeric ODI.

**Note:** I was unable to verify a formal "ATS/ERS joint guideline" document
recommending pulse oximetry as a standalone OSA diagnostic. The literature I
found (PMC7065557, PMC4407425, PMC7676978) treats overnight oximetry as a
**validated screening tool for moderate-severe OSA in high-pretest-probability
populations**, not a diagnostic replacement for polysomnography. Report
wording should match that framing.

### 7.2 STOP-BANG hook (future)

Already scaffolded in `api/import_symptom.py`. STOP-BANG = 8 yes/no items
(Snoring, Tiredness, Observed apnea, Pressure / BP, BMI>35, Age>50, Neck>40 cm,
male Gender). Add a form export in the symptom importer; join into the OSA
report via patient_id.

### 7.3 Probabilistic OSA likelihood (Levy-style lookup)

Levy et al. 2020 (s41746-020-00373-5, Table 5) publish a 2-D lookup of
P(AHI≥15 | ODI_4, T90). Implement as a small CSV fixture:
`analysis/_osa_probability_table.csv`. Emit `osa_probability_moderate_plus` in
the per-session metrics.

## 8. Cross-modal joins

All joins go through `session_id` → `(start_datetime, end_datetime)` window on
`checkme_spo2_sessions`, intersected with external tables on `timestamp`.

### 8.1 Per-sleep-stage SpO₂

```sql
SELECT
  oura_sleep_stage,
  AVG(ck.spo2)  AS spo2_mean,
  MIN(ck.spo2)  AS spo2_min,
  100.0 * SUM(CASE WHEN ck.spo2 < 90 THEN 1 ELSE 0 END) / COUNT(*) AS pct_below_90
FROM checkme_spo2_samples ck
JOIN oura_sleep_5min os
  ON ck.timestamp BETWEEN os.start_datetime AND os.end_datetime
WHERE ck.session_id = ?  AND ck.is_artifact = 0
GROUP BY oura_sleep_stage;
```

Store in `checkme_spo2_sleep_stage_stats` (new small table, one row per
session × stage).

### 8.2 HR-SpO₂ coupling during desaturation

Already captured per-event as `associated_pulse_change` in
`checkme_spo2_events`. Compute correlation between SpO₂ nadir depth and
paired pulse-rate rise: expected positive r (arousal / sympathetic response).
Weak coupling (r < 0.3 in a session with ≥10 events) may indicate autonomic
blunting — relevant for post-HSCT autonomic dysfunction surveillance.

### 8.3 OMRON BP within 30 min of desaturation event

```sql
SELECT e.id, o.datetime, o.sys, o.dia, o.bpm
FROM checkme_spo2_events e
JOIN omron_bp_readings o
  ON o.datetime BETWEEN datetime(e.nadir_ts, '-15 minutes')
                  AND datetime(e.nadir_ts, '+30 minutes')
  AND o.is_artifact = 0
WHERE e.session_id = ?
ORDER BY e.nadir_ts;
```

Report mean BP within the desat-adjacent window vs. session-wide mean BP;
a >10 mmHg SBP rise in the 30-min post-desat window is a nontrivial
cardiovascular signal worth flagging.

### 8.4 Future TH12 ECG join

Once `viatom_ecg_events` is populated:

```sql
SELECT e.nadir_ts, ecg.event_type, ecg.severity
FROM checkme_spo2_events e
JOIN viatom_ecg_events ecg
  ON ecg.timestamp BETWEEN datetime(e.nadir_ts, '-30 seconds')
                      AND datetime(e.nadir_ts, '+60 seconds')
WHERE e.session_id = ?;
```

Focus event types: `afib`, `pvc` cluster, `bradycardia`, `pause`. Post-HSCT
patients with hypoxic arousals have measurable PVC rate increases during
desaturations — worth quantifying.

## 9. Dashboard

Output file: `reports/checkme_spo2_report.html`. Uses `analysis/_theme.py`
for consistency with existing reports.

### 9.1 KPI cards (top row, 4 cards)

- **ODI_4** → severity label + colour (normal/mild/moderate/severe, matching
  `_bos_risk.py` status tokens).
- **T90 %** → % of recording below 90.
- **Min SpO₂** (and nadir mean) → two-tier.
- **BOS composite v2** → if available, otherwise v1 fallback.

### 9.2 Overnight tracing (main panel)

- Scrollable horizontal time series, SpO₂ and pulse rate as twin-y overlay.
- Shaded vertical bands for detected desaturation events, colour-coded by
  depth (≥3 %, ≥4 %, ≥10 %).
- Horizontal reference lines at 90 % and 88 %.
- Artifact regions rendered as a thin grey overlay at the top of the chart
  (so they are visible but do not hide data).

### 9.3 Per-event detail table

Sortable table from `checkme_spo2_events`: start_ts, duration, depth,
recovery slope, associated ΔPR, associated BP if joined.

### 9.4 Trend sparklines (once ≥7 sessions exist)

Small multiples of ODI_4, T90, mean SpO₂, SD SpO₂ across the last N nights.
Matches the visual pattern in `analyze_weekly_tracker.py`.

### 9.5 Event depth-duration scatter

X: event duration (s). Y: event depth (Δ% from baseline). Colour: sleep stage
if cross-modal join available. This is the canonical Levy-2020 visualisation;
it separates OSA-like (deep, short) from central-apnoea-like (deep, long)
morphologies.

### 9.6 BOS risk panel (if v2 payload present)

Decomposition bar chart showing each of the 6 components. Matches the
existing `spo2_bos_screening.html` visual layout.

## 10. Pipeline integration

### 10.1 `scripts/daily_pipeline.sh` additions

Insert between existing step 3b (OMRON) and step 4 (analysis):

```bash
# 3c. Import Checkme O2 Max sessions from the Windows bridge (if any files waiting).
#     Gracefully skips when the inbox is absent or empty.
echo "[3c/5] Importing Checkme O2 Max sessions..."
python "$DIGITAL_TWIN/api/import_checkme_spo2.py" \
    --inbox /mnt/c/Users/ovehe/checkme-bridge/out \
    || echo "  Checkme import non-fatal warning (continuing)"
echo "  Checkme import done."
```

Requires adding `--inbox` flag to the importer (currently takes only `--csv`).
Pattern matches OMRON: scan the inbox, import each file, move to `ingested/`.

### 10.2 `run_all.py` `SCRIPTS` list addition

Add after `analyze_omron_bp.py` (line 55) and before `generate_roadmap.py`:

```python
"analyze_checkme_spo2.py",
```

### 10.3 `SEND_BUNDLE_HTML` / `SEND_BUNDLE_JSON` additions

Add:
```
SEND_BUNDLE_HTML: "checkme_spo2_report.html"
SEND_BUNDLE_JSON: "checkme_spo2_metrics.json"
```

### 10.4 `analysis/_bos_risk.py` addition

New function `load_bos_risk_continuous()` reads
`reports/spo2_bos_metrics_continuous.json` (produced by
`analyze_checkme_spo2.py`). Used by `generate_treatment_report.py` if present.

### 10.5 Tests

- `api/test_import_checkme_spo2.py` already exists — extend with:
  - Binary format round-trip (craft a 40-byte header + 100 records, parse, compare).
  - CSV with non-ASCII (trendmedic-style `°C` / EU decimal comma already handled;
    confirm with a real fixture).
  - Artifact detection edge cases: `oximetry_invalid == 0xFF`.
- New `analysis/test_checkme_metrics.py`:
  - Synthetic signal with exactly 5 desaturation events at known times; assert
    ODI_3 computation returns 5/hour-scaled value; assert T90 matches manual
    count.
  - Azarbarzin area computation against a worked hand example.

### 10.6 Config updates

`config.py` / `config.example.py`: add `CHECKME_INBOX_DIR = Path(...)` with a
default matching OMRON conventions.

## 11. Open questions (do NOT guess — resolve before implementation)

1. **Device SKU ambiguity.** User referred to "Checkme O2 Max Pro". No
   verified SKU by that exact name. Possibilities:
   a. User means "Checkme O2 Max" (most likely — all consumer-facing docs
      call it "Max" and sometimes marketing adds "Pro" loosely).
   b. User means "Checkme O2 Pro" (a different **fingertip** device —
      wrist-worn framing in the prompt argues against this).
   c. User means "Checkme O2 Ultra" (newer wrist model).
   d. Regional SKU variant.
   **Action:** confirm device model string from the engraved back-of-device
   label (Section 2.9 of the user manual explains where) or from the ViHealth
   app "Device info" panel before finalising the importer.

2. **Real ViHealth CSV column names.** The existing importer assumes
   `Timestamp/Time, SpO2/Oxygen, PR/Pulse/HR, PI/Perfusion, Motion/Movement`.
   The specifications page confirms only SpO₂ and pulse rate are recorded.
   **Action:** generate one real export and paste the header row. Do not
   merge the revised importer until this is done.

3. **Real ViHealth binary signature for the O2 Max.** OSCAR's code
   distinguishes `sig==3` (older, double-written 2 s) from `sig==5` (true 2 s
   on Checkme O2 Max). This needs to be verified against a real `.bin` file
   produced by the **current** ViHealth build and the **current** O2 Insight
   Pro build — both may have changed since the OSCAR code was last updated.

4. **Sampling rate after firmware updates.** Viatom has shipped at least one
   firmware update since 2023 that nominally lifted the rate from 4 s to 2 s.
   Confirm by dividing file-header `duration` by record count on the first
   real capture.

5. **O2 Insight Pro file path / filename convention.** We do not know where
   O2 Insight Pro writes exports by default on Windows 11. Needed before
   writing the `checkme-bridge` watcher. (`%USERPROFILE%\Documents\...` is a
   common Viatom default but not verified.)

6. **POBM GPL-3 vs. project licence.** The digital-twin repo is MIT; pobm is
   GPL-3. Because the project is internal and not redistributed, we are
   inside the "internal use" safe harbour — but **if** you ever publish a
   derivative, the subprocess-boundary mitigation or reimplementation is
   required. Confirm distribution intent.

7. **Azarbarzin hypoxic burden search-window definition.** The original paper
   defines a "subject-specific search window from the ensemble average of
   time-aligned SpO₂ curves" — this requires an ensemble of events, which for
   a first single-night session is unstable. Decision needed: use a
   fixed-window approximation (e.g. 100 s from event end, the published
   default) vs. adaptive per-subject vs. omit until ≥10 sessions. Default
   proposed: fixed 100 s for first implementation, switch to ensemble-based
   once we have ≥5 nights of data.

8. **Night-boundary definition.** Does an overnight recording always start
   with the device already on and SpO₂ valid? OSCAR notes that the first
   samples can be noisy "mount-on" transitions. Decision: trim the first N
   seconds (proposed N=30) and last 30 s at artifact flag stage.

9. **Which ODI threshold to feature in the BOS composite v2.** Literature
   favours ODI_4 for OSA screening but ODI_3 for interstitial lung disease /
   early BOS (the BOS setting is more relevant here). Decide after seeing
   Henrik's first few sessions whether ODI_3 or ODI_4 tracks better with
   other markers (peripheral O₂, spirometry).

10. **Perfusion-index workaround.** The device does not export PI, so we
    cannot use low-PI filtering the way OSCAR does. Is the pulse-rate
    discontinuity heuristic (Section 4.5) acceptable, or do we accept a
    higher artifact pass-through rate and compensate with wider rolling
    median windows? Needs a real artifact-heavy night to decide.

## Summary of deliverables this plan implies

| File | Action |
|---|---|
| `api/import_checkme_spo2.py` | Revise: add binary parser, add `--inbox`, fix perfusion/motion assumptions, add chunked insert |
| `api/test_import_checkme_spo2.py` | Extend: binary round-trip, ViHealth-CSV real fixture, artifact edge cases |
| `analysis/analyze_checkme_spo2.py` | NEW: per-session Levy biomarkers, BOS v2 payload, HTML report |
| `analysis/_spo2_events.py` | NEW: AASM desaturation detector + Azarbarzin burden |
| `analysis/test_checkme_metrics.py` | NEW: synthetic-signal unit tests |
| `analysis/_bos_risk.py` | Extend: `load_bos_risk_continuous()` |
| `analysis/_osa_probability_table.csv` | NEW: Levy-2020 lookup |
| `scripts/daily_pipeline.sh` | Add step 3c (Checkme import) |
| `scripts/checkme_bridge_setup.md` | NEW doc: Windows-side O2 Insight Pro watcher |
| `run_all.py` | Add `analyze_checkme_spo2.py` to `SCRIPTS`, report files to `SEND_BUNDLE_*` |
| `config.py` / `config.example.py` | Add `CHECKME_INBOX_DIR` |
| `requirements-full.txt` | Add `pobm>=1.2` with licence caveat comment; add `neurokit2>=0.2.13` if not present |
| `data/*.db` schema | Migrate `checkme_spo2_continuous` → `checkme_spo2_samples`; add `checkme_spo2_metrics`, `checkme_spo2_events`, `checkme_spo2_sleep_stage_stats` |

Total new Python code estimate: ~1 200 lines. Revision of existing: ~300 lines.
No new services, no non-Python dependencies. CPU-only. Works offline.

## Verified references

- Viatom "Checkme O2 Max Smart Wrist Pulse Oximeter User Manual"
  (millenniumsleeplab.com/wp-content/uploads/2024/09/Checkme-O2-Max-Manual.pdf)
  — verified SpO₂/PR accuracy, 4 s record interval (manual), 4 × 10 h storage,
  Bluetooth 4.0 BLE only, no PI/motion CSV columns.
- OSCAR `viatom_loader.cpp` / `viatom_loader.h`
  (gitlab.com/pholy/OSCAR-code) — verified 40-byte header, 5-byte record format,
  signatures 0x0003/0x0005/0x0301, Checkme O2 Max = sig 5 (true 2 s sampling),
  field layout. License: GPL-3.
- POBM 1.2.0 on PyPI + aim-lab/Oximetry_Toolbox GitHub (LICENSE.txt) —
  verified GPL-3 licence, DesaturationsMeasures / HypoxicBurdenMeasures /
  OverallGeneralMeasures classes, ODI_Threshold parameter.
- Levy et al. 2020, npj Digital Medicine article s41746-020-00373-5 — verified
  the "digital oximetry biomarkers" terminology and biomarker set.
- Azarbarzin et al. 2019, PMC6451769 — verified hypoxic-burden definition
  (100 s pre-event baseline, event-anchored area under desaturation curve,
  units %·min / h).
- NeuroKit2 0.2.13 (pypi.org/pypi/neurokit2) — MIT, Python ≥3.10, active
  (last release 2026-03-02).
- STUMPY 1.14.1 (github.com/TDAmeritrade/stumpy LICENSE.txt) — BSD-3-Clause,
  Python ≥3.10, active (last release 2026-02-08).
- FLIRT 0.0.2 (pypi.org/pypi/flirt + github.com/im-ethz/flirt) — MIT,
  pre-alpha, low activity (last release 2021-03, last push 2024-03).
- OSA severity cutoffs — concurrent with multiple peer-reviewed sources
  (PMC7065557, PMC4407425, PMC7676978, Sleep and Breathing 2023 article
  s11325-023-02814-3). ATS/ERS joint "guideline" per se not verified;
  framing in report must be "screening tool", not "diagnostic".
