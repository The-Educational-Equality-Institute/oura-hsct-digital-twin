# 12 — Contec ABPM50 Integration Plan

**Status:** research complete, awaiting first real `.awp` export (~2026-04-28) for validation
**Target device:** Contec ABPM50 (24-hour ambulatory blood pressure monitor, oscillometric)
**Target DB:** `data/oura.db` (project default) or profile-specific (`henrik`, `mitch`)
**Patient context:** Henrik, 36, post-HSCT MDS-AML, on ruxolitinib + bisoprolol 2.5mg daily from 2026-04-08

Existing scaffolding already present in the repo (verified 2026-04-16):
- `api/import_contec_abpm.py` — permissive CSV ingester stub (not yet validated against a real export)
- `api/test_import_contec_abpm.py` — synthetic-fixture tests for the stub
- Schema: `contec_abpm_readings`, `contec_abpm_sessions` (both created by `init_contec_abpm_tables`)

This plan documents (a) what the stub was guessing at, (b) what the research confirmed, (c) what still needs a real-file validation pass, and (d) what to build next (clinical metrics + analysis module).

---

## Research findings (verified 2026-04-16)

### Device characteristics
- Oscillometric NIBP holter; records SYS, DIA, MAP, HR + timestamp.
- Up to **350 groups** of ambulatory NIBP readings per session (per Contec product page; sufficient for 24 h even at 10-min intervals).
- Internal clock; timestamps recorded on-device.
- **Default measurement schedule** (per Contec manual variants): awake **15 min**, asleep **30 min**; awake window 07:00, sleep window 22:00 — user-configurable (selectable intervals 5/10/15/20/30/40/60/90/120/180/240 min).

### Connection + export path
- USB cable (Type-A to mini/micro-B on the device).
- Enumerates as **STM32 Virtual COM Port** on Windows (VID 0x0483, PID 0x5740). STM's VCP driver is usually auto-installed on Windows 10/11; the manual recommends STSW-STM32102 for older Windows.
- Official Contec PC software is Windows-only. It connects over the virtual COM port and writes session files to a local folder.
- Export formats from the vendor software: **PDF report**, and `.awp` native session files. Direct CSV export is **not a first-class feature** of the vendor software — most users either screenshot/PDF the report or export to Excel from within the app.

### `.awp` file format (IMPORTANT: no direct binary readout)

`.awp` is the Contec ABPM50 session format. **It is text-based** (ASCII), not binary — confirmed by two independent reverse-engineering efforts:

- **obidose/abpm50ex** (GitHub) — Python parser, reads `.awp` directly and exports Excel. Parses v2 format only; the `read_v2` function iterates lines, extracts fields by substring match on keys (`ID=`, `Name=`, `Age=`) and parses measurement rows keyed by record number.
- **Alexander Mumm (alexmumm.de)** — independent reverse-engineering document describing the record layout.

**Record layout (per-measurement line), hex-encoded ASCII:**

Per Mumm's reverse-engineering, each record line is `[recnum]=[ssddppaammmm??]` where:
- `ss` (2 hex chars) = systolic, mmHg
- `dd` (2 hex chars) = diastolic, mmHg
- `pp` (2 hex chars) = pulse rate, bpm
- `aa` (2 hex chars) = MAP, mmHg
- `mmmm` (4 hex chars) = minutes elapsed since measurement start
- `??` = unknown trailing bytes, possibly an "edited in PC software" marker

Per `obidose/abpm50ex`'s `parse_data` function, the byte offsets in the hex string are:
- `[4:8]` year, `[8:10]` month, `[10:12]` day, `[12:14]` hour, `[14:16]` minute (hex → int)
- `[20:22]` SYS, `[24:26]` DIA, `[28:30]` MAP, `[32:34]` HR (hex → int)

The two documentations differ — Mumm says elapsed-minutes, `abpm50ex` says absolute date components embedded in the record. Both approaches exist; the `abpm50ex` offsets match the format stored on current-production units. Needs validation against a real file from Henrik's unit.

**Session header fields in `.awp`:**
- `FileVersion_Main=2` (version marker)
- `ID=` patient ID
- `Name=` patient name
- `Age=` age (integer)
- `MinBegin`, `HourBegin`, `DayBegin`, `MonthBegin`, `YearBegin` — session start components

**Validation logic in `abpm50ex`:** `valid = 1 if SYS > DIA + 20 else 0`. Same narrow-pulse-pressure heuristic already in `import_contec_abpm.py::detect_artifact`.

**License status of `abpm50ex`:** repo has **no LICENSE file** (verified with HTTP 404 on `/LICENSE` and `/LICENSE.md`). Not redistributable. We should not vendor or copy it; we can read it as a format reference and write our own parser.

### Python library landscape
- **PyPI search for `abpm` / `ambulatory`:** no mature, MIT/Apache/BSD package exists for ABPM analysis. `pypda` exists but is for synthetic waveform simulation, not ABPM.
- **R `bp` package** (Schwenck, Punjabi, Gaynanova; PLOS One 2022; CRAN) — the mature reference implementation. GPL-2 | GPL-3 license. Implements `bp_arv`, `bp_sv`, `bp_cv`, `bp_mag`, `dip_calc`, `sleep_metrics` (incl. MBPS sleep-trough + prewake), `bp_stats`, `bp_stages`. Does **not** implement AASI, hyperbaric index, or BP-load out of the box.
- **neurokit2** (MIT) and **pyhrv** (BSD) — relevant for the pulse-rate stream as a coarse HRV surrogate, **not** for ABPM metrics. Useful if we want to cross-check cuff pulse against Oura/Viatom HR.
- **Conclusion:** there is no MIT/Apache Python ABPM library worth adopting. Port the needed bp-package metrics to native Python. Do **not** take a hard dependency on `rpy2` — it adds a full R runtime for ~200 lines of equivalent numpy/pandas code.

### Clinical relevance sources (for section 7)
- TA-TMA early-warning literature: elevated BP is an early marker of TA-TMA, detectable weeks before hematologic/renal signals (Jodele et al.; Nature BMT review). Patients requiring >2 BP meds post-HSCT warrant TA-TMA workup. Non-dipping pattern on ABPM is a recognised early autonomic/endothelial signal.
- Ruxolitinib BP effect: real-world data shows mean SBP rising ~5 mmHg at 72 weeks on rux; prevalence of hypertension +~5 percentage points. Mechanism plausibly eNOS downregulation via JAK2 inhibition (Sci Rep 2019). Individual patients can see +20 mmHg. This is a named AE to watch; ABPM catches it before spot-check home readings drift.
- Bisoprolol 2.5 mg onset: some BP reduction within 24 h of first dose, near-steady-state by week 1. Duration of action ~24 h (genuinely once-daily). So for Henrik (started 2026-04-08), an ABPM 2–3 weeks post-start (~2026-04-28) captures both the drug-response plateau and any rux-driven upward drift.

---

## 1. Data extraction

**Current reality: no direct-from-USB pipeline exists in the open-source world for the ABPM50.** The vendor's Windows software is in the path whether we like it or not. We mirror the OMRON M7 pattern: a Windows-side bridge does the vendor-coupled work, writes a deterministic export to an inbox directory, and the Linux-side ingester picks it up.

**Pipeline options, in order of preference:**

**Option A (preferred): `.awp` on Windows → Linux parser.**
1. Contec ABPM50 PC Software on Windows (vendor install, from the bundled CD or `dlsoftw.com` with index `05RK1069` — Henrik to verify on the sticker / packaging; do not ship that URL without re-checking).
2. Connect device via USB cable (STM32 VCP, shows as a COM port).
3. Vendor software reads the session and saves a `.awp` file. Locate the default save folder (typically `Documents\ABPM50\` or similar — Henrik to confirm at first download).
4. Copy/sync the `.awp` to `/mnt/c/Users/ovehe/abpm-bridge/in/` (parallels the existing OMRON bridge layout at `/mnt/c/Users/ovehe/omron-bridge/out/`).
5. Linux-side `api/import_contec_abpm.py` scans the inbox, parses `.awp` directly, moves to `ingested/`.

**Option B (fallback): `.awp` → CSV via vendor Excel export.**
1. Use vendor software's "Export to Excel" (present in most ABPM50 software builds).
2. Save as `.xlsx` or re-export as CSV.
3. Current stub `import_contec_abpm.py` already handles CSV with fuzzy column matching — keep as fallback path.

**Option C (last resort): `.awp` → CSV via `abpm50ex` on Windows.**
- Only if vendor software has no CSV/Excel path.
- Do not vendor `abpm50ex`; use its published Windows `.exe` (in the repo's `/bin/`) **or** write our own parser (preferred — the format is simple enough).

**Recommendation:** Implement Option A (native `.awp` parser in `import_contec_abpm.py`) as the primary path. Keep the existing CSV path as a fallback for any future CSV export.

**Windows bridge `abpm_pull.py` (minimal, parallels `omron_pull.py`):**
The ABPM50 is NOT Bluetooth — unlike the OMRON M7 bridge which uses `omblepy`. The "bridge" here is trivial:
- Watch the vendor's default save folder.
- Copy new `.awp` files to the synced inbox.
- Optionally rename to `abpm_YYYYMMDD_HHMMSS.awp`.
- Write a sibling `abpm_YYYYMMDD_HHMMSS.json` with metadata: `{device_serial, software_version, pulled_at}`.

Or, even simpler, skip the bridge entirely and configure the vendor software's save path to point directly into `/mnt/c/Users/.../abpm-bridge/in/` — WSL mounts it at `/mnt/c/...`.

---

## 2. Python libraries to install

**Already in `requirements.txt`:** `numpy`, `pandas`, `scipy`, `statsmodels`, `plotly`, `scikit-learn`, `pandera`. These cover everything needed for ABPM ingest + core metrics.

**New additions (all MIT/BSD/Apache — confirmed):**

| Package | License | Purpose | Required? |
|---|---|---|---|
| `openpyxl` | MIT | Optional: parse `.xlsx` fallback from vendor Excel export | Optional |

**Deliberately NOT adding:**
- **`rpy2` + R + `bp` package** — rpy2 is MIT but pulling in a full R runtime for ~10 metric functions is excessive for a lean pipeline. The `bp` package is GPL-2|3 which would force license review anyway. We port the metrics natively.
- **`pyhrv` / `neurokit2`** for ABPM — these are HRV packages for continuous ECG/PPG waveforms. Cuff pulse values from ABPM are a discrete-samples stream (one value per measurement, ~50 samples/24h). Not enough density for meaningful HRV. If we want HRV, use Oura or Viatom ECG, not the cuff pulse. Skip for this module.

Net additions to `requirements.txt`: **one line** (`openpyxl`), and even that is optional. The existing numpy/pandas/scipy/statsmodels stack is sufficient.

---

## 3. Schema proposal

The current stub already has `contec_abpm_readings` and `contec_abpm_sessions`. The plan extends with `contec_abpm_metrics` (per-session computed metrics) and tightens the reading fields now that we have the `.awp` format in hand.

**`contec_abpm_readings`** (extend existing):
```
id INTEGER PRIMARY KEY AUTOINCREMENT
datetime TEXT NOT NULL                   -- ISO 8601, parsed from .awp
session_id TEXT NOT NULL                 -- e.g. abpm-20260428-090000
sequence_number INTEGER                  -- 1..N from .awp record key
sys INTEGER NOT NULL                     -- mmHg
dia INTEGER NOT NULL                     -- mmHg
bpm INTEGER                              -- pulse rate from cuff
map_mmhg REAL NOT NULL                   -- from .awp (preferred) else computed
pulse_pressure INTEGER NOT NULL          -- SYS - DIA
error_code INTEGER                       -- vendor error code if present
day_night TEXT                           -- 'day' | 'night' (ESH fixed window initially)
period TEXT                              -- NEW: 'awake' | 'sleep' (Oura-informed)
manual_flag INTEGER DEFAULT 0            -- NEW: 1 if user-initiated (via ABPM50 button)
is_artifact INTEGER DEFAULT 0
artifact_reason TEXT
source_file TEXT
imported_at TEXT DEFAULT CURRENT_TIMESTAMP
UNIQUE(session_id, datetime)
```

Additions vs. current stub: `period` (awake/sleep derived from Oura sleep period overlap — distinct from `day_night` which is a static clock window), `manual_flag` (some ABPM exports expose a "manual" vs "auto" bit; include it now so we don't have to migrate later).

**`contec_abpm_sessions`** (extend existing):
```
id INTEGER PRIMARY KEY AUTOINCREMENT
session_id TEXT UNIQUE NOT NULL
start_datetime TEXT NOT NULL
end_datetime TEXT
duration_hours REAL                       -- NEW
reading_count INTEGER
valid_reading_count INTEGER               -- NEW: after artifact filter
success_rate REAL                         -- NEW: valid / attempted (ESH >70% required)
day_readings INTEGER                      -- fixed-window ESH day
night_readings INTEGER                    -- fixed-window ESH night
awake_readings INTEGER                    -- NEW: Oura-informed awake
sleep_readings INTEGER                    -- NEW: Oura-informed sleep
protocol_code TEXT                        -- NEW: e.g. 'ESH-day15-night30'
device_serial TEXT
source_file TEXT
imported_at TEXT DEFAULT CURRENT_TIMESTAMP
```

**`contec_abpm_metrics`** (NEW — one row per session):
```
id INTEGER PRIMARY KEY AUTOINCREMENT
session_id TEXT UNIQUE NOT NULL

-- Mean BP
mean_sys_24h REAL
mean_dia_24h REAL
mean_map_24h REAL
mean_hr_24h REAL
mean_sys_day REAL
mean_dia_day REAL
mean_sys_night REAL
mean_dia_night REAL
mean_sys_awake REAL                       -- Oura-derived
mean_dia_awake REAL
mean_sys_sleep REAL
mean_dia_sleep REAL

-- Dipping (based on Oura-derived awake/sleep, with fixed-window fallback)
dip_sys_pct REAL                          -- (1 - mean_sleep/mean_awake) * 100
dip_dia_pct REAL
dip_category TEXT                         -- 'extreme' | 'normal' | 'non' | 'reverse'
dip_period_used TEXT                      -- 'oura' | 'fixed_esh'

-- Morning BP surge
mbps_sleep_trough_sys REAL                -- postwake mean - sleep-trough mean
mbps_sleep_trough_dia REAL
mbps_prewake_sys REAL                     -- postwake mean - prewake mean
mbps_prewake_dia REAL

-- BP load
load_day_sys_pct REAL                     -- % of day readings > 135
load_day_dia_pct REAL                     -- > 85
load_night_sys_pct REAL                   -- > 120
load_night_dia_pct REAL                   -- > 70 (ESH; see note §5)

-- Variability
sd_sys_24h REAL
sd_dia_24h REAL
wsd_sys REAL                              -- weighted SD
wsd_dia REAL
cv_sys_pct REAL
cv_dia_pct REAL
arv_sys REAL                              -- average real variability
arv_dia REAL

-- Arterial stiffness / hyperbaric
aasi REAL                                  -- 1 - slope(DIA ~ SYS)
aasi_r_squared REAL
hyperbaric_sys_day REAL                    -- AUC over 135 (mmHg * hours)
hyperbaric_dia_day REAL                    -- AUC over 85
hyperbaric_sys_night REAL                  -- AUC over 120
hyperbaric_dia_night REAL                  -- AUC over 70

-- Smoothness index (populated when baseline session available)
smoothness_sys REAL                        -- effect_size / sd_of_change
smoothness_dia REAL
baseline_session_id TEXT                   -- session used as baseline for smoothness

computed_at TEXT DEFAULT CURRENT_TIMESTAMP
```

Storing computed metrics in-DB rather than recomputing lets downstream comparative scripts (e.g. `analyze_comparative_treatment.py`) consume them without re-importing the raw readings.

---

## 4. Ingester design

Existing `api/import_contec_abpm.py` already implements:
- Tables + indexes (idempotent)
- Permissive CSV parser with fuzzy column detection
- Day/night classification (fixed ESH window 07:00–21:59 / 22:00–06:59)
- Artifact detection (range checks + narrow pulse pressure)
- Upsert with `INSERT OR IGNORE` on `(session_id, datetime)`
- Session summary row

**Keep as-is.** Extend with:

### 4.1. Add `.awp` parser

```
parse_awp(path: Path, session_id: str) -> list[AbpmReading]
```
- Open as text (confirmed ASCII per Mumm + abpm50ex).
- Detect version via `FileVersion_Main=N`. Support v2 (current); raise clear error for v1 until we see one.
- Pull session header: `ID`, `Name`, `Age`, `MinBegin`/`HourBegin`/`DayBegin`/`MonthBegin`/`YearBegin`, device serial if present.
- Iterate non-header lines. For lines matching `^(\d{1,3})=([0-9A-Fa-f]+)$`, parse per the two known layouts:
  - **Primary attempt:** `abpm50ex` offsets (absolute date components in hex at `[4:16]`, SYS/DIA/MAP/HR at `[20:34]`).
  - **Fallback:** Mumm's "elapsed minutes" layout: derive datetime from `YearBegin..MinBegin` + `elapsed_min` parsed from `[offset:offset+4]`.
  - Auto-detect which layout the file uses: if the hex at `[4:16]` decodes to a plausible (year, month, day, hour, minute), use the absolute layout. Otherwise, use elapsed-minutes.
- Reject measurement records that don't decode to physiologic ranges (reuse `detect_artifact` unchanged).

**Per-record parse (pseudocode, NOT a coding task — design contract):**
```
year  = int(hexstr[4:8], 16)
month = int(hexstr[8:10], 16)
day   = int(hexstr[10:12], 16)
hour  = int(hexstr[12:14], 16)
min_  = int(hexstr[14:16], 16)
sys_v = int(hexstr[20:22], 16)
dia_v = int(hexstr[24:26], 16)
map_v = int(hexstr[28:30], 16)
hr_v  = int(hexstr[32:34], 16)
```

### 4.2. Oura-informed period classification

New helper `classify_period(dt, oura_sleep_periods: list[tuple[start, end]]) -> Literal["awake","sleep"]`:
- Load `oura_sleep_periods` from DB (existing table — verified at `api/import_oura.py:247`) for the session's date range ± 1 day.
- For each reading, return `"sleep"` if `dt` falls inside any `(bedtime_start, bedtime_end)` sleep period; else `"awake"`.
- Fall back to ESH fixed window when Oura data is missing for that night.
- Both `period` (Oura-informed) and `day_night` (fixed ESH window) are persisted so the analysis module can compute dipping both ways and document which it used.

### 4.3. Reject-invalid policy (ESH / AAMI)

Current `detect_artifact` checks:
- SYS 60–260
- DIA 30–200
- BPM 25–220
- Pulse pressure ≥ 20
- Future timestamp

Tighten to match ESH 2003/2023 ABPM analysis conventions for analysis-time filtering (distinct from ingest artifact flag — keep all readings, just mark):
- SYS > 260 or < 70 → artifact (current 60 is a touch too permissive for ABPM)
- DIA > 150 or < 40 → artifact (current 30 is too permissive)
- SYS − DIA < 20 → artifact (unchanged)
- SYS − DIA > 150 → artifact (new; catches cuff slippage)
- Vendor `error_code != 0` when present → soft flag (record but mark artifact for metrics)

Apply at analysis time, not at ingest. Ingest stores everything with flags. Rationale: lets us revisit thresholds later without re-importing.

### 4.4. Success-rate check

ESH 2023 requires ≥ 70% valid readings AND ≥ 2 valid readings/hour for a "valid" session.
- Populate `contec_abpm_sessions.success_rate = valid / attempted`.
- Log a WARNING (not error) when success_rate < 0.7. The analysis module should refuse to compute dipping on invalid sessions but still show the raw curve.

---

## 5. Clinical metrics to port (native Python, no R)

All formulas verified against the R `bp` package documentation and the cited guidelines. All native numpy/pandas, placed in a new `analysis/_abpm_metrics.py` module (underscore-prefixed = internal helper, consistent with `_comparative_utils.py`, `_hardening.py`, `_theme.py`).

### 5.1 Mean BP (24h / day / night / awake / sleep)
```
mean_24h = mean over all valid readings
mean_day = mean over readings where day_night == 'day'
mean_night = mean over readings where day_night == 'night'
mean_awake = mean over readings where period == 'awake'   (Oura-informed)
mean_sleep = mean over readings where period == 'sleep'
```

### 5.2 Dipping
```
dip_pct = (1 - mean_sleep / mean_awake) * 100      # prefer Oura-informed
# fallback: (1 - mean_night / mean_day) * 100
```
Categorisation (standard ESH/AHA):
- `extreme`: dip_pct > 20
- `normal`: 10 ≤ dip_pct ≤ 20
- `non`: 0 ≤ dip_pct < 10
- `reverse`: dip_pct < 0

Compute and store both SYS and DIA dipping. Persist `dip_period_used` so the report can caveat "based on Oura sleep periods" vs "fixed ESH window".

### 5.3 Morning BP surge (MBPS)
Two standard variants (R `bp::sleep_metrics` uses both):
- **Sleep-trough MBPS:** `mean(postwake 2h) - mean(lowest BP hour during sleep)`
  - Postwake = 2 hours after Oura sleep-end.
  - Sleep-trough = the 1-hour window containing the minimum BP during sleep.
- **Prewake MBPS:** `mean(postwake 2h) - mean(prewake 2h)`
  - Prewake = 2 hours immediately before Oura sleep-end.

Requires Oura sleep period data. When absent, fall back to a fixed prewake window (05:00–06:59) + postwake (07:00–08:59) and flag as estimate.

### 5.4 BP load
```
load_day_sys = 100 * count(day readings where SYS > 135) / total_day_readings
load_day_dia = 100 * count(day readings where DIA > 85)  / total_day_readings
load_night_sys = 100 * count(night readings where SYS > 120) / total_night_readings
load_night_dia = 100 * count(night readings where DIA > 70)  / total_night_readings
```

**NOTE on the nocturnal DIA threshold:** The user brief specifies 120/75 night; common literature uses both 120/70 and 120/75. ESH is inconsistent across editions. Plan defaults to **120/70** (ESH 2018 recommendation, also AASI paper convention). Parameterise `NIGHT_DIA_THRESHOLD` as a module constant so it can be swapped without a code change.

**NOTE on BP load's clinical status:** ACC/AHA, Hypertension Canada, and NICE do not endorse BP load for adult clinical decisions. ESH says "research only, not the clinical report". The 2022 AHA pediatric update removed BP load from criteria because it did not improve LVH prediction over mean ABP. We compute and show it; the report prose should frame it as "research metric, not a driver of clinical action on its own".

### 5.5 Variability
```
sd_24h = stdev(SYS over all readings)
wsd = ((sd_awake * hours_awake) + (sd_sleep * hours_sleep)) / (hours_awake + hours_sleep)
cv_pct = 100 * stdev / mean
arv = mean(|x[i+1] - x[i]|) over time-adjacent pairs
```
All computed for SYS and DIA. `arv` uses time-ordered readings, not index-ordered; when intervals are irregular, either compute over adjacent readings as-measured (standard) or weight by Δt (less common, flagged as a parameter).

### 5.6 Smoothness index (for longitudinal tracking)
```
delta_hourly = hourly_mean(post_session) - hourly_mean(baseline_session)
smoothness = mean(delta_hourly) / stdev(delta_hourly)
```
Requires a baseline session. For Henrik: if we capture an ABPM pre-treatment increment or use the 2026-04-28 session as baseline, subsequent sessions compute smoothness against it. `contec_abpm_metrics.baseline_session_id` stores which session was the reference. NULL when no baseline is available.

### 5.7 Hyperbaric index (AUC above threshold)
```
hyperbaric_sys_day = integral over day period of max(0, SYS(t) - 135) dt
# Trapezoid rule over (timestamp, max(0, SYS - threshold)) pairs.
# Unit: mmHg * hours.
```
Same for DIA and for night thresholds. Trapezoidal integration handles irregular ABPM sampling correctly.

### 5.8 AASI (ambulatory arterial stiffness index)
```
from scipy.stats import linregress
slope, intercept, r_value, p_value, stderr = linregress(sys_values, dia_values)
aasi = 1 - slope
aasi_r_squared = r_value ** 2
```
Standard calculation: OLS of DIA on SYS (DIA is Y, SYS is X). Persist R² because low R² (< ~0.36) means AASI is unreliable — many papers flag this. Alternative regression methods (exponential, Deming) exist but are not standard; stick with OLS and note the limitation.

### 5.9 Module layout

New file `analysis/_abpm_metrics.py`:
- `compute_session_metrics(readings_df, sleep_periods_df=None, baseline_session_id=None) -> dict`
- Individual metric functions all pure (take DataFrame in, return float or dict) so they're unit-testable in isolation.
- One public `persist_metrics(conn, session_id, metrics)` helper that writes to `contec_abpm_metrics`.

Analysis module (next section) imports from `_abpm_metrics` and adds visualization/HTML.

---

## 6. Analysis module — `analysis/analyze_abpm.py`

Mirror the structure of `analyze_omron_bp.py`: pandas load, metric computation, plotly figures, HTML wrap via `_theme.py`, JSON export for downstream scripts.

**Outputs:**
- `reports/abpm_report.html` — interactive dashboard (clinical_dark theme)
- `reports/abpm_metrics.json` — structured for `generate_index.py` and `generate_treatment_report.py`
- `reports/abpm_comparison_omron.json` — optional cross-device comparison

**Report sections (each a `make_section` wrapped block):**

1. **KPI strip** (`make_kpi_row`): 24-h mean SYS/DIA, dip %, dip category, success rate, morning surge (sleep-trough), AASI.
2. **24-h BP curve** — SYS and DIA traces on one plotly figure; shaded night region (Oura-informed if available, else fixed ESH window); horizontal reference lines at day thresholds 135/85 and night 120/70 in dashed `ACCENT_AMBER`.
3. **Hourly mean with 95% CI** — per clock hour, aggregate valid readings, show mean with bootstrap CI whiskers. Useful for spotting afternoon surge / morning white-coat effect.
4. **Dipping block** — category chip (extreme/normal/non/reverse), numeric dip % for SYS and DIA, bar comparing `mean_awake` vs `mean_sleep`, footnote noting whether Oura or fixed window was used.
5. **Morning surge visualisation** — from 2 h before to 4 h after Oura-wake, overlay hourly means; mark the sleep-trough hour; annotate the two MBPS values.
6. **Load bar chart** — 4 bars: day SYS, day DIA, night SYS, night DIA load %.
7. **AASI scatter** — DIA (Y) vs SYS (X) with OLS line, annotate AASI + R². If R² < 0.36, annotate "low R² — interpret AASI cautiously".
8. **Variability block** — SD, wSD, CV, ARV for SYS and DIA in a table.
9. **OMRON M7 cross-check** — merge OMRON morning readings from the same date on the ABPM day. Compute Bland–Altman style delta for the morning window. Does the home monitor agree with the ambulatory device? Drift between the two over multiple ABPM sessions is a recalibration signal.
10. **Longitudinal strip** — if prior ABPM sessions exist, show mean SYS/DIA/dip% across sessions with a trend line. Mark bisoprolol start (2026-04-08) and any future ruxolitinib dose changes as vertical lines via the existing `TREATMENT_MARKERS` pattern in `_theme.py`.

**Usage:**
```
python analysis/analyze_abpm.py                      # default DB
python analysis/analyze_abpm.py --profile henrik
python analysis/analyze_abpm.py --session abpm-20260428-090000
```

**Hooks into the existing pipeline:**
- Add to `run_all.py` as an optional stage (only runs if there is at least one row in `contec_abpm_sessions`).
- Expose metrics via `generate_index.py` so the dashboard index links to the ABPM report.
- `generate_treatment_report.py` reads `reports/abpm_metrics.json` and includes the dip %, MBPS, and any new hypertension trend into the treatment response narrative.

---

## 7. Clinical interpretation for Henrik

### 7.1 Why ABPM is the right instrument right now
Henrik has three specific drivers for ABPM > home cuff:
- **Ruxolitinib-associated BP drift** — JAK1/2 inhibition downregulates endothelial NOS; real-world data shows ~5 mmHg mean SBP rise at 72 weeks and clinically meaningful (+20 mmHg) outliers. Home-BP spot checks at one clock time catch only the segment of the circadian curve that's being sampled. ABPM captures the whole curve — including the nocturnal segment where rux effects are reported to show first.
- **TA-TMA early signal** — post-HSCT thrombotic microangiopathy presents with elevated BP weeks before lab/renal changes. The ABPM pattern that most predicts TA-TMA is **sustained non-dipping or reverse-dipping** — exactly what the ambulatory monitor is designed to detect and what a home cuff fundamentally cannot show. Requiring >2 antihypertensives post-HSCT is listed as a trigger for TA-TMA workup.
- **Bisoprolol titration** — 2.5 mg daily from 2026-04-08 is a conservative dose. ABPM 3 weeks post-start (~2026-04-28) answers: (a) is the drug covering the whole 24 h? (bisoprolol claims true once-daily but this varies individually); (b) is the morning surge blunted? (c) is the overnight trough too low? — all invisible to a morning + evening home reading.

### 7.2 Expected findings at 2026-04-28 ABPM (pre-registered hypotheses)

| Metric | Expected | Red flag | Interpretation path |
|---|---|---|---|
| 24-h SYS | 120–135 | >140 | rux drift or insufficient bisoprolol; consider titration to 5 mg |
| Dip % SYS | 10–20% (normal) | <10% | non-dipping → TA-TMA workup trigger |
| Dip % SYS | >20% | extreme dipper | hypotension risk overnight with beta-blocker; monitor |
| MBPS (sleep-trough) | <35 mmHg | >40 mmHg | insufficient overnight coverage of bisoprolol |
| Morning SYS (6–10 am) | comparable to OMRON M7 morning triplet | big divergence | instrument drift; recalibrate |
| Success rate | >70% | <70% | session invalid; repeat |
| AASI | <0.55 (healthy 36 y/o) | >0.55 | worth re-measuring on next ABPM; single reading is noisy |

### 7.3 Longitudinal plan
- **Session 1:** 2026-04-28 — captures bisoprolol steady-state + current rux exposure. Serves as baseline for smoothness index in later sessions.
- **Subsequent sessions:** every 3–6 months during rux treatment, or sooner if (a) home BP >140/90 for >1 week, (b) new proteinuria or LDH rise, (c) bisoprolol dose change. Compare dip%, mean 24-h, and MBPS to prior sessions.
- **Integration with existing pipeline:** ABPM results feed `generate_treatment_report.py` alongside OMRON home-BP and Oura HR/HRV. A composite autonomic score (existing) can then include the BP dip pattern as a component.

---

## 8. Open questions (flagged — do not guess)

1. **Vendor software URL.** The Contec manual references `www.dlsoftw.com` with index `05RK1069`. The domain exists but software distribution URLs change — **Henrik verify from the device packaging/CD before downloading**. Do not shell out to that URL from code.
2. **Default save folder for `.awp` files.** The vendor software location depends on installer version and Windows user profile. Verify on first install and set it deliberately (either to `/mnt/c/Users/.../abpm-bridge/in/` via WSL-mountable path, or adjust the Linux ingester's inbox to wherever the software lands).
3. **`.awp` file version on Henrik's unit.** `abpm50ex` only handles v2 (`FileVersion_Main=2`). If Henrik's unit produces v1 files, we need to see one before writing the parser. Primary parser targets v2; handle v1 with a clear error and capture the file for inspection.
4. **Exact record offsets.** Two independent reverse-engineering documents (Mumm and abpm50ex) disagree on what lives at offsets `[4:16]` (absolute date components per abpm50ex, elapsed-minutes + session-start per Mumm). The ingester should auto-detect which layout a file uses and log which path it took. Confirm on first real file.
5. **Does Contec's PC software ever export CSV directly?** The Amazon listings and manuals mention PDF + software suite but not CSV. Some versions expose "Export Excel" from the report view. **Before writing the CSV fallback path any further, confirm whether Henrik's software version has Excel/CSV export; if not, the `.awp` parser is the only path and we can delete the CSV-specific code paths.**
6. **`abpm50ex` license.** Repo has no LICENSE file (verified 404 on the raw LICENSE URL). Not legally redistributable. We use it as a *documentation reference* for the file format; we do **not** copy code or vendor the `.exe`. Email or open an issue with the repo owner if we want permission to include it.
7. **Device serial extraction.** The `.awp` header fields documented so far don't include a device serial. Either the vendor software keeps that in a separate config file, or the device stores it only in its firmware. If we need serial (for multi-device tracking), check whether the PC software UI exposes it and capture it manually in `--device-serial` on import.
8. **Manual-measurement flag.** The ABPM50 has a button for patient-initiated readings during an event. Whether this bit is preserved in the `.awp` record (i.e. whether the mystery `??` trailing bytes encode it) is unclear. Needs inspection of a real file with a known manual reading.
9. **Nocturnal DIA load threshold.** 120/70 vs 120/75 is genuinely ambiguous in ESH literature. Defaulting to 120/70 (ESH 2018). Keep as a module constant; revisit if we settle on 2023 ESH wording.
10. **Protocol code for `contec_abpm_sessions.protocol_code`.** Not exported by the device. Populate from `--protocol-code` CLI arg (e.g. `ESH-day15-night30`), or derive post-hoc from observed sampling intervals.

---

## Implementation order (recommended)

1. Build `.awp` parser with auto-layout detection, add to `import_contec_abpm.py`, validate on first real file (~2026-04-28).
2. Add `contec_abpm_metrics` table + safe migration in `init_contec_abpm_tables`.
3. Implement `analysis/_abpm_metrics.py` (pure functions, fully unit-tested against synthetic data + fixtures from the real file).
4. Implement `analysis/analyze_abpm.py` using the existing `_theme.py` primitives.
5. Wire into `run_all.py` as an optional stage gated on row count.
6. Fold ABPM outputs into `generate_treatment_report.py` narrative.
7. After session 2, enable smoothness-index path in `_abpm_metrics.py` and `analyze_abpm.py`.

All steps are net-additive; none modifies existing OMRON or Oura code paths.

---

## Sources (verified 2026-04-16)

- [obidose/abpm50ex GitHub](https://github.com/obidose/abpm50ex) — .awp parser (no LICENSE file; not redistributable, reference only)
- [Alexander Mumm — ABPM50 file format reverse-engineering](http://www.alexmumm.de/pgAbpm50_en.htm)
- [Contec ABPM50 product page — contecmed.com](https://www.contecmed.com/productinfo/894136.html)
- [ABPM50 user manual PDF — amperorblog.com](http://www.amperorblog.com/doc-lib/ABPM50usermanual10.06.24.pdf)
- [AmperorDirect ABPM-50 support/reference page](https://www.amperordirect.com/pc/r-product-support/z-abpm50-reference.html)
- [bp R package on CRAN](https://cran.r-project.org/web/packages/bp/bp.pdf) (GPL-2|GPL-3; used as design reference only, not a runtime dep)
- [bp — Blood pressure analysis in R (PLOS One 2022, PMC9462781)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9462781/)
- [dip_calc reference — bp package](https://search.r-project.org/CRAN/refmans/bp/html/dip_calc.html)
- [AASI original paper — Hypertension 2006 (Dolan/O'Brien)](https://www.ahajournals.org/doi/10.1161/01.hyp.0000200695.34024.4c)
- [AASI rationale and methodology (Dolan, eoinobrien.org, 2006)](http://www.eoinobrien.org/wp-content/uploads/2007/07/aasi-rationaledolanjbpmonitmarch-2006.pdf)
- [BP load literature review (PMC10417809, 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10417809/)
- [Target BP values in ABPM (PMC9908722)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9908722/)
- [TA-TMA theoretical considerations and practical approach — Nature BMT 2021](https://www.nature.com/articles/s41409-021-01283-0)
- [Ruxolitinib weight gain + hypertension — Scientific Reports 2019](https://www.nature.com/articles/s41598-019-53056-x)
- [Adverse events associated with JAK inhibitors — Scientific Reports 2022](https://www.nature.com/articles/s41598-022-10777-w)
- [Bisoprolol — StatPearls (NCBI Bookshelf)](https://www.ncbi.nlm.nih.gov/books/NBK551623/)
- [rpy2 on PyPI (considered, rejected for this module)](https://pypi.org/project/rpy2/)
- [STM32 VCP driver (STSW-STM32102)](https://www.st.com/en/development-tools/stsw-stm32102.html)
