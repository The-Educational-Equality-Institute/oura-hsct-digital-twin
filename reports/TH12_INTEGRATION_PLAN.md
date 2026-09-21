# Viatom TH12 12-Lead ECG Holter — Integration Plan

Status: Plan only, no code. Written 2026-04-16.
Target: `/home/henrik/projects/helse/oura-hsct-digital-twin/`
Python: 3.12 venv at `.venv/`
DB: SQLite at `data/oura.db`

## 0. What already exists in the repo

Important: this is not a greenfield. Two files already target the TH12 and
need to be the starting point, not duplicated:

- `api/import_viatom_ecg.py` — skeleton ingester for metadata XML/JSON and AI
  event CSV. Creates `viatom_ecg_recordings` and `viatom_ecg_events` tables.
  Header comment documents: "The Viatom TH12 PC software export format has
  NOT been validated against a real export. When the first real export
  arrives (~2026-04-28), validate and fix within 24h."
- `api/test_import_viatom_ecg.py` — synthetic-fixture tests exercising the
  CSV and XML parser paths.

Neither is currently wired into `scripts/daily_pipeline.sh` or
`run_all.py` SCRIPTS. No `analysis/analyze_th12_ecg.py` exists yet. That is
the gap this plan closes.

---

## 1. Data acquisition

### 1.1 What the TH12 actually is (verified)

- Vendor: Shenzhen Viatom Technology (sold by Viatom + Wellue + Livenpace +
  TrendMedic + Lepu resellers).
- 12-lead continuous Holter recorder, 250 Hz sample rate, pacemaker-capable,
  continuous recording up to 24 h per session, SD-card storage.
- Paired Viatom AI engine claims 17 arrhythmia categories (afib, aflutter,
  VT, PAC/PVC, pause, bradycardia, tachycardia, ST elevation/depression,
  long QT, wide QRS, heart block, ischemia, noise, etc.).

Not verified / unknown:
- Exact ADC bit depth (typical for this family is 16-bit, but spec not
  published; assume 16-bit until validated).
- Frequency response / built-in filters on the device.
- Whether "24h" is a hard cap or the recommended session length for the
  seven-day usage pattern the marketing text mentions.

### 1.2 How data leaves the device

Two paths, confirmed by vendor FAQ:

1. **SD card**: plug the TH12 (or its SD card) into the PC via USB.
   The device then mounts as two block devices:
   - `TH12` drive: contains per-session files whose names start with `R`
     (e.g., `R_2026-04-28_13-00-00.<ext>`).
   - `NO NAME` drive: contains a single `HolterECGData.dat` raw-waveform
     file per recording session.
2. **Wellue "ECG Browser" desktop app** (Windows and Mac). Installer
   either ships on the included USB stick or is downloaded from
   `getwellue.com/pages/pc-software`. The app reads the device/SD,
   runs the AI analysis locally (Windows) or via Viatom cloud
   (default-off, user opt-in), and produces PDF reports.

Not verified / unknown:
- Whether the ECG Browser exposes a CSV / XML / EDF "raw export" from its
  UI. Vendor marketing talks only about PDF reports. Historical precedent
  from other Viatom devices (ER1, Checkme O2) suggests it probably does
  offer a CSV or XML export from the "Event list" / "Beat list" panels,
  but that must be checked hands-on.
- Whether the ECG Browser has a documented command-line mode. None found
  in public docs. Likely GUI-only.

### 1.3 Windows-bridge plumbing

We already use the Windows-inbox pattern for OMRON (see
`api/import_omron.py:47`: `DEFAULT_INBOX = /mnt/c/Users/ovehe/omron-bridge/out`).
Re-use exactly the same pattern:

- Create `C:\Users\<user>\viatom-bridge\` with:
  - `out\` — inbox for WSL-side ingester to consume. Henrik (or a light
    PowerShell runner) drops exports here.
  - `pc-software\` — ECG Browser installer + any preset export profile.
  - `raw\<recording_id>\` — raw waveform copy (HolterECGData.dat) if we
    want to keep the binary for later reprocessing.
- WSL side reads from `/mnt/c/Users/<user>/viatom-bridge/out/`. Same
  "move to `ingested/` after successful import" pattern OMRON uses.

No BLE bridge needed. TH12 has no documented BLE data-dump path — the
device is configured entirely via SD-card contents and the desktop app.

### 1.4 File formats we will actually parse

Per-session drop in `viatom-bridge/out/<recording_id>/`:

| File                        | Source                      | What we do with it |
| --------------------------- | --------------------------- | ------------------ |
| `HolterECGData.dat`         | NO NAME disk, raw           | Copy to `data/th12_raw/<recording_id>/` and store path only. Never load into SQLite. |
| `R_*.xml` or `session.xml`  | TH12 disk or ECG Browser    | Session metadata -> `viatom_ecg_recordings`. Parser already drafted. |
| `events.csv`                | ECG Browser event-list export (if available) | AI event table -> `viatom_ecg_events`. Parser already drafted. |
| `report.pdf`                | ECG Browser                 | Archive only, not parsed. |
| `<our-own>.edf`             | Post-ingest conversion      | Produced by us (see 2.5) for downstream ML. Not a device output. |
| `<our-own>.parquet`         | Post-ingest conversion      | Optional Arrow/Parquet dump of the 12-lead waveform for fast random-access analysis. |

Critical unknown (flag explicitly, do not guess): the **binary layout of
`HolterECGData.dat` for TH12 is not publicly documented**. The O2Ring /
Checkme family is 40-byte header + 5-byte records little-endian, but that
is pulse-ox, not 12-lead ECG. For TH12 we need to hex-dump a real file
and reverse the header + record-size + scaling on first delivery.

---

## 2. Python libraries

### 2.1 Recommended install set (baseline, CPU-only, Python 3.12)

```bash
# From repo root, inside .venv
source .venv/bin/activate

# Core ECG feature extraction (R/P/T peaks, QRS width, HRV, QTc).
# MIT license, 3.12-supported, ~15 MB installed.
pip install "neurokit2>=0.2.13"

# Holter / EDF / WFDB I/O, pacemaker-ready writer support.
# BSD-2, pure Python, ~5 MB.
pip install "wfdb>=4.1"
pip install "pyedflib>=0.1.38"

# torch_ecg: MIT license, CPU-capable PyTorch ECG models (NSR/AF/PVC/SPB).
# ~80 MB with torch CPU wheel already present.
pip install "torch_ecg>=0.0.31"

# ecglib: Apache-2, pretrained DenseNet1D-121 AFib classifier
# (pathology='AFIB', frequency=500). Pure PyTorch, CPU ok.
pip install "ecglib>=1.1.1"

# Arrow/Parquet for waveform dumps (optional but recommended for fast
# random-access slicing of a 24h x 12-lead x 250Hz = 259M-sample file).
pip install "pyarrow>=15.0"
```

These go into a new `requirements-ecg.txt` mirroring the existing
`requirements-full.txt` convention. They are additive to `requirements.txt`.

### 2.2 Model-by-model recommendation

| Model | License | Install | CPU? | Recommendation |
| --- | --- | --- | --- | --- |
| **NeuroKit2** | MIT | `pip install neurokit2>=0.2.13` | Yes | **Install.** Non-negotiable. This gives us R/P/Q/S/T peak delineation, QRS width, QT interval, HRV time+freq+nonlinear metrics, all in pure NumPy/SciPy/Pandas. Python 3.10–3.14 supported; 0.2.13 released 2026-03-02. Zero GPU dep. |
| **torch_ecg** | MIT | `pip install torch_ecg>=0.0.31` | Yes | **Install.** Provides pre-built CNN/CRNN models for 4-class rhythm (NSR / AF / PVC / SPB) on 12-lead input, plus building blocks for QRS detection and sequence labelling. Python >=3.9. Torch CPU wheel is ~200 MB but we can pin CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu` first. |
| **ecglib** | Apache-2 | `pip install ecglib>=1.1.1` | Yes | **Install.** Adds a pretrained `densenet1d121` AFib classifier callable as `create_model(model_name='densenet1d121', pathology='AFIB', pretrained=True)`. Also supports 1AVB, STACH, SBRAD, IRBBB, CRBBB, PVC pathologies. Trained at 500 Hz, so we resample the TH12 250 Hz input. Last release 2023-06 (stale but stable). |
| **ECG-FM** (Wang Lab) | MIT | Not pip-installable. Clone `github.com/Jwoo5/fairseq-signals` + `bowang-lab/ECG-FM`. | Yes but painful | **Skip for first pass.** 90 M-param wav2vec2 model trained on MIMIC-IV-ECG. Hard dependency on **fairseq_signals**, which is a fork of Meta's fairseq. Fairseq is known to break on Python 3.12 due to mutable-dataclass defaults (facebookresearch/fairseq#5634). Would require either downgrade to Python 3.11 or a maintained upstream fix. Revisit after initial pipeline is working. |
| **ECGFounder** (PKU Digital Health / NEJM AI 2025) | MIT | `git clone` + `conda create -n ECGFounder python=3.10; pip install -r requirements.txt` | Unverified | **Skip for first pass.** Officially pinned to Python 3.10 via conda. 150-label classifier, weights on Hugging Face (wanglab-style gated). CPU latency not published. Would require a separate venv. Revisit if the 150-label taxonomy is worth the isolation cost. |
| Viatom's own on-device AI | Proprietary | N/A | N/A | **Consume its outputs.** The PDF report and (where available) the events CSV already contain Viatom's AI labels. Parsing them into `viatom_ecg_events` is already scaffolded. Treat Viatom labels as one opinion, our ML labels as a second opinion, cross-compare in the dashboard. |

### 2.3 Dependency conflicts to watch

- **torch**: installing `torch_ecg` pulls `torch`. If we later add
  `chronos-forecasting` (already in `requirements-full.txt`), both must
  agree on the torch version. Pin to the CPU wheel index URL so we
  don't accidentally grab the ~2 GB CUDA build on a CPU-only box.
- **NumPy 2.x**: all five libraries above publish NumPy-2-compatible
  releases as of early 2026. The project already uses `numpy>=1.24`
  (see `requirements.txt:12`), no pin bump needed.
- **fairseq / fairseq_signals** (ECG-FM path only): pulls `omegaconf`,
  `hydra-core`, `cython`, and is the single biggest reason to defer
  ECG-FM. Do not add to the main venv.
- **pyedflib** wheels exist for 3.12 on Linux x86_64. If the wheel is
  missing, it compiles from C source and needs `gcc`. Acceptable.

### 2.4 Approximate install size

- Baseline (NeuroKit2 + wfdb + pyedflib + pyarrow): ~30 MB.
- Plus torch CPU: +200 MB.
- Plus torch_ecg model weights (lazy-downloaded on first use): +50–300
  MB depending on which model.
- Plus ecglib DenseNet1D-121 weights: ~30 MB.
- Total realistic ceiling for the recommended set: ~500 MB on top of
  the current venv. That is still well under our existing full-stack
  footprint (chronos-forecasting alone is >1 GB).

### 2.5 Why convert to EDF post-ingest

We store `HolterECGData.dat` verbatim because it is the authoritative
artifact, but we **also** emit a companion EDF+ file after first parse:

- EDF+ is the lingua franca for polysomnography/Holter tooling (EDFbrowser,
  MNE-Python, WFDB). Makes downstream re-analysis trivial.
- Most ECG foundation models and open ECG datasets load EDF or WFDB
  natively — we won't have to rewrite parsers per model.
- 12 × 250 Hz × 24 h × 2 bytes ≈ 518 MB per session uncompressed. Keep
  both the `.dat` and the `.edf` under `data/th12_raw/<recording_id>/`.
  Parquet column-store drops this to ~250 MB with Snappy.

---

## 3. Schema proposal (SQLite)

Two tables **already exist** in `api/import_viatom_ecg.py`. Three more
are needed. All prefixed with `viatom_ecg_` to match the repo's
`oura_*`, `omron_*`, `checkme_*` device-prefixed convention.

### 3.1 Existing (keep)

`viatom_ecg_recordings`
- `id INTEGER PK`
- `recording_id TEXT UNIQUE NOT NULL`
- `start_datetime TEXT NOT NULL`
- `end_datetime TEXT`
- `duration_seconds INTEGER`
- `sample_rate_hz INTEGER`
- `leads_count INTEGER`
- `device_serial TEXT`
- `firmware_version TEXT`
- `raw_file_path TEXT` -- path to HolterECGData.dat on our FS
- `source_file TEXT`
- `imported_at TEXT`

`viatom_ecg_events`
- `id INTEGER PK`
- `recording_id TEXT NOT NULL`
- `timestamp TEXT NOT NULL`
- `offset_seconds REAL`
- `event_type TEXT NOT NULL` -- afib, pvc, pac, bradycardia, ...
- `severity TEXT, lead TEXT, duration_seconds REAL`
- `hr_at_event INTEGER, confidence REAL, notes TEXT`
- UNIQUE(recording_id, timestamp, event_type)

These already have the right shape. The only change needed is adding a
`source` column distinguishing `"viatom_ai"` (on-device / ECG Browser)
from `"our_model_ecgfm"`, `"our_model_torchecg"`, `"our_model_ecglib"`,
`"our_model_neurokit"`. That lets us store multiple opinions per event
without breaking the unique constraint, by extending the UNIQUE to
`(recording_id, timestamp, event_type, source)`.

### 3.2 New tables to add

`viatom_ecg_waveform_files`
One row per channel dump / conversion artifact, not per sample. Keeps
SQLite out of the waveform-storage business. Waveforms live on disk.

```sql
CREATE TABLE viatom_ecg_waveform_files (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  recording_id TEXT NOT NULL,
  file_kind TEXT NOT NULL,          -- 'dat_raw' | 'edf' | 'parquet' | 'wfdb_hea'
  file_path TEXT NOT NULL,
  sha256 TEXT,
  n_samples INTEGER,
  n_channels INTEGER,
  sample_rate_hz INTEGER,
  bytes INTEGER,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(recording_id, file_kind, file_path),
  FOREIGN KEY (recording_id) REFERENCES viatom_ecg_recordings(recording_id)
);
```

`viatom_ecg_analysis`
One row per (recording_id, analysis_version). Versioned so re-runs
after a model or NeuroKit2 upgrade don't overwrite prior results.

```sql
CREATE TABLE viatom_ecg_analysis (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  recording_id TEXT NOT NULL,
  analysis_version TEXT NOT NULL,   -- e.g. 'v1_neurokit_0.2.13_torchecg_0.0.31'

  -- Rhythm burden (percent of analysed windows)
  pct_nsr REAL,
  pct_afib REAL,
  pct_other REAL,
  pct_noise REAL,

  -- Ectopy
  pvc_count INTEGER,
  pvc_per_hour REAL,
  pac_count INTEGER,
  pac_per_hour REAL,

  -- HRV time domain (ms)
  sdnn_ms REAL,
  rmssd_ms REAL,
  pnn50_pct REAL,
  sdnn_index_ms REAL,              -- mean of 5-min-window SDNNs

  -- HRV frequency domain
  hrv_lf_power REAL,
  hrv_hf_power REAL,
  hrv_lf_hf_ratio REAL,
  hrv_vlf_power REAL,

  -- HRV nonlinear
  sample_entropy REAL,
  approx_entropy REAL,
  dfa_alpha1 REAL,
  dfa_alpha2 REAL,

  -- Repolarisation
  qtc_bazett_ms REAL,
  qtc_fridericia_ms REAL,
  qt_dispersion_ms REAL,

  -- Heart rate (bpm)
  hr_mean REAL,
  hr_min REAL,
  hr_max REAL,
  hr_range REAL,

  -- Model-specific confidence summary
  model_afib_burden REAL,          -- mean of per-5s window p(AFib)
  model_pvc_burden REAL,

  -- Audit
  analysed_at TEXT DEFAULT CURRENT_TIMESTAMP,
  analysis_notes TEXT,

  UNIQUE(recording_id, analysis_version),
  FOREIGN KEY (recording_id) REFERENCES viatom_ecg_recordings(recording_id)
);
```

`viatom_ecg_analysis_windows`
Per-window inference results. 5-second windows at 250 Hz match ECG-FM's
input length and are small enough to store many per recording (24h / 5s
= 17 280 rows per recording, well within SQLite's comfort zone).

```sql
CREATE TABLE viatom_ecg_analysis_windows (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  recording_id TEXT NOT NULL,
  analysis_version TEXT NOT NULL,
  window_start_sec REAL NOT NULL,
  window_end_sec REAL NOT NULL,
  predicted_rhythm TEXT,           -- 'nsr' | 'afib' | 'other' | 'noise'
  p_nsr REAL, p_afib REAL, p_other REAL, p_noise REAL,
  hr_bpm REAL,
  rmssd_ms REAL,
  notes TEXT,
  UNIQUE(recording_id, analysis_version, window_start_sec),
  FOREIGN KEY (recording_id) REFERENCES viatom_ecg_recordings(recording_id)
);

CREATE INDEX idx_th12_windows_recording ON viatom_ecg_analysis_windows(recording_id);
CREATE INDEX idx_th12_windows_time ON viatom_ecg_analysis_windows(window_start_sec);
```

Optional (revisit after first real data):

`viatom_ecg_beats`
One row per detected beat. A 24h recording at 70 bpm = 100 800 beats.
Storing them enables per-beat arrhythmia classification and PVC/PAC
timestamps, at the cost of ~10 MB per recording. Only add if rhythm
burden analysis alone isn't enough. Columns: `recording_id,
beat_index, timestamp, rr_prev_ms, qrs_width_ms, qt_ms, beat_class`.

### 3.3 "Don't do this"

Do NOT create `th12_ecg_samples` as a sample-level table. A single 24h
recording is 259 M samples × 12 channels = 3.1 billion cells. SQLite
can technically hold it but every read query becomes an I/O bomb.
Samples live in EDF/Parquet on the filesystem; the DB stores file
pointers and derived metrics only.

---

## 4. Analysis module design: `analysis/analyze_th12_ecg.py`

Mirrors `analysis/analyze_omron_bp.py` in structure.

### 4.1 Public shape

```
analyze_th12_ecg.py
├── ANALYSIS_VERSION = "v1_neurokit_0.2.13_torchecg_0.0.31"
├── load_recordings(db) -> DataFrame
├── load_waveform(recording_id) -> numpy array (n_channels, n_samples)
├── run_feature_extraction(sig, sr) -> dict of NeuroKit2 features
├── run_rhythm_classifier(sig, sr) -> DataFrame of per-window probs
├── compute_clinical_metrics(recording_id, sig, features, windows) -> dict
├── cross_join_oura(db, recording) -> dict
├── cross_join_omron(db, recording) -> dict
├── cross_join_checkme_spo2(db, recording) -> dict
├── render_html(all_metrics) -> writes reports/th12_ecg_report.html
├── write_json(all_metrics) -> writes reports/th12_ecg_report.json
└── main()
```

### 4.2 Pipeline inside `main()`

1. Open DB read-only via `_hardening.safe_connect` (same pattern as
   `analyze_omron_bp.py`).
2. `SELECT * FROM viatom_ecg_recordings WHERE recording_id NOT IN (SELECT recording_id FROM viatom_ecg_analysis WHERE analysis_version = ?)`.
3. For each unanalysed recording:
   - Resolve the EDF file from `viatom_ecg_waveform_files`. If only the
     `.dat` exists, convert dat→EDF first via a helper in
     `api/import_viatom_ecg.py` (new function `convert_dat_to_edf`,
     callable from both ingester and analyser).
   - Load with `pyedflib.EdfReader`. Returns 12 channels × n_samples.
   - Run lightweight preprocessing: NeuroKit2's bandpass 0.5–40 Hz,
     baseline-wander removal, 50 Hz notch.
   - Call NeuroKit2: `signals, info = nk.ecg_process(lead_ii, sampling_rate=250)`
     on lead II for R-peak / HRV / QT extraction, then per-lead
     QT delineation for QT dispersion.
   - Call torch_ecg CRNN on 5 s windows of all 12 leads stacked
     (NSR/AF/PVC/SPB per window). Write window rows.
   - Call ecglib DenseNet1D-121 AFib model at 500 Hz on resampled
     windows for a second AFib opinion.
   - Aggregate into the `viatom_ecg_analysis` shape.
4. Cross-joins:
   - **Oura**: pull `oura_heart_rate` rows where `timestamp BETWEEN
     recording.start AND recording.end` (same join as
     `analyze_omron_bp.py:load_oura_hr`). Compute nearest-timestamp
     agreement of Oura wrist HR vs TH12 beat-to-beat HR
     (Bland–Altman bias, LoA, correlation).
   - **OMRON**: pull `omron_bp_readings` where
     `datetime BETWEEN recording.start AND recording.end`. Annotate
     each BP reading with the TH12 rhythm predicted at that minute and
     any AFib/PVC event within ±60 s. This directly validates or
     refutes the `afib_candidate` flag the OMRON ingester sets
     (`api/import_omron.py:152`).
   - **Checkme SpO2**: pull `checkme_spo2_continuous` for the same
     window. Flag joint HR+SpO2 desaturation events that also have a
     TH12 rhythm abnormality.
5. Render HTML via `_theme.wrap_html` + `plotly.graph_objects`. Match
   the clinical-white palette in `config.py` (C_PRIMARY `#0056B3`, etc.).
6. Write `reports/th12_ecg_report.html` and `.json`.

### 4.3 Performance budget

- 24h × 12 lead × 250 Hz preprocessing in NeuroKit2 on CPU: ~2–3 min
  per recording (benchmarked on comparable Holter files by the
  NeuroKit2 maintainers).
- torch_ecg CRNN on 17 280 windows, CPU: ~5–8 min per recording.
- Total realistic wall-clock per recording on CPU: ~10–15 min.
- `run_all.py` already has a 600 s per-script timeout
  (`run_all.py:247`). Since this script processes only *new* recordings
  (gated by analysis_version), the amortised daily cost stays low.

---

## 5. Clinical metrics to compute per recording

All land in `viatom_ecg_analysis`. Grouped by category:

**Rhythm analysis**
- `pct_nsr` — fraction of analysed 5 s windows classified NSR.
- `pct_afib` — AFib burden; the clinically-relevant number for
  anticoagulation decisions is >6 minutes in 24 h for CHA2DS2-VASc
  patients (2024 ESC/AHA guidance; compute and flag).
- `pct_other` — catch-all for VT, SVT, heart block, paced, artifact-not-noise.
- `pct_noise` — windows rejected as unreadable.

**Ectopy counts**
- `pvc_count`, `pac_count` — per recording.
- `pvc_per_hour`, `pac_per_hour` — clinical thresholds: >30/h PVC is
  "frequent", >10% of total beats often prompts cardiology referral
  (Latchamsetty 2016). Compute, don't interpret.

**HRV time domain** (from NeuroKit2 `hrv_time`)
- SDNN (24h norm >100 ms, low <50 ms; Malik 1996)
- RMSSD
- pNN50
- SDNN index (mean of 5-min SDNNs) — more robust for Holter than 24h SDNN.

**HRV frequency domain** (from NeuroKit2 `hrv_frequency`)
- LF, HF, VLF power (Welch).
- LF/HF ratio. Interpret with caution in Holter — respiratory coupling
  dominates HF at rest and during sleep.

**HRV nonlinear** (from NeuroKit2 `hrv_nonlinear`)
- Sample entropy (SampEn)
- Approximate entropy (ApEn)
- DFA α1 and α2
- Poincaré SD1/SD2 (already in the function; include as a secondary row).

**Repolarisation**
- QTc Bazett: QT / sqrt(RR). Over-corrects at high HR.
- QTc Fridericia: QT / RR^(1/3). Preferred for tachycardia.
- QT dispersion: max(QT) - min(QT) across the 12 leads, per beat,
  averaged across the recording. Controversial metric but trivially
  cheap to compute once we have the 12-lead delineation.

**Heart rate**
- hr_mean, hr_min, hr_max, hr_range.
- Diurnal split: mean sleep HR vs mean wake HR, using Oura sleep
  periods from `oura_sleep_periods` as the sleep mask (natural
  cross-join we already have).

**Integration signals (derived at cross-join time, not stored in the
`viatom_ecg_analysis` row itself)**
- OMRON BP–TH12 agreement: are OMRON-flagged AFib triplets confirmed
  by TH12 rhythm at the same minute?
- Oura HR–TH12 HR agreement: Bland–Altman bias, 95% LoA, Lin's
  concordance correlation.
- Checkme SpO2–TH12 rhythm coupling: AFib/pause onset vs desat nadir
  timing.

---

## 6. Daily pipeline wiring

Two files to touch:

### 6.1 `scripts/daily_pipeline.sh`

Insert between the existing OMRON step (line 51–53) and the
analysis-pipeline step (line 56–58):

```
# 3c. Import Viatom TH12 ECG exports from the Windows bridge (if any).
#     Non-fatal: skips gracefully when nothing is waiting.
echo "[3c/5] Importing Viatom TH12 ECG..."
python "$DIGITAL_TWIN/api/import_viatom_ecg.py" --profile henrik \
  || echo "  Viatom TH12 import non-fatal warning (continuing)"
echo "  Viatom TH12 import done."
```

That requires extending `api/import_viatom_ecg.py` with an `--inbox`
flag and inbox-sweep logic modelled on `api/import_omron.py:310`
(currently the ingester only accepts explicit `--metadata` /
`--events` paths).

### 6.2 `run_all.py`

Add one entry to `SCRIPTS` (line 29) after the OMRON BP script:

```python
    "analyze_omron_bp.py",
    "analyze_th12_ecg.py",           # new
```

Add two entries to `SEND_BUNDLE_HTML` / `SEND_BUNDLE_JSON`
(lines 62 / 95):

```python
    "omron_bp_report.html",
    "th12_ecg_report.html",          # new

    "omron_bp_report.json",
    "th12_ecg_report.json",          # new
```

That is the *entire* wiring surface. The orchestrator's resilience
logic (partial success allowed, strict mode available) already covers
the case where no TH12 data has arrived yet — the analysis script will
print "no unanalysed recordings" and exit 0.

---

## 7. Open questions / unknowns

Flag explicitly, do not guess. Each needs resolution before or during
implementation.

1. **HolterECGData.dat binary layout for TH12 is not publicly
   documented.** O2Ring / pulse-ox reverse-engineering work exists
   (Nelson's Log, OSCAR sleep app) but does not cover 12-lead ECG.
   Action: when the first real file arrives (~2026-04-28 per the
   existing ingester header comment), hex-dump the first 1024 bytes,
   compare file size against expected samples-per-second × duration ×
   channel-count × bytes-per-sample, and write a parser. Plan 24 h
   fix window.

2. **Whether ECG Browser exports events CSV vs only PDF.** Public docs
   mention "ECG report conclusion" and PDF export. An "event list"
   export to CSV is a plausible feature (precedent in Viatom's ER1
   software) but not verified. Action: launch ECG Browser on Windows,
   open a test recording, try every export menu item, document what
   appears. If no CSV export exists, we may need to:
   - OCR the PDF event table (`pdfplumber` already Python 3.12-safe),
   - OR trigger Viatom's cloud AI API (requires account, not local),
   - OR build our own AI-event list from first principles via
     `torch_ecg` + NeuroKit2 (already the plan for our second
     opinion, just add "promote to primary" fallback).

3. **ADC bit depth and mV-per-LSB scaling.** 16-bit is the family
   default for Viatom; TH12 datasheet does not publish it explicitly.
   Must be confirmed before any signal-based analysis is clinically
   meaningful. Action: use a known-amplitude calibration pulse (many
   Holter monitors self-calibrate; if TH12 doesn't, compare R-peak
   amplitudes against a same-subject 12-lead hospital ECG done the
   same week).

4. **Sample-rate variance across firmware versions.** Vendor says
   250 Hz. Some EU-market TH12 units are rumoured to ship at 500 Hz
   after firmware 2.x. Action: store actual sample rate from the XML
   metadata (already in schema) and never hard-code 250 Hz in analysis.

5. **Lead ordering in the raw binary.** Standard 12-lead is
   I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6 but some vendors
   store only 8 independent channels (I, II, V1–V6) and reconstruct
   III, aVR, aVL, aVF on demand. Action: check what the ECG Browser
   displays vs what is in the .dat. If only 8 channels, we reconstruct
   the augmented leads with standard formulae (Einthoven / Goldberger)
   and document it.

6. **ECG-FM upgrade path.** If we later want the 90 M-param foundation
   model, the blocker is fairseq + Python 3.12. Options:
   - Wait for upstream fairseq fix (issue #5634 is open).
   - Port ECG-FM to plain PyTorch (others have done this for
     wav2vec 2.0 — reasonable effort).
   - Spin up an isolated Python 3.11 venv just for ECG-FM inference
     and call it via subprocess from the main pipeline.
   Decision deferred until after first-pass pipeline is working.

7. **OMRON → TH12 AFib cross-validation metric.** The OMRON ingester
   flags `afib_candidate = 1` when a triplet has ≥2 IHB positives
   (`api/import_omron.py:182`). We should explicitly publish the
   sensitivity / specificity of OMRON-IHB vs TH12-AFib in the
   cross-join report. Action: once we have ≥5 TH12 recordings with
   ≥1 OMRON triplet each, compute the 2×2 and report it.

8. **Clinical label authority.** We will end up with three simultaneous
   opinions on every 5-second window: Viatom on-device AI, torch_ecg,
   ecglib DenseNet. They will disagree. Action: define a precedence
   rule before the first report ships, rather than after. Suggested
   default: majority vote of the three, tied going to Viatom (it is
   the FDA/CE-cleared source). Log disagreements so they can be
   audited.

9. **Private data hygiene.** Raw ECG is identifiable biometric data
   under GDPR Art. 9. Action: keep `data/th12_raw/` out of git (add
   to `.gitignore`) and out of any send bundle. Confirm neither
   `reports/send_bundle/` nor any HTML report embeds base64-encoded
   waveforms. The current `analyze_omron_bp.py` model is fine — it
   only serialises derived metrics.
