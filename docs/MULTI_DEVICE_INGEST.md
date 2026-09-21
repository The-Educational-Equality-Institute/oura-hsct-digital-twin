# Multi-Device Ingest Guide

**Scope:** Hardware-monitoring pipeline extensions added 2026-04-16 to cover the
Henrik multimodal trial (FreeStyle Libre CGM, Viatom TH12 12-lead Holter, Contec
ABPM50 24h BP, Checkme O2 Max continuous SpO2), complementing the existing Oura
Ring and OMRON M7 integrations.

**Clinical context:** Pre-Mayo-consultation (2026-05-19) multimodal baseline for
post-HSCT survivorship. See also
[`reports/cgm_hypotheses_pre_registered.md`](../reports/cgm_hypotheses_pre_registered.md)
for the pre-registered analysis plan and
[`checkme_o2_max_integration_plan.md`](./checkme_o2_max_integration_plan.md) for
vendor-format research notes.

## Device overview

| Device | Importer | Tables | Format status |
|---|---|---|---|
| FreeStyle Libre 3 Plus | `api/import_glucose.py` | `glucose_readings`, `symptom_events` | LibreView CSV validated on synthetic fixtures; real file validation pending sensor arrival ~2026-04-21 |
| Viatom TH12 | `api/import_viatom_ecg.py` | `viatom_ecg_recordings`, `viatom_ecg_events` | Speculative — built from Viatom/Wellue family patterns, real format validation pending ~2026-04-28 |
| Contec ABPM50 | `api/import_contec_abpm.py` | `contec_abpm_readings`, `contec_abpm_sessions` | Speculative — general ABPM CSV patterns, real validation pending ~2026-04-28 |
| Checkme O2 Max | `api/import_checkme_spo2.py` | `checkme_spo2_continuous`, `checkme_spo2_sessions` | Speculative — see integration plan notes on OSCAR binary format and PI/motion caveats |
| OMRON M7 (HEM-7380T1) | `api/import_omron.py` | `omron_bp_readings`, `omron_bp_sessions` | Validated against real BLE bridge output |
| Oura Ring Gen 3/4 | `api/import_oura.py` | `oura_*` (17 tables) | Production — OAuth2 against Oura API v2 |
| Manual symptom logging | `api/import_symptom.py` | `symptom_events` | CLI-driven |

## Shared ingest infrastructure

`api/_ingest_common.py` provides helpers every CSV importer uses:

- `decode_bytes(raw)` — tries utf-8-sig / utf-8 / cp1252 / latin-1 in order
- `find_any(header_lower, *needles)` — first column where every needle substring matches
- `find_col(header_lower, *alternatives)` — tries each AND-group of needles; avoids the 0-is-falsy pitfall
- `get_cell(row, idx)` — safe extract + trim, returns None on missing/empty
- `int_or_none(v)` / `float_or_none(v)` — accepts comma-decimal European format
- `resolve_db_path(db_override, profile_name)` — precedence: explicit > profile > config.DATABASE_PATH

Bug fixes or format additions only need one edit rather than five.

## Per-device usage

### FreeStyle Libre 3 Plus (glucose)

```bash
# Initialize schema before sensor arrives
.venv/bin/python api/import_glucose.py --init-only

# Import a LibreView CSV export
.venv/bin/python api/import_glucose.py --csv ~/Downloads/libreview_export.csv

# Override profile (uses profiles.PROFILES["mitch"]["database"])
.venv/bin/python api/import_glucose.py --csv export.csv --profile mitch
```

Handles: UTF-8 BOM, 1-3 metadata rows, mmol/L and mg/dL units, comma-decimal
(Norwegian) or period-decimal. Filters record types to `{0, 1}` (historic + scan),
drops blood strip tests / insulin / food / notes rows.

### Viatom TH12 (12-lead ECG Holter)

```bash
.venv/bin/python api/import_viatom_ecg.py --init-only

# Import session metadata only
.venv/bin/python api/import_viatom_ecg.py --metadata session.xml

# Import AI events for a specific recording
.venv/bin/python api/import_viatom_ecg.py \
    --events events.csv \
    --recording-id R-2026-04-28-01 \
    --recording-start 2026-04-28T13:00:00

# Combined: metadata + events
.venv/bin/python api/import_viatom_ecg.py \
    --metadata session.xml --events events.csv
```

Raw waveforms are **not** stored in the database — path reference only. Events
are normalized through `_normalize_event_type` (aliases like "Atrial Fibrillation"
→ "afib", "ST_elev" → "st_elevation") into a fixed vocabulary: `afib`, `pac`,
`pvc`, `bradycardia`, `tachycardia`, `pause`, `st_depression`, `st_elevation`,
`long_qt`, `noise`, `wide_qrs`, `heart_block`, `ischemia`, `other`.

### Contec ABPM50 (24h ambulatory BP)

```bash
.venv/bin/python api/import_contec_abpm.py --init-only

.venv/bin/python api/import_contec_abpm.py --csv abpm.csv

# With explicit session tagging
.venv/bin/python api/import_contec_abpm.py \
    --csv abpm.csv \
    --session-id henrik-2026-04-28 \
    --device-serial ABPM50-XXX
```

Enrichment at ingest: MAP (computed if not in export), pulse pressure,
day/night classification (ESH convention: day 07:00-21:59, night 22:00-06:59),
artifact flag (implausible SYS/DIA/HR, narrow pulse pressure <20 mmHg, future
timestamps). Session summary aggregates reading count, artifact count, and
day/night split.

### Checkme O2 Max (continuous pulse oximeter)

```bash
.venv/bin/python api/import_checkme_spo2.py --init-only

# Absolute timestamps in CSV
.venv/bin/python api/import_checkme_spo2.py --csv checkme.csv

# Relative timestamps (HH:MM:SS from session start)
.venv/bin/python api/import_checkme_spo2.py \
    --csv checkme.csv \
    --session-start 2026-04-28T22:00:00
```

Enrichment: artifact flag (SpO2 out of 50-100%, PR out of 25-220 bpm, motion=1
when column present). Session summary computes mean/min SpO2 and T90 (% time
below 90%).

**Caveat:** The integration plan in
[`checkme_o2_max_integration_plan.md`](./checkme_o2_max_integration_plan.md)
notes the device likely exports only SpO2 + pulse rate, not perfusion index.
The current parser tolerates missing PI/motion columns but the schema reserves
space for them; validate column presence against first real export.

### Symptom event logging (manual)

```bash
.venv/bin/python api/import_symptom.py \
    --symptom chest_pain --severity 4 \
    --context "30min postprandial" --note "after lunch"

# With explicit timestamp
.venv/bin/python api/import_symptom.py \
    --symptom nausea --severity 3 \
    --timestamp 2026-04-22T13:15:00

# Against a different patient's database
.venv/bin/python api/import_symptom.py \
    --symptom dizziness --severity 3 --profile mitch
```

Controlled vocabulary: `chest_pain`, `nausea`, `dizziness`, `fatigue`,
`palpitations`, `shortness_of_breath`, `headache`, `sweating`, `tremor`,
`confusion`, `other`. Extend `SYMPTOM_TYPES` tuple in `api/import_symptom.py`
as trial surfaces new types.

## Schema summary

All tables live in the same SQLite database as Oura data (`data/oura.db` by
default, or the per-profile database via `--profile`).

```sql
-- Time-series readings
glucose_readings           (timestamp, glucose_mmol_l, record_type, sensor_serial, ...)
viatom_ecg_events          (recording_id, timestamp, offset_seconds, event_type, lead, confidence, ...)
contec_abpm_readings       (datetime, session_id, sys, dia, bpm, map_mmhg, day_night, is_artifact, ...)
checkme_spo2_continuous    (timestamp, session_id, spo2, pulse_rate, perfusion_index, motion_flag, ...)
omron_bp_readings          (datetime, user_slot, sys, dia, bpm, triplet_id, afib_candidate, ...)

-- Session / recording metadata
viatom_ecg_recordings      (recording_id, start_datetime, duration_seconds, sample_rate_hz, leads_count, ...)
contec_abpm_sessions       (session_id, start_datetime, reading_count, day_readings, night_readings, ...)
checkme_spo2_sessions      (session_id, start_datetime, mean_spo2, min_spo2, time_below_90_pct, ...)
omron_bp_sessions          (pulled_at, device_model, device_serial, reading_count, ...)

-- Manual annotation
symptom_events             (timestamp, symptom_type, severity, context, note)
```

Full column lists are in each importer's `init_*_tables` function.

## Running tests

All importer tests live under `tests/ingest/` and use stdlib only (no pytest
required, though pytest discovery works too).

```bash
# Run everything
.venv/bin/python run_tests.py

# Stop on first failure
.venv/bin/python run_tests.py --fast

# Run one suite directly
.venv/bin/python tests/ingest/test_import_glucose.py
```

Current coverage: 79 tests across 5 suites (20 common helpers, 10 glucose,
19 Viatom, 14 Contec, 16 Checkme).

## Format-validation workflow (when real exports arrive)

For each device once the first real CSV/XML lands:

1. Save export under a versioned name (e.g. `data/raw/th12_2026-04-29_initial.csv`)
2. Run `import_<device>.py` against it
3. If parsing fails or yields zero rows, inspect the header line and compare
   column names against the importer's `find_col(...)` calls
4. Add missing column aliases to the `find_col` tuples — no schema changes
5. Re-run tests; they should still pass (synthetic fixtures unchanged)
6. Commit the fix with the real export format documented in the commit message

The 24-hour fix window assumes the real format doesn't introduce a fundamentally
different structure (e.g. binary-only export, or JSON instead of CSV). If it
does, rewrite the parser and keep the schema — the schema is decoupled from
file format on purpose.

## Known speculation

These files were written against published-spec patterns, NOT real exports:

- `api/import_viatom_ecg.py` — event-type vocabulary assumes Viatom AI output patterns
- `api/import_contec_abpm.py` — CSV columns assume Contec-standard naming
- `api/import_checkme_spo2.py` — assumes perfusion index + motion columns (may not exist; see integration plan)

`api/import_glucose.py` is built from LibreView's well-documented CSV format but
is also pending first real file validation.

## Related docs

- [`../reports/cgm_hypotheses_pre_registered.md`](../reports/cgm_hypotheses_pre_registered.md) — pre-registered H1-H4 hypotheses for CGM trial
- [`../reports/DATA_METHODOLOGY.md`](../reports/DATA_METHODOLOGY.md) — Oura pipeline methodology
- [`checkme_o2_max_integration_plan.md`](./checkme_o2_max_integration_plan.md) — Viatom Checkme vendor-format research
- Memory: `~/.claude/projects/-home-henrik-projects-teei-oura-hsct-digital-twin/memory/`
  - `project_cgm_trial.md`
  - `project_hardware_monitoring_expansion.md`
  - `project_omron_m7.md`
