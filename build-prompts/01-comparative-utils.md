# Build Task: analysis/_comparative_utils.py

## What to build

Create `analysis/_comparative_utils.py` — a shared utility module that all 5 comparative analysis scripts will import. It sits alongside `_theme.py` and `_hardening.py` in the `analysis/` directory.

## Project location

`/home/henrik/projects/teei/oura-hsct-digital-twin/`

## Two patients

- **Henrik** (demo.db): Post-HSCT (Nov 23, 2023), 81 days of data (Jan 8 – Apr 4, 2026). HRV ~9ms, resting HR ~80bpm, ~2700 steps/day. On Ruxolitinib since Mar 16, 2026. Ring Gen 4.
- **Mitchell** (mitch.db): Post-stroke (~Dec 2024, bilateral carotid/vertebral artery dissection), 518 days of data (Feb 2021 – Apr 2026). HRV ~43ms, resting HR ~50bpm, ~10000 steps/day. Ring Gen 3.

## Database schema (both DBs identical)

```sql
oura_sleep: date, score, total_sleep_duration, rem_sleep_duration, deep_sleep_duration, 
            light_sleep_duration, awake_time, efficiency, latency, restless_periods,
            bedtime_start, bedtime_end, hr_lowest, hr_average, hrv_average, 
            breath_average, temperature_delta

oura_readiness: date, score, temperature_deviation, activity_balance, body_temperature,
                hrv_balance, previous_day_activity, previous_night, recovery_index,
                resting_heart_rate, sleep_balance
                -- NOTE: ALL fields are contributor SCORES (0-100), NOT physiological values!
                -- resting_heart_rate is a SCORE, not bpm!

oura_activity: date, score, active_calories, total_calories, steps, daily_movement,
               inactive_time, rest_time, low_activity_time, medium_activity_time, 
               high_activity_time

oura_sleep_periods: period_id, day, type, average_hrv, average_heart_rate, average_breath,
                    total_sleep_duration, rem_sleep_duration, deep_sleep_duration,
                    light_sleep_duration, awake_time, efficiency, latency, restless_periods,
                    lowest_heart_rate, bedtime_start, bedtime_end, time_in_bed
```

## Existing imports available

From `_hardening.py`: `safe_connect(path, read_only=True)`, `safe_read_sql(sql, conn, label="", required=False)`
From `_theme.py`: colors (`ACCENT_BLUE`, `ACCENT_GREEN`, `ACCENT_PURPLE`, `ACCENT_CYAN`, `BG_PRIMARY`, etc.), `wrap_html`, `make_section`, `make_kpi_card`
From `profiles.py`: `PROFILES` dict with `database`, `major_event_date`, `major_event_label`, `label`, `age`
From `config.py`: `REPORTS_DIR`, `DATABASE_PATH`

## What the module must provide

### 1. Data Structures

```python
@dataclass(frozen=True)
class PatientConfig:
    patient_id: str          # "henrik" or "mitch"
    display_name: str        # "Henrik (post-HSCT)"
    db_path: Path
    event_date: date         # HSCT or stroke date
    event_label: str         # "HSCT" or "Stroke"
    color: str               # Primary plot color

@dataclass
class NormalizedResult:
    patient_id: str
    raw: pd.Series
    z_scores: pd.Series
    percentiles: pd.Series
    baseline_mean: float
    baseline_std: float
    n_observations: int
```

`COMPARABLE_METRICS`: A list of tuples or dicts defining all metrics safe for cross-patient comparison, with table, column, display name, unit, and whether higher is better.

`default_patients() -> tuple[PatientConfig, PatientConfig]`: Factory reading from `PROFILES`.

### 2. Data Loading (3 functions)

- `load_patient_data(patient, table, columns="*", date_range=None) -> pd.DataFrame`: Connect read-only, query, return DataFrame indexed by date.
- `load_both_patients(patients=None, table="oura_sleep", columns="*") -> dict[str, pd.DataFrame]`: Load from both DBs.
- `load_metric(metric_name, table, column, patients=None) -> dict[str, pd.Series]`: Load a single metric for both patients.

### 3. Z-Score Normalization (2 functions)

- `zscore_normalize(series, baseline_period=None) -> NormalizedResult`: Z-scores relative to patient's own mean/std. If baseline_period given, use only that window for computing mean/std.
- `zscore_both(data, baseline_periods=None) -> dict[str, NormalizedResult]`: Apply to both patients.

### 4. Percentile-of-Self (2 functions)

- `percentile_of_self(series) -> pd.Series`: Rank each day within patient's own distribution (0-100).
- `percentile_both(data) -> dict[str, pd.Series]`

### 5. Overlap & Alignment (4 functions)

- `find_date_overlap(data) -> pd.DatetimeIndex`: Intersection of dates where both have data.
- `align_to_overlap(data) -> dict[str, pd.DataFrame]`: Filter both to overlap only.
- `days_since_event(dates, event_date) -> pd.Series`: Convert date index to integer days-since-event.
- `align_by_event(data, patients) -> dict[str, pd.Series]`: Re-index by days-since-event.

### 6. Statistical Comparison (3 functions)

- `compare_distributions(a, b, test="auto") -> dict`: Mann-Whitney U (default), returns `test_name`, `statistic`, `p_value`, `significant`, `effect_size` (Cohen's d), `effect_label`, `ci_95`.
- `effect_size_cohens_d(a, b) -> float`: Pooled SD Cohen's d.
- `bootstrap_ci(a, b, func=None, n_bootstrap=10000, ci=0.95) -> tuple[float, float]`

### 7. Shared Plot Helpers (3 functions)

All use Plotly. CRITICAL: never use `add_vline` with `annotation_text` on datetime axes — use `add_shape` + `add_annotation` separately.

- `dual_patient_timeseries(data, patients, title="", y_label="", show_rolling=7, normalize=None, event_lines=True) -> go.Figure`: Both patients overlaid, optional rolling mean, optional z-score/percentile normalization, optional vertical event date lines.
- `dual_patient_distribution(data, patients, title="", kind="violin") -> go.Figure`: Side-by-side violin/box/histogram.
- `event_aligned_comparison(data, patients, title="", window=(-30, 365)) -> go.Figure`: X-axis = days-since-event, both patients overlaid.

## Design constraints

- Use `safe_connect` from `_hardening.py` for all DB access (read-only)
- Handle NaN gracefully throughout — drop before computing baselines, return NaN for missing days
- All functions should work even if only one patient has data for a given date range
- Import `_theme.py` colors gracefully with try/except fallback
- Path resolution: `Path(__file__).resolve().parent.parent` gives project root, `/ "data"` for databases

## Do NOT

- Do not create test files
- Do not modify existing analysis scripts
- Do not add any dependencies not already in requirements.txt (pandas, numpy, scipy, plotly are all available)
- Do not import from config.py directly in this module — get paths from PatientConfig/PROFILES instead
