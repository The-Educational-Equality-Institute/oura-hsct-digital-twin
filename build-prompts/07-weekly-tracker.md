# Build Task: analysis/analyze_weekly_tracker.py

## What to build
A weekly trend tracker that compares this week vs last week across all key metrics. Designed for doctor visits — clear, simple, actionable.

## Output
- `reports/weekly_tracker.html` — clean summary with trend arrows
- `reports/weekly_tracker.json` — structured metrics

## Core logic
1. Load last 14 days of data from demo.db (oura_sleep, oura_readiness, oura_activity, oura_sleep_periods type='long_sleep')
2. Split into "this week" (last 7 days) and "last week" (7-14 days ago)
3. For each key metric, compute:
   - This week mean, last week mean, delta, % change
   - Direction arrow (improving/stable/declining) based on clinical direction (higher HRV = good, lower HR = good)
   - Flag if change is > 1 SD from rolling baseline
4. Generate a "traffic light" status for each metric (green/amber/red)

## Metrics to track
- HRV (RMSSD) — from oura_sleep.hrv_average or oura_sleep_periods.average_hrv
- Lowest HR — from oura_sleep_periods.lowest_heart_rate (Henrik's oura_sleep has NULL hr_lowest)
- Average HR — from oura_sleep_periods.average_heart_rate
- Sleep duration (hours) — from oura_sleep_periods.total_sleep_duration/3600
- Deep sleep (hours)
- REM sleep (hours)
- Sleep efficiency (%)
- Readiness score — oura_readiness.score
- Recovery index — oura_readiness.recovery_index
- Steps — oura_activity.steps
- Temperature deviation — oura_readiness.temperature_deviation
- Breath rate — oura_sleep.breath_average or oura_sleep_periods.average_breath

## CRITICAL data notes
- Henrik's oura_sleep has NULL for hr_average, hr_lowest — MUST use oura_sleep_periods (type='long_sleep')
- oura_readiness.resting_heart_rate is a SCORE (0-100), NOT bpm
- Sleep durations in oura_sleep_periods are in SECONDS
- Use safe_connect from _hardening.py for read-only DB access

## Visualization
- KPI card grid: one card per metric with value, change arrow, traffic light
- Sparkline for each metric (14-day mini chart)
- "Doctor Summary" section: 3-4 bullet points auto-generated based on biggest changes
- Ruxolitinib section: days since start, cumulative improvement since baseline

## HTML structure
Use wrap_html from _theme.py with report_id (add to REPORT_REGISTRY as {"id": "weekly", "file": "weekly_tracker.html", "title": "Weekly Tracker", "group": "Core"}).

## JSON structure
```json
{
  "generated_at": "...",
  "week_ending": "2026-04-06",
  "days_on_ruxolitinib": 21,
  "metrics": {
    "hrv_average": { "this_week": 10.5, "last_week": 9.8, "delta": 0.7, "pct_change": 7.1, "direction": "improving", "status": "amber" },
    ...
  },
  "doctor_summary": ["HRV improved 7.1% week-over-week", ...]
}
```

## Follow existing patterns
- main() -> int, return 0 on success
- Wrap sections in try/except with section_html_or_placeholder
- pio.templates.default = "clinical_dark"
- NEVER use add_vline with annotation_text on datetime axes
