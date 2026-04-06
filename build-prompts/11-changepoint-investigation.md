# Build Task: analysis/analyze_mitch_changepoints.py

## What to build
Deep investigation of Mitchell's 14 automatically discovered changepoints. Cross-reference with known life events to validate the detection system.

## Output
- `reports/mitch_changepoint_investigation.html`
- `reports/mitch_changepoint_metrics.json`

## Known context from Mitchell (via WhatsApp)
- Dec 18, 2021: Arrived in Australia (first time in years)
- May 4, 2022: Left Australia, international flight via Singapore, masked 24hrs
- Ring off for extended periods (see gap list below)
- Post-stroke patient, bilateral carotid/vertebral artery dissection

## The 14 discovered changepoints
1. 2022-01-23 (consensus=9) — While in Australia
2. 2021-05-31 (consensus=8)
3. 2021-10-29 (consensus=8)
4. 2022-05-06 (consensus=8) — Right after leaving Australia
5. 2023-07-06 (consensus=8)
6. 2023-12-30 (consensus=8)
7. 2022-08-08 (consensus=7)
8. 2024-06-03 (consensus=6)
9. 2025-02-03 (consensus=6)
10. 2024-01-22 (consensus=5)
11. 2025-02-11 (consensus=5)
12. 2025-03-25 (consensus=5)
13. 2021-02-20 (consensus=2)
14. 2025-12-02 (consensus=2)

## Data gaps (ring off)
- 2022-02-19 → 2022-05-03 (73 days)
- 2022-05-04 → 2022-08-07 (95 days)
- 2022-08-08 → 2023-04-29 (264 days)
- 2023-09-07 → 2023-12-13 (97 days)
- 2024-01-25 → 2024-06-02 (129 days)
- 2024-06-05 → 2025-01-30 (239 days)
- 2025-02-11 → 2025-03-25 (42 days)
- 2025-03-26 → 2025-12-01 (250 days)
- 2025-12-03 → 2026-03-25 (112 days)

## Core analyses

### 1. Per-changepoint deep dive
For each of the 14 changepoints:
- Show 7 days before and after with all metrics
- Identify which metrics shifted and by how much
- Classify the changepoint type:
  - "Travel/timezone" (near data gaps, temp spikes)
  - "Health event" (HR spike + HRV drop + sleep disruption)
  - "Lifestyle shift" (gradual multi-metric change)
  - "Recovery milestone" (sustained improvement)
  - "Data artifact" (near ring on/off boundary)

### 2. Gap proximity analysis
- Many changepoints are near ring on/off boundaries
- Flag which ones are likely artifacts (first/last day of a wearing period) vs genuine
- Compute: days from nearest gap start/end

### 3. Australia trip analysis
- Dec 18, 2021 arrival → May 4, 2022 departure
- Full biometric profile during the ~4.5 month stay
- Jet lag recovery pattern (Dec 18-25)
- Adaptation over time
- Departure crash (May 4)

### 4. Long-term trajectory segmentation
- Split Mitchell's 5 years into segments between gaps
- Compute mean/SD for each segment
- Show how his baseline shifts across segments
- Is there an overall recovery trajectory or decline?

### 5. Validation scoring
- For changepoints near known events (Australia trip): mark as VALIDATED
- For changepoints at gap boundaries: mark as ARTIFACT
- For unexplained changepoints: mark as INVESTIGATE — generate questions for Mitchell

## Visualizations
1. Full 5-year timeline with all 14 changepoints marked + data gaps shaded
2. Per-changepoint detail cards (before/after comparison)
3. Australia trip deep dive panel
4. Gap proximity chart
5. Validation summary table

## Database
Use mitch.db only. Tables: oura_sleep, oura_readiness, oura_activity, oura_sleep_periods.
Same data gotchas: oura_readiness fields are scores, not physiological values.

## Notes
- This is a Mitchell-only report, not comparative
- Use _hardening.py for safe_connect
- Use _theme.py for wrap_html, make_section, make_kpi_card
- Register as {"id": "mitch_changepoints", "file": "mitch_changepoint_investigation.html", "title": "Mitch Changepoints", "group": "Comparative"}
