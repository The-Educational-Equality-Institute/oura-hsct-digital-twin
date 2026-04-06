# Prerequisites — Run this first

Before building the comparative modules, make these two changes:

## 1. Update profiles.py

Add `major_event_date` and `major_event_label` to both profiles so the comparative scripts can normalize timelines by days-since-event.

In `/home/henrik/projects/teei/oura-hsct-digital-twin/profiles.py`, update the `henrik` profile to add:
```python
"major_event_date": date(2023, 11, 23),  # HSCT date
"major_event_label": "HSCT",
```

And update the `mitch` profile to add:
```python
"major_event_date": date(2024, 12, 15),  # Approximate stroke date
"major_event_label": "Stroke",
```

## 2. Add comparative reports to REPORT_REGISTRY in _theme.py

Add these entries to the `REPORT_REGISTRY` list in `analysis/_theme.py`, at the end, as a new "Comparative" group:

```python
{"id": "comp_autonomic", "file": "comparative_autonomic_report.html", "title": "Autonomic Comparison", "group": "Comparative"},
{"id": "comp_treatment", "file": "comparative_treatment_response.html", "title": "Treatment Response", "group": "Comparative"},
{"id": "comp_sleep", "file": "comparative_sleep_analysis.html", "title": "Sleep Architecture", "group": "Comparative"},
{"id": "comp_coupling", "file": "comparative_activity_recovery_coupling.html", "title": "Activity-Recovery", "group": "Comparative"},
{"id": "comp_anomalies", "file": "comparative_anomaly_report.html", "title": "Anomaly Patterns", "group": "Comparative"},
```

## 3. Verify the import fix is working

Run this to confirm both databases have full data:
```bash
cd /home/henrik/projects/teei/oura-hsct-digital-twin
source .venv/bin/activate
python -c "
import sqlite3
for name, path in [('Henrik', 'data/demo.db'), ('Mitch', 'data/mitch.db')]:
    c = sqlite3.connect(path)
    row = c.execute('SELECT COUNT(*), SUM(CASE WHEN total_sleep_duration IS NOT NULL THEN 1 ELSE 0 END), SUM(CASE WHEN hr_lowest IS NOT NULL THEN 1 ELSE 0 END) FROM oura_sleep').fetchone()
    print(f'{name}: {row[0]} rows, {row[1]} with duration, {row[2]} with HR')
    c.close()
"
```

Expected: both patients should have matching counts (all fields populated).
