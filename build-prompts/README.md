# Oura Comparative Analysis — Build Prompts

Feed these to the Claude Code session on ryzen (`launch-h4oura.sh`) in order.

## Build Order

1. **00-prerequisites.md** — Update profiles.py + _theme.py registry. Quick.
2. **01-comparative-utils.md** — Shared utility module. All others depend on this.
3. **02-comparative-autonomic.md** — Module 1: HRV/HR recovery trajectories
4. **03-comparative-treatment.md** — Module 2: Changepoint detection (~1500 lines, most complex)
5. **04-comparative-sleep.md** — Module 3: Sleep architecture comparison
6. **05-comparative-coupling.md** — Module 4: Activity→recovery lag analysis
7. **06-comparative-anomalies.md** — Module 5: Anomaly fingerprinting

## After each module

Test with:
```bash
cd /home/henrik/projects/teei/oura-hsct-digital-twin
source .venv/bin/activate
python analysis/<script_name>.py
```

Check that HTML and JSON appear in `reports/`.

## After all modules

Add to run_all.py SCRIPTS list:
```python
"analyze_comparative_autonomic.py",
"analyze_comparative_treatment.py",
"analyze_comparative_sleep.py",
"analyze_comparative_coupling.py",
"analyze_comparative_anomalies.py",
```

And add to SEND_BUNDLE_HTML and SEND_BUNDLE_JSON accordingly.

Then run: `python run_all.py`
