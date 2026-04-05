You are building 5 new comparative analysis modules for the oura-hsct-digital-twin project, plus a shared utility module. The build specs are in `build-prompts/`. Read them all first.

## Build plan

### Phase 1: Prerequisites (do yourself, no agent)
Read and execute `build-prompts/00-prerequisites.md`. Update profiles.py and _theme.py directly. Then verify the databases are healthy.

### Phase 2: Shared Utils (do yourself, no agent)
Read `build-prompts/01-comparative-utils.md` and build `analysis/_comparative_utils.py`. This is the foundation — all 5 modules import from it. Test it works:
```bash
python -c "from analysis._comparative_utils import default_patients, load_patient_data, zscore_normalize; print('OK')"
```

### Phase 3: Build all 5 modules in parallel using agents
Once _comparative_utils.py is tested and working, launch 5 agents in parallel — one per module. Each agent should:
1. Read its build prompt from build-prompts/
2. Read _comparative_utils.py, _theme.py, _hardening.py, profiles.py, config.py for context
3. Build the complete script in analysis/
4. Test it runs without errors: `python analysis/<script>.py`
5. Verify HTML + JSON appear in reports/

The 5 agents to launch simultaneously:
- Agent 1: Read `build-prompts/02-comparative-autonomic.md` → build `analysis/analyze_comparative_autonomic.py`
- Agent 2: Read `build-prompts/03-comparative-treatment.md` → build `analysis/analyze_comparative_treatment.py`
- Agent 3: Read `build-prompts/04-comparative-sleep.md` → build `analysis/analyze_comparative_sleep.py`
- Agent 4: Read `build-prompts/05-comparative-coupling.md` → build `analysis/analyze_comparative_coupling.py`
- Agent 5: Read `build-prompts/06-comparative-anomalies.md` → build `analysis/analyze_comparative_anomalies.py`

### Phase 4: Integration (do yourself after agents complete)
1. Add all 5 scripts to `run_all.py` SCRIPTS list
2. Add HTML/JSON to SEND_BUNDLE_HTML and SEND_BUNDLE_JSON
3. Run `python run_all.py` to verify the full pipeline
4. Fix any failures

## Critical rules for all modules
- NEVER use `add_vline` with `annotation_text` on datetime axes — use `add_shape` + `add_annotation`
- `oura_readiness.resting_heart_rate` is a 0-100 SCORE, not bpm
- Henrik's `oura_sleep` has NULL for hr_average/hr_lowest — use `oura_sleep_periods` (type='long_sleep')
- All DB access via `safe_connect` from `_hardening.py` (read-only)
- Each script must follow the existing pattern: `main() -> int`, return 0 on success
- Wrap major sections in try/except with `section_html_or_placeholder` for resilience
- Use `pio.templates.default = "clinical_dark"` for all Plotly figures
