# Build report, 2026-09-20: spor A3, A4, C1, C2, C3

Scope: the pipeline, the gates, the tests and the docs of
`oura-hsct-digital-twin`. Nothing was deployed, no `wrangler` was run, no git
command was run, `.env` was not touched and no secret value was printed.

Plan: `~/projects/advisor/reports/helse/2026-09-20-world-class-plan-digital-twin-genomics.md`.

Interpreter for everything below: `.venv/bin/python`, **Python 3.14.7**. No
second interpreter was needed; every optional package has a working 3.14 wheel
or built one from source. `scripts/daily-pipeline-local.sh` and
`scripts/deploy-local.sh` keep calling `.venv/bin/python` unchanged.

---

## Headline result

| Measure | Before | After |
|---|---|---|
| Pipeline scripts passing | 30/31 (`analyze_sequential_ci.py` crashed) | **34/34** |
| Total runtime | 442.5 s | **538.0 s** |
| Modules publishing "not installed" / "No module named" / "No model backend" / "dependency unavailable" | 3 | **0** |
| `chronos_available` | `false` (torchvision ImportError) | **`true`**, `amazon/chronos-bolt-base` on CPU |
| Claim audit | not wired; last run 13.04: 12 pages, 60 claims, 7 mismatches, `pass: false` | **wired as a gate: 36 pages, 673 claims, 0 mismatches, `pass: true`** |
| Claim audit published as a page | no | **`reports/claims.html`** |
| Dead internal links in the build | 1 dangling, plus no `404.html` (soft-404 on every unknown path) | **0 broken, `404.html` present**, gated |
| Test suites | 5 (importer units only) | **6** (plus report-output smoke tests), 6/6 green |
| Pipeline failure notification | none (failed silently 4 days) | **critical desktop notification plus `last-failure.txt`**, verified from a systemd user unit |

---

## A3: full stack install, CPU-only Chronos, one clean pipeline run

### Install

The background install finished on its own; no version pin had to change and no
Python 3.12 venv was needed. Verified in `.venv/bin/python` (3.14.7):

```
$ .venv/bin/python -c "import causalimpact, hmmlearn, torch, chronos, tigramite, \
    stumpy, tsfresh, antropy, nolds, filterpy, pykalman"
ALL 11 OK
```

Versions measured: `pycausalimpact 0.1.1`, `hmmlearn 0.3.3` (built a `cp314`
wheel from source), `torch 2.14.0+cpu`, `chronos-forecasting 2.3.2`,
`tigramite 5.2.10.1`, `stumpy 1.14.1`, `tsfresh 0.21.2`, `antropy 0.2.2`,
`nolds 0.6.3`, `filterpy 1.4.5`, `pykalman 0.11.2`, `transformers 5.17.0`,
`numpy 2.4.6`, `scipy 1.18.0`, `pandas 3.0.3`.

Two more were installed during the run, both required for a module to produce a
real report rather than a placeholder:

- **`duckdb 1.5.5`**. `analysis/_frames.py` imports it. Without it
  `analyze_multimodal_anomalies.py` published only "Optional dependency
  unavailable: ... requires stumpy and the DuckDB frame backend". `stumpy` was
  already fine; `duckdb` was the missing half. Added to `requirements-full.txt`.
  The module now reports **99 days, 6 channels, 55 univariate discords, 0 joint
  discords**.
- **`torchvision 0.29.0+cpu`**. See the Chronos section below.

`ssm` was left uninstalled, as instructed. `analyze_oura_gvhd_predict.py` uses
its documented fallback and logs `[HMM] ssm not available, using hmmlearn
GaussianHMM`. That string is not a failure marker and is not in the gate.

### GPU is not required, and is not used

Measured: `nvidia-smi` reported **633 MiB free of 8188 MiB**. Chronos must not
touch it.

Two independent guarantees now hold:

1. `torch` is the CPU-only build (`2.14.0+cpu`), so
   `torch.cuda.is_available()` is `False`.
2. `analysis/analyze_oura_foundation_models.py:382` no longer asks the GPU
   whether it exists. It defaults to CPU and requires an explicit opt-in:

   ```python
   requested = os.environ.get("CHRONOS_DEVICE", "cpu").strip().lower()
   device = "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"
   ```

### The Chronos forecast was silently dead, and why

`chronos_available` was `false` with
`ImportError: cannot import name 'ImageReadMode' from 'torchvision.io' (unknown
location)`. "unknown location" is the tell: `torchvision` was not installed at
all, and `_install_torchvision_compat_stub()` was installing a **stub** module
in its place unconditionally. `transformers` then asked the stub for a symbol
the stub does not define.

Fix, two parts:

- Installed the real CPU torchvision matching torch:
  `pip install --index-url https://download.pytorch.org/whl/cpu torchvision`,
  giving `torchvision 0.29.0+cpu`.
- `analysis/analyze_oura_foundation_models.py:123`: the stub is now a
  **fallback**. The real torchvision is imported first, and the stub is only
  installed if that import raises. A machine without a usable torchvision still
  gets a forecast; this machine gets the real library.

Result: `reports/foundation_model_metrics.json` now carries
`"chronos_available": true`, `"chronos_error": null`,
`"model": "amazon/chronos-bolt-base"`, runtime 280.7 s on the first (cold,
model-download) run and **50.9 s** in the final pipeline run.

### One script was genuinely broken: duplicate dates

`analyze_sequential_ci.py` failed with
`ValueError: cannot reindex on an axis with duplicate labels`.

Root cause, measured, not guessed:

```
$ sqlite3 data/oura.db "SELECT day, COUNT(*) c FROM oura_sleep_periods
                        WHERE type='long_sleep' GROUP BY day HAVING c>1;"
2026-05-19|2
2026-06-01|2
```

Two nights carry two scored `long_sleep` periods each. The daily matrix was
**251 rows for 249 distinct dates**. Run A's window (Jan 8 to Apr 7) misses both
dates, so it succeeded; Run B's window contains them and crashed on
`ts.reindex(full_range)`.

`analysis/analyze_oura_causal.py` already had the right convention ("keep the
longest so every date appears exactly once"). I applied that same convention to
every loader that lacked it:

| File | Line | Effect |
|---|---|---|
| `analysis/analyze_sequential_ci.py` | 245 | fixes the crash; matrix is now 249 rows / 249 dates |
| `analysis/analyze_piecewise_its.py` | 150 | those two dates were counted twice in the ITS regression |
| `analysis/analyze_tau_u.py` | 208 | same, in the Tau-U daily series |
| `analysis/analyze_placebo_tests.py` | 144 | latent only: its window ends at `TREATMENT_START`, which excludes both dates today, but the window is config-driven |
| `analysis/analyze_rux_forecast.py` | 120 | those two days were double-weighted in the phase regressions |

`analysis/analyze_weekly_tracker.py` was **already** deduplicating
(`df_sp[~df_sp.index.duplicated(keep="last")]`, line 188) and was left alone: a
different tie-break, but on a 14-day window that contains neither date.

You independently found and fixed the same class in
`analysis/analyze_oura_causal.py`. Two sessions reaching the same root cause
from different symptoms is good evidence it is the real one.

### The pipeline run

```
$ .venv/bin/python run_all.py
  Passed: 34/34  Failed: 0/34  Skipped: 0/34
  Total runtime: 538.0s
  STATCHECK PASS - 673 claims across 36 pages, 0 mismatches.
  Send bundle: reports/send_bundle   (31 HTML, 29 JSON)
```

Database: `config.py` resolves `data/oura.db`, which is **byte-identical** to
`data/demo.db` (`sha256 3de3abc7...26493e38` for both), so this is the demo
dataset. No Oura API call was made.

Slowest three, from `reports/run_summary.json`:
`analyze_oura_advanced_hrv.py` 280.4 s, `analyze_oura_anomalies.py` 75.7 s,
`analyze_oura_foundation_models.py` 50.9 s.

### A3 gate: clean

```
$ rg -n -i 'FAILED|not installed|No module named|No model backend|package not installed|dependency unavailable' \
     reports/*.html reports/*_metrics.json | wc -l
14
```

All 14 read, all benign, **zero runtime failures**:

| Where | Text | Verdict |
|---|---|---|
| `reports/digital_twin_report.html:2778` | "...structure the model **failed** to absorb" | prose |
| `reports/piecewise_regression.html:2596` | "support gaps or **failed** fits remain" (legend) | prose |
| `reports/piecewise_regression.html:2599,2617,2635,2653` | "0 were underpowered and 0 fit(s) **failed**" | prose, and the count is 0 |
| `reports/piecewise_regression_metrics.json` (8 hits) | `"failed_total": 0`, `"failed_offsets": []` | JSON field names, all empty or zero |

No hit is in any of the five files I was not allowed to edit.

---

## A4: the claim audit became a real gate

`analysis/statcheck_reports.py` claimed in its own docstring to run "as a
post-generation step in `run_all.py`". It did not; nothing called it. It is now
true, and the docstring says exactly what is wired.

### 1. It runs, and it stops the pipeline

- `run_all.py:223` `run_statcheck()` runs the audit as a subprocess after all
  scripts, reads `reports/statcheck_audit.json`, and returns pass only when
  **both** the exit code is 0 and `pass` is `true`.
- `run_all.py:387`: the send bundle is assembled only when
  `failures == 0 and statcheck_ok`.
- `run_all.py:416`: `sys.exit(1)` when the audit does not pass.

### 2. It covers every page, not 12

`analysis/statcheck_reports.py:119`: `HTML_TO_JSON` went from 12 hardcoded
entries to **34**, derived by reading every `REPORTS_DIR / "*.json"` write in
each generator (the mapping table is in the file, one comment per group). Values
may now be a **list**, so a page that aggregates numbers from several modules
can be checked against all of them. `roadmap.html` is checked against
`causal_inference_metrics.json` and `composite_biomarkers.json`, as you asked.
Pages with no companion JSON (`index.html`, `404.html`, `how_built.html`,
`anthropic_case.html`) are still scanned for sanity issues; their claims are
reported as UNMATCHED rather than silently skipped.

`analysis/statcheck_reports.py:170`: `AUDIT_OUTPUT_PAGES = {"claims.html"}`.
The register restates every other page's numbers, so scanning it would
re-extract the whole site as if it were a fresh set of claims.

### 3. The mismatches it found were mostly its own blind spots

The first run over all pages reported **40 mismatches**. I checked them one by
one before changing a single generator. They were not page errors.

Worked example. `comparative_activity_recovery_coupling.html` prints
`p=0.640` in a Kruskal-Wallis dose-response table. The audit flagged it against
`regression.henrik.steps_vs_total_sleep_duration.p_value = 0.6548`. The page is
right: the JSON holds `dose_response.henrik.kruskal_p = 0.6399584975513617`.
The audit never loaded that value, because its key patterns were `p_value` and
`p_val` only, and `kruskal_p` matches neither. With no legitimate reference to
match, the claim was paired with whatever number of the same *kind* happened to
be numerically nearest, and the difference was reported as a contradiction the
page had not made.

Surveying every numeric key across all metrics JSONs showed how wide the blind
spot was: `p` (214 occurrences), `mann_whitney_p` (16), `b4_pvalue` and
`b5_pvalue` (28 each), `bonferroni_p`, `levene_p`, `fisher_p`, `spearman_p`,
`kruskal_p`, `ljung_box_p`, plus `r` (214), `rho` (18), `tau` (15), `r2` (14),
all invisible to the audit.

Four fixes, all in the audit, none in a generator:

| Fix | Where | Why |
|---|---|---|
| Key classifier for the names the files actually use | `statcheck_reports.py:367` `classify_stat_key()` | recognises `p`, `*_p`, `*_pvalue`, `r`, `rho`, `tau*`, `*_corr`, `r2`, `r_squared*`, `cohens_d`. Deliberately **excludes** `p_flare`, `p_preflare`, `p_threshold`, `p_indirect`, which are probabilities, not p-values. A range check (`STAT_RANGES`) drops values that cannot be that statistic, so `tau_max = 7` never becomes a "correlation". |
| Context anchoring before calling anything a mismatch | `statcheck_reports.py:480` `_is_anchored()` | a value difference is evidence the page is wrong only if the surrounding text names the statistic the reference holds. Otherwise the claim is UNMATCHED, which is the truth: unverifiable, not contradicted. |
| Inequalities judged as inequalities | `statcheck_reports.py:497` `_satisfies()` | `Ljung-Box p > 0.05` is a threshold statement. It is correct when the JSON value is *above* 0.05, not when it *equals* it. This alone accounted for the last 3 mismatches, all true statements. |
| Significance legends are notation, not claims | `extract_claims()`, asterisk guard | `* p<0.05, ** p<0.01, *** p<0.001` under a table is a key, not an assertion. |

Measured effect, same pages, same data, nothing regenerated between runs:

| | Before | After |
|---|---|---|
| JSON references loaded | 824 | **1449** |
| Claims verified OK | 375 | **423** |
| Mismatches | **40** | **0** |

### 4. Mismatches to resolve in generators: none

After the audit was made correct, **zero** mismatches remain, so no generator
prints a number its JSON does not carry. There is nothing on the list for you,
and nothing I had to leave in a file I could not touch.

Final audit, from the full pipeline run:

```
  Reports checked:   36
  Claims extracted:  673
  JSON references:   1457
  Matched (OK):      433
  Mismatches:        0
  Sanity issues:     114
  Unmatched HTML:    240
  pass: true
```

**The 114 sanity issues are all `warning` severity and do not block** (the gate
counts only `error`):

- **60 x `p_zero_exact`**: a page prints `p=0.0000`. Not wrong, but `p < 0.0001`
  would be honest about the precision. Most are inside Plotly chart annotations.
- **54 x `inconsistent_self_report`**: the same short context string carries two
  different p-values, usually two rows of a table that begin with the same words.

Neither is a contradiction of the JSON. Flagging them as errors would block the
deploy on a style preference. They are visible in `reports/statcheck_audit.json`
if you want to drive either to zero.

**240 unmatched claims** is the honest residue: numbers printed on a page that
the audit cannot locate in any JSON it is allowed to consult, largely
`index.html`, `roadmap.html`, `anthropic_case.html` and prose in the narrative
pages. They are listed individually in `claims.html` with an UNMATCHED verdict,
so they are declared rather than hidden. Driving that number down means having
generators write the numbers their prose quotes; that is a follow-up, not a
regression.

### 5. `reports/claims.html`

`analysis/statcheck_reports.py:935` `render_claims_page()`. It imports and calls
`_theme.wrap_html` exactly as `generate_roadmap.py` does (`sys.path.insert` of
the analysis dir, then `from _theme import wrap_html, make_section,
make_kpi_card, make_kpi_row`). **`_theme.py` was not modified.**

Measured output: 283 KB, `<title>Claims Register | Oura Digital Twin</title>`,
a KPI row (pages checked / claims extracted / verified / mismatches), a legend
explaining the three verdicts, and one table row per claim: **654 rows, 423 OK,
0 MISMATCH, 231 UNMATCHED** at the time it was rendered (673/433/240 after the
final run). Each row shows page, statistic, the value on the page, the value in
the JSON, the JSON path, the verdict and the surrounding context.

You added `claims.html` to `REPORT_REGISTRY`, so it is in the navigation. The
smoke tests treat it as an audit output and do not require it in `HTML_TO_JSON`.

### 6. Verified in both directions

Not just "it passed". I built a two-file fixture and ran the real audit on it:

```
# page says p = 0.310, JSON says 0.004, context names the same comparison
  Mismatches: 1   [!!] tau_u_effects.html
      JSON path: comparisons.baseline_vs_treatment.tau_u_p_value
  RESULT: FAIL - 1 PROBLEM(S) FOUND       exit=1

# same fixture with the page corrected to p = 0.004
  Matched (OK): 1   Mismatches: 0
  RESULT: PASS                            exit=0
```

---

## C3: deploy gates

### `scripts/check-links.py` (new)

Stdlib only, no network. Crawls a build directory (default `deploy/`), parses
every `.html` with `html.parser`, and resolves every internal `href`, `src`,
`poster`, `data-src` and `srcset` candidate. External schemes and bare
fragments are counted and skipped. A link that resolves to a directory must find
`index.html` inside it; a link that escapes the build directory is a failure.
`404.html` is required at the root, with the reason stated in the failure text.
`assets/` needs no special case: it is a real directory in the build and
resolves like any other path.

On the current build:

```
$ .venv/bin/python scripts/check-links.py deploy
  HTML files:       101
  Internal links:   3982 (33 distinct targets)
  External links:   782 (not fetched)
  Broken links:     0
  Required files:   1/1 present
  RESULT: PASS                            exit=0
```

It found a real defect on the first run, before any of this was wired:
`glucose_autonomic_coupling.html` linked to `cgm_hypotheses_pre_registered.md`,
which exists in `reports/` but was never copied into the build because the
assembly copied only `*.html` and `*.json`. You have since deleted that page;
the assembly now copies `*.md` as well, so the class of defect is closed either
way.

Failure paths verified, not assumed:

```
# build with a dangling link and no 404.html
  FAIL: 404.html is missing from ...
    page.html:1  ->  nope.html    (no such file in the build)
  RESULT: FAIL                            exit=1

# same build with 404.html and the targets present
  RESULT: PASS                            exit=0
```

### Both deploy scripts now gate before wrangler

`scripts/daily-pipeline-local.sh` and `scripts/deploy-local.sh` each run, in
order: the analyses, the claim audit, assembly, the link crawl, and only then
`wrangler`. The audit gate is a byte-identical snippet in both (verified with
`diff`), and it was tested in all three states: `pass: false` gives exit 1,
`pass: true` gives exit 0, audit file missing gives exit 1.

Assembly in both scripts now:

- is a **clean rebuild** (`rm -rf deploy && mkdir -p deploy`), so a page you
  delete from `reports/` disappears from the build. This was already the
  existing behaviour and I kept it; the two orphans you removed
  (`compare_henrik_mitch.html`, `glucose_autonomic_coupling.*`) are gone from
  the rebuilt `deploy/`.
- copies `reports/*.html`, `reports/*.json`, `reports/*.md` and `reports/assets/`.
  The dated snapshots (`oura_full_analysis_YYYYMMDD.html`) are carried, because
  they match `*.html`: **64 of them** in the current build.

Measured build: **211 files, 101 HTML** = 37 current pages plus 64 dated
snapshots, plus `assets/og.png`.

### `daily-pipeline-local.sh` fails loudly

`scripts/daily-pipeline-local.sh:68` installs `trap on_exit EXIT`. A named
`STEP` variable is set before each stage (`token refresh`, `import`, `run_all`,
`statcheck`, `demo.db sync`, `assemble deploy`, `link check`, `deploy`,
`log rotation`). On any non-zero exit it:

1. writes `~/.local/state/oura-pipeline/last-failure.txt` with the timestamp,
   the step, the exit code, the first error line and the log path;
2. raises `notify-send -u critical "Oura pipeline failed" "<step>: <first error
   line>. See <log>"`;
3. on success, removes the failure file, so a stale one never lingers.

The first error line is scraped only from **this run's** portion of the day log
(the line count is recorded before the `tee` starts), with a one-second pause so
the `tee` subshell has flushed.

**notify-send was verified from a real systemd user unit**, not assumed:

```
$ systemctl --user show-environment | grep DBUS
DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus

$ systemd-run --user --unit=oura-notify-selftest --wait --collect <selftest>
...
notify-test.sh[...]: PIPELINE FAILED at step 'statcheck' (exit 1): STATCHECK FAIL: ...
notify-test.sh[...]: notify-send rc=0
```

`last-failure.txt` was written correctly by that test and then removed, along
with the self-test log, so nothing looks like a real failure.

The script also exports a fallback
`DBUS_SESSION_BUS_ADDRESS=unix:path=$XDG_RUNTIME_DIR/bus` when the variable is
absent, so a cron or ssh invocation still reaches the desktop. There is a
pre-existing `OnFailure=oura-pipeline-notify.service` on the unit; it stays, and
is now the second line of defence behind a message that names the step.

### The token message is accurate

I read `scripts/oura-reauth.py` first. **It does not open a browser.** It
prints an authorize URL for you to open and then catches the callback on
`http://localhost:8080/callback`. The message at
`scripts/daily-pipeline-local.sh:109` says exactly that:

> Oura refresh token rejected. Henrik: run .venv/bin/python scripts/oura-reauth.py
> (it prints an authorize URL for you to open, then catches the callback on
> http://localhost:8080/callback and writes the new tokens into .env),
> or put OURA_PAT in .env

The `OURA_PAT`-first logic is unchanged.

---

## C2: smoke tests

`tests/analysis/test_report_outputs.py` (new). Stdlib `unittest`, discovered by
`run_tests.py` via `tests/**/test_*.py`, run as a subprocess, exits non-zero on
failure and prints an `N/M passed` line like the existing suites. **It reads
`reports/` and never regenerates anything.**

The page registry is `statcheck_reports.HTML_TO_JSON`: one list, not a second
one that drifts.

12 tests in four groups:

| Group | Asserts |
|---|---|
| `ReportPagesExist` | `reports/` exists; every registry page was generated; every page reachable from `_theme.REPORT_REGISTRY` is covered by the page registry (this is what would catch a nav entry pointing at a page nobody generates) |
| `ReportPagesAreComplete` | every page is at least 20 000 bytes (`404.html` at least 4 000, it is deliberately lean); no page contains a runtime-failure string; every page closes its `</html>` |
| `MetricsJsonContract` | every companion JSON exists, parses, and is a dict; every one still carries the top-level keys its page depends on; every JSON referenced by the registry has a declared key contract |
| `ChronosRanForReal` | `foundation_model_metrics.json` has `chronos_available: true`, as you asked. False is a failure, with `chronos_error` printed |

The failure strings are **case-sensitive on purpose**:
`No module named`, `No model backend`, `not installed`,
`Optional dependency unavailable`, `dependency unavailable`,
`Traceback (most recent call last)`, `FAILED`. Lowercase `failed` is not in the
list, because "failed fits remain" and `"failed_total": 0` are legitimate. I
verified no uppercase `FAILED` and no `Traceback` exists in any current report.

The required-key contract was **derived from the generators**, not from one
run's output: for each metrics JSON I took the structural sections (dict or list
values) plus the provenance stamp, and kept a key only when the literal also
appears in the generator's source. `oura_full_analysis.json` is a flat scalar
payload, so it gets a named subset of the numbers the page leads with. 31 JSON
files have a contract; a new metrics file with no contract fails
`test_every_checked_json_has_a_contract`.

```
$ .venv/bin/python run_tests.py
  PASS  tests/analysis/test_report_outputs.py  12/12 passed
  PASS  tests/ingest/test_import_checkme_spo2.py  16/16 passed
  PASS  tests/ingest/test_import_contec_abpm.py  14/14 passed
  PASS  tests/ingest/test_import_glucose.py  10/10 passed
  PASS  tests/ingest/test_import_viatom_ecg.py  19/19 passed
  PASS  tests/ingest/test_ingest_common.py  20/20 passed
  6/6 suites passed
```

---

## C1: truth in docs

### `README.md`

Every count is now measured, and the source of each is named in the file.

| Was | Is | Measured how |
|---|---|---|
| "runs 12 analysis scripts" | "runs **34** analysis scripts" | `len(SCRIPTS)` in `run_all.py` |
| "all 12 scripts, ~35 seconds ... ~2 minutes" | "all **34** scripts" (no invented runtime; `reports/run_summary.json` carries the real one) | |
| "generates 12 self-contained HTML reports" | "generates **37** current self-contained HTML pages ... the build additionally carries **64** dated snapshot pages, **101** HTML files in total" | `ls deploy/*.html`, split on the `_YYYYMMDD.html` suffix |
| "Current window: 79 modeled days, 8 post-ruxolitinib days" | removed; replaced with wording computed from `config.py` (`DATA_START`, `TREATMENT_START`, `HEV_DIAGNOSIS_DATE`) and a note that each page prints the window it actually modelled | |
| "demo.db (79 days of real Oura data)" | "covers **2026-01-08 to 2026-09-15** (**240** nights with a scored long-sleep period)" | `SELECT MIN(day), MAX(day), COUNT(DISTINCT day) ...` |
| 12-row script table | **34-row table** regenerated from `SCRIPTS`, with each script's docstring first line as the method and its real `REPORTS_DIR` HTML output | script over `run_all.py` plus `ast.get_docstring` |
| 12-row report table | full table generated from `_theme.REPORT_REGISTRY`, grouped as the navigation groups them | `REPORT_REGISTRY` |
| "Python 3.12" | "Python 3.10+ ... (developed and run on **Python 3.14.7**)" | `sys.version` |
| "GPU recommended" for Chronos | "Chronos runs on CPU by default. Set `CHRONOS_DEVICE=cuda` to opt in" | |

Also added: a **Gates** section (what the three gates enforce and where they
live), a **Tests** section, the `DATABASE_PATH` override, `duckdb` and
`torchvision` in the dependency list with what each unlocks, `CHANGELOG.md` and
the new `scripts/`, `tests/analysis/` and `analysis/_frames.py` entries in the
structure tree, and the one-row-per-date rule under Methodology.

The live-site link and the MIT licence line are unchanged.

Verified no stale number survived:

```
$ grep -nE '\b12 (analysis )?scripts|all 12|12 self-contained|79 modeled|8 post-ruxolitinib|79 days' README.md
(none)
```

### `run_all.py` docstring

"Executes 12 pipeline scripts sequentially" now describes `SCRIPTS` without
restating a number that will rot, and documents the audit gate and
`run_summary.json`.

### `CHANGELOG.md` (new)

One entry, dated **2026-09-20**, facts only, grouped: optional backends, the
one-row-per-date fix, the claim audit, the pipeline gates, configuration, tests,
documentation. It opens by stating that nothing in it describes a deployment.

---

## Items you asked for mid-task

| # | Ask | Status |
|---|---|---|
| 1 | Add `generate_404.py` and `generate_how_built.py` to SCRIPTS, skip gracefully if absent | Done. `run_all.py:76` `OPTIONAL_SCRIPTS`; a missing optional generator records `SKIPPED`, which is excluded from the failure count, so the pipeline does not turn red for a page that does not exist yet. Both existed by the final run and both passed. |
| 2 | `reports/run_summary.json` with `{generated_at, scripts:[{name, ok, runtime_s}], total_runtime_s, passed, failed}` | Done, `run_all.py:265`. Exactly that shape, plus `skipped` (leaving skipped scripts invisible would misreport the totals). Current file: `total_runtime_s: 538.0`, `passed: 34`, `failed: 0`, `skipped: 0`, 34 script entries. |
| 3 | Copy `reports/assets/` and `404.html` in both deploy scripts; treat `assets/` as valid link targets | Done. `assets/og.png` is in the build; `404.html` arrives with `*.html`; `check-links.py` resolves `assets/` with no special case, and requires `404.html`. |
| 4 | Check `roadmap.html` against `causal_inference_metrics.json` and `composite_biomarkers.json` | Done. `HTML_TO_JSON` values accept a list; `roadmap.html` names both. |
| 5 | Add `generate_anthropic_case.py` before `generate_index.py`, keep its helseoversikt copy | Done, `run_all.py:69`. The script is unmodified, so its second write to `~/projects/helseoversikt/36_Anthropic_Case/technical/anthropic_case.html` is intact. It passed in 0.5 s. |
| 6 | Fix the multimodal "DuckDB frame backend" placeholder; add "dependency unavailable" to the gate | Done. `duckdb` was the missing package. Real report: 99 days, 6 channels, 55 univariate discords. String added to the smoke-test gate. |
| 7 | Clean rebuild of `deploy/`, keep the dated snapshots | Done and verified: 101 HTML = 37 current plus 64 snapshots; the two orphans you deleted are gone. |
| 8 | `chronos_available: false` = fail | Done: asserted in `ChronosRanForReal`, and `chronos_available` is a required key in the JSON contract. Currently `true`. |
| 9 | Re-run `run_all.py` at the end and report the totals | Done: **34/34, 538.0 s, statcheck pass, send bundle assembled.** |

---

## The five files I was not allowed to edit

`analysis/_theme.py`, `analysis/generate_roadmap.py`,
`analysis/generate_index.py`, `analysis/analyze_oura_causal.py`,
`analysis/generate_anthropic_case.py`: **not modified.**

**No change to any of them is required by A3, A4, C1, C2 or C3.** Specifically:

- No A3 failure-string hit falls in any of them.
- No statcheck mismatch falls in any of them (there are none anywhere).
- `claims.html` is built by *calling* `_theme.wrap_html`; the theme needed no new
  API. It picked up `claims.html` in the nav from the registry entry you added.
- The duplicate-date fix in `analyze_oura_causal.py` was yours, already landed.

Two observations, offered as findings rather than requests. Both are yours to
decide and neither blocks anything:

1. **`analysis/_theme.py`, `wrap_html()`** emits
   `<html lang="en" data-theme="dark">` on every page, while the palette the same
   file defines is the light clinical set (`BG_PRIMARY = "#F7F7F5"`,
   `TEXT_PRIMARY = "#14161A"`). If any CSS ever keys off `[data-theme]`, the
   attribute and the pigment disagree. Proposed edit, if you want it:

   ```diff
   -<html lang="en" data-theme="dark">
   +<html lang="en" data-theme="light">
   ```

2. **`analysis/_theme.py`, `wrap_html()`** emits
   `<meta name="robots" content="noindex, nofollow">`. The plan lists indexing as
   an open decision (live has `noindex`, the fresh build does not), and you have
   already parameterised it as `{robots}`, so this is a call for you and Henrik,
   not an edit for me.

---

## What remains

- **240 unmatched claims** in `claims.html`. Not errors; numbers the audit
  cannot trace to a JSON. Reducing them means generators writing the figures
  their prose quotes. Most of the residue is `index.html`, `roadmap.html` and
  `anthropic_case.html`.
- **60 `p=0.0000` warnings.** Printing `p < 0.0001` instead would be more honest
  about precision. Mostly inside Plotly annotations.
- **54 `inconsistent_self_report` warnings.** Two different p-values sharing a
  30-character context prefix, usually adjacent table rows. Worth a look to
  confirm they are all benign; I spot-checked several and they were.
- **`mitch_standalone_report.html` and `wenche_standalone_report.html`** are in
  the navigation and in the page registry, but `analyze_patient_standalone.py`
  is **not** in `run_all.py` SCRIPTS, so those two pages never refresh with the
  rest. They are stale by construction. Adding the generator to SCRIPTS (it
  takes a `--profile`) or removing the pages from the nav would close it; I did
  not choose for you.
- **Chronos model download.** The first run fetched `amazon/chronos-bolt-base`
  into the HF cache (280.7 s cold versus 50.9 s warm). A machine with no cache
  and no network will fall back; the page says so.
- **`ssm`** remains uninstalled by design; `hmmlearn` carries the GvHD model.

## Everything that changed

| File | What |
|---|---|
| `config.py:22` | `DATABASE_PATH` honours the env var of the same name |
| `run_all.py` | docstring; `OPTIONAL_SCRIPTS` (76); `run_statcheck()` (223); `write_run_summary()` (265); statcheck gate before the send bundle (387); non-zero exit (416); SCRIPTS plus `generate_404.py`, `generate_how_built.py`, `generate_anthropic_case.py` |
| `analysis/statcheck_reports.py` | docstring; `HTML_TO_JSON` 12 to 34 with list values (119); `AUDIT_OUTPUT_PAGES` (170); `classify_stat_key()` plus `STAT_RANGES` (367); `_is_anchored()` (480); `_satisfies()` (497); legend guard; multi-JSON `run_audit`; `audit_passed()`; `render_claims_page()` (935) |
| `analysis/analyze_oura_foundation_models.py:123,382` | torchvision stub is a fallback; CPU is the default device |
| `analysis/analyze_sequential_ci.py:245` | one row per date (fixes the crash) |
| `analysis/analyze_piecewise_its.py:150` | one row per date |
| `analysis/analyze_tau_u.py:208` | one row per date |
| `analysis/analyze_placebo_tests.py:144` | one row per date |
| `analysis/analyze_rux_forecast.py:120` | one row per date |
| `scripts/check-links.py` | new: internal-link and `404.html` gate |
| `scripts/daily-pipeline-local.sh` | loud failure (68); token message (109); statcheck gate; `*.md` and `assets/` in the assembly; link gate (193) |
| `scripts/deploy-local.sh` | newest-DB selection by mtime; statcheck gate; `*.md` and `assets/`; link gate (87) |
| `tests/analysis/test_report_outputs.py` | new: 12 report-output smoke tests |
| `requirements-full.txt` | `duckdb>=1.0` |
| `README.md` | measured counts; regenerated tables; gates, tests, `DATABASE_PATH` |
| `CHANGELOG.md` | new: one entry, 2026-09-20 |
| `docs/BUILD-2026-09-20-pipeline.md` | this report |

No `*.bak` file was created, no code was commented out, no TODO was added.
