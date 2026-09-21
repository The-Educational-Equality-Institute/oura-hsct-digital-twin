# Changelog

Dated record of changes to the pipeline, the gates and the published pages.
Numbers here are measured, not estimated. Nothing in this file describes a
deployment; see the live site for what is currently published.

## 2026-09-20

### Optional backends now actually run

- Installed the full optional stack into `.venv` (Python 3.14.7): `pycausalimpact`
  0.1.1, `hmmlearn` 0.3.3, `torch` 2.14.0+cpu, `chronos-forecasting` 2.3.2,
  `tigramite` 5.2.10.1, `stumpy` 1.14.1, `tsfresh` 0.21.2, `antropy` 0.2.2,
  `nolds` 0.6.3, `filterpy` 1.4.5, `pykalman` 0.11.2, plus `duckdb` 1.5.5 and
  `torchvision` 0.29.0+cpu. `ssm` remains uninstalled; the GvHD module uses its
  documented `hmmlearn` fallback.
- `analysis/analyze_oura_foundation_models.py`: Chronos now selects CPU by
  default. It previously chose CUDA whenever a GPU was visible, which fails on a
  machine whose VRAM is nearly full. `CHRONOS_DEVICE=cuda` opts back in.
- `analysis/analyze_oura_foundation_models.py`: the torchvision compatibility
  stub is now a fallback rather than an unconditional replacement. Installing it
  over a working torchvision made `transformers` fail with
  `cannot import name 'ImageReadMode' from 'torchvision.io'`, which left
  `chronos_available: false` in the metrics and no forecast on the page.
- `requirements-full.txt`: added `duckdb>=1.0`. Without it
  `analysis/_frames.py` cannot import and `analyze_multimodal_anomalies.py`
  publishes "Optional dependency unavailable" instead of a report.

### One row per date

A night can carry more than one scored `long_sleep` period. Two dates in the
current data do: 2026-05-19 and 2026-06-01. Loaders that did not collapse them
counted those dates twice, and `analyze_sequential_ci.py` crashed outright with
`cannot reindex on an axis with duplicate labels`.

Each of these now keeps the longest period per date, matching the convention
already used in `analyze_oura_causal.py`:

- `analysis/analyze_sequential_ci.py`
- `analysis/analyze_piecewise_its.py`
- `analysis/analyze_tau_u.py`
- `analysis/analyze_placebo_tests.py`
- `analysis/analyze_rux_forecast.py`

### The claim audit became a gate

`analysis/statcheck_reports.py` existed and was documented as "a post-generation
step in run_all.py", but nothing ran it.

- It now covers every published page (34 registry entries, up from 12 hardcoded),
  and a page may declare several authoritative JSON files.
- Reference extraction recognises the key names the metrics files actually use
  (`kruskal_p`, `mann_whitney_p`, `b4_pvalue`, plain `p`, `rho`, `tau`, `r2` and
  the rest). It previously recognised only `p_value` and `p_val`, so most printed
  statistics had no authority to match against and were paired with whichever
  unrelated value happened to be numerically nearest.
- A value difference is reported as a mismatch only when the claim's surrounding
  text names the statistic the reference holds. Otherwise the claim is reported
  as unmatched, which is what it is.
- Threshold statements (`p > 0.05`) are judged as inequalities, not equalities.
- Significance legends under tables (`* p<0.05, ** p<0.01`) are notation and are
  no longer extracted as claims.
- Effect of the four items above, same reports, same data: 40 reported
  mismatches to 0, with JSON references rising from 824 to 1449 and verified
  claims from 375 to 423.
- It writes `reports/claims.html` alongside `reports/statcheck_audit.json`:
  every extracted claim with its JSON value and an OK / MISMATCH / UNMATCHED
  verdict, built with the shared theme.

### The pipeline stops instead of shipping

- `run_all.py` runs the claim audit after the analyses, refuses to assemble
  `reports/send_bundle` when it does not pass, and exits non-zero.
- `run_all.py` writes `reports/run_summary.json`: per-script pass/fail and
  runtime, total runtime, and the passed/failed/skipped counts.
- `run_all.py` gained `generate_404.py`, `generate_how_built.py` and
  `generate_anthropic_case.py`. The first two are optional: if the file is not
  present the run records SKIPPED rather than failing.
- `scripts/check-links.py` (new): crawls a build directory, checks every
  internal `href`/`src` resolves, and requires `404.html`. Without a 404 page
  Cloudflare Pages answers every unknown path with `index.html` and HTTP 200, so
  a dead link is indistinguishable from a working one.
- `scripts/daily-pipeline-local.sh` and `scripts/deploy-local.sh` run the claim
  audit and the link crawl before the deploy step and stop there on failure.
  Both now also copy `reports/assets/` and `reports/*.md` into the build; the
  markdown fixes a link that had no target in the build.
- `scripts/daily-pipeline-local.sh` fails loudly: on any failing step it writes
  the step, exit code, first error line and log path to
  `~/.local/state/oura-pipeline/last-failure.txt` and raises a critical desktop
  notification. Verified from a systemd user unit (`notify-send` returned 0).
  The pipeline had failed silently for four days before this.
- `scripts/daily-pipeline-local.sh`: a rejected Oura refresh token now prints
  what to do about it, naming `scripts/oura-reauth.py` and the `OURA_PAT`
  alternative.

### Configuration and deploy selection

- `config.py`: `DATABASE_PATH` honours the environment variable of the same
  name. `api/import_oura.py` already did; both pipeline scripts set it before
  invoking `run_all.py`; until now the analysis scripts ignored it and always
  read `data/oura.db`.
- `scripts/deploy-local.sh` picks the newer of `data/demo.db` and
  `data/demo-latest.db` by modification time and prints which, instead of a
  hardcoded preference that had gone stale.

### Tests

- `tests/analysis/test_report_outputs.py` (new): for every page in the registry,
  asserts the HTML exists, is a complete render, carries no text saying a module
  could not run, and that its metrics JSON parses and still has the top-level
  keys the page depends on. It reads `reports/` and never regenerates. It also
  fails if the navigation registry and the page registry disagree, and if the
  Chronos forecast did not run.

### Documentation

- `README.md`: script and report tables regenerated from `run_all.py` SCRIPTS;
  the "12 scripts" and "12 reports" counts replaced with measured ones; the
  stale fixed-window line ("79 modeled days, 8 post-ruxolitinib days") replaced
  with wording computed from `config.py`; gates, tests and the `DATABASE_PATH`
  override documented.
- `CHANGELOG.md` (this file, new).
