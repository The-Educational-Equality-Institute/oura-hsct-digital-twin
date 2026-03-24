# HTML Report Audit: analyze_oura_gvhd_predict.py + _theme.py

**Date:** 2026-03-24
**Scope:** HTML generation quality, Plotly embedding, XSS, responsiveness, accessibility, metadata accuracy
**Files audited:**
- `analysis/analyze_oura_gvhd_predict.py` (2680 lines)
- `analysis/_theme.py` (2159 lines)

---

## 1. HTML Template Well-Formedness

**Verdict: PASS (good)**

The `wrap_html()` function in `_theme.py` (line 2033) produces a complete, well-formed HTML5 document:

- `<!DOCTYPE html>` declaration present
- `<html lang="en" data-theme="dark">` with language attribute
- Proper `<head>` with `<meta charset="UTF-8">`, viewport meta, theme-color meta, robots noindex
- `<title>` properly set: `{title} -- Oura Digital Twin`
- Body structure: `<nav>` -> context strip -> `<div class="odt-header">` -> `<div class="odt-container">` -> `<div class="odt-footer">`
- All opened tags are closed

**CSS analysis:**
- CSS is generated via `get_base_css()` (lines 417-1798) inside a single `<style>` block
- Uses CSS custom properties (`:root` variables) consistently
- All braces are balanced (verified by f-string double-brace escaping pattern `{{` / `}}`)
- Extra report-specific CSS is injected via `extra_css` parameter wrapped in its own `<style>` block (line 2082)

**Minor issues found:**
- None. The CSS is properly scoped and all selectors are well-formed.

---

## 2. Plotly Figure Embedding

**Verdict: PASS (good)**

**CDN version:** Plotly 2.35.2 loaded from `https://cdn.plot.ly/plotly-2.35.2.min.js` (config.py line 51). This is a recent stable version.

**Embedding method:**
- Each figure is converted via `fig.to_html(full_html=False, include_plotlyjs=False)` (line 2233)
- This correctly avoids duplicate Plotly.js bundles -- the CDN script is loaded once in `<head>`
- Figures are inserted as inline div fragments with JSON data attributes

**Plotly config:**
- `hovermode="x unified"` set on all figures (template default + per-figure)
- `responsive: true` set in lazy-load config (line 2102)
- `displayModeBar: true`, `scrollZoom: true` enabled
- `displaylogo: false` (removes Plotly branding)
- `modeBarButtonsToRemove: ['lasso2d', 'select2d']` -- sensible for time-series

**Plotly enhancer JS (`get_plotly_enhancer_js()`, line 273):**
- Runtime patch that enforces minimum font sizes, margins, automargin
- Runs on `window.load` with two passes (120ms and 900ms delay) for reliability
- Resize handler registered for `window.resize` event
- `Plotly.relayout()` + `Plotly.Plots.resize()` chain with error handling

**Lazy-load (IntersectionObserver):**
- Available via `chart_data` parameter in `wrap_html()` but NOT used by gvhd_predict (it uses inline embedding)
- For this report, all 5 figures are embedded inline -- no lazy-loading. This means all 5 Plotly charts render immediately, which could be slow on mobile but is acceptable for 5 charts.

**One concern:** The inline figures from `to_html()` bypass the Plotly config object (no `{responsive: true, displaylogo: false}` etc). Only the lazy-loaded charts get the full config. For the inline figures, the template defaults (`clinical_dark`) and the enhancer JS compensate, but the modebar buttons (lasso/select removal) are NOT applied to inline figures.

---

## 3. XSS Risk Analysis

**Verdict: LOW RISK (acceptable for this use case)**

All data inserted into HTML comes from:
1. **SQLite database values** (numeric biometric data) -- no user-supplied strings
2. **Hardcoded constants** (STATE_NAMES, feature labels, clinical dates)
3. **Computed statistics** (floats, integers, date strings)

**No `html.escape()` is used anywhere in the analysis scripts.** Specific injection points:

| Location | Data Source | Risk |
|----------|------------|------|
| State distribution table (line 2364-2368) | `state_dist` dict keys = STATE_NAMES constant | None -- hardcoded |
| Alert table (lines 2408-2418) | `alert['date']`, `alert['level']`, numeric probs | None -- all computed |
| Feature importance table (lines 2444-2450) | `r['label']` from feature column names | None -- hardcoded labels |
| `fallback_reason` (line 2381-2383) | `model_info` dict from internal computation | Low -- but unescaped |
| `top_feature_labels` (line 2264) | Joined feature labels | None -- hardcoded |
| BOS section values (lines 2474-2492) | Computed scores + loaded from JSON file | Low -- JSON file is self-generated |

**The `fallback_reason` on line 2388 is the only value that could theoretically contain unexpected characters** (it comes from exception messages). In practice, this is an internal error string, not user input. However, if an exception message contained `<script>`, it would be injected unescaped.

**Recommendation:** Add `html.escape()` for `fallback_reason` as defensive coding. Priority: low.

---

## 4. Responsiveness / Mobile-Friendliness

**Verdict: PASS (good)**

**Viewport meta:** `<meta name="viewport" content="width=device-width, initial-scale=1.0">` -- correct.

**Responsive breakpoints in `_theme.py`:**
- `@media (max-width: 900px)` -- tablet/small laptop:
  - Nav collapses to hamburger menu (`.odt-nav-toggle` shown, `.odt-nav-links` hidden until `.open`)
  - Container padding reduced (32px -> 16px)
  - KPI grid: `repeat(2, 1fr)` (2 columns)
  - Section padding reduced, scroll-margin adjusted for sticky nav
  - Context strip stacks vertically
- `@media (max-width: 480px)` -- phone:
  - KPI grid: `1fr` (single column)
  - Header h1 font reduced to 1.625rem
  - KPI head stacks vertically

**Plotly charts:**
- CSS forces `width: 100% !important` on `.plotly-graph-div`, `.js-plotly-plot`, `.plot-container`, `.svg-container`
- Enhancer JS sets `autosize: true` and calculates container width
- Resize event listener triggers `Plotly.Plots.resize()`

**Issue:** Fixed chart heights (900px, 800px, 950px, 700px, 450px+) may cause awkward layouts on small screens. The charts will be full-width but very tall relative to viewport. No `max-height` or scroll wrapper is applied. This is a minor UX concern on phones.

**Hamburger menu:** Uses inline `onclick` with `classList.toggle('open')` and proper `aria-label`/`aria-expanded` attributes -- accessible.

---

## 5. Color Accessibility

**Verdict: MIXED -- some concerns**

### Colorblind Safety
The COLORWAY array (line 114) uses 8 colors:
- Blue (#3B82F6), Green (#10B981), Amber (#F59E0B), Purple (#8B5CF6)
- Pink (#EC4899), Cyan (#06B6D4), Orange (#F97316), Indigo (#6366F1)

**State colors** are the critical set:
- Remission = Green (#10B981)
- Pre-flare = Amber (#F59E0B)
- Active Flare = Red (#EF4444)
- Recovery = Blue (#3B82F6)

**For deuteranopia (red-green colorblindness):** Red (#EF4444) and Green (#10B981) will be difficult to distinguish. This is the most common form of colorblindness (~8% of males). The state model chart (Section 3) shows stacked probability areas in these 4 colors -- red/green confusion is a real concern.

**Mitigations already present:**
- State names appear in tooltips (hover data includes state name text)
- Viterbi path panel uses discrete colored blocks -- pattern/position helps
- KPI cards have text labels ("Flare", "Stable") alongside color coding

**Contrast ratios (dark theme):**
| Element | Foreground | Background | Approx Ratio | WCAG AA? |
|---------|-----------|------------|-------------|----------|
| Body text | #E8E8ED on #0F1117 | -- | ~14:1 | PASS |
| Secondary text | #9CA3AF on #0F1117 | -- | ~7.5:1 | PASS |
| Tertiary text | #6B7280 on #0F1117 | -- | ~4.3:1 | PASS (large) |
| KPI label | rgba(255,255,255,0.5) on dark | -- | ~4.5:1 | Borderline |
| KPI detail | rgba(255,255,255,0.4) on dark | -- | ~3.5:1 | FAIL AA normal |
| `.odt-kpi-explainer` | rgba(255,255,255,0.35) | -- | ~3.0:1 | FAIL AA |
| Disclaimer text | #9CA3AF on #242837 | -- | ~6.1:1 | PASS |
| Alert-red row | on rgba(239,68,68,0.1) | -- | Fine | PASS |

**Issues:**
1. **KPI detail text** at `rgba(255,255,255,0.4)` fails WCAG AA (4.5:1 minimum for normal text). Currently ~3.5:1.
2. **KPI explainer text** at `rgba(255,255,255,0.35)` also fails. Currently ~3.0:1.
3. **Red/green state distinction** problematic for colorblind users.

**Recommendation:**
- Increase KPI detail/explainer opacity to at least 0.55 (rgba(255,255,255,0.55)) for AA compliance
- Consider adding pattern fills or distinct markers to state probability areas

---

## 6. Report Title and Metadata Accuracy

**Verdict: PASS (accurate)**

**HTML title:** `"GvHD Prediction Model -- Oura Digital Twin"` (line 2564)
**Header h1:** `"GvHD Prediction Model"` -- matches the report content.

**Subtitle (line 2557-2559):**
```
Oura Ring, {DATA_START} to {DATA_END} ({n_days} days) | State model: {display_model_type} | Ruxolitinib started {RUXOLITINIB_START} | HEV diagnosed {HEV_DIAGNOSIS}
```
This accurately describes:
- Data range (dynamically resolved)
- Which state model was used (rSLDS or HMM fallback)
- Treatment start date
- HEV confound date

**Navigation:** `report_id="gvhd"` correctly matches the registry entry `{"id": "gvhd", "file": "gvhd_prediction_report.html", "title": "GvHD Prediction"}`.

**Footer metadata:**
- Data range displayed
- Post-intervention days computed
- N=1 disclaimer present
- Generation timestamp shown

**Content sections match the docstring claims:**
1. Temperature Fluctuation Analysis -- present
2. Multi-Stream GVHD Composite Score -- present
3. rSLDS/HMM State Model -- present (adapts label)
4. Retrospective Alert Burden -- present
5. Predictive Feature Importance -- present
6. BOS Risk Integration -- present

All 6 sections documented in the module docstring are rendered.

---

## 7. Interactive Elements (Tooltips, Hover Data)

**Verdict: PASS (good)**

All Plotly traces use explicit `hovertemplate` strings. Analysis of each chart:

**Chart 1 - Temperature (3 panels):**
- Panel 1: `"<b>%{x|%b %d, %Y}</b><br>Deviation: %{y:+.2f} °C<br><extra></extra>"` -- correct unit, signed value
- Panel 2: `"%{x|%b %d}: %{y:.3f} °C<br><extra></extra>"` -- 7d variability
- Panel 3: `"%{x|%b %d}: %{y:+.3f} °C/night<br><extra></extra>"` -- gradient

**Chart 2 - Composite Score (2 panels):**
- Daily: `"<b>%{x|%b %d}</b><br>Composite: %{y:.1f}/100<br><extra></extra>"`
- 7-day rolling: `"%{x|%b %d}: %{y:.1f}<br><extra></extra>"`
- Component traces: `"%{x|%b %d}: %{y:.1f}<br><extra></extra>"`

**Chart 3 - State Model (3 panels):**
- State probs: `"P({name}): %{y:.2f}<br><extra></extra>"` where `{name}` is the state name
- Convergence: `"Value: %{y:.1f}<br><extra></extra>"`

**Chart 4 - Alerts (2 panels):**
- Composite line: has standard hovertemplate
- Probability traces: standard format

**Chart 5 - Feature Importance:**
- `"<b>%{y}</b><br>Importance: %{x:.3f}<br><extra></extra>"`

All hovertemplates use `<extra></extra>` to suppress the default trace name box. Date formatting is consistent (`%b %d` or `%b %d, %Y`). Units are correctly specified in tooltips.

**`hovermode="x unified"`** is set on all figures -- this shows all traces at the same x-position in a single tooltip, which is ideal for multi-trace time-series.

**Crosshair spikes** are enabled on all axes (`spikemode="across"`, `spikethickness=1`, dot dash) -- provides visual guidance when hovering.

---

## 8. Theme Consistency (_theme.py Application)

**Verdict: PASS (consistent)**

The GVHD report uses the theme system correctly:

**Imports from `_theme.py` (line 86-90):**
- `wrap_html` -- page assembly
- `make_kpi_card`, `make_kpi_row` -- KPI dashboard
- `make_section` -- section cards
- Color constants (`COLORWAY`, `STATUS_COLORS`, `BG_*`, `TEXT_*`, `ACCENT_*`, `BORDER_*`)

**Plotly template:** `pio.templates.default = "clinical_dark"` set at line 93. This ensures all figures inherit the dark theme (backgrounds, fonts, gridlines, colorway).

**Report-specific CSS (lines 2510-2551):**
- `.clinical-note` -- amber-bordered context box (uses `ACCENT_AMBER`, `TEXT_PRIMARY`)
- `.methodology` -- blue-bordered method description (uses hardcoded rgba matching `ACCENT_BLUE`)
- `.disclaimer` -- elevated background card (uses `BG_ELEVATED`, `BORDER_SUBTLE`, `TEXT_SECONDARY`)
- `.alert-red` / `.alert-yellow` -- table row highlights

All custom CSS uses the same color palette as the theme. No rogue colors or conflicting styles.

**Consistency check across components:**
- KPI cards: 4 cards via `make_kpi_card()` -> `make_kpi_row()` -- correct
- Sections: 6 sections via `make_section()` -- correct
- Navigation: `report_id="gvhd"` highlights correct nav link
- Footer: generated by `wrap_html()` with proper date range

---

## Summary of Findings

| Category | Status | Priority |
|----------|--------|----------|
| HTML well-formedness | PASS | -- |
| Plotly CDN + embedding | PASS | -- |
| Inline figure modebar config | MINOR | Low |
| XSS risk | LOW | Low |
| `fallback_reason` unescaped | MINOR | Low |
| Responsive layout | PASS | -- |
| Fixed chart heights on mobile | MINOR | Low |
| Text contrast (KPI details) | FAIL WCAG AA | Medium |
| Red/green colorblind distinction | CONCERN | Medium |
| `prefers-reduced-motion` | PASS | -- |
| Print styles | PASS | -- |
| Report title/metadata | PASS | -- |
| Hover tooltips | PASS | -- |
| Theme consistency | PASS | -- |

### Actionable Recommendations

1. **Medium priority:** Increase `rgba(255,255,255,0.4)` and `rgba(255,255,255,0.35)` in KPI detail/explainer to at least `rgba(255,255,255,0.55)` for WCAG AA compliance.

2. **Medium priority:** Add pattern differentiation (dash patterns, distinct markers) to state probability area chart for colorblind users, or use a colorblind-safe 4-color palette for the states.

3. **Low priority:** Add `html.escape()` to `fallback_reason` string before HTML insertion (line 2388).

4. **Low priority:** Consider adding `max-height` + `overflow-y: auto` wrapper around Plotly charts for mobile viewports.

5. **Low priority:** Inline-embedded figures (non-lazy) do not receive `displaylogo: false` or `modeBarButtonsToRemove` config. The enhancer JS partially compensates but does not remove buttons.
