"""
Counterfactual "digital twin" fork for HRV: the whole point of the twin.

We take the PRE-treatment nightly HRV (RMSSD) level and hold it forward across the whole
observation window as the modelled "without treatment" trajectory, with a 95% band. The
ACTUAL trajectory is overlaid, with both medicine starts marked (ruxolitinib, then the
beta-blocker three weeks later). The gap between the projected band and the actual line is
the modelled effect of treatment; which medicine carries it is a question for the phase
analyses (piecewise ITS, Tau-U), not for this chart. Honest for an N=1: association, not
proof; HEV is a confounder.

No em-dashes anywhere (repo rule). Numbers computed live from data, never hardcoded.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(REPO))
from config import BETA_BLOCKER_START  # noqa: E402
INK = "#14161A"
ACCENT = "#3A3AD6"          # indigo, the actual on-Jakavi line
COUNTER = "#8A909C"         # grey, the projected no-Jakavi line
BAND = "rgba(138,144,156,0.16)"
GAP = "rgba(58,58,214,0.10)"
GRID = "rgba(20,22,26,0.06)"


def _load_series() -> tuple[list[str], list[float], int]:
    """dates, actual HRV, intervention index, from the precomputed causal timeseries."""
    d = json.loads((REPO / "reports" / "causal_timeseries.json").read_text())
    s = d["streams"]["mean_rmssd"]
    return s["dates"], s["actual"], int(s["intervention_idx"])


def compute_counterfactual() -> dict:
    """Project the no-Jakavi counterfactual forward and compare against the actual.

    The pre-treatment nightly HRV is flat-with-noise: its upward drift is NOT statistically
    significant (ITS time coefficient p=0.11), so extrapolating a trend far forward is not
    defensible. The honest counterfactual is the pre-treatment LEVEL held forward, with a 95%
    band from the pre-treatment scatter. The actual on-Jakavi trajectory is compared to it; the
    nights that sit above the whole band are nights outside what the twin projects without the
    medicine. Returns every series the chart needs plus the headline numbers.
    """
    dates, actual, ivx = _load_series()
    y = np.array(actual, dtype=float)

    pre_y = y[:ivx]
    mean = float(np.nanmean(pre_y))
    sd = float(np.nanstd(pre_y, ddof=1))
    half = 1.96 * sd

    n = len(actual)
    cf = [round(mean, 2)] * n
    cf_lo = [round(mean - half, 2)] * n
    cf_hi = [round(mean + half, 2)] * n

    recent = slice(n - 30, n)
    actual_recent = float(np.nanmean(y[recent]))
    gap_pct = (actual_recent - mean) / mean * 100.0
    post = np.arange(ivx, n)
    above = int(np.sum(y[post] > (mean + half)))

    # Second intervention: the beta-blocker. Split the post period at its start so the
    # chart and the copy can say where the nights above the band actually sit.
    bb_str = BETA_BLOCKER_START.isoformat() if BETA_BLOCKER_START else None
    bb_ivx = next((i for i, d in enumerate(dates) if d >= bb_str), n) if bb_str else n
    jakavi_only = np.arange(ivx, bb_ivx)
    bb_post = np.arange(bb_ivx, n)
    above_jakavi_only = int(np.sum(y[jakavi_only] > (mean + half)))
    above_bb = int(np.sum(y[bb_post] > (mean + half)))

    return {
        "dates": dates,
        "actual": [None if np.isnan(v) else round(float(v), 2) for v in y],
        "cf": cf,
        "cf_lo": cf_lo,
        "cf_hi": cf_hi,
        "ivx": ivx,
        "intervention_date": dates[ivx],
        "pre_mean": round(mean, 1),
        "band_lo": round(mean - half, 1),
        "band_hi": round(mean + half, 1),
        "band_half": round(half, 1),
        "actual_recent": round(actual_recent, 1),
        "cf_recent": round(mean, 1),
        "gap_pct": round(gap_pct, 0),
        "nights_above_band": above,
        "post_nights": int(len(post)),
        "bb_ivx": bb_ivx,
        "bb_date": dates[bb_ivx] if bb_ivx < n else None,
        "jakavi_only_nights": int(len(jakavi_only)),
        "nights_above_band_jakavi_only": above_jakavi_only,
        "bb_nights": int(len(bb_post)),
        "nights_above_band_bb": above_bb,
    }


def _read_json(name: str) -> dict:
    path = REPO / "reports" / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError:
        return {}


def phase_estimates(metric: str = "mean_rmssd") -> dict:
    """Phase-corrected estimates for one metric from the pipeline's own outputs.

    Reads the piecewise ITS (AR(1)) steps at each medicine start, the Tau-U phase
    contrasts, the phase means, and the placebo calibration of the simple pre/post test.
    Everything is optional; a missing file just leaves its key out.
    """
    out: dict = {}
    co = _read_json("piecewise_regression_metrics.json").get("metrics", {}).get(metric, {}).get("coefficients", {})
    if co:
        out["its_jakavi"] = co.get("jakavi", {})
        out["its_bb"] = co.get("bb", {})
    comps = _read_json("tau_u_metrics.json").get("comparisons", {})
    ab = comps.get("A_vs_B", {}).get("metrics", {}).get(metric, {})
    bc = comps.get("B_vs_C", {}).get("metrics", {}).get(metric, {})
    if ab:
        out["phase_a_mean"] = ab.get("phase_a_mean")
        out["phase_b_mean"] = ab.get("phase_b_mean")
        out["n_a"] = ab.get("n_a")
        out["n_b"] = ab.get("n_b")
        out["tau_ab"] = ab.get("tau_u", {}).get("tau")
        out["tau_ab_p"] = ab.get("tau_u", {}).get("p_value")
    if bc:
        out["phase_c_mean"] = bc.get("phase_b_mean")
        out["n_c"] = bc.get("n_b")
        out["tau_bc"] = bc.get("tau_u", {}).get("tau")
        out["tau_bc_p"] = bc.get("tau_u", {}).get("p_value")
    fpr = _read_json("placebo_calibration_metrics.json").get("false_positive_rates", {}).get(metric, {}).get("mann_whitney", {}).get("fpr")
    if fpr is not None:
        out["mw_fpr"] = float(fpr)
    return out


def format_p(p: float | None) -> str:
    if p is None:
        return "p=n/a"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}" if p < 0.01 else f"p={p:.2f}"


def fork_chart_html(div_id: str = "cf-fork", height: int = 460) -> tuple[str, dict]:
    """Light clinical-premium plotly fork chart. Returns (html_div, computed numbers)."""
    import plotly.graph_objects as go
    import plotly.io as pio

    c = compute_counterfactual()
    dates = c["dates"]
    ivx = c["ivx"]
    marker_x = dates[ivx]

    fig = go.Figure()
    # Projected no-Jakavi 95% band (upper then lower with fill).
    fig.add_trace(go.Scatter(x=dates, y=c["cf_hi"], mode="lines",
                             line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=dates, y=c["cf_lo"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=BAND, hoverinfo="skip", showlegend=False))
    # Projected no-Jakavi center line (dashed grey).
    fig.add_trace(go.Scatter(x=dates, y=c["cf"], mode="lines", hoverinfo="skip", showlegend=False,
                             line=dict(color=COUNTER, width=1.6, dash="dash")))
    # Actual on-Jakavi line (indigo, solid).
    fig.add_trace(go.Scatter(x=dates, y=c["actual"], mode="lines", connectgaps=True,
                             name="HRV", hovertemplate="%{y:.0f} ms<extra></extra>",
                             showlegend=False, line=dict(color=ACCENT, width=2.4)))

    # Treatment marker (kept clear of the top edge; label sits just inside, left-anchored).
    fig.add_vline(x=marker_x, line=dict(color=INK, width=1.4, dash="dot"))
    fig.add_annotation(x=marker_x, y=1.0, yref="paper", showarrow=False,
                       text="Ruxolitinib start", font=dict(size=12.5, color=INK, family="Inter"),
                       xanchor="left", xshift=8, yshift=-6, align="left")
    if c["bb_date"]:
        fig.add_vline(x=c["bb_date"], line=dict(color=INK, width=1.4, dash="dot"))
        fig.add_annotation(x=c["bb_date"], y=1.0, yref="paper", showarrow=False,
                           text="Beta-blocker added", font=dict(size=12.5, color=INK, family="Inter"),
                           xanchor="left", xshift=8, yshift=-24, align="left")
    # The gap = the effect. Actual label lifted above the cloud; projected sits on the band line.
    fig.add_annotation(x=dates[-1], y=52, showarrow=False,
                       text=f"<b>{c['actual_recent']:.0f} ms</b> actual on treatment",
                       font=dict(size=12.5, color=ACCENT), xanchor="right", align="right")
    fig.add_annotation(x=dates[-1], y=c["cf_recent"], showarrow=False,
                       text=f"{c['cf_recent']:.0f} ms projected", font=dict(size=11.5, color=COUNTER),
                       xanchor="right", yshift=-12, align="right")

    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=48, r=88, t=26, b=34),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, system-ui, -apple-system, sans-serif", color=INK, size=13),
        showlegend=False,
        xaxis=dict(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID,
                   tickfont=dict(size=11, color="#8A909C")),
        yaxis=dict(title=dict(text="HRV (RMSSD, ms)", font=dict(size=12, color="#5B616E")),
                   showgrid=True, gridcolor=GRID, zeroline=False,
                   tickfont=dict(size=11, color="#8A909C")),
        hovermode="x",
    )
    html = pio.to_html(fig, include_plotlyjs=False, full_html=False,
                       div_id=div_id, config={"displayModeBar": False, "responsive": True})
    return html, c


if __name__ == "__main__":
    c = compute_counterfactual()
    print(json.dumps({k: v for k, v in c.items()
                      if k not in ("dates", "actual", "cf", "cf_lo", "cf_hi")}, indent=2))
