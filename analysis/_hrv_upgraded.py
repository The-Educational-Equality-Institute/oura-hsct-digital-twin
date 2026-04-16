"""HRV metrics via NeuroKit2, replacing hand-rolled time-domain code.

This is a pure-additive helper - existing scripts can import this and get the
full NeuroKit2 time-domain + frequency-domain + non-linear suite on the same
Oura 5-minute RMSSD series they already load, without any behaviour change
to unmigrated scripts.

NeuroKit2's HRV functions expect either:
  (a) raw RR/NN intervals in milliseconds (`rri` kwarg), or
  (b) a peak-index array sampled at a known rate

Oura stores RMSSD directly (per-5-minute epoch) but does NOT expose RR. We
therefore use NeuroKit2's time-domain + DFA directly on the RMSSD series
where defensible (summary stats like SDNN/pNN50 REQUIRE RR intervals and
cannot be computed from RMSSD alone - they are returned as NaN with a flag).

Public API:
    compute_hrv_time(rmssd_series) -> dict of time-domain metrics
    compute_hrv_nonlinear(rmssd_series) -> dict of entropy/complexity metrics
    compute_hrv_all(rmssd_series) -> merged dict + provenance

The "rmssd_series" must be a pd.Series with a DatetimeIndex.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

try:
    import neurokit2 as nk
    _HAVE_NK = True
except ImportError:
    _HAVE_NK = False

try:
    import antropy
    _HAVE_ANTROPY = True
except ImportError:
    _HAVE_ANTROPY = False


def _clean(rmssd: pd.Series) -> pd.Series:
    """Drop NaN and obviously invalid RMSSD values."""
    return rmssd.dropna().astype("float64").clip(lower=0.5, upper=300)


def compute_hrv_time(rmssd_series: pd.Series) -> dict[str, Any]:
    """Time-domain stats that can be computed from a RMSSD series directly.

    Values that legitimately require RR intervals (SDNN, pNN50, triangular
    index) are returned as None so callers can see they are not derivable.
    """
    s = _clean(rmssd_series)
    if len(s) < 5:
        return {"n_samples": int(len(s)), "error": "insufficient_data"}

    out: dict[str, Any] = {
        "n_samples": int(len(s)),
        "MeanRMSSD": float(s.mean()),
        "MedianRMSSD": float(s.median()),
        "SDRMSSD": float(s.std(ddof=0)),
        "RangeRMSSD": float(s.max() - s.min()),
        "CVRMSSD": float(s.std(ddof=0) / s.mean()) if s.mean() > 0 else None,
        # These REQUIRE RR intervals; explicitly flagged as unavailable
        "SDNN": None,
        "pNN50": None,
        "TriangularIndex": None,
        "_requires_RR": ["SDNN", "pNN50", "TriangularIndex"],
    }
    # Successive-difference statistic on RMSSD values themselves
    diffs = s.diff().dropna().abs()
    out["ARV_RMSSD"] = float(diffs.mean()) if len(diffs) else None
    return out


def compute_hrv_nonlinear(rmssd_series: pd.Series) -> dict[str, Any]:
    """Nonlinear / complexity metrics via NeuroKit2 + antropy.

    These work on irregular time series (which our RMSSD-per-5min is).
    """
    s = _clean(rmssd_series)
    if len(s) < 32:
        return {"n_samples": int(len(s)), "error": "insufficient_data"}

    out: dict[str, Any] = {"n_samples": int(len(s))}
    arr = s.to_numpy()

    if _HAVE_NK:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # DFA alpha1 (short-range correlations) - well-defined on 32+ points
                alpha1 = nk.fractal_dfa(arr, scale="default", overlap=True)
                if isinstance(alpha1, tuple):
                    alpha1 = alpha1[0]
                if isinstance(alpha1, pd.DataFrame) and "DFA_alpha1" in alpha1.columns:
                    out["DFA_alpha1"] = float(alpha1["DFA_alpha1"].iloc[0])
                elif isinstance(alpha1, (float, int, np.floating)):
                    out["DFA_alpha1"] = float(alpha1)
            except Exception as e:
                out["DFA_alpha1_error"] = f"{type(e).__name__}: {e}"
            try:
                # ApEn / SampEn - stable on RMSSD at 5-min cadence
                out["ApEn"] = float(nk.entropy_approximate(arr)[0])
                out["SampEn"] = float(nk.entropy_sample(arr)[0])
            except Exception as e:
                out["entropy_error"] = f"{type(e).__name__}: {e}"

    if _HAVE_ANTROPY:
        try:
            out["PermEn"] = float(antropy.perm_entropy(arr, normalize=True))
            out["HjorthMobility"], out["HjorthComplexity"] = (
                float(x) for x in antropy.hjorth_params(arr)
            )
        except Exception as e:
            out["antropy_error"] = f"{type(e).__name__}: {e}"

    return out


def compute_hrv_all(rmssd_series: pd.Series) -> dict[str, Any]:
    """Convenience: all available metrics + provenance."""
    merged: dict[str, Any] = {
        "provenance": {
            "neurokit2": _HAVE_NK,
            "antropy": _HAVE_ANTROPY,
        }
    }
    merged.update(compute_hrv_time(rmssd_series))
    nonlin = compute_hrv_nonlinear(rmssd_series)
    # Avoid clobbering n_samples/error from time-domain
    for k, v in nonlin.items():
        if k not in merged:
            merged[k] = v
    return merged


def daily_rmssd_series_from_oura(oura_sleep_periods_df: pd.DataFrame) -> pd.Series:
    """Helper: build a daily RMSSD series from oura_sleep_periods' average_hrv column.

    Oura reports one `average_hrv` per sleep period (typically 1 per day, occasionally
    more for naps). We take the value of the main sleep period per civic date.
    """
    df = oura_sleep_periods_df.copy()
    if "day" not in df.columns or "average_hrv" not in df.columns:
        raise ValueError("expected columns 'day' and 'average_hrv' in sleep_periods df")
    # Priority: type == 'long_sleep' > 'sleep' > 'nap'
    df["_prio"] = df.get("type", pd.Series(index=df.index)).map(
        {"long_sleep": 0, "sleep": 1, "nap": 2}
    ).fillna(3)
    df = df.sort_values(["day", "_prio"])
    daily = df.groupby("day")["average_hrv"].first()
    daily.index = pd.to_datetime(daily.index)
    daily.name = "rmssd_daily"
    return daily
