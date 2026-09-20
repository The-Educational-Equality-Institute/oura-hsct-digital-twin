"""Reanalysis: Autonomic / cardiovascular domain, N=1 post-HSCT treatment response.

Metrics (real physiological VALUES only, from oura_sleep_periods, type='long_sleep'):
  - average_hrv         HRV RMSSD in ms                (higher is better)
  - average_heart_rate  average sleeping heart rate bpm (lower is better)
  - lowest_heart_rate   lowest nightly heart rate bpm   (lower is better)

EXCLUDED: oura_readiness.resting_heart_rate and hrv_balance are 0-100 CONTRIBUTOR
SCORES (range 1..100, capped at 100 on 24 nights, min values of 1/7/8), not real
bpm/ms values, so they are not usable as physiological measurements.

Treatment cutoff: 2026-03-16. Pre = day < cutoff, Post = day >= cutoff.
Recent = last 30 days ending at max(day) in data.
Durations in oura_sleep_periods are SECONDS (not used in this domain).
"""

import sqlite3
import numpy as np
from scipy import stats
from datetime import datetime, timedelta

DB = "/home/henrik/projects/helse/oura-hsct-digital-twin/data/oura.db"
CUTOFF = "2026-03-16"

METRICS = [
    # (label, column, unit, higher_is_better)
    ("HRV RMSSD (average_hrv)", "average_hrv", "ms", True),
    ("Average sleeping heart rate", "average_heart_rate", "bpm", False),
    ("Lowest nightly heart rate", "lowest_heart_rate", "bpm", False),
]


def fetch(con, column):
    """Return list of (day, value) for long_sleep nights, non-null, ordered by day."""
    q = (
        f"SELECT day, {column} FROM oura_sleep_periods "
        f"WHERE type='long_sleep' AND {column} IS NOT NULL "
        f"ORDER BY day"
    )
    return con.execute(q).fetchall()


def cohens_d(pre, post):
    """Pooled-SD Cohen's d (post minus pre)."""
    n1, n2 = len(pre), len(post)
    if n1 < 2 or n2 < 2:
        return float("nan")
    v1, v2 = np.var(pre, ddof=1), np.var(post, ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled == 0:
        return float("nan")
    return (np.mean(post) - np.mean(pre)) / pooled


def main():
    con = sqlite3.connect(DB)

    # global max day across long_sleep nights -> defines recent window
    max_day = con.execute(
        "SELECT MAX(day) FROM oura_sleep_periods WHERE type='long_sleep'"
    ).fetchone()[0]
    recent_start = (
        datetime.strptime(max_day, "%Y-%m-%d") - timedelta(days=30)
    ).strftime("%Y-%m-%d")

    print(f"max day in data (long_sleep): {max_day}")
    print(f"recent window: day >= {recent_start}  (last 30 days)")
    print(f"treatment cutoff: {CUTOFF}  (pre = day < cutoff, post = day >= cutoff)")
    print("=" * 72)

    results = []
    for label, col, unit, hib in METRICS:
        rows = fetch(con, col)
        days = np.array([r[0] for r in rows])
        vals = np.array([float(r[1]) for r in rows])

        pre = vals[days < CUTOFF]
        post = vals[days >= CUTOFF]
        recent = vals[days >= recent_start]

        pre_mean = float(np.mean(pre)) if len(pre) else float("nan")
        post_mean = float(np.mean(post)) if len(post) else float("nan")
        recent_mean = float(np.mean(recent)) if len(recent) else float("nan")

        recent_vs_pre_pct = (recent_mean - pre_mean) / pre_mean * 100.0
        tested_pct = (post_mean - pre_mean) / pre_mean * 100.0

        u, p = stats.mannwhitneyu(pre, post, alternative="two-sided")
        d = cohens_d(pre, post)

        print(f"\n### {label}  [{unit}]  higher_is_better={hib}")
        print(f"  n_pre={len(pre)}  n_post={len(post)}  n_recent={len(recent)}")
        print(f"  pre_baseline_mean  (day < {CUTOFF}) = {pre_mean:.3f} {unit}")
        print(f"  recent30_mean      (day >= {recent_start}) = {recent_mean:.3f} {unit}")
        print(f"  whole_pre_mean     = {pre_mean:.3f} {unit}")
        print(f"  whole_post_mean    = {post_mean:.3f} {unit}")
        print(f"  recent_vs_pre_pct  = {recent_vs_pre_pct:+.2f} %")
        print(f"  tested_pct (post vs pre) = {tested_pct:+.2f} %")
        print(f"  Mann-Whitney U     = {u:.1f}   p = {p:.6f}")
        print(f"  Cohen's d (post-pre) = {d:+.3f}")

        results.append(
            dict(
                metric=label, unit=unit, higher_is_better=hib,
                n_pre=len(pre), n_recent=len(recent),
                pre_baseline_mean=pre_mean, recent30_mean=recent_mean,
                whole_pre_mean=pre_mean, whole_post_mean=post_mean,
                recent_vs_pre_pct=recent_vs_pre_pct, tested_pct=tested_pct,
                mann_whitney_p=float(p), cohens_d=float(d),
            )
        )

    print("\n" + "=" * 72)
    print("EXCLUDED (0-100 contributor scores, not physiological values):")
    print("  oura_readiness.resting_heart_rate  (range 1..100, capped at 100)")
    print("  oura_readiness.hrv_balance         (range 7..100, capped at 100)")
    con.close()
    return results


if __name__ == "__main__":
    main()
