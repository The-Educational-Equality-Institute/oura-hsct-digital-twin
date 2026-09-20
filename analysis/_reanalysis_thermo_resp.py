"""Reanalysis: Thermoregulation and respiration domain (N=1 post-HSCT).

Treatment split day: 2026-03-16.
- pre-treatment baseline: day < 2026-03-16
- whole_post: day >= 2026-03-16
- recent 30 nights: day >= (max day - 30 days)

Metrics (REAL physiological values only, never 0-100 scores):
  1. Nocturnal skin temperature deviation (oura_readiness.temperature_deviation), degrees C.
     NOTE: oura_readiness.body_temperature is a 0-100 SCORE -> EXCLUDED.
  2. SpO2 average (oura_spo2.spo2_average), percent. Drops NULL and 0.0 sentinels.
  3. Respiratory / breath rate (oura_sleep_periods.average_breath, type='long_sleep'), breaths/min.

Durations in oura_sleep_periods are SECONDS (not used here; no duration metric in this domain).
"""

import sqlite3
from datetime import date, timedelta

import numpy as np
from scipy import stats

DB = "/home/henrik/projects/helse/oura-hsct-digital-twin/data/oura.db"
SPLIT = "2026-03-16"


def cohens_d(pre, post):
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    n1, n2 = len(pre), len(post)
    s1, s2 = pre.std(ddof=1), post.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return (post.mean() - pre.mean()) / pooled


def fetch(con, sql, params=()):
    rows = con.execute(sql, params).fetchall()
    return [(r[0], float(r[1])) for r in rows if r[1] is not None]


def analyze(name, unit, higher_is_better, rows, max_day):
    """rows = list of (day_str, value). Returns a dict of computed numbers."""
    days = {d: v for d, v in rows}
    recent_cutoff = (date.fromisoformat(max_day) - timedelta(days=30)).isoformat()

    pre = [v for d, v in days.items() if d < SPLIT]
    post = [v for d, v in days.items() if d >= SPLIT]
    recent = [v for d, v in days.items() if d >= recent_cutoff]

    pre_a, post_a, recent_a = map(lambda x: np.asarray(x, float), (pre, post, recent))

    pre_mean = pre_a.mean()
    post_mean = post_a.mean()
    recent_mean = recent_a.mean()

    recent_vs_pre_pct = (recent_mean - pre_mean) / pre_mean * 100
    tested_pct = (post_mean - pre_mean) / pre_mean * 100

    u, p = stats.mannwhitneyu(pre_a, post_a, alternative="two-sided")
    d = cohens_d(pre_a, post_a)

    print(f"\n===== {name} ({unit}) =====")
    print(f"  higher_is_better        : {higher_is_better}")
    print(f"  recent cutoff (>= )     : {recent_cutoff}  (max day {max_day})")
    print(f"  n_pre                   : {len(pre)}")
    print(f"  n_post                  : {len(post)}")
    print(f"  n_recent (30 nights)    : {len(recent)}")
    print(f"  pre_baseline_mean       : {pre_mean:.4f}")
    print(f"  whole_post_mean         : {post_mean:.4f}")
    print(f"  recent30_mean           : {recent_mean:.4f}")
    print(f"  tested_pct (post vs pre): {tested_pct:+.2f}%")
    print(f"  recent_vs_pre_pct       : {recent_vs_pre_pct:+.2f}%")
    print(f"  Mann-Whitney U          : U={u:.1f}  p={p:.6g}")
    print(f"  Cohen's d (pre->post)   : {d:+.4f}")

    return {
        "metric": name,
        "unit": unit,
        "higher_is_better": higher_is_better,
        "n_pre": len(pre),
        "n_recent": len(recent),
        "pre_baseline_mean": round(pre_mean, 4),
        "recent30_mean": round(recent_mean, 4),
        "recent_vs_pre_pct": round(recent_vs_pre_pct, 2),
        "whole_pre_mean": round(pre_mean, 4),
        "whole_post_mean": round(post_mean, 4),
        "tested_pct": round(tested_pct, 2),
        "mann_whitney_p": float(p),
        "cohens_d": round(float(d), 4),
    }


def main():
    con = sqlite3.connect(DB)

    # Determine a common max day across the domain (use the metric's own table).
    # Metric 1: skin temperature deviation (degrees C), oura_readiness.temperature_deviation
    temp_rows = fetch(
        con,
        "SELECT date, temperature_deviation FROM oura_readiness "
        "WHERE temperature_deviation IS NOT NULL",
    )
    temp_max = max(d for d, _ in temp_rows)

    # Metric 2: SpO2 average (percent), oura_spo2.spo2_average. Drop NULL and 0.0 sentinels.
    spo2_rows = fetch(
        con,
        "SELECT date, spo2_average FROM oura_spo2 "
        "WHERE spo2_average IS NOT NULL AND spo2_average > 0",
    )
    spo2_max = max(d for d, _ in spo2_rows)

    # Metric 3: respiratory rate (breaths/min), oura_sleep_periods.average_breath, long_sleep
    breath_rows = fetch(
        con,
        "SELECT day, average_breath FROM oura_sleep_periods "
        "WHERE type='long_sleep' AND average_breath IS NOT NULL",
    )
    breath_max = max(d for d, _ in breath_rows)

    print("Data coverage:")
    print(f"  temperature_deviation rows: {len(temp_rows)}  max day {temp_max}")
    print(f"  spo2 (>0) rows            : {len(spo2_rows)}  max day {spo2_max}")
    print(f"  breath long_sleep rows    : {len(breath_rows)}  max day {breath_max}")

    findings = []
    findings.append(
        analyze(
            "Nocturnal skin temperature deviation",
            "degrees C",
            False,  # temperature-deviation: lower/near-zero better
            temp_rows,
            temp_max,
        )
    )
    findings.append(
        analyze(
            "SpO2 average",
            "percent",
            True,  # SpO2: higher better
            spo2_rows,
            spo2_max,
        )
    )
    findings.append(
        analyze(
            "Respiratory rate",
            "breaths/min",
            False,  # respiratory rate: lower better
            breath_rows,
            breath_max,
        )
    )

    con.close()
    return findings


if __name__ == "__main__":
    main()
