"""Treatment-response re-analysis: Readiness and activity domain (N=1, post-HSCT).

Domain metrics:
  - resting_heart_rate  (oura_readiness, real bpm VALUE) -> higher_is_better=False
  - steps               (oura_activity, real count VALUE) -> higher_is_better=True
  - active_calories     (oura_activity, real kcal VALUE)  -> higher_is_better=True

EXCLUDED as a 0-100 SCORE (not a physiological value):
  - hrv_balance (oura_readiness) is a 0-100 balance INDEX, not HRV in ms.
    Same-day it reads ~10-16 while the real sleep HRV is ~7-8 ms. The real
    HRV-in-ms value lives in oura_sleep_periods.average_hrv (sleep domain),
    so hrv_balance is reported here only as an excluded score.

Split date: treatment/observation boundary = 2026-03-16.
Pre  = day < 2026-03-16 ; Post = day >= 2026-03-16.
Recent = last 30 nights (day >= max_day - 30 days).

Data-quality filter: resting_heart_rate has corrupt Oura rows (values 1, 7, 8,
14, 16, 19, 26, 30 bpm) that are physiologically impossible for a resting HR.
These are dropped with a plausibility floor of >= 35 bpm (treated like NULLs).
No such issue for steps/active_calories.
"""

import sqlite3
from datetime import date, timedelta

import numpy as np
from scipy import stats

DB = "/home/henrik/projects/helse/oura-hsct-digital-twin/data/oura.db"
SPLIT = "2026-03-16"
RHR_FLOOR = 35  # bpm; drop corrupt sub-physiological readiness rows


def fetch(con, table, col, datecol, extra=""):
    q = (
        f"SELECT {datecol} AS d, {col} AS v FROM {table} "
        f"WHERE {col} IS NOT NULL {extra} ORDER BY d"
    )
    rows = con.execute(q).fetchall()
    return [(r[0], float(r[1])) for r in rows]


def cohens_d(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return float("nan")
    # post minus pre, so positive d = post higher than pre
    return (b.mean() - a.mean()) / sp


def analyze(name, unit, rows, higher_is_better):
    days = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    max_day = date.fromisoformat(max(days))
    recent_cut = (max_day - timedelta(days=30)).isoformat()

    pre = [v for d, v in rows if d < SPLIT]
    post = [v for d, v in rows if d >= SPLIT]
    recent = [v for d, v in rows if d >= recent_cut]

    pre_mean = float(np.mean(pre))
    post_mean = float(np.mean(post))
    recent_mean = float(np.mean(recent))

    recent_vs_pre_pct = (recent_mean - pre_mean) / pre_mean * 100
    tested_pct = (post_mean - pre_mean) / pre_mean * 100

    u, p = stats.mannwhitneyu(pre, post, alternative="two-sided")
    d = cohens_d(pre, post)

    print(f"\n===== {name} ({unit}) =====")
    print(f"  higher_is_better        : {higher_is_better}")
    print(f"  n_pre (day<{SPLIT})   : {len(pre)}")
    print(f"  n_post (day>={SPLIT}) : {len(post)}")
    print(f"  n_recent (last 30d>={recent_cut}) : {len(recent)}")
    print(f"  pre_baseline_mean       : {pre_mean:.3f}")
    print(f"  whole_post_mean         : {post_mean:.3f}")
    print(f"  recent30_mean           : {recent_mean:.3f}")
    print(f"  recent_vs_pre_pct       : {recent_vs_pre_pct:+.2f} %")
    print(f"  tested_pct (post vs pre): {tested_pct:+.2f} %")
    print(f"  Mann-Whitney U p-value  : {p:.6g}")
    print(f"  Cohen's d (post-pre)    : {d:+.4f}")

    return {
        "metric": name,
        "unit": unit,
        "n_pre": len(pre),
        "n_recent": len(recent),
        "pre_baseline_mean": round(pre_mean, 4),
        "whole_pre_mean": round(pre_mean, 4),
        "whole_post_mean": round(post_mean, 4),
        "recent30_mean": round(recent_mean, 4),
        "recent_vs_pre_pct": round(recent_vs_pre_pct, 2),
        "tested_pct": round(tested_pct, 2),
        "mann_whitney_p": float(p),
        "cohens_d": round(float(d), 4),
        "higher_is_better": higher_is_better,
    }


def main():
    con = sqlite3.connect(DB)
    results = []

    rhr = fetch(
        con, "oura_readiness", "resting_heart_rate", "date",
        extra=f"AND resting_heart_rate >= {RHR_FLOOR}",
    )
    print(f"[note] resting_heart_rate: dropped rows < {RHR_FLOOR} bpm as corrupt.")
    results.append(analyze("resting_heart_rate", "bpm", rhr, higher_is_better=False))

    steps = fetch(con, "oura_activity", "steps", "date")
    results.append(analyze("steps", "count", steps, higher_is_better=True))

    acal = fetch(con, "oura_activity", "active_calories", "date")
    results.append(analyze("active_calories", "kcal", acal, higher_is_better=True))

    # Reported-but-excluded score, for transparency only (not a physiological value).
    hb = fetch(con, "oura_readiness", "hrv_balance", "date")
    pre = [v for d, v in hb if d < SPLIT]
    post = [v for d, v in hb if d >= SPLIT]
    print("\n===== hrv_balance (0-100 SCORE, EXCLUDED) =====")
    print("  This is a 0-100 balance INDEX, not HRV in ms. Excluded from findings.")
    print(f"  pre_mean={np.mean(pre):.2f} post_mean={np.mean(post):.2f} (index units)")
    print("  Real HRV-in-ms is oura_sleep_periods.average_hrv (sleep domain).")

    con.close()
    return results


if __name__ == "__main__":
    main()
