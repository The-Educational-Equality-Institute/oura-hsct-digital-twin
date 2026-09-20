"""Rigorous treatment-response reanalysis for the SLEEP ARCHITECTURE domain (N=1, post-HSCT).

Data source: data/oura.db, table oura_sleep_periods, filtered to type='long_sleep'.
Durations in that table are SECONDS. We convert to hours (sleep stage durations) or
minutes (latency, awake_time). efficiency is a real physiological percent (0-100), NOT
an Oura 0-100 SCORE, so it is a valid VALUE. latency is seconds -> minutes.

Treatment split date: 2026-03-16.
  pre  = day < '2026-03-16'
  post = day >= '2026-03-16'
Recent-30 window = last 30 calendar days ending at max(day) in the data.

For each metric:
  pre_baseline_mean (whole pre), recent30 mean, recent_vs_pre_pct,
  whole_pre mean, whole_post mean, tested_pct,
  Mann-Whitney U p (whole pre vs whole post), Cohen's d (pooled SD),
  n_pre, n_recent.
"""

import sqlite3
from datetime import date, timedelta

import numpy as np
from scipy import stats

DB = "data/oura.db"
SPLIT = "2026-03-16"

con = sqlite3.connect(DB)
cur = con.cursor()

# Max day in the long_sleep data defines the recent-30 window end.
cur.execute("SELECT MAX(day) FROM oura_sleep_periods WHERE type='long_sleep'")
max_day_str = cur.fetchone()[0]
max_day = date.fromisoformat(max_day_str)
recent_start = (max_day - timedelta(days=30)).isoformat()
print(f"Max day in long_sleep data: {max_day_str}")
print(f"Recent-30 window: day >= {recent_start}")
print(f"Treatment split: day < {SPLIT} (pre) vs day >= {SPLIT} (post)")
print("=" * 78)


def fetch(col, transform):
    """Return dict of pre, post, recent numpy arrays for a column, NULLs dropped.

    transform converts raw stored units into the reporting unit.
    """
    cur.execute(
        f"SELECT day, {col} FROM oura_sleep_periods "
        f"WHERE type='long_sleep' AND {col} IS NOT NULL"
    )
    rows = cur.fetchall()
    pre, post, recent = [], [], []
    for day, val in rows:
        v = transform(val)
        if day < SPLIT:
            pre.append(v)
        else:
            post.append(v)
        if day >= recent_start:
            recent.append(v)
    return np.array(pre, float), np.array(post, float), np.array(recent, float)


def cohens_d(a, b):
    """Cohen's d with pooled SD. Sign: positive means post > pre (b > a)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return float("nan")
    return (b.mean() - a.mean()) / pooled


def analyze(label, unit, pre, post, recent, higher_is_better):
    pre_mean = pre.mean()
    post_mean = post.mean()
    recent_mean = recent.mean() if len(recent) else float("nan")
    recent_vs_pre = (recent_mean - pre_mean) / pre_mean * 100 if pre_mean else float("nan")
    tested = (post_mean - pre_mean) / pre_mean * 100 if pre_mean else float("nan")
    if len(pre) and len(post):
        u, p = stats.mannwhitneyu(pre, post, alternative="two-sided")
    else:
        p = float("nan")
    d = cohens_d(pre, post)
    print(f"\n### {label}  [{unit}]  higher_is_better={higher_is_better}")
    print(f"  n_pre={len(pre)}  n_post={len(post)}  n_recent={len(recent)}")
    print(f"  pre_baseline_mean (whole pre) = {pre_mean:.4f}")
    print(f"  recent30_mean                 = {recent_mean:.4f}")
    print(f"  whole_post_mean               = {post_mean:.4f}")
    print(f"  recent_vs_pre_pct = {recent_vs_pre:+.2f}%")
    print(f"  tested_pct (post vs pre) = {tested:+.2f}%")
    print(f"  Mann-Whitney p = {p:.6g}")
    print(f"  Cohen's d (pre->post) = {d:+.4f}")
    return {
        "metric": label,
        "unit": unit,
        "n_pre": len(pre),
        "n_recent": len(recent),
        "pre_baseline_mean": pre_mean,
        "recent30_mean": recent_mean,
        "recent_vs_pre_pct": recent_vs_pre,
        "whole_pre_mean": pre_mean,
        "whole_post_mean": post_mean,
        "tested_pct": tested,
        "mann_whitney_p": p,
        "cohens_d": d,
        "higher_is_better": higher_is_better,
    }


results = []

# --- Duration metrics: seconds -> hours ---
sec_to_hr = lambda s: s / 3600.0
sec_to_min = lambda s: s / 60.0

pre, post, rec = fetch("total_sleep_duration", sec_to_hr)
results.append(analyze("Total sleep duration", "hours", pre, post, rec, True))

pre, post, rec = fetch("deep_sleep_duration", sec_to_hr)
results.append(analyze("Deep sleep duration", "hours", pre, post, rec, True))

pre, post, rec = fetch("rem_sleep_duration", sec_to_hr)
results.append(analyze("REM sleep duration", "hours", pre, post, rec, True))

pre, post, rec = fetch("light_sleep_duration", sec_to_hr)
results.append(analyze("Light sleep duration", "hours", pre, post, rec, True))

# --- efficiency: real percent VALUE (0-100), higher better ---
pre, post, rec = fetch("efficiency", lambda v: float(v))
results.append(analyze("Sleep efficiency", "percent", pre, post, rec, True))

# --- latency: seconds -> minutes, lower better ---
pre, post, rec = fetch("latency", sec_to_min)
results.append(analyze("Sleep latency", "minutes", pre, post, rec, False))

# --- awake_time: seconds -> minutes, lower better ---
pre, post, rec = fetch("awake_time", sec_to_min)
results.append(analyze("Awake time in bed", "minutes", pre, post, rec, False))

# --- REM as percent of total sleep: computed per-night, higher better ---
cur.execute(
    "SELECT day, rem_sleep_duration, total_sleep_duration FROM oura_sleep_periods "
    "WHERE type='long_sleep' AND rem_sleep_duration IS NOT NULL "
    "AND total_sleep_duration IS NOT NULL AND total_sleep_duration > 0"
)
pre_p, post_p, rec_p = [], [], []
for day, rem, tot in cur.fetchall():
    pct = rem / tot * 100.0
    if day < SPLIT:
        pre_p.append(pct)
    else:
        post_p.append(pct)
    if day >= recent_start:
        rec_p.append(pct)
results.append(
    analyze(
        "REM percent of total sleep",
        "percent",
        np.array(pre_p, float),
        np.array(post_p, float),
        np.array(rec_p, float),
        True,
    )
)

con.close()

print("\n" + "=" * 78)
print("SUMMARY TABLE")
print("=" * 78)
for r in results:
    print(
        f"{r['metric']:<28} pre={r['pre_baseline_mean']:8.3f} "
        f"post={r['whole_post_mean']:8.3f} recent={r['recent30_mean']:8.3f} "
        f"rec_vs_pre={r['recent_vs_pre_pct']:+7.2f}% p={r['mann_whitney_p']:.4g} "
        f"d={r['cohens_d']:+.3f}"
    )
