# CGM Trial Pre-Registered Hypotheses

**Patient:** Henrik (post-HSCT, MDS-AML, 2023-11-23 transplant)
**Sensor:** FreeStyle Libre 3 Plus, 1 sensor / 15-day trial
**Order date:** 2026-04-16
**Estimated activation:** 2026-04-21
**Estimated trial end:** ~2026-05-06
**Mayo consultation:** 2026-05-19
**Committed before data collection:** 2026-04-16

## Purpose of pre-registration

This document fixes the analytical hypotheses, thresholds, time windows, and
falsification criteria **before** any glucose data is collected. Pre-registration
prevents post-hoc selection of favorable statistics from a 21,600-point dataset.
Any deviation from this plan during or after analysis must be documented as
exploratory, not confirmatory.

## Clinical background

Henrik has:
- Documented autonomic dysfunction (reduced HRV, elevated resting HR, circadian HR blunting)
- Chronic GvHD on ruxolitinib 10mg BID (from 2026-03-16) + bisoprolol 2.5mg daily (from 2026-04-08)
- HEV diagnosed 2026-03-18 (active confound)
- Unexplained postprandial chest pain episodes (origin unclear: cardiac vs autonomic vs metabolic)
- No prior PTDM (post-transplant diabetes mellitus) screening under Norwegian standard protocol

## Primary hypothesis

**H1.** During the 15-day trial, Henrik will exhibit at least one episode of
postprandial hyperglycemia (interstitial glucose ≥ 10.0 mmol/L within 120 min
after a recorded meal timestamp).

- **Test:** Count of qualifying post-meal spikes, stratified by meal type if logged.
- **Null:** No episode ≥ 10.0 mmol/L across trial.
- **Clinical significance if true:** Supports referral for HbA1c + OGTT on Mayo visit
  to formally evaluate PTDM. PTDM prevalence after allogeneic HSCT is reported
  15–50% depending on cohort and screening intensity.

## Secondary hypotheses

### H2 — Glycemic-autonomic coupling

**Claim:** Rapid glucose excursions (Δglucose ≥ 2.0 mmol/L within 30 min) are
followed by a transient HRV depression (RMSSD decrease ≥ 5 ms relative to the
preceding 60-min baseline) within the subsequent 60 min.

- **Sample:** All glucose excursions meeting the Δ threshold during nighttime
  windows (where Oura's 5-min RMSSD is available).
- **Statistic:** Paired comparison of pre-excursion vs post-excursion RMSSD (Wilcoxon
  signed-rank, two-sided).
- **Minimum sample for testing:** n ≥ 10 excursion events.
- **Falsification:** p > 0.20 OR median ΔRMSSD ≥ 0 ms.
- **Known confound:** Ruxolitinib and bisoprolol both influence HRV independently.
  Finding is hypothesis-generating regardless of direction.

### H3 — Postprandial chest pain coincidence

**Claim:** Logged chest-pain events (`symptom_events.symptom_type = 'chest_pain'`)
occur more frequently within 0–120 min after glucose readings ≥ 9.0 mmol/L than
during baseline (non-postprandial) windows.

- **Sample:** All chest_pain symptom events in the trial.
- **Statistic:** Binomial test comparing proportion of events in post-hyperglycemic
  windows vs the overall proportion of time spent in those windows.
- **Minimum sample:** n ≥ 3 chest_pain events.
- **Falsification:** Fewer than 2 chest_pain events occur in a post-hyperglycemic
  window, OR proportion does not exceed baseline rate.
- **Explicit limitation:** n=1 case, low event count, cannot establish causation.
  A positive result motivates structured provocation testing under medical supervision,
  not clinical decision-making.

### H4 — Glycemic variability as autonomic-dysfunction marker

**Claim:** Days with higher glycemic variability (coefficient of variation > 36%,
the ADA clinical threshold for unstable glucose) show lower nightly RMSSD
(Spearman ρ < 0 across trial days).

- **Sample:** Daily aggregates (CV of glucose per 24h vs following night's mean RMSSD).
- **Statistic:** Spearman rank correlation.
- **Minimum sample:** 10 trial days with complete glucose + HRV coverage.
- **Falsification:** ρ ≥ 0 OR p > 0.20.

## Exclusion criteria

Observations excluded before analysis:
1. First 12 hours after sensor activation (warm-up + potential tissue calibration drift).
2. Glucose readings with `record_type` outside {0, 1} (strip tests, insulin, food, notes).
3. Glucose readings below 2.2 mmol/L or above 27 mmol/L (physiologically implausible
   for a non-insulin-dependent patient, likely sensor artifact).
4. Symptom events without a matched glucose reading within ±15 min (insufficient
   temporal alignment for coupling analysis).

## Known confounds that cannot be separated with n=1

- Ruxolitinib (JAK1/2 inhibitor) has documented metabolic effects.
- Bisoprolol (β1-blocker) blunts HR variability and could mask H2/H4 signals.
- HEV infection (active) alters immune and metabolic state.
- Diet composition is uncontrolled (patient eats normally, no provocation meals).
- Sleep, stress, exercise all confound HRV independently.

Any "significant" finding must be reported with these confounds listed. The
realistic publication path is a multimodal-monitoring methods paper, not a
mechanistic finding paper, until a larger cohort is available.

## What would make me extend the trial

- Multiple H1-qualifying spikes with H3-qualifying symptom coincidence.
- H2 direction-correct with p < 0.10 (not significance — a signal worth pursuing).
- Any clinically alarming glucose pattern (frequent >13 mmol/L, nocturnal <3.5 mmol/L).

In these cases, order 4–6 additional sensors for continuous 12-month monitoring
and add LibreLink-Up API integration.

## What would stop me from pursuing this further

- No spike ≥ 10.0 mmol/L across 15 days.
- No chest_pain events logged (baseline symptom not captured).
- H2 falsified with adequate sample (n ≥ 10 excursions).

## Amendments

Any amendments to this document after 2026-04-21 (sensor activation) must be
dated, justified, and kept visible. Original version preserved in git history.
