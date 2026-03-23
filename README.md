# Oura Ring Post-HSCT Digital Twin

A 13-module biometric analysis platform built on Oura Ring Gen 4 data for post-transplant monitoring.

## What this is

I got a blood cancer diagnosis at 33 and had a stem cell transplant in November 2023. Survived. Then spent two years getting worse while 60+ doctors told me I was fine.

In January 2026 I put on an Oura Ring. Within two days I could see they were all wrong. HRV in the single digits. Heart rate never dropping below 80 in my sleep. No recovery, ever.

I built this platform to prove it. The data led to advanced imaging that found pathology in ten organ systems. Nobody had looked.

## What it does

| Module | Method | What it found |
|--------|--------|--------------|
| Digital Twin | Linear + Extended Kalman Filter | Drug response at p=0.02 within 3 days |
| Foundation Models | Amazon Chronos-2 (200M params, zero-shot) | Flagged acute event 12 hours before ER visit |
| GVHD Prediction | 4-state Gaussian HMM | Predicted disease flare 31 days in advance |
| Causal Inference | CausalImpact + PCMCI+ + Transfer Entropy | 98.2% posterior probability of treatment effect |
| Anomaly Detection | Matrix Profile + Isolation Forest + LSTM + SPC | Ensemble alerts across 5 independent methods |
| Advanced HRV | Lomb-Scargle, DFA, RQA, multiscale entropy | Parasympathetic deficiency below ESC threshold |
| SpO2 Screening | Trend analysis + desaturation index | Early bronchiolitis obliterans detection |
| Composite Biomarkers | 6 novel indices (ADSI, GVHD Score, CV Risk) | Integrated multi-system severity tracking |
| Sleep Architecture | Markov chains + ultradian FFT | Fragmentation and staging analysis |
| Ruxolitinib Response | PELT/BOCPD change-point detection | Treatment response quantification |

Plus: comprehensive dashboard, HRV clinical mapping, biometrics evidence report.

## The numbers

~99,000 readings across 74 days. ~25,000 lines of Python. 28 libraries. One ring.

RMSSD averaged 9.96 ms (87.9% of nights below the 15 ms parasympathetic deficiency threshold). Sleep heart rate averaged 85.2 bpm. REM sleep 14.8%.

Four international transplant centers — Dana-Farber, Mayo Clinic, UZ Leuven, University of Regensburg — reviewed the case. Treatment started.

## Status

Code cleanup and PHI removal in progress. Modules will be added as they are ready.

## License

MIT
