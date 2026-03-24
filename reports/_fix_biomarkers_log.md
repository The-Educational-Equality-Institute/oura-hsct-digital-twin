# Fix M5 - RMSSD Norm Inconsistency in analyze_oura_biomarkers.py

**Date:** 2026-03-24
**Status:** In progress

## Findings

1. **Line 74-75**: `NORM_RMSSD_MEAN = 42.0` and `NORM_RMSSD_SD = 15.0` hardcoded locally
   - Comment says "Nunan 2010, Shaffer & Ginsberg 2017" (acceptable)
2. **Line 101**: Allostatic threshold for RMSSD references `ESC_RMSSD_DEFICIENCY` from config (good),
   but comment cites "Kleiger 1987 / Bigger 1992" - needs update to "ESC/NASPE Task Force 1996; Shaffer & Ginsberg 2017"
3. **Config currently exports**: `ESC_RMSSD_DEFICIENCY` but NOT `POPULATION_RMSSD_MEAN` / `POPULATION_RMSSD_SD`
   - Config agent adding those now with values 42.0 / 15.0

## Changes Made

- [ ] Add POPULATION_RMSSD_MEAN, POPULATION_RMSSD_SD to config import
- [ ] Replace local hardcoded NORM_RMSSD_MEAN/SD with aliases from config
- [ ] Update Kleiger/Bigger citation to ESC/NASPE + Shaffer & Ginsberg
