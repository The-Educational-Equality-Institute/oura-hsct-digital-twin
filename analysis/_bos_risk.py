"""Shared BOS risk helpers for cross-report consistency."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


BOS_COMPONENT_LABELS: dict[str, str] = {
    "spo2_slope": "SpO2 Trend",
    "spo2_variability": "SpO2 Variability",
    "desaturation_freq": "Desaturation Frequency",
    "bdi": "Breathing Disturbance Index",
    "hr_decoupling": "HR-SpO2 Decoupling",
}


def _coerce_float(value: Any) -> float | None:
    """Best-effort numeric coercion for JSON payload fields."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_bos_level(level: Any) -> str:
    """Normalize BOS level labels to a stable uppercase vocabulary."""
    if level is None:
        return "N/A"
    normalized = str(level).strip().upper()
    if not normalized:
        return "N/A"
    return normalized


def bos_status(level: Any) -> str:
    """Map BOS risk level to theme status token used by KPI cards."""
    normalized = normalize_bos_level(level)
    if normalized == "HIGH":
        return "critical"
    if normalized in {"ELEVATED", "MODERATE"}:
        return "warning"
    if normalized == "LOW":
        return "normal"
    return "neutral"


def format_bos_label(bos_risk: dict[str, Any] | None) -> str:
    """Return a compact BOS label like 'LOW (17/100)'."""
    if not bos_risk:
        return "N/A"
    level = normalize_bos_level(bos_risk.get("risk_level"))
    score = _coerce_float(bos_risk.get("composite_score"))
    if score is None and level == "N/A":
        return "N/A"
    if score is None:
        return level
    return f"{level} ({score:.0f}/100)"


def load_bos_risk(reports_dir: Path) -> dict[str, Any]:
    """Load canonical BOS payload from spo2_bos_metrics.json."""
    payload_path = reports_dir / "spo2_bos_metrics.json"
    if not payload_path.exists():
        return {}
    try:
        with payload_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    bos = payload.get("bos_risk")
    if not isinstance(bos, dict):
        return {}
    if "risk_level" in bos:
        bos["risk_level"] = normalize_bos_level(bos.get("risk_level"))
    return bos
