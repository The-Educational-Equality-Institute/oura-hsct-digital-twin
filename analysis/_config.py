"""Shared path configuration for all Oura analysis scripts.

Re-exports from the top-level config.py for backwards compatibility.
New scripts should import from config directly.
"""
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent  # oura-digital-twin/

sys.path.insert(0, str(REPO_ROOT))
from config import DATABASE_PATH, REPORTS_DIR, validate_config  # noqa: E402

# Backwards-compatible aliases
BIOMETRICS_DB = DATABASE_PATH


def _resolve_symlink(p: Path) -> Path | None:
    """Return the resolved target if *p* is a valid, non-dangling symlink or
    a regular file.  Returns None when the path is missing or dangling."""
    if p.is_symlink():
        target = p.resolve()
        if target.exists() and os.access(str(target), os.R_OK):
            return target
        # Dangling symlink
        print(
            f"WARNING: Symlink exists but target is missing: "
            f"{p} -> {target}"
        )
        return None
    if p.exists():
        return p.resolve()
    return None


# Verify DATABASE_PATH symlink target actually exists
_db_resolved = _resolve_symlink(DATABASE_PATH)
if _db_resolved is None and DATABASE_PATH.is_symlink():
    print(
        f"WARNING: DATABASE_PATH symlink is dangling: {DATABASE_PATH} "
        f"-> {DATABASE_PATH.resolve()}"
    )

# Investigation DB (used by analyze_oura_full.py)
# If not found, set to None instead of crashing
INVESTIGATION_DB: Path | None = None
_inv_candidate = REPO_ROOT / "data" / "investigation.db"
_inv_resolved = _resolve_symlink(_inv_candidate)
if _inv_resolved is not None:
    INVESTIGATION_DB = _inv_candidate
else:
    _parent_inv = REPO_ROOT.parent / "database" / "investigation.db"
    _parent_resolved = _resolve_symlink(_parent_inv)
    if _parent_resolved is not None:
        INVESTIGATION_DB = _parent_inv

if INVESTIGATION_DB is None:
    print(
        "INFO: Investigation DB not found — timeline features will be "
        "disabled. Looked in data/investigation.db and "
        "../database/investigation.db"
    )
