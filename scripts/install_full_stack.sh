#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# install_full_stack.sh - Full dependency installer for oura-digital-twin
#
# Handles the Cython/ssm build issue: ssm==0.0.1 requires Cython at build
# time, but pip's build isolation means Cython isn't available during the
# build. This script pre-installs build dependencies and then installs ssm
# with --no-build-isolation.
#
# Portable: Linux / macOS / WSL
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- Constants ------------------------------------------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQUIREMENTS="$ROOT_DIR/requirements.txt"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_SSM=1
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=10

# ---- Colors (disabled when not a terminal) --------------------------------

if [[ -t 1 ]]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    GREEN='' RED='' YELLOW='' BLUE='' BOLD='' NC=''
fi

# ---- Helpers --------------------------------------------------------------

info()  { printf "${BLUE}[INFO]${NC}  %s\n" "$*"; }
ok()    { printf "${GREEN}[OK]${NC}    %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
fail()  { printf "${RED}[FAIL]${NC}  %s\n" "$*" >&2; exit 1; }

# ---- Usage ----------------------------------------------------------------

if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    cat <<EOF
${BOLD}Usage:${NC} bash scripts/install_full_stack.sh [--no-ssm]

Installs all dependencies for oura-digital-twin, including the optional
ssm rSLDS backend that requires special build handling.

${BOLD}Steps:${NC}
  1. Check Python version (${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+)
  2. Pre-install build dependencies (cython, numpy, setuptools, wheel)
  3. Install ssm with --no-build-isolation
  4. Install remaining requirements from requirements.txt
  5. Verify installation by importing key packages

${BOLD}Flags:${NC}
  --no-ssm   Skip ssm installation (use hmmlearn HMM fallback instead)
  -h, --help Show this help

${BOLD}Environment:${NC}
  PYTHON_BIN   Python executable to use (default: python3)

${BOLD}Examples:${NC}
  bash scripts/install_full_stack.sh            # Full install
  bash scripts/install_full_stack.sh --no-ssm   # Skip ssm
  PYTHON_BIN=python3.12 bash scripts/install_full_stack.sh
EOF
    exit 0
fi

# ---- Parse flags ----------------------------------------------------------

if [[ "${1:-}" == "--no-ssm" ]]; then
    INSTALL_SSM=0
    info "Skipping ssm installation (--no-ssm flag)"
fi

# ---- Step 1: Check Python version ----------------------------------------

info "Step 1/5: Checking Python version..."

if ! command -v "$PYTHON_BIN" &>/dev/null; then
    fail "Python executable not found: $PYTHON_BIN
    Set PYTHON_BIN to point to your Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ binary, e.g.:
      PYTHON_BIN=/usr/bin/python3.12 bash scripts/install_full_stack.sh"
fi

PYTHON_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.minor)")

if [[ "$PYTHON_MAJOR" -lt "$MIN_PYTHON_MAJOR" ]] || \
   { [[ "$PYTHON_MAJOR" -eq "$MIN_PYTHON_MAJOR" ]] && [[ "$PYTHON_MINOR" -lt "$MIN_PYTHON_MINOR" ]]; }; then
    fail "Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ required, found ${PYTHON_VERSION}.
    Install a newer Python or set PYTHON_BIN to point to it."
fi

ok "Python ${PYTHON_VERSION} ($("$PYTHON_BIN" --version 2>&1))"

# ---- Step 2: Pre-install build dependencies -------------------------------

info "Step 2/5: Installing build dependencies..."

if ! "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel cython numpy; then
    fail "Failed to install build dependencies (pip, setuptools, wheel, cython, numpy).
    Check your network connection and pip configuration."
fi

ok "Build dependencies installed"

# ---- Step 3: Install ssm with --no-build-isolation ------------------------

if [[ "$INSTALL_SSM" -eq 1 ]]; then
    info "Step 3/5: Installing ssm==0.0.1 with --no-build-isolation..."
    info "  (This compiles C extensions via Cython - requires a C compiler)"

    # Check for a C compiler before attempting the build
    if ! command -v gcc &>/dev/null && ! command -v cc &>/dev/null && ! command -v clang &>/dev/null; then
        warn "No C compiler found (gcc/cc/clang). ssm build will likely fail."
        warn "On Ubuntu/Debian: sudo apt install build-essential"
        warn "On macOS: xcode-select --install"
        warn "On Fedora/RHEL: sudo dnf install gcc gcc-c++"
    fi

    if "$PYTHON_BIN" -m pip install --no-build-isolation ssm==0.0.1; then
        ok "ssm==0.0.1 installed (rSLDS backend available)"
    else
        warn "ssm installation failed. This is expected on some systems."
        warn "The GVHD analysis script will fall back to hmmlearn (HMM) automatically."
        warn "To debug: install build-essential (Linux) or Xcode CLI tools (macOS) and retry."
    fi
else
    info "Step 3/5: Skipping ssm installation (--no-ssm)"
    ok "ssm skipped - GVHD analysis will use hmmlearn HMM fallback"
fi

# ---- Step 4: Install remaining requirements ------------------------------

info "Step 4/5: Installing requirements from requirements.txt..."

if [[ ! -f "$REQUIREMENTS" ]]; then
    fail "requirements.txt not found at: $REQUIREMENTS
    Run this script from the oura-digital-twin directory:
      bash scripts/install_full_stack.sh"
fi

# Filter out the commented ssm line (already handled above) and install the rest
if ! "$PYTHON_BIN" -m pip install -r "$REQUIREMENTS"; then
    fail "Failed to install requirements.txt.
    Check the error output above for the failing package."
fi

ok "All requirements installed"

# ---- Step 5: Verify installation ------------------------------------------

info "Step 5/5: Verifying key package imports..."

VERIFY_SCRIPT=$(cat <<'PYEOF'
import sys

packages = {
    "numpy": "numpy",
    "pandas": "pandas",
    "plotly": "plotly",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "sklearn": "scikit-learn",
    "filterpy": "filterpy",
    "pykalman": "pykalman",
    "antropy": "antropy",
    "nolds": "nolds",
    "pandera": "pandera",
}

optional = {
    "hmmlearn": "hmmlearn",
    "ssm": "ssm",
}

failed = []
for module, pkg in packages.items():
    try:
        __import__(module)
    except ImportError:
        failed.append(pkg)

opt_available = []
opt_missing = []
for module, pkg in optional.items():
    try:
        __import__(module)
        opt_available.append(pkg)
    except ImportError:
        opt_missing.append(pkg)

if failed:
    print(f"FAIL: Could not import required packages: {', '.join(failed)}")
    sys.exit(1)

print(f"REQUIRED: All {len(packages)} core packages imported successfully")

if opt_available:
    print(f"OPTIONAL: Available: {', '.join(opt_available)}")
if opt_missing:
    print(f"OPTIONAL: Not installed: {', '.join(opt_missing)}")

# Check ssm vs hmmlearn for GVHD analysis
ssm_ok = "ssm" in opt_available
hmm_ok = "hmmlearn" in opt_available
if ssm_ok:
    print("GVHD: rSLDS backend (ssm) available - full state-space model enabled")
elif hmm_ok:
    print("GVHD: HMM fallback (hmmlearn) available - Gaussian HMM will be used")
else:
    print("WARN: Neither ssm nor hmmlearn available - GVHD prediction will not work")
    sys.exit(1)

sys.exit(0)
PYEOF
)

if "$PYTHON_BIN" -c "$VERIFY_SCRIPT"; then
    ok "Verification passed"
else
    fail "Verification failed. Some required packages could not be imported.
    Check the output above and re-run the installer."
fi

# ---- Done -----------------------------------------------------------------

echo ""
printf "${GREEN}${BOLD}Installation complete.${NC}\n"
echo ""
info "Next steps:"
info "  1. cp .env.example .env   # Add your Oura API credentials"
info "  2. cp config.example.py config.py   # Set your patient constants"
info "  3. python api/import_oura.py --days 90   # Import Oura data"
info "  4. python run_all.py   # Run the full analysis pipeline"
