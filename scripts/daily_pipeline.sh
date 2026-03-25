#!/usr/bin/env bash
# Daily pipeline: import Oura data → run analysis → (optionally) deploy
#
# Configure these paths for your environment, then add to crontab:
#   crontab -e
#   15 6 * * * /path/to/oura-digital-twin/scripts/daily_pipeline.sh
set -euo pipefail

# --- Configuration (edit these) ---
VENV_DIR="${VENV_DIR:-$(dirname "$0")/../../.venv}"
DIGITAL_TWIN="$(cd "$(dirname "$0")/.." && pwd)"
IMPORT_SCRIPT="${IMPORT_SCRIPT:-$DIGITAL_TWIN/api/import_oura.py}"
LOGFILE="${LOGFILE:-/tmp/oura_daily_pipeline.log}"

# Optional: Astro site deployment (uncomment and set paths)
# SITE_DIR="/path/to/your/astro-site"
# NODE_BIN="$HOME/.nvm/versions/node/v22.21.1/bin"

# --- Setup ---
exec >> "$LOGFILE" 2>&1
echo ""
echo "========================================"
echo "  Daily pipeline — $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# 1. Activate venv
source "$VENV_DIR/bin/activate"

# 2. Refresh OAuth2 token (prevents expired-token failures)
echo "[1/4] Refreshing OAuth2 token..."
python "$DIGITAL_TWIN/api/oura_oauth2_setup.py" --refresh || echo "  Token refresh skipped (no refresh token or already valid)"

# 3. Import fresh Oura data (last 3 days for overlap safety)
echo "[2/4] Importing Oura data..."
python "$IMPORT_SCRIPT" --days 3
echo "  Import done."

# 4. Run all analysis scripts
echo "[3/4] Running analysis pipeline..."
cd "$DIGITAL_TWIN"
python run_all.py
echo "  Analysis done."

# 5. Optional: deploy to static site
# Uncomment the block below if you have an Astro/Cloudflare Pages setup
#
# export PATH="$NODE_BIN:$PATH"
# echo "[3/3] Deploying..."
# cd "$SITE_DIR"
# node scripts/sync-reports.mjs
# npx astro build
# set -a && source .env && set +a
# npx wrangler pages deploy dist --project-name=digital-twin --branch=main
# echo "  Deploy done."

echo ""
echo "Pipeline complete — $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
