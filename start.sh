#!/usr/bin/env bash
# Sentinel-Graph — One-command startup
# Usage: ./start.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        SENTINEL-GRAPH  ENGINE            ║${NC}"
echo -e "${BLUE}║   Graph-Native ITDR — Team Velosta       ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo ""

# ── 1. Activate venv ─────────────────────────────────────────────────────────
if [ ! -f "venv/bin/activate" ]; then
  echo -e "${YELLOW}Creating virtual environment...${NC}"
  python3 -m venv venv
fi
source venv/bin/activate
echo -e "${GREEN}✓ venv activated${NC}"

# ── 2. Install Python deps if needed ─────────────────────────────────────────
if ! python -c "import fastapi" 2>/dev/null; then
  echo -e "${YELLOW}Installing Python dependencies...${NC}"
  pip install -r requirements.txt -q
fi
echo -e "${GREEN}✓ Python dependencies ready${NC}"

# ── 3. Run pipeline if data not generated ────────────────────────────────────
if [ ! -f "data/generated/scored_events.csv" ]; then
  echo ""
  echo -e "${YELLOW}First run — generating data, training GraphSAGE...${NC}"
  python run_pipeline.py --no-dash
  echo -e "${GREEN}✓ Pipeline complete${NC}"
else
  echo -e "${GREEN}✓ Pipeline data already exists — skipping training${NC}"
fi

# ── 4. Kill anything on port 8000 ────────────────────────────────────────────
lsof -ti tcp:8000 | xargs kill -9 2>/dev/null || true
sleep 1

# ── 5. Start FastAPI (serves API + static dashboard + landing page) ──────────
echo ""
echo -e "${YELLOW}Starting Sentinel-Graph Engine...${NC}"
PYTHONPATH="$SCRIPT_DIR" uvicorn api.main:app --host 0.0.0.0 --port 8000 \
  > /tmp/sentinel-api.log 2>&1 &
API_PID=$!

# Wait for API to be ready
echo -n "  Waiting for engine"
for i in $(seq 1 30); do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e " ${GREEN}✓${NC}"
    break
  fi
  echo -n "."
  sleep 1
done
echo -e "${GREEN}✓ Engine running (PID $API_PID)${NC}"

# ── 6. Done ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              SENTINEL-GRAPH IS RUNNING                  ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  🏠  Landing Page →  http://localhost:8000               ║${NC}"
echo -e "${GREEN}║  🖥️  Dashboard   →  http://localhost:8000/static/dashboard.html  ║${NC}"
echo -e "${GREEN}║  ⚡  API Docs    →  http://localhost:8000/docs           ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Log: /tmp/sentinel-api.log"
echo -e "Press ${RED}Ctrl+C${NC} to stop"
echo ""

# ── Cleanup on exit ──────────────────────────────────────────────────────────
cleanup() {
  echo ""
  echo -e "${YELLOW}Stopping Sentinel-Graph...${NC}"
  kill $API_PID 2>/dev/null || true
  echo -e "${GREEN}Done.${NC}"
}
trap cleanup INT TERM

# Keep script alive
wait
