#!/usr/bin/env bash
# (Re)start the listening UI in the background: stops whatever holds the port first.
set -euo pipefail
PORT="${1:-8765}"
cd "$(dirname "$0")/.."
pid=$(ss -ltnpH "sport = :$PORT" | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1 || true)
if [ -n "$pid" ]; then kill "$pid"; sleep 1; fi
setsid nohup uv run lab review --port "$PORT" > ../data/review.log 2>&1 < /dev/null &
echo "review server starting on port $PORT (log: data/review.log)"
