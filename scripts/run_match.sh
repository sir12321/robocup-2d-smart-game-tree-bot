#!/usr/bin/env bash
# Play one match: our Agent1 (agent.py in this repo) on the left.
#
#   scripts/run_match.sh                  vs. the HELIOS reference team
#   scripts/run_match.sh starter          vs. the course's starter agent
#   scripts/run_match.sh path/to/agent.py vs. another team's agent.py (class Agent1)
#
# Anything after the opponent is passed to `a3.py match`, e.g.
#   scripts/run_match.sh full -p 6010     (another port, to run matches in parallel)
#   scripts/run_match.sh full --dry-run   (print what would run)
#
# Needs the starter set up first: see "Running it" in README.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY2D="$ROOT/py2d"

if [ ! -f "$PY2D/a3.py" ]; then
    echo "py2d/ is empty: run 'git submodule update --init' and set it up (README.md)." >&2
    exit 1
fi

# agent.py is generated from agent/: rebuild it if the source changed.
if ! python3 "$ROOT/tools/bundle.py" --check >/dev/null; then
    echo "agent/ changed since agent.py was built: rebuilding"
    python3 "$ROOT/tools/bundle.py"
fi

opp="${1:-full}"
if [ $# -gt 0 ]; then shift; fi

case "$opp" in
    full)    right=(FULL --right full) ;;
    starter) right=(STARTER) ;;  # no spec: py2d/src/student/agent.py, the starter stub
    *)       right=(OPPONENT --right "$(cd "$(dirname "$opp")" && pwd)/$(basename "$opp")") ;;
esac

cd "$PY2D"
exec python3 a3.py match AGENT1 "${right[0]}" --left "$ROOT/agent.py" "${right[@]:1}" "$@"
