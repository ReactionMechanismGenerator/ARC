#!/usr/bin/env bash
set -euo pipefail

# 1) Micromamba setup (for both interactive & non-interactive)
eval "$(micromamba shell hook --shell bash)"

# 1.1) Ensure the Code directory is owned by rmguser
if [ ! -O /home/rmguser/Code ]; then
  chown -R rmguser:rmguser /home/rmguser/Code
fi

# 2) Interactive mode: login shells or docker exec … bash
if [ -t 0 ] && [ -t 1 ]; then
#   # Show the aliases cheat-sheet
#   /home/rmguser/.aliases_print.sh
  # Drop into an interactive Bash
  exec /bin/bash -l
fi

# 3) Non-interactive command mode: rmg or arc
case "${1:-}" in
  rmg)
    micromamba activate rmg_env
    exec python-jl /home/rmguser/Code/RMG-Py/rmg.py "${2:-input.py}"
    ;;
  arc)
    micromamba activate arc_env
    exec python /home/rmguser/Code/ARC/ARC.py "${2:-input.yml}"
    ;;
  *)
    echo "Usage: <rmg|arc> [input-file]" >&2
    exit 1
    ;;
esac
