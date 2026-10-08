#!/usr/bin/env bash
# Wrapper kept for convenience: see scripts/make_grid.py
exec python "$(dirname "$0")/make_grid.py" "$@"
