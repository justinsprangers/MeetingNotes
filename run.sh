#!/usr/bin/env bash
# Starts Meeting Notes. The first run sets up a private Python environment (a few minutes).
set -e
cd "$(dirname "$0")"
if [ ! -f .venv/setup-done ]; then
    echo "First run: installing what Meeting Notes needs. This takes a few minutes..."
    [ -d .venv ] || python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
    touch .venv/setup-done
fi
exec .venv/bin/python app.py
