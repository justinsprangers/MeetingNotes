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
# After an update adds packages to requirements.txt, install them (quick when nothing changed).
if ! cmp -s requirements.txt .venv/requirements.installed; then
    echo "Installing new packages Meeting Notes needs..."
    .venv/bin/python -m pip install -q -r requirements.txt && cp requirements.txt .venv/requirements.installed
fi
exec .venv/bin/python app.py
