#!/usr/bin/env bash

REAL_SCRIPT_PATH="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
. "$REAL_SCRIPT_PATH/env/bin/activate"
exec python3 "$REAL_SCRIPT_PATH/main.py"
