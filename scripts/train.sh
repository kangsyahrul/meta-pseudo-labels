#!/usr/bin/env bash

# Arguments
CONFIG_PATH="${1:-configs/mnist.yml}"   # default config
shift || true                           # pop $1 so $@ = “extra args”

# Project location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
PROJECT_ROOT="$( realpath "${SCRIPT_DIR}/.." )"   # assumes scripts/ is one level down
cd "$PROJECT_ROOT"


# Activate environment
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

# Run
echo "Starting MPL-TF training with config: ${CONFIG_PATH}"
python -m mpl_tf.cli train --config "$CONFIG_PATH" "$@"
