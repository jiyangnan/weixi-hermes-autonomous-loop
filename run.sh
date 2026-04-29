#!/bin/bash
# Run EpisodicMemory scripts with correct Python environment
# Uses system Python (numpy) + venv packages (neo4j, etc.)

PYTHON=/usr/local/bin/python3.14
VENV_PKGS=$(~/.hermes/venv/bin/python -c "import site; print(site.getsitepackages()[0])")

exec $PYTHON -W ignore::RuntimeWarning \
  "$@"
