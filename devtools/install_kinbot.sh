#!/bin/bash -l
set -eo pipefail
ENV_NAME=kinbot_env
KINBOT_VERSION=2.3.0
PYTHON_VERSION=3.12

echo "📦 Installing KinBot..."

if command -v micromamba &> /dev/null; then
    echo "✔️ Micromamba is installed."
    COMMAND_PKG=micromamba
elif command -v mamba &> /dev/null; then
    echo "✔️ Mamba is installed."
    COMMAND_PKG=mamba
elif command -v conda &> /dev/null; then
    echo "✔️ Conda is installed."
    COMMAND_PKG=conda
else
    echo "❌ Micromamba, Mamba, or Conda is required. Please install one."
    exit 1
fi

if [ "$COMMAND_PKG" = "micromamba" ]; then
    eval "$(micromamba shell hook --shell=bash)"
else
    BASE=$(conda info --base)
    . "$BASE/etc/profile.d/conda.sh"
    eval "$($COMMAND_PKG shell hook --shell=bash)"
fi

# KinBot lives in its own environment (it requires Python >= 3.11 and is
# imported by arc/job/adapters/scripts/kinbot_script.py through
# run_in_conda_env), independent of arc_env.

echo ">>> Removing any existing '$ENV_NAME' environment..."
if [ "$COMMAND_PKG" = "micromamba" ] || [ "$COMMAND_PKG" = "mamba" ]; then
    $COMMAND_PKG env remove -n "$ENV_NAME" --yes 2>/dev/null || true
else
    conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
fi

echo ">>> Creating the '$ENV_NAME' environment (python=$PYTHON_VERSION)..."
$COMMAND_PKG create -n "$ENV_NAME" -c conda-forge "python=$PYTHON_VERSION" pip -y

# Upstream (https://github.com/zadorlab/KinBot) recommends pip;
# the conda-forge package lags behind PyPI releases.
echo ">>> Installing KinBot $KINBOT_VERSION into '$ENV_NAME' via pip..."
# PyYAML is needed by ARC's kinbot_script.py worker for its input/output files.
$COMMAND_PKG run -n "$ENV_NAME" python -m pip install "kinbot==$KINBOT_VERSION" pyyaml

echo ">>> Sanity-checking the KinBot installation..."
$COMMAND_PKG run -n "$ENV_NAME" python -c "
from kinbot.modify_geom import modify_coordinates
from kinbot.parameters import Parameters
from kinbot.qc import QuantumChemistry
from kinbot.reaction_finder import ReactionFinder
from kinbot.reaction_generator import ReactionGenerator
from kinbot.stationary_pt import StationaryPoint
print('KinBot imports OK')
"

echo "✅ Done installing KinBot $KINBOT_VERSION in the '$ENV_NAME' environment."
