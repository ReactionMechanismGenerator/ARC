#!/bin/bash -l
# Usage: install_kinbot.sh [--uma]
#   --uma: additionally install KinBot's 'fc' extra (fairchem-core) to enable the
#          optional UMA machine-learned-potential refinement of KinBot TS guesses
#          (see 'kinbot_uma_settings' in arc/settings/settings.py).
#          NOTE: the UMA checkpoints themselves are license-gated on HuggingFace —
#          they require a HuggingFace login and accepting Meta's UMA license
#          (https://huggingface.co/facebook/UMA), or a locally downloaded checkpoint
#          file. The default install therefore stays lean and skips fairchem.
set -eo pipefail
ENV_NAME=kinbot_env
KINBOT_VERSION=2.3.0
PYTHON_VERSION=3.12

INSTALL_UMA=false
for arg in "$@"; do
    case $arg in
        --uma)
            INSTALL_UMA=true
            ;;
        *)
            echo "❌ Unknown argument: $arg (supported: --uma)"
            exit 1
            ;;
    esac
done

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
KINBOT_SPEC="kinbot==$KINBOT_VERSION"
if [ "$INSTALL_UMA" = "true" ]; then
    # The 'fc' extra pulls in fairchem-core (and torch) for UMA refinement.
    KINBOT_SPEC="kinbot[fc]==$KINBOT_VERSION"
    echo ">>> --uma given: installing KinBot with the fairchem ('fc') extra."
fi
echo ">>> Installing $KINBOT_SPEC into '$ENV_NAME' via pip..."
# PyYAML is needed by ARC's kinbot_script.py worker for its input/output files.
$COMMAND_PKG run -n "$ENV_NAME" python -m pip install "$KINBOT_SPEC" pyyaml

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

if [ "$INSTALL_UMA" = "true" ]; then
    echo ">>> Sanity-checking the fairchem installation..."
    $COMMAND_PKG run -n "$ENV_NAME" python -c "
from fairchem.core import FAIRChemCalculator
print('fairchem imports OK')
"
    echo "ℹ️  Reminder: UMA checkpoints are license-gated on HuggingFace. Either log in"
    echo "    with 'huggingface-cli login' after accepting the Meta UMA license, or set"
    echo "    a local checkpoint path in kinbot_uma_settings['model_path'] (~/.arc/settings.py)."
fi

echo "✅ Done installing KinBot $KINBOT_VERSION in the '$ENV_NAME' environment."
