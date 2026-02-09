#!/usr/bin/env bash
set -eo pipefail

# Enable tracing of each command, but tee it to a logfile
LOGFILE="tani_env_setup.log"
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' EXIT
exec 1> >(tee "$LOGFILE") 2>&1
set -x

echo ">>> Starting TANI environment setup at $(date)"

# 2) Pick a conda front-end
if command -v micromamba &>/dev/null; then
    echo "✔️  Using micromamba"
    COMMAND_PKG=micromamba
elif command -v mamba &>/dev/null; then
    echo "✔️  Using mamba"
    COMMAND_PKG=mamba
elif command -v conda &>/dev/null; then
    echo "✔️  Using conda"
    COMMAND_PKG=conda
else
    echo "❌  No micromamba/mamba/conda found in PATH"
    exit 1
fi

# 3) Initialize shell integration
if [ "$COMMAND_PKG" = "micromamba" ]; then
    eval "$(micromamba shell hook --shell=bash)"
elif [ "$COMMAND_PKG" = "mamba" ] || [ "$COMMAND_PKG" = "conda" ]; then
    BASE=$(conda info --base)
    source "$BASE/etc/profile.d/conda.sh"
fi


# 4) Clean caches to free space (pre-env)
echo ">>> Cleaning package caches (pre-env)"
$COMMAND_PKG clean -a -y || true

# 5) Determine the env name
ENV_YAML="devtools/tani_environment.yml"
ENV_NAME=$(grep -E '^ *name:' "$ENV_YAML" | head -1 | awk '{print $2}')

# 6) Remove any existing env (emulate --force)
echo ">>> Removing any existing '$ENV_NAME' env"
if [[ $COMMAND_PKG == micromamba || $COMMAND_PKG == mamba ]]; then
    $COMMAND_PKG env remove -n "$ENV_NAME" --yes 2>/dev/null || true
else
    conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
fi

# 7) Create the environment
echo ">>> Creating conda env from $ENV_YAML (name=$ENV_NAME)"
if ! $COMMAND_PKG env create -n "$ENV_NAME" -f "$ENV_YAML" -v; then
    echo "❌  Environment creation failed. Dumping last 200 lines of log:"
    tail -n 200 "$LOGFILE"
    echo "---- Disk usage at failure ----"
    df -h .
    exit 1
fi

# 8) Clean caches again to reclaim space (post-env)
echo ">>> Cleaning package caches (post-env)"
$COMMAND_PKG clean -a -y || true

# 9) Show final disk usage & env list
echo "---- Disk usage after env create ----"
df -h .

echo "---- Conda env list ----"
$COMMAND_PKG env list

# 10) Ensure pkg_resources exists, then sanity-check import without shell activation
echo ">>> Ensuring setuptools/pkg_resources is present in '$ENV_NAME'"
$COMMAND_PKG install -n "$ENV_NAME" -y setuptools

echo ">>> Verifying pkg_resources in '$ENV_NAME'"
if ! $COMMAND_PKG run -n "$ENV_NAME" python -c "import pkg_resources" >/dev/null 2>&1; then
    echo ">>> pkg_resources missing after conda install; forcing setuptools reinstall via pip"
    $COMMAND_PKG run -n "$ENV_NAME" python -m pip install --upgrade --force-reinstall "setuptools<81"
fi

# Hard fail early if pkg_resources is still unavailable.
$COMMAND_PKG run -n "$ENV_NAME" python -c "import pkg_resources"

echo ">>> Sanity-checking TANI import"
set +x
$COMMAND_PKG run -n "$ENV_NAME" python - <<'PYCODE'
import pkg_resources
import torchani
print("pkg_resources:", pkg_resources.__name__)
print("torchani version:", torchani.__version__)
PYCODE

echo "✅  TANI environment '$ENV_NAME' setup completed at $(date)"
