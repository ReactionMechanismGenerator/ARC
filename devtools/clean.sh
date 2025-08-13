#!/usr/bin/env bash
set -eo pipefail

# -----------------------------------------------------------------------------
# Helper: aggressively clean conda/micromamba caches & remove any known build
# directories in our workspace.  Ignores any permission errors.
# -----------------------------------------------------------------------------

clean_conda() {
    echo ">>> Cleaning conda/micromamba caches…"
    echo ">>> Current disk usage:"
    df -h . | sed '1!p;d'   # show before-clean

    if command -v micromamba &>/dev/null; then
        micromamba clean --all --yes || true
    elif command -v mamba &>/dev/null; then
        mamba clean --all --yes || true
    elif command -v conda &>/dev/null; then
        conda clean -afy || true
    fi

    echo ">>> After cleaning conda/micromamba caches:"
    
    df -h . | sed '1!p;d'   # show after-clean

    echo ">>> Cleaned conda/micromamba caches."
}

clean_pip() {
    echo ">>> Cleaning pip cache…"
    
    rm -rf "$HOME/.cache/pip" || true

    echo ">>> Cleaned pip cache."
}

clean_python_builds() {
    echo ">>> Cleaning Python build directories, .pyc, .pyd, and .so files…"

    # Remove any "build/" or "dist/" dirs left behind in our repo clones
    find . -type d \( -name build -o -name dist \) -prune -exec rm -rf {} + 2>/dev/null || true

    # Remove any __pycache__
    find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true

    # remove .pyc .pyd and .so files
    find . -type f \( -name "*.pyc" -o -name "*.pyd" -o -name "*.so" \) -exec rm -f {} + 2>/dev/null || true

    echo ">>> Cleaned."

}
# -----------------------------------------------------------------------------

# Main script execution
# USER Passed Arguments
if [[ "${1:-}" == "--conda" ]]; then
    clean_conda
fi

if [[ "${1:-}" == "--pip" ]]; then
    clean_pip
fi

if [[ "${1:-}" == "--python-builds" ]]; then
    clean_python_builds
fi

if [[ "${1:-}" == "--all" ]]; then
    clean_conda
    clean_pip
    clean_python_builds
fi

# What if the user didn't pass any arguments?
if [[ -z "${1:-}" ]]; then
    echo "No arguments provided. Please specify --conda, --pip, --python-builds, or --all."
    echo "Example: ./devtools/clean.sh --all"
    exit 1
elif [[ "${1:-}" == "--help" ]]; then
    echo "Usage: ./devtools/clean.sh [--conda | --pip | --python-builds | --all]"
    echo "Options:"
    echo "  --conda          Clean conda/micromamba caches"
    echo "  --pip            Clean pip cache"
    echo "  --python-builds  Clean Python build directories and files"
    echo "  --all            Clean all caches and build directories"
    exit 0
elif [[ "${1:-}" != "--conda" && "${1:-}" != "--pip" && "${1:-}" != "--python-builds" && "${1:-}" != "--all" ]]; then
    echo "Invalid argument: $1"
    echo "Use --help for usage information."
    exit 1
fi
