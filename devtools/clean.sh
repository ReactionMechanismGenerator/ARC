#!/usr/bin/env bash
set -eo pipefail

echo "🧹 Cleaning ARC, molecule, and RMG-Py build artifacts..."

# Run make clean in molecule and RMG-Py if directories exist
if [[ -d "molecule" ]]; then
    echo "🧹 Running 'make clean' in molecule..."
    (cd molecule && make clean)
fi

if [[ -d "RMG-Py" ]]; then
    echo "🧹 Running 'make clean' in RMG-Py..."
    (cd RMG-Py && make clean)
fi

# Remove Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} +

# Remove coverage files
rm -f .coverage coverage.xml

echo "✅ Clean completed."
