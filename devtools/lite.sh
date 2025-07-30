#!/bin/bash -l
set -eo pipefail

echo ">>> Running lite installation script..."

if [ ! -d arc ] || [ ! -f environment.yml ]; then
    echo "❌ This script must be run from the ARC root directory."
    exit 1
fi

echo "🧹 Removing testing-related directories..."
rm -rf functional ipython arc/testing

echo "🧹 Removing test files matching '*Test.py'..."
find arc -name '*Test.py' -type f -print -delete

echo "✅ Lite installation complete."
