#!/bin/bash -l
set -e

echo ">>> Installing RMG-database..."

pushd ..

if [ -d RMG-database ]; then
    echo "✔️ RMG-database already exists. Checking for update..."
    cd RMG-database

    REMOTE=$(git remote | grep -E '^(origin|official)$' | head -n1)
    if [ -z "$REMOTE" ]; then
        echo "❌ No valid remote found (expected 'origin' or 'official')"
        exit 1
    fi

    CURRENT_BRANCH=$(git symbolic-ref --short HEAD)
    if [ "$CURRENT_BRANCH" = "main" ]; then
        echo "ℹ️ Updating 'main' branch from $REMOTE..."
        git pull "$REMOTE" main
    else
        echo "ℹ️ Skipping update: current branch is '$CURRENT_BRANCH'"
    fi
else
    echo "📦 Cloning RMG-database..."
    git clone https://github.com/ReactionMechanismGenerator/RMG-database
    cd RMG-database
fi

RMGDB_PATH=$(pwd)

export PYTHONPATH="$PYTHONPATH:$RMGDB_PATH"
LINE_PYTHONPATH="export PYTHONPATH=\$PYTHONPATH:$RMGDB_PATH"
if ! grep -q "$RMGDB_PATH" ~/.bashrc; then
    echo "$LINE_PYTHONPATH" >> ~/.bashrc
    echo "✔️ RMG-database path added to ~/.bashrc"
else
    echo "ℹ️ RMG-database path already in ~/.bashrc"
fi

export RMGDB="$RMGDB_PATH"
LINE_RMGDB="export RMGDB=$RMGDB_PATH"
if ! grep -q "export RMGDB=" ~/.bashrc || ! grep -q "$RMGDB_PATH" ~/.bashrc; then
    sed -i "/export RMGDB=/d" ~/.bashrc
    echo "$LINE_RMGDB" >> ~/.bashrc
    echo "✔️ RMGDB environment variable set and added to ~/.bashrc"
else
    echo "ℹ️ RMGDB environment variable already set in ~/.bashrc"
fi

echo "PYTHONPATH=$PYTHONPATH"
echo "RMGDB=$RMGDB"

popd > /dev/null
echo "✅ Done installing RMG-database."
