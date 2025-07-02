#!/usr/bin/env bash
set -euo pipefail

# ensure the aliases defined in /etc/profile.d/ are loaded
# (interactive login shells source them automatically, but
# non-login shells like `docker exec … bash` do not)
shopt -q login_shell || source /etc/profile

cat <<'EOF'
╭─────────────────────────────────────────────────────────────╮
│  Built-in aliases & helpers                                │
╰─────────────────────────────────────────────────────────────╯

‣ Environment switches
    rmge        →  micromamba activate rmg_env
    arce        →  micromamba activate arc_env
    deact       →  micromamba deactivate

‣ Jump to source trees
    rmgcode     →  cd $rmgpy_path
    dbcode      →  cd $rmgdb_path
    arcode      →  cd $arc_path

‣ One-liner runners
    rmg         →  python    $rmgpy_path/rmg.py    input.py
    arkane      →  python    $rmgpy_path/Arkane.py input.py
    arc         →  python    $arc_path/ARC.py      input.yml
    arcrestart  →  python    $arc_path/ARC.py      restart.yml

Type  aliases   again at any time to reopen this cheat-sheet.
EOF
