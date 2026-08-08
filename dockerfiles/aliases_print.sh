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

‣ Job runners  (arguments are forwarded; the file shown is the no-argument default)
    rmg  [args] <input.py>     →  RMG in rmg_env        (default: input.py)
    arkane [args] <input.py>   →  Arkane in rmg_env     (default: input.py)
    arc  [args] <input.yml>    →  ARC in arc_env        (default: input.yml)
    arcrestart [args] [file]   →  ARC in arc_env        (default: restart.yml)

    e.g.  arc my_case/input.yml
    stdout/stderr are also appended to ./stdout.log and ./stderr.log

Type  aliases   again at any time to reopen this cheat-sheet.
EOF

if [ -r "${ARC_IMAGE_VERSIONS:-/home/mambauser/Code/image_versions.env}" ]; then
    echo
    echo "‣ Image provenance"
    sed 's/^/    /' "${ARC_IMAGE_VERSIONS:-/home/mambauser/Code/image_versions.env}"
fi
