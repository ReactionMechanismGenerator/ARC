#!/usr/bin/env bash
# Cases for release_inputs.sh. Run from the repository root; exits non-zero if any case fails.
set -uo pipefail

script="${1:-.github/scripts/release_inputs.sh}"
good=dockerfiles/release_pins.env
sha=0123456789abcdef0123456789abcdef01234567
d=$(mktemp -d)
trap 'rm -r "$d"' EXIT
printf "VERSION = '2.0.0'\n" > "$d/common.py"
failed=0

# run NAME EXPECTED_EXIT(0|1) PINS_FILE VERSION_FILE [VAR=value ...]
run() {
  local name=$1 want=$2 pins=$3 version_file=$4 out rc
  shift 4
  : > "$d/out"
  out=$(env -i PATH="$PATH" GITHUB_OUTPUT="$d/out" PINS_FILE="$pins" VERSION_FILE="$version_file" "$@" bash "$script" 2>&1)
  rc=$?
  [ "$rc" -ne 0 ] && rc=1
  if [ "$rc" = "$want" ]; then
    echo "PASS $name"
  else
    echo "FAIL $name (exit $rc, wanted $want): $out"
    failed=1
  fi
}

c="$d/common.py"
run tag 0 "$good" "$c" EVENT_NAME=push REF_NAME=2.0.0 TAG_SHA=$sha
run v-tag 0 "$good" "$c" EVENT_NAME=push REF_NAME=v2.0.0 TAG_SHA=$sha
run pr 0 "$good" "$c" EVENT_NAME=pull_request HEAD_REF=some-branch HEAD_SHA=$sha
run tag-not-version 1 "$good" "$c" EVENT_NAME=push REF_NAME=2.0.1 TAG_SHA=$sha
run tag-two-parts 1 "$good" "$c" EVENT_NAME=push REF_NAME=2.0 TAG_SHA=$sha
run tag-prerelease 1 "$good" "$c" EVENT_NAME=push REF_NAME=v2.0.0-rc1 TAG_SHA=$sha
run tag-short-sha 1 "$good" "$c" EVENT_NAME=push REF_NAME=2.0.0 TAG_SHA=abc
run pr-no-ref 1 "$good" "$c" EVENT_NAME=pull_request HEAD_REF= HEAD_SHA=$sha
run other-event 1 "$good" "$c" EVENT_NAME=workflow_dispatch

grep -v '^RMG_PY_SHA=' "$good" > "$d/p1"
run pin-missing 1 "$d/p1" "$c" EVENT_NAME=push REF_NAME=2.0.0 TAG_SHA=$sha
sed 's/^RMG_PY_SHA=.*/RMG_PY_SHA=5c33e07e0999/' "$good" > "$d/p2"
run pin-short-sha 1 "$d/p2" "$c" EVENT_NAME=push REF_NAME=2.0.0 TAG_SHA=$sha
sed 's/@sha256:.*//' "$good" > "$d/p3"
run base-no-digest 1 "$d/p3" "$c" EVENT_NAME=push REF_NAME=2.0.0 TAG_SHA=$sha
{ cat "$good"; echo "RMG_DATABASE_SHA=$sha"; } > "$d/p4"
run pin-twice 1 "$d/p4" "$c" EVENT_NAME=push REF_NAME=2.0.0 TAG_SHA=$sha
run no-pins-file 1 "$d/absent" "$c" EVENT_NAME=push REF_NAME=2.0.0 TAG_SHA=$sha

printf "x = 1\n" > "$d/c2"
run no-version 1 "$good" "$d/c2" EVENT_NAME=pull_request HEAD_REF=b HEAD_SHA=$sha
printf 'VERSION = "2.0.0"\n' > "$d/c3"
run version-double-quotes 0 "$good" "$d/c3" EVENT_NAME=push REF_NAME=v2.0.0 TAG_SHA=$sha
run repo-version-file 0 "$good" arc/common.py EVENT_NAME=pull_request HEAD_REF=b HEAD_SHA=$sha

exit "$failed"
