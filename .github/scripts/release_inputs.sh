#!/usr/bin/env bash
# Resolve and validate the inputs of the release image build. Fails closed on any missing or
# malformed value. Reads its inputs from the environment and writes key=value lines to the file
# named by GITHUB_OUTPUT.
#
#   EVENT_NAME    push | pull_request
#   REF_NAME      the tag name (push)
#   TAG_SHA       the tag's commit (push)
#   HEAD_REF      the PR head branch (pull_request)
#   HEAD_SHA      the PR head commit (pull_request)
#   PINS_FILE     default dockerfiles/release_pins.env
#   VERSION_FILE  default arc/common.py
set -euo pipefail

pins="${PINS_FILE:-dockerfiles/release_pins.env}"
version_file="${VERSION_FILE:-arc/common.py}"

fail() { echo "::error::$*" >&2; exit 1; }

pin() {
  local key="$1" pattern="$2" lines
  lines=$(grep -E "^${key}=" "$pins" || true)
  [ -n "$lines" ] || fail "$pins: $key is missing"
  [ "$(printf '%s\n' "$lines" | wc -l)" -eq 1 ] || fail "$pins: $key is set more than once"
  local value="${lines#*=}"
  [[ "$value" =~ $pattern ]] || fail "$pins: $key='$value' does not match $pattern"
  printf '%s' "$value"
}

[ -f "$pins" ] || fail "$pins not found"
sha_re='^[0-9a-f]{40}$'
ref_re='^[A-Za-z0-9._/-]+$'
base_image=$(pin BASE_IMAGE '^[a-z0-9./_-]+(:[A-Za-z0-9._-]+)?@sha256:[0-9a-f]{64}$')
rmg_py_ref=$(pin RMG_PY_REF "$ref_re")
rmg_py_sha=$(pin RMG_PY_SHA "$sha_re")
rmg_db_ref=$(pin RMG_DATABASE_REF "$ref_re")
rmg_db_sha=$(pin RMG_DATABASE_SHA "$sha_re")

version=$(sed -nE "s/^VERSION = ['\"]([^'\"]+)['\"]\$/\1/p" "$version_file")
[ -n "$version" ] || fail "no VERSION = '...' line in $version_file"
[ "$(printf '%s\n' "$version" | wc -l)" -eq 1 ] || fail "more than one VERSION line in $version_file"

case "${EVENT_NAME:-}" in
  push)
    [[ "${REF_NAME:-}" =~ ^v?([0-9]+\.[0-9]+\.[0-9]+)$ ]] || fail "tag '${REF_NAME:-}' is not [v]X.Y.Z"
    tag_version="${BASH_REMATCH[1]}"
    [ "$tag_version" = "$version" ] || fail "tag ${REF_NAME} gives version $tag_version, but VERSION in $version_file is $version"
    arc_ref="$REF_NAME"
    arc_sha="${TAG_SHA:-}"
    ;;
  pull_request)
    arc_ref="${HEAD_REF:-}"
    arc_sha="${HEAD_SHA:-}"
    ;;
  *)
    fail "unsupported event '${EVENT_NAME:-}'"
    ;;
esac
[[ "$arc_ref" =~ $ref_re ]] || fail "ARC ref '$arc_ref' is empty or malformed"
[[ "$arc_sha" =~ $sha_re ]] || fail "ARC commit '$arc_sha' is not a full SHA"

{
  echo "base_image=$base_image"
  echo "rmg_py_ref=$rmg_py_ref"
  echo "rmg_py_sha=$rmg_py_sha"
  echo "rmg_database_ref=$rmg_db_ref"
  echo "rmg_database_sha=$rmg_db_sha"
  echo "arc_ref=$arc_ref"
  echo "arc_sha=$arc_sha"
  echo "version=$version"
} | tee -a "${GITHUB_OUTPUT:?GITHUB_OUTPUT is not set}"
