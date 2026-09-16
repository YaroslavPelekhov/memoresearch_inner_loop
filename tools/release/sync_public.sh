#!/usr/bin/env bash
# sync_public.sh — Sync internal repo to public FusionBrainLab/gigaevo-core
#
# Usage:
#   bash tools/release/sync_public.sh                  # dry-run (no push)
#   bash tools/release/sync_public.sh --push           # actually push
#   bash tools/release/sync_public.sh --keep-tempdir   # keep temp clone for inspection
#
# Requires: git-filter-repo (pip install git-filter-repo)

set -euo pipefail

# ─── Constants ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXCLUDE_FILE="$SCRIPT_DIR/exclude_paths.txt"
REPLACE_FILE="$SCRIPT_DIR/replace_ips.txt"
DEFAULT_PUBLIC_REMOTE="git@github.com:FusionBrainLab/gigaevo-core.git"
DEFAULT_BRANCH="main"
FILTER_REPO="${GIGAEVO_PYTHON:-python3} -m git_filter_repo"

# ─── Argument parsing ────────────────────────────────────────────────
DO_PUSH=false
SOURCE_BRANCH="$DEFAULT_BRANCH"
PUBLIC_REMOTE="$DEFAULT_PUBLIC_REMOTE"
KEEP_TEMPDIR=false

usage() {
    echo "Usage: bash $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --push              Actually push to public remote (default: dry-run)"
    echo "  --branch BRANCH     Source branch to sync (default: main)"
    echo "  --public-remote URL Public repo URL (default: $DEFAULT_PUBLIC_REMOTE)"
    echo "  --keep-tempdir      Keep temp directory after run (for inspection)"
    echo "  -h, --help          Show this help"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --push)          DO_PUSH=true; shift ;;
        --branch)        SOURCE_BRANCH="$2"; shift 2 ;;
        --public-remote) PUBLIC_REMOTE="$2"; shift 2 ;;
        --keep-tempdir)  KEEP_TEMPDIR=true; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               echo "ERROR: Unknown option: $1"; usage; exit 1 ;;
    esac
done

# ─── Helpers ─────────────────────────────────────────────────────────
info()  { echo "==> $*"; }
warn()  { echo "WARNING: $*" >&2; }
die()   { echo "ERROR: $*" >&2; exit 1; }

cleanup() {
    if [[ "$KEEP_TEMPDIR" == "false" && -n "${TMPDIR_PATH:-}" ]]; then
        info "Cleaning up $TMPDIR_PATH"
        rm -rf "$TMPDIR_PATH"
    fi
}
trap cleanup EXIT

# ─── Phase 0: Preflight ─────────────────────────────────────────────
info "Phase 0: Preflight checks"

$FILTER_REPO --help >/dev/null 2>&1 \
    || die "git-filter-repo not found. Install: pip install git-filter-repo"

# Must be run from repo root
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" \
    || die "Not inside a git repository"
[[ -f "$REPO_ROOT/pyproject.toml" && -d "$REPO_ROOT/gigaevo" ]] \
    || die "Not in gigaevo-core-internal root (expected pyproject.toml + gigaevo/)"

[[ -f "$EXCLUDE_FILE" ]] \
    || die "Exclude file not found: $EXCLUDE_FILE"
[[ -f "$REPLACE_FILE" ]] \
    || die "Replace file not found: $REPLACE_FILE"

git rev-parse --verify "$SOURCE_BRANCH" >/dev/null 2>&1 \
    || die "Branch '$SOURCE_BRANCH' does not exist"

# Strip comment lines from exclude file into a clean temp file
CLEAN_EXCLUDE=$(mktemp)
grep -v '^\s*#' "$EXCLUDE_FILE" | grep -v '^\s*$' > "$CLEAN_EXCLUDE"

# Strip comment lines from replace file
CLEAN_REPLACE=$(mktemp)
grep -v '^\s*#' "$REPLACE_FILE" | grep -v '^\s*$' > "$CLEAN_REPLACE"

COMMIT_COUNT=$(git rev-list --count "$SOURCE_BRANCH")
info "Source: $SOURCE_BRANCH ($COMMIT_COUNT commits)"
info "Public remote: $PUBLIC_REMOTE"
info "Excluding $(wc -l < "$CLEAN_EXCLUDE" | tr -d ' ') paths"
info "IP replacements: $(wc -l < "$CLEAN_REPLACE" | tr -d ' ') rules"
echo ""

# ─── Phase 1: Clone ─────────────────────────────────────────────────
info "Phase 1: Cloning to temp directory"

TMPDIR_PATH=$(mktemp -d "/tmp/gigaevo-sync-XXXXXX")
git clone --no-local --branch "$SOURCE_BRANCH" --single-branch "$REPO_ROOT" "$TMPDIR_PATH/repo"
cd "$TMPDIR_PATH/repo"

info "Cloned to $TMPDIR_PATH/repo"
echo ""

# ─── Phase 2: Filter excluded paths ─────────────────────────────────
info "Phase 2: Filtering excluded paths"

# Build --path arguments from exclude file
FILTER_ARGS=()
while IFS= read -r line; do
    FILTER_ARGS+=(--path "$line")
done < "$CLEAN_EXCLUDE"

$FILTER_REPO --invert-paths "${FILTER_ARGS[@]}" --force

FILTERED_COUNT=$(git rev-list --count HEAD)
REMOVED=$((COMMIT_COUNT - FILTERED_COUNT))
info "After filtering: $FILTERED_COUNT commits ($REMOVED empty commits removed)"
echo ""

# ─── Phase 3: Sanitize IPs ──────────────────────────────────────────
info "Phase 3: Sanitizing internal IPs"

$FILTER_REPO --replace-text "$CLEAN_REPLACE" --force

info "IP sanitization complete"
echo ""

# Clean up temp files
rm -f "$CLEAN_EXCLUDE" "$CLEAN_REPLACE"

# ─── Phase 4: Verify ────────────────────────────────────────────────
info "Phase 4: Verification"

PROBLEMS=0

# Check excluded paths are gone
while IFS= read -r path; do
    # Strip trailing slash for test
    clean_path="${path%/}"
    if [[ -e "$clean_path" ]]; then
        warn "EXCLUDED PATH STILL PRESENT: $clean_path"
        PROBLEMS=$((PROBLEMS + 1))
    fi
done < <(grep -v '^\s*#' "$EXCLUDE_FILE" | grep -v '^\s*$')

# Check no internal IPs remain
IP_HITS=$(grep -r '10\.232\.' --include='*.py' --include='*.yaml' --include='*.yml' --include='*.sh' . 2>/dev/null | grep -v '.git/' | wc -l || true)
if [[ "$IP_HITS" -gt 0 ]]; then
    warn "Found $IP_HITS files still containing internal IPs:"
    grep -r '10\.232\.' --include='*.py' --include='*.yaml' --include='*.yml' --include='*.sh' . 2>/dev/null | grep -v '.git/' | head -10
    PROBLEMS=$((PROBLEMS + 1))
fi

# Check key public files exist
for required in README.md pyproject.toml run.py gigaevo tests config; do
    if [[ ! -e "$required" ]]; then
        warn "MISSING REQUIRED: $required"
        PROBLEMS=$((PROBLEMS + 1))
    fi
done

FINAL_COUNT=$(git rev-list --count HEAD)
FINAL_FILES=$(git ls-files | wc -l)

echo ""
echo "════════════════════════════════════════════════════════"
echo "  SYNC SUMMARY"
echo "════════════════════════════════════════════════════════"
echo "  Source branch:    $SOURCE_BRANCH"
echo "  Commits:          $FINAL_COUNT (was $COMMIT_COUNT)"
echo "  Tracked files:    $FINAL_FILES"
echo "  Problems found:   $PROBLEMS"
echo "  Temp directory:   $TMPDIR_PATH/repo"
echo "════════════════════════════════════════════════════════"
echo ""

if [[ "$PROBLEMS" -gt 0 ]]; then
    die "Verification failed with $PROBLEMS problem(s). Fix before pushing."
fi

# ─── Phase 5: Push or dry-run ───────────────────────────────────────
if [[ "$DO_PUSH" == "true" ]]; then
    info "Phase 5: Pushing to $PUBLIC_REMOTE"
    git remote add public "$PUBLIC_REMOTE"

    echo ""
    echo "About to FORCE-PUSH $FINAL_COUNT commits to $PUBLIC_REMOTE ($SOURCE_BRANCH)"
    read -r -p "Confirm? [y/N] " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo "Aborted."
        exit 0
    fi

    git push public HEAD:"$SOURCE_BRANCH" --force
    info "Push complete!"
    echo ""
    echo "=== Post-push checklist ==="
    echo "[ ] Check https://github.com/FusionBrainLab/gigaevo-core"
    echo "[ ] Verify README renders correctly"
    echo "[ ] Verify no internal IPs: clone + grep -r '10.232.'"
    echo "[ ] Verify CI passes"
else
    info "Phase 5: DRY RUN — no push"
    echo ""
    echo "To inspect the filtered repo:"
    echo "  cd $TMPDIR_PATH/repo"
    echo "  git log --oneline | head -20"
    echo "  grep -r '10.232.' --include='*.py' ."
    echo ""
    echo "To push for real:"
    echo "  bash tools/release/sync_public.sh --push"

    if [[ "$KEEP_TEMPDIR" == "false" ]]; then
        echo ""
        echo "Tip: use --keep-tempdir to inspect the filtered clone"
    fi
fi
