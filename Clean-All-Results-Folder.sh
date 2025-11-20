#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Thesis Project Cleanup Script (Ubuntu)
# Removes all results and cache directories across architectures.
# ---------------------------------------------------------------------

set -euo pipefail  # Safe mode: exit on error, undefined var, or pipefail

PROJECT_ROOT="/home/ubuntu/skripsi_2602214545"   # <-- adjust if your project is elsewhere
ARCHS=("lstm" "tcn" "tft")
SUBDIRS=("efficiency" "stopping_embargo" "stopping_no_embargo")

echo "=============================================================="
echo "[Cleanup Script] Starting cleanup at: $(date)"
echo "Project root: $PROJECT_ROOT"
echo "=============================================================="

cd "$PROJECT_ROOT"

for arch in "${ARCHS[@]}"; do
    echo ""
    echo "--- Cleaning architecture: $arch ---"

    for sub in "${SUBDIRS[@]}"; do
        TARGET_DIR="$arch/$sub/results"
        if [ -d "$TARGET_DIR" ]; then
            echo "Removing: $TARGET_DIR"
            rm -rf "$TARGET_DIR"
        else
            echo "Skip: $TARGET_DIR (not found)"
        fi
    done

    # Remove architecture-level __pycache__
    PYCACHE_DIR="$arch/$sub/__pycache__"
    if [ -d "$PYCACHE_DIR" ]; then
        echo "Removing: $PYCACHE_DIR"
        rm -rf "$PYCACHE_DIR"
    else
        echo "Skip: $PYCACHE_DIR (not found)"
    fi
done

echo ""
echo "=============================================================="
echo "[Cleanup Script] Completed successfully at: $(date)"
echo "=============================================================="
