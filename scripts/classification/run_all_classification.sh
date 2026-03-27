#!/usr/bin/env bash
# Run all logreg and frozen-finetune classification experiments.
# SAM uses gap only (no CLS token); CineMA and DINOv3 run both cls and gap.

set -euo pipefail

SCRIPT="scripts/classification/run_acdc_classification.py"

run() {
    echo "=== $* ==="
    python "$SCRIPT" "$@"
}

# ── CineMA ──
for pooling in cls gap; do
    run --backbone cinema --eval-mode logreg    --pooling "$pooling"
    run --backbone cinema --eval-mode finetune  --pooling "$pooling"
done

# ── DINOv3 ──
for pooling in cls gap; do
    run --backbone dinov3 --eval-mode logreg    --pooling "$pooling"
    run --backbone dinov3 --eval-mode finetune  --pooling "$pooling"
done

# ── SAM (gap only — no CLS token) ──
run --backbone sam --eval-mode logreg   --pooling gap
run --backbone sam --eval-mode finetune --pooling gap

echo "All experiments complete."
