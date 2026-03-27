#!/usr/bin/env bash
# Smoke test: all model versions, logreg only, one pooling each, minimal patients.
# 50 patients is the minimum that reliably covers all 5 pathology classes for 10-fold CV.

set -euo pipefail

SCRIPT="scripts/classification/run_acdc_classification.py"
MAX_PATIENTS=50

run() {
    echo "=== $* ==="
    python "$SCRIPT" "$@" --max-patients "$MAX_PATIENTS"
}

# ── CineMA ──
run --backbone cinema --eval-mode logreg --pooling cls

# ── DINOv3 ──
for model in dinov3_vits16 dinov3_vitb16; do
    run --backbone dinov3 --dinov3-model-name "$model" --eval-mode logreg --pooling cls
done

# ── SAM (gap only — no CLS token) ──
for model in facebook/sam-vit-base facebook/sam-vit-large facebook/sam-vit-huge; do
    run --backbone sam --sam-model-id "$model" --eval-mode logreg --pooling gap
done

echo "Smoke test complete."
