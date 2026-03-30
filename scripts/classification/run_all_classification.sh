#!/usr/bin/env bash
# Run all logreg, frozen-finetune, and full-finetune classification experiments.
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
    run --backbone cinema --eval-mode finetune  --pooling "$pooling" --no-freeze-backbone
done

# ── DINOv3 ──
for model in dinov3_vits16 dinov3_vitb16; do
    for pooling in cls gap; do
        run --backbone dinov3 --dinov3-model-name "$model" --eval-mode logreg   --pooling "$pooling"
        run --backbone dinov3 --dinov3-model-name "$model" --eval-mode finetune --pooling "$pooling"
        run --backbone dinov3 --dinov3-model-name "$model" --eval-mode finetune --pooling "$pooling" --no-freeze-backbone
    done
done

# ── SAM (gap only — no CLS token) ──
for model in facebook/sam-vit-base facebook/sam-vit-large facebook/sam-vit-huge; do
    run --backbone sam --sam-model-id "$model" --eval-mode logreg   --pooling gap
    run --backbone sam --sam-model-id "$model" --eval-mode finetune --pooling gap
    run --backbone sam --sam-model-id "$model" --eval-mode finetune --pooling gap --no-freeze-backbone
done

echo "All experiments complete."
