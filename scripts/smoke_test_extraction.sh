#!/usr/bin/env bash
# Local smoke test for feature extraction across every dataset x model x task.
#
# Two lanes, because extraction and training have very different minimum-patient
# needs:
#
#   Lane A (default) — --cache-only extraction for all 102 cache keys at
#     MAX_PATIENTS_EXTRACT patients.  No labels are needed, so a couple of
#     patients is enough.  This is the dataset x model x task sweep.
#     ACDC has no val split, so a couple of patients are carved out of train --
#     hence a floor of 3.
#
#   Lane B (TRAIN=1) — one short training run per model on ACDC only, to prove
#     the decoders/probes actually consume the new caches.  Extraction alone
#     cannot catch a decoder shape mismatch, which is the real risk in the
#     never-before-run SAM1-segmentation and SAM2-classification paths.
#     Classification needs many more patients here: the pathology label *is* the
#     target, and ACDC's 10-fold stratified CV needs >=10 patients per class.
#     --max-patients is stratified by pathology (plain head() would give ACDC only
#     3 of 5 classes at n=50, since its metadata is sorted by pathology).
#
# Everything is written under $CACHE_ROOT so real caches and results/ stay clean.
#
# Usage:
#   bash scripts/smoke_test_extraction.sh                 # Lane A, all datasets
#   DATASETS="acdc" bash scripts/smoke_test_extraction.sh # Lane A, ACDC only
#   TRAIN=1 bash scripts/smoke_test_extraction.sh         # Lane A + Lane B
#
# Env overrides:
#   CACHE_ROOT              scratch tree for caches + smoke results
#   DATASETS                space-separated subset of "acdc mnm mnm2"
#   MAX_PATIENTS_EXTRACT    patients per split for Lane A (default 4; ACDC needs >=3
#                           so the val carve-out leaves training data behind)
#   MAX_PATIENTS_SEG_TRAIN  patients per split for Lane B segmentation (default 4)
#   MAX_PATIENTS_CLS_TRAIN  patients per split for Lane B classification (default 50).
#                           ACDC has no val split so it uses 10-fold stratified CV,
#                           which needs >=10 patients per class; 50 stratified over
#                           ACDC's 5 classes gives exactly 10 each.
#   TRAIN                   set to 1 to also run Lane B
#   PYTHON                  interpreter to use (default "python")

set -euo pipefail

CACHE_ROOT=${CACHE_ROOT:-/tmp/heartfm_smoke}
DATASETS=${DATASETS:-"acdc mnm mnm2"}
MAX_PATIENTS_EXTRACT=${MAX_PATIENTS_EXTRACT:-4}
MAX_PATIENTS_SEG_TRAIN=${MAX_PATIENTS_SEG_TRAIN:-4}
MAX_PATIENTS_CLS_TRAIN=${MAX_PATIENTS_CLS_TRAIN:-50}
TRAIN=${TRAIN:-0}
PYTHON=${PYTHON:-python}

SEG="scripts/segmentation/run_segmentation.py"
CLS="scripts/classification/run_classification.py"

# DINOv3 model names.
DINO_MODELS=(dinov3_vits16 dinov3_vitb16 dinov3_vitl16)
# SAM v1 checkpoints.
SAM_MODELS=(facebook/sam-vit-base facebook/sam-vit-large facebook/sam-vit-huge)
# SAM2 checkpoints.  hiera-tiny is excluded — nothing in results/ uses it.
SAM2_MODELS=(
    facebook/sam2.1-hiera-small
    facebook/sam2.1-hiera-base-plus
    facebook/sam2.1-hiera-large
)

n_pass=0
n_fail=0
failures=()

run() {
    local label="$1"
    shift
    printf '\n=== %s ===\n' "$label"
    if "$@"; then
        n_pass=$((n_pass + 1))
    else
        n_fail=$((n_fail + 1))
        failures+=("$label")
        printf '!!! FAILED: %s\n' "$label"
    fi
}

# ── Lane A: extraction only, every dataset x model x task ────────────────────
# Segmentation: --decoder conv_decoder covers the 2D cache (shared with
# linear_probe, which slices the last layer out of it at train time) and
# --decoder unetr covers the 3D volume cache.  A third linear_probe run would
# extract nothing new.
echo "########## Lane A: extraction (--cache-only, ${MAX_PATIENTS_EXTRACT} patients) ##########"

for dataset in $DATASETS; do
    for decoder in conv_decoder unetr; do
        for model in "${DINO_MODELS[@]}"; do
            run "extract seg $dataset dinov3/$model $decoder" \
                "$PYTHON" "$SEG" --dataset "$dataset" --backbone dinov3 \
                --dinov3-model-name "$model" --decoder "$decoder" \
                --cache-only --max-patients "$MAX_PATIENTS_EXTRACT" \
                --cache-dir "$CACHE_ROOT/seg/$dataset/$model/$decoder"
        done

        run "extract seg $dataset cinema $decoder" \
            "$PYTHON" "$SEG" --dataset "$dataset" --backbone cinema --decoder "$decoder" \
            --cache-only --max-patients "$MAX_PATIENTS_EXTRACT" \
            --cache-dir "$CACHE_ROOT/seg/$dataset/cinema/$decoder"

        for model in "${SAM_MODELS[@]}"; do
            tag="${model##*/}"
            run "extract seg $dataset sam/$tag $decoder" \
                "$PYTHON" "$SEG" --dataset "$dataset" --backbone sam \
                --sam-model-id "$model" --decoder "$decoder" \
                --cache-only --max-patients "$MAX_PATIENTS_EXTRACT" \
                --cache-dir "$CACHE_ROOT/seg/$dataset/$tag/$decoder"
        done

        for model in "${SAM2_MODELS[@]}"; do
            tag="${model##*/}"
            run "extract seg $dataset sam2/$tag $decoder" \
                "$PYTHON" "$SEG" --dataset "$dataset" --backbone sam2 \
                --sam2-model-id "$model" --decoder "$decoder" \
                --cache-only --max-patients "$MAX_PATIENTS_EXTRACT" \
                --cache-dir "$CACHE_ROOT/seg/$dataset/$tag/$decoder"
        done
    done

    # Classification: logreg and finetune share one cache, so extract once per
    # (model, pooling).  Both SAM families are gap-only (no CLS token).
    for pooling in cls gap; do
        for model in "${DINO_MODELS[@]}"; do
            run "extract cls $dataset dinov3/$model $pooling" \
                "$PYTHON" "$CLS" --dataset "$dataset" --backbone dinov3 \
                --dinov3-model-name "$model" --eval-mode logreg --pooling "$pooling" \
                --cache-only --max-patients "$MAX_PATIENTS_EXTRACT" \
                --cls-cache-dir "$CACHE_ROOT/cls/$dataset/$model/$pooling"
        done

        run "extract cls $dataset cinema $pooling" \
            "$PYTHON" "$CLS" --dataset "$dataset" --backbone cinema \
            --eval-mode logreg --pooling "$pooling" \
            --cache-only --max-patients "$MAX_PATIENTS_EXTRACT" \
            --cls-cache-dir "$CACHE_ROOT/cls/$dataset/cinema/$pooling"
    done

    for model in "${SAM_MODELS[@]}"; do
        tag="${model##*/}"
        run "extract cls $dataset sam/$tag gap" \
            "$PYTHON" "$CLS" --dataset "$dataset" --backbone sam \
            --sam-model-id "$model" --eval-mode logreg --pooling gap \
            --cache-only --max-patients "$MAX_PATIENTS_EXTRACT" \
            --cls-cache-dir "$CACHE_ROOT/cls/$dataset/$tag/gap"
    done

    for model in "${SAM2_MODELS[@]}"; do
        tag="${model##*/}"
        run "extract cls $dataset sam2/$tag gap" \
            "$PYTHON" "$CLS" --dataset "$dataset" --backbone sam2 \
            --sam2-model-id "$model" --eval-mode logreg --pooling gap \
            --cache-only --max-patients "$MAX_PATIENTS_EXTRACT" \
            --cls-cache-dir "$CACHE_ROOT/cls/$dataset/$tag/gap"
    done
done

# ── Lane B: one short training run per model, ACDC only ──────────────────────
# Decoder shapes depend on the backbone, not the dataset, so ACDC is enough.
if [[ "$TRAIN" == "1" ]]; then
    echo ""
    echo "########## Lane B: training (ACDC only) ##########"
    OUT="$CACHE_ROOT/results"

    for decoder in linear_probe conv_decoder unetr; do
        for model in "${DINO_MODELS[@]}"; do
            run "train seg dinov3/$model $decoder" \
                "$PYTHON" "$SEG" --dataset acdc --backbone dinov3 \
                --dinov3-model-name "$model" --decoder "$decoder" \
                --max-patients "$MAX_PATIENTS_SEG_TRAIN" --n-epochs 1 --patience 1 \
                --cache-dir "$CACHE_ROOT/segtrain/$model" --output-dir "$OUT/segmentation/acdc"
        done

        run "train seg cinema $decoder" \
            "$PYTHON" "$SEG" --dataset acdc --backbone cinema --decoder "$decoder" \
            --max-patients "$MAX_PATIENTS_SEG_TRAIN" --n-epochs 1 --patience 1 \
            --cache-dir "$CACHE_ROOT/segtrain/cinema" --output-dir "$OUT/segmentation/acdc"

        for model in "${SAM_MODELS[@]}"; do
            tag="${model##*/}"
            run "train seg sam/$tag $decoder" \
                "$PYTHON" "$SEG" --dataset acdc --backbone sam --sam-model-id "$model" \
                --decoder "$decoder" \
                --max-patients "$MAX_PATIENTS_SEG_TRAIN" --n-epochs 1 --patience 1 \
                --cache-dir "$CACHE_ROOT/segtrain/$tag" --output-dir "$OUT/segmentation/acdc"
        done

        for model in "${SAM2_MODELS[@]}"; do
            tag="${model##*/}"
            run "train seg sam2/$tag $decoder" \
                "$PYTHON" "$SEG" --dataset acdc --backbone sam2 --sam2-model-id "$model" \
                --decoder "$decoder" \
                --max-patients "$MAX_PATIENTS_SEG_TRAIN" --n-epochs 1 --patience 1 \
                --cache-dir "$CACHE_ROOT/segtrain/$tag" --output-dir "$OUT/segmentation/acdc"
        done
    done

    # Classification needs enough patients for stratified CV over all classes.
    for pooling in cls gap; do
        for model in "${DINO_MODELS[@]}"; do
            run "train cls dinov3/$model $pooling" \
                "$PYTHON" "$CLS" --dataset acdc --backbone dinov3 \
                --dinov3-model-name "$model" --eval-mode logreg --pooling "$pooling" \
                --max-patients "$MAX_PATIENTS_CLS_TRAIN" \
                --cls-cache-dir "$CACHE_ROOT/clstrain/$model/$pooling" \
                --output-dir "$OUT/classification/acdc"
        done

        run "train cls cinema $pooling" \
            "$PYTHON" "$CLS" --dataset acdc --backbone cinema \
            --eval-mode logreg --pooling "$pooling" \
            --max-patients "$MAX_PATIENTS_CLS_TRAIN" \
            --cls-cache-dir "$CACHE_ROOT/clstrain/cinema/$pooling" \
            --output-dir "$OUT/classification/acdc"
    done

    for model in "${SAM_MODELS[@]}"; do
        tag="${model##*/}"
        run "train cls sam/$tag gap" \
            "$PYTHON" "$CLS" --dataset acdc --backbone sam --sam-model-id "$model" \
            --eval-mode logreg --pooling gap --max-patients "$MAX_PATIENTS_CLS_TRAIN" \
            --cls-cache-dir "$CACHE_ROOT/clstrain/$tag/gap" \
            --output-dir "$OUT/classification/acdc"
    done

    for model in "${SAM2_MODELS[@]}"; do
        tag="${model##*/}"
        run "train cls sam2/$tag gap" \
            "$PYTHON" "$CLS" --dataset acdc --backbone sam2 --sam2-model-id "$model" \
            --eval-mode logreg --pooling gap --max-patients "$MAX_PATIENTS_CLS_TRAIN" \
            --cls-cache-dir "$CACHE_ROOT/clstrain/$tag/gap" \
            --output-dir "$OUT/classification/acdc"
    done
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "########## Summary ##########"
echo "passed: $n_pass"
echo "failed: $n_fail"
if ((n_fail > 0)); then
    printf 'failing configs:\n'
    printf '  - %s\n' "${failures[@]}"
    exit 1
fi
echo "Cache tree: $CACHE_ROOT (du -sh to check size; safe to delete)"
