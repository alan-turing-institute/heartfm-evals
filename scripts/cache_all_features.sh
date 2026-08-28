#!/usr/bin/env bash
# Extract and cache features for every model x dataset x task, without training.
#
# SAM2 is deliberately excluded -- SAM v1 only.
#
# Enumerates *cache keys*, not experiments, because several experiments share one
# cache:
#
#   segmentation  linear_probe and conv_decoder read the same 2D cache (the probe
#                 slices the layer it wants out of it at train time), so only
#                 --decoder conv_decoder is run for the 2D cache.  --decoder unetr
#                 covers the separate 3D volume cache.
#   classification  logreg and finetune read the same cache, so only
#                 --eval-mode logreg is run.  Both are per (model, pooling).
#
# That makes 75 runs instead of the 141 the full experiment grid would need:
#
#   segmentation    7 models x 2 cache families x 3 datasets = 42
#   classification  11 keys              x 3 datasets        = 33
#
# Caches are written to the default locations:
#   feature_cache_segmentation/{dataset}/{model}[_unetr3d]/{split}/
#   feature_cache_classification/{dataset}/{model}/{pooling}/{split}/
#
# DISK: the full grid is roughly 600 GB (~190 GB of 2D caches, ~430 GB of 3D
# volume caches; classification is ~1 GB).  Check free space before starting, and
# consider going one dataset at a time with DATASETS=.
#
# RESUMABLE: every cache function skips files that already exist, so re-running
# after an interruption picks up where it stopped.  The flip side is that a stale
# file is never overwritten -- delete a cache directory to force regeneration.
#
# A failing config is recorded and the run continues; the exit status is non-zero
# if anything failed.
#
# Usage:
#   bash scripts/cache_all_features.sh                        # everything
#   DATASETS="acdc" bash scripts/cache_all_features.sh        # one dataset
#   TASKS="seg" bash scripts/cache_all_features.sh            # segmentation only
#   DRY_RUN=1 bash scripts/cache_all_features.sh              # print, do not run
#
# Env overrides:
#   DATASETS   space-separated subset of "acdc mnm mnm2"   (default: all three)
#   TASKS      space-separated subset of "seg cls"          (default: both)
#   LOG_DIR    per-config logs (default: logs/cache_all)
#   PYTHON     interpreter to use                           (default: python)
#   DRY_RUN    set to 1 to print the commands without running them
#
# NOTE: do not pass a non-default --seed.  ACDC has no val_metadata.csv, so its
# validation split is carved out of train by seed, and the cache path does not
# record the seed -- extracting different models under different seeds would put
# the same patient in train/ for one and val/ for another.

set -euo pipefail

DATASETS=${DATASETS:-"acdc mnm mnm2"}
TASKS=${TASKS:-"seg cls"}
LOG_DIR=${LOG_DIR:-logs/cache_all}
PYTHON=${PYTHON:-python}
DRY_RUN=${DRY_RUN:-0}

SEG="scripts/segmentation/run_segmentation.py"
CLS="scripts/classification/run_classification.py"

DINO_MODELS=(dinov3_vits16 dinov3_vitb16 dinov3_vitl16)
SAM_MODELS=(facebook/sam-vit-base facebook/sam-vit-large facebook/sam-vit-huge)

mkdir -p "$LOG_DIR"

n_pass=0
n_fail=0
n_total=0
failures=()

# run <label> <command...>
run() {
    local label="$1"
    shift
    n_total=$((n_total + 1))

    if [[ "$DRY_RUN" == "1" ]]; then
        printf '[dry-run] %s\n           %s\n' "$label" "$*"
        return 0
    fi

    local log_file="$LOG_DIR/$(echo "$label" | tr ' /' '__').log"
    printf '\n=== [%s] %s ===\n' "$(date +%H:%M:%S)" "$label"

    if "$@" >"$log_file" 2>&1; then
        n_pass=$((n_pass + 1))
        printf '    ok  (log: %s)\n' "$log_file"
    else
        n_fail=$((n_fail + 1))
        failures+=("$label")
        printf '    !!! FAILED -- see %s\n' "$log_file"
        tail -n 15 "$log_file" | sed 's/^/    | /'
    fi
}

# ── Segmentation ─────────────────────────────────────────────────────────────
# conv_decoder -> the 2D slice cache (also used by linear_probe)
# unetr        -> the 3D volume cache
cache_segmentation() {
    local dataset="$1"

    for decoder in conv_decoder unetr; do
        for model in "${DINO_MODELS[@]}"; do
            run "seg $dataset dinov3/$model $decoder" \
                "$PYTHON" "$SEG" \
                --dataset "$dataset" \
                --backbone dinov3 \
                --dinov3-model-name "$model" \
                --decoder "$decoder" \
                --cache-only
        done

        run "seg $dataset cinema $decoder" \
            "$PYTHON" "$SEG" \
            --dataset "$dataset" \
            --backbone cinema \
            --decoder "$decoder" \
            --cache-only

        for model in "${SAM_MODELS[@]}"; do
            run "seg $dataset sam/${model##*/} $decoder" \
                "$PYTHON" "$SEG" \
                --dataset "$dataset" \
                --backbone sam \
                --sam-model-id "$model" \
                --decoder "$decoder" \
                --cache-only
        done
    done
}

# ── Classification ───────────────────────────────────────────────────────────
# One cache per (model, pooling); logreg and finetune share it.
# SAM v1 has no CLS token, so it is gap only -- the driver rejects --pooling cls.
cache_classification() {
    local dataset="$1"

    for pooling in cls gap; do
        for model in "${DINO_MODELS[@]}"; do
            run "cls $dataset dinov3/$model $pooling" \
                "$PYTHON" "$CLS" \
                --dataset "$dataset" \
                --backbone dinov3 \
                --dinov3-model-name "$model" \
                --eval-mode logreg \
                --pooling "$pooling" \
                --cache-only
        done

        run "cls $dataset cinema $pooling" \
            "$PYTHON" "$CLS" \
            --dataset "$dataset" \
            --backbone cinema \
            --eval-mode logreg \
            --pooling "$pooling" \
            --cache-only
    done

    for model in "${SAM_MODELS[@]}"; do
        run "cls $dataset sam/${model##*/} gap" \
            "$PYTHON" "$CLS" \
            --dataset "$dataset" \
            --backbone sam \
            --sam-model-id "$model" \
            --eval-mode logreg \
            --pooling gap \
            --cache-only
    done
}

# ── Main ─────────────────────────────────────────────────────────────────────
started_at=$(date +%s)
echo "datasets: $DATASETS"
echo "tasks:    $TASKS"
echo "logs:     $LOG_DIR"
[[ "$DRY_RUN" == "1" ]] && echo "mode:     DRY RUN (nothing will be executed)"

for dataset in $DATASETS; do
    echo ""
    echo "########## dataset: $dataset ##########"
    for task in $TASKS; do
        case "$task" in
            seg) cache_segmentation "$dataset" ;;
            cls) cache_classification "$dataset" ;;
            *) echo "Unknown task: $task (expected 'seg' or 'cls')" >&2; exit 2 ;;
        esac
    done
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "########## Summary ##########"
if [[ "$DRY_RUN" == "1" ]]; then
    echo "configs that would run: $n_total"
    exit 0
fi

elapsed=$(( $(date +%s) - started_at ))
printf 'configs: %d   passed: %d   failed: %d   elapsed: %dh%02dm\n' \
    "$n_total" "$n_pass" "$n_fail" $((elapsed / 3600)) $(((elapsed % 3600) / 60))

if ((n_fail > 0)); then
    echo "failing configs:"
    printf '  - %s\n' "${failures[@]}"
    echo "Re-run the script to retry them; completed caches are skipped."
    exit 1
fi

echo "All caches written."
du -sh feature_cache_segmentation feature_cache_classification 2>/dev/null || true
