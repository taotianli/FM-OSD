#!/bin/bash
# Backbone ablation experiment: compare DINO-s, DINO-b, DINOv2-s, DINOv2-b, SAM-B, SAM-L
# Pipeline per backbone:
#   Step 0: Precompute features (one-shot + test + train patches) → cache/feat/<model_type>/
#   Step 1: train1 (global, 9450 iters) — loads cache, very fast
#   Step 2: train2 (local fine-tune, 20 iters)
#   Step 3: test — loads cache
# Results saved to output/<backbone_name>/
# Uses conda environment: bmad

CONDA_ENV="bmad"
PYTHON="conda run -n ${CONDA_ENV} python"
LOG_DIR="/home/taotl/Desktop/FM-OSD/logs/backbone_ablation"
CACHE_BASE="/home/taotl/Desktop/FM-OSD/cache/feat"
mkdir -p "$LOG_DIR"

BASE_ARGS="--dataset_pth /home/taotl/Desktop/FM-OSD/dataset/Cephalometric/ \
           --save_dir /home/taotl/Desktop/FM-OSD/output \
           --load_size 224 --id_shot 125 --layer 8 --facet key --bin True \
           --num_workers 0"

# ────────────────────────────────────────────────────────────────────────────
# Helper: run a backbone variant sequentially
# Usage: run_backbone <name> <model_type>
# ────────────────────────────────────────────────────────────────────────────
run_backbone() {
    local NAME=$1
    local MODEL_TYPE=$2
    local FEAT_CACHE="${CACHE_BASE}/${MODEL_TYPE}"

    echo "======================================================"
    echo "[backbone_ablation] START: $NAME  ($MODEL_TYPE)"
    echo "======================================================"

    # ── Step 0: Precompute features ────────────────────────────────────────
    echo "[${NAME}] Step 0: precompute features → ${FEAT_CACHE}"
    $PYTHON precompute_backbone_features.py \
        --model_type "$MODEL_TYPE" \
        --dataset_pth /home/taotl/Desktop/FM-OSD/dataset/Cephalometric/ \
        --train_patch_dir /home/taotl/Desktop/FM-OSD/data/head/image \
        --cache_dir "${CACHE_BASE}" \
        --load_size 224 --id_shot 125 --layer 8 --facet key --bin True \
        2>&1 | tee "${LOG_DIR}/${NAME}_precompute.log"
    echo "[${NAME}] Precompute done."

    # ── Step 1: Train global model ─────────────────────────────────────────
    echo "[${NAME}] Step 1: train1 (global, 9450 iters, using feature cache)"
    $PYTHON -u train1.py $BASE_ARGS \
        --model_type "$MODEL_TYPE" \
        --exp "global_${NAME}" \
        --max_iterations 9450 \
        --feat_cache_dir "${FEAT_CACHE}" \
        2>&1 | tee "${LOG_DIR}/${NAME}_train1.log"

    # Find the best global checkpoint (lowest MRE)
    GLOBAL_CKPT=$(ls /home/taotl/Desktop/FM-OSD/models/global_${NAME}/model_post_iter_*.pth \
                  2>/dev/null | python3 -c "
import sys, re
files = [f for f in sys.stdin.read().split() if re.search(r'iter_\d+_([0-9.]+)\.pth', f)]
if not files:
    print('', end=''); exit(1)
best = min(files, key=lambda f: float(re.search(r'iter_\d+_([0-9.]+)\.pth', f).group(1)))
print(best)
")
    if [ -z "$GLOBAL_CKPT" ]; then
        echo "[${NAME}] ERROR: no global checkpoint found, skipping."
        return 1
    fi
    echo "[${NAME}] Best global checkpoint: $GLOBAL_CKPT"

    # ── Step 2: Fine-tune local model ─────────────────────────────────────
    echo "[${NAME}] Step 2: train2 (local fine-tune, 20 iters)"
    $PYTHON -u train2.py $BASE_ARGS \
        --model_type "$MODEL_TYPE" \
        --exp "local_${NAME}" \
        --max_iterations 20 \
        2>&1 | tee "${LOG_DIR}/${NAME}_train2.log"

    LOCAL_CKPT=$(ls /home/taotl/Desktop/FM-OSD/models/local_${NAME}/model_post_fine_iter_*.pth \
                 2>/dev/null | python3 -c "
import sys, re
files = [f for f in sys.stdin.read().split() if re.search(r'iter_\d+_([0-9.]+)\.pth', f)]
if not files:
    print('', end=''); exit(1)
best = min(files, key=lambda f: float(re.search(r'iter_\d+_([0-9.]+)\.pth', f).group(1)))
print(best)
")
    if [ -z "$LOCAL_CKPT" ]; then
        echo "[${NAME}] ERROR: no local checkpoint found, skipping test."
        return 1
    fi
    echo "[${NAME}] Best local checkpoint: $LOCAL_CKPT"

    # ── Step 3: Evaluate ──────────────────────────────────────────────────
    echo "[${NAME}] Step 3: test (using feature cache)"
    $PYTHON -u test.py $BASE_ARGS \
        --model_type "$MODEL_TYPE" \
        --exp "$NAME" \
        --global_model_path "$GLOBAL_CKPT" \
        --local_model_path  "$LOCAL_CKPT" \
        --feat_cache_dir "${FEAT_CACHE}" \
        2>&1 | tee "${LOG_DIR}/${NAME}_test.log"

    echo "[${NAME}] DONE. Results in output/${NAME}/"
}

# ────────────────────────────────────────────────────────────────────────────
# Run each backbone sequentially
# DINO-s: baseline — already done (MRE 1.819)
# DINO-b:    dino_vitb8,    stride=4, in_channels=13056
# DINOv2-s:  dinov2_vits14, stride=7, in_channels=6528
# DINOv2-b:  dinov2_vitb14, stride=7, in_channels=13056
# SAM-B:     sam_vit_b,     stride=16, in_channels=variable
# SAM-L:     sam_vit_l,     stride=16, in_channels=variable
# ────────────────────────────────────────────────────────────────────────────

run_backbone "dino_b"     "dino_vitb8"
run_backbone "dinov2_s"   "dinov2_vits14"
run_backbone "dinov2_b"   "dinov2_vitb14"
run_backbone "sam_b"      "sam_vit_b"
run_backbone "sam_l"      "sam_vit_l"

echo "======================================================"
echo "[backbone_ablation] All backbones finished."
echo "Summary CSV locations:"
for NAME in dino_s dino_b dinov2_s dinov2_b sam_b sam_l; do
    echo "  $NAME: output/${NAME}/*.csv"
done
echo "======================================================"
