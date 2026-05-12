#!/bin/bash
# Full MLMF + TCGR pipeline for Hand X-ray dataset (37 landmarks, 909 images)
# Steps:
#   0. data_generate_hand.py  → data/hand/ (500 augmented template patches)
#   1. train1_mlmf_hand.py    → models/global_mlmf_hand/
#   2. train2_mlmf_hand.py    → models/local_mlmf_hand/
#   3. precompute_hand_cache.py → data/tcgr_cache_hand/{train,test}/
#   4. train_tcgr_cv.py (hand)  → 5-fold TCGR CV results

CONDA_ENV="bmad"
PYTHON="conda run -n ${CONDA_ENV} python -u"
LOG_DIR="/home/taotl/Desktop/FM-OSD/logs/hand"
mkdir -p "$LOG_DIR"

BASE="/home/taotl/Desktop/FM-OSD"
DATASET_PTH="${BASE}/dataset/Hand/hand/"
MODEL_DIR="${BASE}/models"
CACHE_DIR="${BASE}/data/tcgr_cache_hand"
FEAT_CACHE_DIR="${BASE}/cache/feat_hand"
OUTPUT_DIR="${BASE}/output"
MODEL_TYPE="dino_vits8"

echo "============================================"
echo "[hand] Pipeline start: $(date)"
echo "============================================"

# ── Step 0: Generate augmented template patches ───────────────────────────
echo "[hand] Step 0: data_generate_hand (500 patches)"
$PYTHON "${BASE}/data_generate_hand.py" \
    --dataset_pth "${DATASET_PTH}" \
    --id_shot 0 \
    --max_iter 500 \
    2>&1 | tee "${LOG_DIR}/step0_datagen.log"
echo "[hand] Step 0 done."

# ── Step 0.5: Precompute MLMF backbone features ───────────────────────────
echo "[hand] Step 0.5: precompute_hand_features (MLMF backbone encode)"
$PYTHON "${BASE}/precompute_hand_features.py" \
    --model_type "${MODEL_TYPE}" \
    --load_size 224 \
    --stride 4 \
    --bin True \
    --mlmf_layers 5,8,11 \
    --mlmf_facets key,value \
    --dataset_pth "${DATASET_PTH}" \
    --train_patch_dir "${BASE}/data/hand/image" \
    --cache_dir "${FEAT_CACHE_DIR}" \
    --id_shot 0 \
    2>&1 | tee "${LOG_DIR}/step0_precompute.log"
echo "[hand] Step 0.5 done."

# ── Step 1: Train global MLMF model ──────────────────────────────────────
echo "[hand] Step 1: train1_mlmf_hand (8000 iters)"
$PYTHON "${BASE}/train1_mlmf_hand.py" \
    --dataset_pth "${DATASET_PTH}" \
    --save_dir "${OUTPUT_DIR}" \
    --load_size 224 \
    --id_shot 0 \
    --layer 8 \
    --facet key \
    --bin True \
    --mlmf_layers 5,8,11 \
    --mlmf_facets key,value \
    --max_iterations 8000 \
    --exp global_mlmf_hand \
    --feat_cache_dir "${FEAT_CACHE_DIR}" \
    2>&1 | tee "${LOG_DIR}/step1_train1.log"

# Find best global checkpoint
GLOBAL_CKPT=$(ls "${MODEL_DIR}/global_mlmf_hand"/model_post_iter_*.pth 2>/dev/null | python3 -c "
import sys, re
files = [f for f in sys.stdin.read().split() if re.search(r'iter_\d+_([0-9.]+)\.pth', f)]
if not files: exit(1)
best = min(files, key=lambda f: float(re.search(r'iter_\d+_([0-9.]+)\.pth', f).group(1)))
print(best)
")
if [ -z "$GLOBAL_CKPT" ]; then
    echo "[hand] ERROR: no global checkpoint found. Exiting."
    exit 1
fi
echo "[hand] Best global: $GLOBAL_CKPT"

# ── Step 2: Fine-tune local model ─────────────────────────────────────────
echo "[hand] Step 2: train2_mlmf_hand (100 iters)"
$PYTHON "${BASE}/train2_mlmf_hand.py" \
    --dataset_pth "${DATASET_PTH}" \
    --save_dir "${OUTPUT_DIR}" \
    --load_size 224 \
    --id_shot 0 \
    --layer 8 \
    --facet key \
    --bin True \
    --mlmf_layers 5,8,11 \
    --mlmf_facets key,value \
    --max_iterations 100 \
    --exp local_mlmf_hand \
    --global_ckpt "${GLOBAL_CKPT}" \
    2>&1 | tee "${LOG_DIR}/step2_train2.log"

# Find best local checkpoint
LOCAL_CKPT=$(ls "${MODEL_DIR}/local_mlmf_hand"/model_post_fine_iter_*.pth 2>/dev/null | python3 -c "
import sys, re
files = [f for f in sys.stdin.read().split() if re.search(r'iter_\d+_([0-9.]+)\.pth', f)]
if not files: exit(1)
best = min(files, key=lambda f: float(re.search(r'iter_\d+_([0-9.]+)\.pth', f).group(1)))
print(best)
")
if [ -z "$LOCAL_CKPT" ]; then
    echo "[hand] ERROR: no local checkpoint found. Exiting."
    exit 1
fi
echo "[hand] Best local: $LOCAL_CKPT"

# ── Step 3: Precompute MLMF predictions for TCGR ──────────────────────────
echo "[hand] Step 3: precompute_hand_cache"
$PYTHON "${BASE}/precompute_hand_cache.py" \
    --hand_path "${DATASET_PTH}" \
    --load_size 224 \
    --mlmf_layers 5,8,11 \
    --mlmf_facets key,value \
    --local_model_path "${LOCAL_CKPT}" \
    --cache_dir "${CACHE_DIR}" \
    --oneshot_idx 0 \
    2>&1 | tee "${LOG_DIR}/step3_precompute.log"
echo "[hand] Step 3 done."

# ── Step 4: TCGR 5-fold CV on Hand ────────────────────────────────────────
echo "[hand] Step 4: train_tcgr_cv (hand, 5-fold, 3000 iters)"
$PYTHON "${BASE}/train_tcgr_cv.py" \
    --dataset hand \
    --cache_dir "${CACHE_DIR}" \
    --save_dir "${OUTPUT_DIR}" \
    --exp tcgr_cv_hand \
    --n_folds 5 \
    --max_iterations 3000 \
    --bs 32 \
    --lr 1e-4 \
    --tcgr_num_layers 4 \
    --tcgr_hidden_dim 128 \
    --tcgr_use_attention True \
    2>&1 | tee "${LOG_DIR}/step4_tcgr_cv.log"

echo "============================================"
echo "[hand] Pipeline complete: $(date)"
echo "Results: ${OUTPUT_DIR}/tcgr_cv_hand*"
echo "============================================"
