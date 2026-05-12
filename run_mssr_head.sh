#!/bin/bash
# Full MLMF + MSSR + TCGR pipeline for Cephalometric (head) dataset
# Steps:
#   0. precompute_head_features.py → cache/feat_head/  (one-time, ~5 min)
#   1. train1_mssr.py   → models/global_mssr/
#   2. train2_mssr.py   → models/local_mssr/
#   3. precompute_mlmf_cache.py   → data/tcgr_cache_mssr/
#   4. train_tcgr_cv.py → 5-fold TCGR CV results

CONDA_ENV="bmad"
PYTHON="conda run -n ${CONDA_ENV} python -u"
LOG_DIR="/home/taotl/Desktop/FM-OSD/logs/mssr_head"
mkdir -p "$LOG_DIR"

BASE="/home/taotl/Desktop/FM-OSD"
DATASET_PTH="${BASE}/dataset/Cephalometric/"
MODEL_DIR="${BASE}/models"
CACHE_DIR="${BASE}/data/tcgr_cache_mssr"
FEAT_CACHE_DIR="${BASE}/cache/feat_head"
OUTPUT_DIR="${BASE}/output"
MODEL_TYPE="dino_vits8"

echo "============================================"
echo "[mssr-head] Pipeline start: $(date)"
echo "============================================"

# ── Step 0: Precompute MLMF backbone features (skip if already done) ─────────
FEAT_DONE="${FEAT_CACHE_DIR}/${MODEL_TYPE}/shot_125.pt"
if [ -f "$FEAT_DONE" ]; then
    echo "[mssr-head] Step 0: skipped (feat_head cache already exists)"
else
    echo "[mssr-head] Step 0: precompute_head_features (one-time, ~5 min)"
    $PYTHON "${BASE}/precompute_head_features.py" \
        --model_type   "${MODEL_TYPE}" \
        --load_size    224 \
        --stride       4 \
        --bin          True \
        --mlmf_layers  5,8,11 \
        --mlmf_facets  key,value \
        --dataset_pth  "${DATASET_PTH}" \
        --train_patch_dir "${BASE}/data/head/image" \
        --cache_dir    "${FEAT_CACHE_DIR}" \
        --id_shot      125 \
        2>&1 | tee "${LOG_DIR}/step0_precompute.log"
    echo "[mssr-head] Step 0 done."
fi

# ── Step 1: Train global MSSR model ──────────────────────────────────────────
echo "[mssr-head] Step 1: train1_mssr (max 1000 iters, early stop patience=5)"
$PYTHON "${BASE}/train1_mssr.py" \
    --dataset_pth  "${DATASET_PTH}" \
    --save_dir     "${OUTPUT_DIR}" \
    --load_size    224 \
    --id_shot      125 \
    --bin          True \
    --mlmf_layers  5,8,11 \
    --mlmf_facets  key,value \
    --mssr_d_state 16 \
    --mssr_expand  2 \
    --max_iterations 1000 \
    --eval_freq    100 \
    --early_stop_patience 5 \
    --exp          global_mssr \
    --feat_cache_dir "${FEAT_CACHE_DIR}" \
    2>&1 | tee "${LOG_DIR}/step1_train1.log"

# Find best global checkpoint
GLOBAL_CKPT=$(ls "${MODEL_DIR}/global_mssr"/model_post_mssr_iter_*_*.pth 2>/dev/null | python3 -c "
import sys, re
files = [f.strip() for f in sys.stdin.read().split('\n') if re.search(r'iter_\d+_([0-9.]+)\.pth', f)]
if not files: exit(1)
best = min(files, key=lambda f: float(re.search(r'iter_\d+_([0-9.]+)\.pth', f).group(1)))
print(best)
")
if [ -z "$GLOBAL_CKPT" ]; then
    echo "[mssr-head] ERROR: no global checkpoint found. Exiting."
    exit 1
fi
echo "[mssr-head] Best global: $GLOBAL_CKPT"

# ── Step 2: Fine-tune local model ─────────────────────────────────────────────
echo "[mssr-head] Step 2: train2_mssr (50 iters)"
$PYTHON "${BASE}/train2_mssr.py" \
    --dataset_pth  "${DATASET_PTH}" \
    --save_dir     "${OUTPUT_DIR}" \
    --load_size    224 \
    --id_shot      125 \
    --bin          True \
    --mlmf_layers  5,8,11 \
    --mlmf_facets  key,value \
    --mssr_d_state 16 \
    --mssr_expand  2 \
    --max_iterations 50 \
    --exp          local_mssr \
    --global_ckpt  "${GLOBAL_CKPT}" \
    2>&1 | tee "${LOG_DIR}/step2_train2.log"

# Find best local checkpoint
LOCAL_CKPT=$(ls "${MODEL_DIR}/local_mssr"/model_post_fine_iter_*_*.pth 2>/dev/null | python3 -c "
import sys, re
files = [f.strip() for f in sys.stdin.read().split('\n') if re.search(r'iter_\d+_([0-9.]+)\.pth', f)]
if not files: exit(1)
best = min(files, key=lambda f: float(re.search(r'iter_\d+_([0-9.]+)\.pth', f).group(1)))
print(best)
")
if [ -z "$LOCAL_CKPT" ]; then
    echo "[mssr-head] ERROR: no local checkpoint found. Exiting."
    exit 1
fi
echo "[mssr-head] Best local: $LOCAL_CKPT"

# ── Step 3: Precompute MSSR predictions for TCGR ──────────────────────────────
echo "[mssr-head] Step 3: precompute_mlmf_cache (MSSR ckpt)"
$PYTHON "${BASE}/precompute_mlmf_cache.py" \
    --mlmf_layers 5,8,11 \
    --mlmf_facets key,value \
    --ckpt        "${LOCAL_CKPT}" \
    --cache_dir   "${CACHE_DIR}" \
    --force       False \
    2>&1 | tee "${LOG_DIR}/step3_precompute.log"
echo "[mssr-head] Step 3 done."

# ── Step 4: TCGR 5-fold CV ────────────────────────────────────────────────────
echo "[mssr-head] Step 4: train_tcgr_cv (5-fold, 3000 iters)"
$PYTHON "${BASE}/train_tcgr_cv.py" \
    --cache_dir      "${CACHE_DIR}" \
    --save_dir       "${OUTPUT_DIR}" \
    --exp            tcgr_cv_mssr \
    --max_iterations 3000 \
    --bs 32 --lr 1e-4 \
    --tcgr_num_layers    4 \
    --tcgr_hidden_dim    128 \
    --tcgr_use_attention True \
    2>&1 | tee "${LOG_DIR}/step4_tcgr_cv.log"

echo "============================================"
echo "[mssr-head] Pipeline complete: $(date)"
echo "Results: ${OUTPUT_DIR}/tcgr_cv_mssr*"
echo "============================================"
