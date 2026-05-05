#!/bin/bash
# A-series MLMF feature ablations: full pipeline per variant
#   train2_mlmf (local fine-tune) -> precompute_mlmf_cache -> train_tcgr_cv
set -e
source /home/taotl/miniconda3/etc/profile.d/conda.sh
conda activate bmad
cd /home/taotl/Desktop/FM-OSD

LOG=logs/ablation_a
mkdir -p $LOG

# ---------------------------------------------------------------------------
# Helper: run one A-variant end-to-end
#   $1 = variant tag (a2/a3/a5)
#   $2 = mlmf_layers
#   $3 = mlmf_facets
#   $4 = global ckpt path
# ---------------------------------------------------------------------------
run_a_variant() {
  local TAG=$1
  local LAYERS=$2
  local FACETS=$3
  local GLOBAL_CKPT=$4

  echo ""
  echo "========================================================"
  echo "A-variant: $TAG  layers=$LAYERS  facets=$FACETS"
  echo "========================================================"

  # ---- Step 1: Local fine-tune ----
  echo "[${TAG}] Step 1: train2_mlmf (local fine-tune, 20 iters)" | tee -a $LOG/${TAG}.log
  python -u train2_mlmf.py \
    --mlmf_layers "$LAYERS" \
    --mlmf_facets "$FACETS" \
    --global_ckpt "$GLOBAL_CKPT" \
    --exp local_mlmf_${TAG} \
    --max_iterations 20 \
    2>&1 | tee -a $LOG/${TAG}.log
  echo "[${TAG}] Step 1 done"

  LOCAL_CKPT=models/local_mlmf_${TAG}/model_post_final.pth
  if [ ! -f "$LOCAL_CKPT" ]; then
    echo "ERROR: local checkpoint not found at $LOCAL_CKPT"
    exit 1
  fi

  # ---- Step 2: Precompute MLMF cache ----
  echo "[${TAG}] Step 2: precompute_mlmf_cache" | tee -a $LOG/${TAG}.log
  python -u precompute_mlmf_cache.py \
    --mlmf_layers "$LAYERS" \
    --mlmf_facets "$FACETS" \
    --ckpt "$LOCAL_CKPT" \
    --cache_dir data/tcgr_cache_mlmf_${TAG} \
    --force False \
    2>&1 | tee -a $LOG/${TAG}.log
  echo "[${TAG}] Step 2 done"

  # ---- Step 3: TCGR 5-fold CV ----
  echo "[${TAG}] Step 3: train_tcgr_cv" | tee -a $LOG/${TAG}.log
  python -u train_tcgr_cv.py \
    --cache_dir data/tcgr_cache_mlmf_${TAG} \
    --exp tcgr_cv_${TAG} \
    --max_iterations 3000 \
    2>&1 | tee -a $LOG/${TAG}.log
  echo "[${TAG}] Step 3 done - see logs/ablation_a/${TAG}.log"
}

# ---------------------------------------------------------------------------
# A2: multi-layer (5,8,11) key-only
# ---------------------------------------------------------------------------
run_a_variant a2 "5,8,11" "key" \
  "models/global_mlmf_a2_multilayer_keyonly/model_post_mlmf_iter_2000_1.9884305318409219.pth"

# ---------------------------------------------------------------------------
# A3: single-layer (8) key+value
# ---------------------------------------------------------------------------
run_a_variant a3 "8" "key,value" \
  "models/global_mlmf_a3_singlelayer_kv/model_post_mlmf_iter_1500_1.9488647771800263.pth"

# ---------------------------------------------------------------------------
# A5: full MLMF layers (5,8,11) key+value with uniform (non-adaptive) fusion
#     Local fine-tune starts from the uniform-init global ckpt; source_scale
#     will be fine-tuned from the uniform starting point.
# ---------------------------------------------------------------------------
run_a_variant a5 "5,8,11" "key,value" \
  "models/global_mlmf_a5_uniform/model_post_mlmf_iter_2000_1.9642905504771004.pth"

echo ""
echo "==== All A-series ablations done ===="
