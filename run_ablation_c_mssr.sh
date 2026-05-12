#!/bin/bash
# C-series MSSR ablations (full pipeline per variant)
#   train1_mssr → train2_mssr → precompute_mlmf_cache → train_tcgr_cv
#
# Ablation groups:
#   C1 — scan direction:  bidir(proposed) / forward / backward
#   C2 — d_state:         8 / 16(proposed) / 32
#   C3 — expand factor:   1 / 2(proposed) / 4
#   C4 — no MSSR control: MLMF only (no SSM scan)
#   C5 — MSSR position:   global-only(proposed) / global+local
#
# Usage:
#   bash run_ablation_c_mssr.sh          # run all groups
#   bash run_ablation_c_mssr.sh c1       # run only C1
#   bash run_ablation_c_mssr.sh c4       # run only C4 (no-MSSR control)

set -e
source /home/taotl/miniconda3/etc/profile.d/conda.sh
conda activate bmad
cd /home/taotl/Desktop/FM-OSD

LOG=logs/ablation_c
mkdir -p $LOG

DATASET=/home/taotl/Desktop/FM-OSD/dataset/Cephalometric/
MODELS=/home/taotl/Desktop/FM-OSD/models
OUTPUT=/home/taotl/Desktop/FM-OSD/output
FEAT_CACHE=/home/taotl/Desktop/FM-OSD/cache/feat_head

# ---------------------------------------------------------------------------
# Helper: run one C-variant end-to-end
#   $1  = variant tag (e.g. c1_bidir)
#   $2+ = extra args forwarded to train1_mssr.py
#
# Each variant:
#   Step 1 — train1_mssr  (global, 20000 iters)
#   Step 2 — train2_mssr  (local fine-tune, 50 iters, eval_freq=999999)
#   Step 3 — precompute_mlmf_cache  (cache predictions for TCGR)
#   Step 4 — train_tcgr_cv  (5-fold, 3000 iters)
# ---------------------------------------------------------------------------
run_c_variant() {
    local TAG=$1; shift
    local EXTRA_TRAIN1="$@"

    echo ""
    echo "========================================================"
    echo "C-variant: $TAG   extra_args: $EXTRA_TRAIN1"
    echo "========================================================"

    # ---- Step 1: global branch ----
    echo "[${TAG}] Step 1: train1_mssr (max 1000 iters, early stop patience=5)" | tee -a $LOG/${TAG}.log
    python -u train1_mssr.py \
        --dataset_pth    "$DATASET" \
        --save_dir       "$OUTPUT" \
        --mlmf_layers    5,8,11 \
        --mlmf_facets    key,value \
        --max_iterations 1000 \
        --eval_freq      100 \
        --early_stop_patience 5 \
        --exp            global_mssr_${TAG} \
        --feat_cache_dir "$FEAT_CACHE" \
        $EXTRA_TRAIN1 \
        2>&1 | tee -a $LOG/${TAG}.log

    # pick best global ckpt (lowest MRE in filename)
    GLOBAL_CKPT=$(ls ${MODELS}/global_mssr_${TAG}/model_post_mssr_iter_*_*.pth 2>/dev/null \
        | python3 -c "
import sys, re
files=[f.strip() for f in sys.stdin if re.search(r'iter_\d+_([0-9.]+)\.pth',f)]
print(min(files, key=lambda f: float(re.search(r'iter_\d+_([0-9.]+)\.pth',f).group(1)))) if files else exit(1)
")
    [ -z "$GLOBAL_CKPT" ] && { echo "ERROR: no global ckpt for $TAG"; exit 1; }
    echo "[${TAG}] Best global: $GLOBAL_CKPT"

    # ---- Step 2: local fine-tune ----
    echo "[${TAG}] Step 2: train2_mssr (50 iters)" | tee -a $LOG/${TAG}.log
    python -u train2_mssr.py \
        --dataset_pth "$DATASET" \
        --save_dir    "$OUTPUT" \
        --mlmf_layers 5,8,11 \
        --mlmf_facets key,value \
        --max_iterations 50 \
        --eval_freq 999999 \
        --exp local_mssr_${TAG} \
        --global_ckpt "$GLOBAL_CKPT" \
        $EXTRA_TRAIN1 \
        2>&1 | tee -a $LOG/${TAG}.log

    LOCAL_CKPT=${MODELS}/local_mssr_${TAG}/model_post_final.pth
    [ ! -f "$LOCAL_CKPT" ] && { echo "ERROR: local ckpt not found for $TAG"; exit 1; }

    # ---- Step 3: precompute cache ----
    echo "[${TAG}] Step 3: precompute_mlmf_cache" | tee -a $LOG/${TAG}.log
    python -u precompute_mlmf_cache.py \
        --mlmf_layers 5,8,11 \
        --mlmf_facets key,value \
        --ckpt        "$LOCAL_CKPT" \
        --cache_dir   data/tcgr_cache_mssr_${TAG} \
        --force False \
        2>&1 | tee -a $LOG/${TAG}.log

    # ---- Step 4: TCGR 5-fold CV ----
    echo "[${TAG}] Step 4: train_tcgr_cv (3000 iters)" | tee -a $LOG/${TAG}.log
    python -u train_tcgr_cv.py \
        --cache_dir      data/tcgr_cache_mssr_${TAG} \
        --save_dir       "$OUTPUT" \
        --exp            tcgr_cv_mssr_${TAG} \
        --max_iterations 3000 \
        --bs 32 --lr 1e-4 \
        --tcgr_num_layers 4 \
        --tcgr_hidden_dim 128 \
        --tcgr_use_attention True \
        2>&1 | tee -a $LOG/${TAG}.log

    echo "[${TAG}] Done — see $LOG/${TAG}.log"
}

# ---------------------------------------------------------------------------
# C4 control: MLMF only, no MSSR  (uses train1_mlmf / train2_mlmf)
# ---------------------------------------------------------------------------
run_c4_no_mssr() {
    local TAG=c4_no_mssr
    echo ""
    echo "========================================================"
    echo "C4 control: MLMF only (no MSSR)"
    echo "========================================================"

    echo "[${TAG}] Step 1: train1_mlmf (20000 iters)" | tee -a $LOG/${TAG}.log
    python -u train1_mlmf.py \
        --dataset_pth "$DATASET" \
        --save_dir    "$OUTPUT" \
        --mlmf_layers 5,8,11 \
        --mlmf_facets key,value \
        --max_iterations 20000 \
        --exp global_mlmf_${TAG} \
        2>&1 | tee -a $LOG/${TAG}.log

    GLOBAL_CKPT=$(ls ${MODELS}/global_mlmf_${TAG}/model_post_mlmf_iter_*_*.pth 2>/dev/null \
        | python3 -c "
import sys, re
files=[f.strip() for f in sys.stdin if re.search(r'iter_\d+_([0-9.]+)\.pth',f)]
print(min(files, key=lambda f: float(re.search(r'iter_\d+_([0-9.]+)\.pth',f).group(1)))) if files else exit(1)
")
    [ -z "$GLOBAL_CKPT" ] && { echo "ERROR: no global ckpt for $TAG"; exit 1; }

    echo "[${TAG}] Step 2: train2_mlmf (50 iters)" | tee -a $LOG/${TAG}.log
    python -u train2_mlmf.py \
        --dataset_pth "$DATASET" \
        --save_dir    "$OUTPUT" \
        --mlmf_layers 5,8,11 \
        --mlmf_facets key,value \
        --max_iterations 50 \
        --eval_freq 999999 \
        --exp local_mlmf_${TAG} \
        --global_ckpt "$GLOBAL_CKPT" \
        2>&1 | tee -a $LOG/${TAG}.log

    LOCAL_CKPT=${MODELS}/local_mlmf_${TAG}/model_post_final.pth
    [ ! -f "$LOCAL_CKPT" ] && { echo "ERROR: local ckpt not found for $TAG"; exit 1; }

    echo "[${TAG}] Step 3: precompute_mlmf_cache" | tee -a $LOG/${TAG}.log
    python -u precompute_mlmf_cache.py \
        --mlmf_layers 5,8,11 \
        --mlmf_facets key,value \
        --ckpt        "$LOCAL_CKPT" \
        --cache_dir   data/tcgr_cache_mssr_${TAG} \
        --force False \
        2>&1 | tee -a $LOG/${TAG}.log

    echo "[${TAG}] Step 4: train_tcgr_cv (3000 iters)" | tee -a $LOG/${TAG}.log
    python -u train_tcgr_cv.py \
        --cache_dir      data/tcgr_cache_mssr_${TAG} \
        --save_dir       "$OUTPUT" \
        --exp            tcgr_cv_mssr_${TAG} \
        --max_iterations 3000 \
        --bs 32 --lr 1e-4 \
        --tcgr_num_layers 4 \
        --tcgr_hidden_dim 128 \
        --tcgr_use_attention True \
        2>&1 | tee -a $LOG/${TAG}.log

    echo "[${TAG}] Done"
}

# ---------------------------------------------------------------------------
# Dispatch: run all groups, or a specific one via $1
# ---------------------------------------------------------------------------
GROUP=${1:-all}

run_c1() {
    # C1: scan direction
    run_c_variant c1_bidir    --mssr_direction bidir    --mssr_d_state 16 --mssr_expand 2
    run_c_variant c1_forward  --mssr_direction forward  --mssr_d_state 16 --mssr_expand 2
    run_c_variant c1_backward --mssr_direction backward --mssr_d_state 16 --mssr_expand 2
}

run_c2() {
    # C2: d_state
    run_c_variant c2_dstate8  --mssr_direction bidir --mssr_d_state 8  --mssr_expand 2
    # c2_dstate16 == proposed (c1_bidir), skip re-running
    run_c_variant c2_dstate32 --mssr_direction bidir --mssr_d_state 32 --mssr_expand 2
}

run_c3() {
    # C3: expand factor
    run_c_variant c3_expand1 --mssr_direction bidir --mssr_d_state 16 --mssr_expand 1
    # c3_expand2 == proposed (c1_bidir), skip re-running
    run_c_variant c3_expand4 --mssr_direction bidir --mssr_d_state 16 --mssr_expand 4
}

run_c5() {
    # C5: MSSR on local branch too
    run_c_variant c5_global_local \
        --mssr_direction bidir --mssr_d_state 16 --mssr_expand 2 --mssr_local True
}

case "$GROUP" in
    c1)   run_c1 ;;
    c2)   run_c2 ;;
    c3)   run_c3 ;;
    c4)   run_c4_no_mssr ;;
    c5)   run_c5 ;;
    all)
        run_c1
        run_c2
        run_c3
        run_c4_no_mssr
        run_c5
        ;;
    *)
        echo "Unknown group: $GROUP. Use c1/c2/c3/c4/c5/all"
        exit 1
        ;;
esac

echo ""
echo "==== C-series MSSR ablations done ===="
echo "Logs: $LOG/"
echo "Results: $OUTPUT/tcgr_cv_mssr_*"
