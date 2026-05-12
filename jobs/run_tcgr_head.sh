#!/bin/bash
#SBATCH --job-name=fmosd_tcgr
#SBATCH --output=/home/u6da/taotl.u6da/FM-OSD/logs/%j_tcgr.log
#SBATCH --error=/home/u6da/taotl.u6da/FM-OSD/logs/%j_tcgr.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

WORK=/home/u6da/taotl.u6da/FM-OSD
CACHE=/projects/u6da/fmosd_cache
DATA=/home/u6da/taotl.u6da/FM-OSD/data/Cephalometric

mkdir -p $CACHE/tcgr_cache_mlmf
mkdir -p $WORK/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate fmosd
cd $WORK

# ── Step 1: Precompute TCGR input cache (model predictions) ──────────────────
echo "=== Precompute MLMF cache for TCGR ==="
GLOBAL_CKPT=$(ls $WORK/models/global_mssr/model_post_mssr_iter_*_*.pth 2>/dev/null | sort -t_ -k6 -rn | head -1)
LOCAL_CKPT=$(ls $WORK/models/local_mssr/model_post_mssr_iter_*_*.pth 2>/dev/null | sort -t_ -k6 -rn | head -1)

python precompute_mlmf_cache.py \
    --dataset_pth $DATA \
    --global_ckpt $GLOBAL_CKPT \
    --local_ckpt $LOCAL_CKPT \
    --cache_dir $CACHE/tcgr_cache_mlmf \
    --bin False

# ── Step 2: Train TCGR (5-fold CV) ───────────────────────────────────────────
echo "=== Train TCGR (5-fold CV) ==="
python train_tcgr_cv.py \
    --dataset_pth $DATA \
    --cache_dir $CACHE/tcgr_cache_mlmf \
    --save_dir $WORK/output \
    --tcgr_hidden_dim 128 \
    --tcgr_num_layers 3 \
    --epochs 100 \
    --lr 1e-3

echo "=== TCGR complete ==="
