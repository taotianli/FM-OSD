#!/bin/bash
#SBATCH --job-name=fmosd_head
#SBATCH --output=/home/u6da/taotl.u6da/FM-OSD/logs/%j_head.log
#SBATCH --error=/home/u6da/taotl.u6da/FM-OSD/logs/%j_head.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# ── Paths ────────────────────────────────────────────────────────────────────
WORK=/home/u6da/taotl.u6da/FM-OSD
CACHE=/projects/u6da/fmosd_cache
DATA=/home/u6da/taotl.u6da/FM-OSD/data/Cephalometric

mkdir -p $CACHE/feat_head
mkdir -p $WORK/logs
mkdir -p $WORK/models/global_mssr
mkdir -p $WORK/models/local_mssr
mkdir -p $WORK/output

# ── Environment ───────────────────────────────────────────────────────────────
source ~/miniforge3/etc/profile.d/conda.sh
conda activate fmosd
cd $WORK

# ── Step 1: data_generate (augmented training images) ─────────────────────────
echo "=== Step 1: data_generate ==="
python data_generate.py \
    --dataset_pth $DATA \
    --id_shot 125

# ── Step 2: precompute MLMF feature cache (1 ViT pass per image, saved to /projects) ──
echo "=== Step 2: precompute_head_features ==="
python precompute_head_features.py \
    --dataset_pth $DATA \
    --cache_dir $CACHE/feat_head \
    --bin False \
    --model_type dino_vits8 \
    --stride 4

# ── Step 3: Train global branch (MSSR) ────────────────────────────────────────
echo "=== Step 3: train1_mssr (global) ==="
python train1_mssr.py \
    --dataset_pth $DATA \
    --feat_cache_dir $CACHE/feat_head \
    --bin False \
    --bs 8 \
    --max_iterations 3000 \
    --eval_freq 200 \
    --early_stop_patience 8 \
    --exp global_mssr \
    --save_dir $WORK/output

# ── Step 4: Train local branch (MSSR coarse-to-fine) ──────────────────────────
echo "=== Step 4: train2_mssr (local) ==="
GLOBAL_CKPT=$(ls $WORK/models/global_mssr/model_post_mssr_iter_*_*.pth 2>/dev/null | sort -t_ -k6 -rn | head -1)
if [ -z "$GLOBAL_CKPT" ]; then
    GLOBAL_CKPT=$WORK/models/global_mssr/model_post_mssr_final.pth
fi
echo "  Using global ckpt: $GLOBAL_CKPT"

python train2_mssr.py \
    --dataset_pth $DATA \
    --feat_cache_dir $CACHE/feat_head \
    --bin False \
    --bs 8 \
    --max_iterations 2000 \
    --eval_freq 200 \
    --early_stop_patience 8 \
    --global_ckpt $GLOBAL_CKPT \
    --exp local_mssr \
    --save_dir $WORK/output

echo "=== Pipeline complete ==="
