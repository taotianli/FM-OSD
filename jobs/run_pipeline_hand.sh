#!/bin/bash
#SBATCH --job-name=fmosd_hand
#SBATCH --output=/home/u6da/taotl.u6da/FM-OSD/logs/%j_hand.log
#SBATCH --error=/home/u6da/taotl.u6da/FM-OSD/logs/%j_hand.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

WORK=/home/u6da/taotl.u6da/FM-OSD
CACHE=/projects/u6da/fmosd_cache
DATA=/home/u6da/taotl.u6da/FM-OSD/data/Hand/hand

mkdir -p $CACHE/feat_hand
mkdir -p $WORK/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate fmosd
cd $WORK

# Step 1 (data_generate_hand) skipped — training images already exist in data/hand/

echo "=== Step 1: precompute_hand_features ==="
python precompute_hand_features.py \
    --dataset_pth $DATA \
    --cache_dir $CACHE/feat_hand \
    --bin False

echo "=== Step 2: train1_mssr_hand ==="
python train1_mssr_hand.py \
    --dataset_pth $DATA \
    --feat_cache_dir $CACHE/feat_hand \
    --bin False \
    --bs 8 \
    --max_iterations 3000 \
    --eval_freq 200 \
    --early_stop_patience 8 \
    --exp global_mssr_hand \
    --save_dir $WORK/output

echo "=== Step 3: train2_mssr_hand ==="
GLOBAL_CKPT=$(ls $WORK/models/global_mssr_hand/model_post_mssr_iter_*_*.pth 2>/dev/null | sort -t_ -k6 -rn | head -1)
if [ -z "$GLOBAL_CKPT" ]; then
    GLOBAL_CKPT=$WORK/models/global_mssr_hand/model_post_mssr_final.pth
fi

python train2_mssr_hand.py \
    --dataset_pth $DATA \
    --feat_cache_dir $CACHE/feat_hand \
    --bin False \
    --bs 8 \
    --max_iterations 2000 \
    --eval_freq 200 \
    --early_stop_patience 8 \
    --global_ckpt $GLOBAL_CKPT \
    --exp local_mssr_hand \
    --save_dir $WORK/output

echo "=== Hand pipeline complete ==="
