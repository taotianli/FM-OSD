#!/bin/bash
# B-series TCGR ablations using MLMF+FM-OSD cache as input (data/tcgr_cache_mlmf)
# Varying only TCGR architecture; main experiment (B1) = tcgr_cv_mlmf (already done, MRE=1.776mm)
set -e
source /home/taotl/miniconda3/etc/profile.d/conda.sh
conda activate bmad
cd /home/taotl/Desktop/FM-OSD

LOG=logs/ablation_b
mkdir -p $LOG

MLMF_CACHE=data/tcgr_cache_mlmf
FMOSD_CACHE=data/tcgr_cache

run_tcgr() {
  local EXP=$1; shift
  echo "==== $EXP ====" | tee -a $LOG/${EXP}.log
  python -u train_tcgr_cv.py \
    --cache_dir $MLMF_CACHE \
    --exp "$EXP" \
    --max_iterations 3000 \
    "$@" 2>&1 | tee -a $LOG/${EXP}.log
  echo "Done: $EXP"
}

# B2: ablate number of TCGR layers (main=3 layers)
run_tcgr tcgr_abl_b2_layers1 --tcgr_num_layers 1
run_tcgr tcgr_abl_b2_layers2 --tcgr_num_layers 2
run_tcgr tcgr_abl_b2_layers4 --tcgr_num_layers 4

# B3: ablate topological loss
run_tcgr tcgr_abl_b3_notopo --topo_loss_weight 0.0

# B4: ablate graph attention
run_tcgr tcgr_abl_b4_noattn --tcgr_use_attention False

# B5: FM-OSD input instead of MLMF (control)
echo "==== tcgr_abl_b5_fmosd_input ====" | tee -a $LOG/tcgr_abl_b5_fmosd_input.log
python -u train_tcgr_cv.py \
  --cache_dir $FMOSD_CACHE \
  --exp tcgr_abl_b5_fmosd_input \
  --max_iterations 3000 2>&1 | tee -a $LOG/tcgr_abl_b5_fmosd_input.log

echo "==== All B-series done ===="
