#!/bin/bash
# Run A-series MLMF ablations sequentially (each ~20 min for 2000 iters)
set -e
LOG=/tmp/ablation_a
mkdir -p $LOG

echo "==== A2: multi-layer (5,8,11) key-only ====" | tee $LOG/a2.log
python train1_mlmf.py \
  --mlmf_layers 5,8,11 --mlmf_facets key \
  --exp global_mlmf_a2_multilayer_keyonly \
  --max_iterations 2000 2>&1 | tee -a $LOG/a2.log
echo "A2 done" && echo "A2_MRE=$(grep 'best' $LOG/a2.log | tail -1)"

echo "==== A3: single-layer (8) key+value ====" | tee $LOG/a3.log
python train1_mlmf.py \
  --mlmf_layers 8 --mlmf_facets key,value \
  --exp global_mlmf_a3_singlelayer_kv \
  --max_iterations 2000 2>&1 | tee -a $LOG/a3.log
echo "A3 done"

echo "==== A5: full MLMF with uniform weights (no attention) ====" | tee $LOG/a5.log
python train1_mlmf.py \
  --mlmf_layers 5,8,11 --mlmf_facets key,value \
  --fusion_uniform True \
  --exp global_mlmf_a5_uniform \
  --max_iterations 2000 2>&1 | tee -a $LOG/a5.log
echo "A5 done"

echo "==== All A-series done ===="
