#!/bin/bash
# SGLang ("baseline") and Libra ("rebalance") imbalance-ratio fluctuation (paper Fig. 9, bottom row),
# 4096 tokens per GPU per iteration. CPU only. Run from prefetch_rebalance/ after building the planner:
#   python setup.py build_ext --inplace
# Reads  $LIBRA_ARTIFACT_DIR/expt_token2expert_results/<alias>_<dataset>_train/1000/{real,pred1}/layer_L.feather
# Writes $LIBRA_ARTIFACT_DIR/imbalance_ratio_fluctuation/<alias>/<dataset>/4096/step1/{baseline,rebalance}_layer_L.csv
#        and the per-run logs log_<alias>_<dataset>_step1_4096.txt in the current directory
LIBRA_ARTIFACT_DIR=${LIBRA_ARTIFACT_DIR:-$HOME/libra_artifacts}
LIBRA_ARTIFACT_DIR=${LIBRA_ARTIFACT_DIR/#\~/$HOME}
MODEL_ALIASES=("glm45" "qwen3")
DATASETS=("bookcorpus" "codeforces" "deepseek_prover" "fineweb" "gsm8k" "hellaswag" "humanevalplus" "lmsys_chat_1m")
MAX_NUM_BATCHES=1000
EXPTS=("step1")
NUM_GPUS=8
SEQ_LENS=(4096)
P=7
Q=0

for SEQ_LEN in "${SEQ_LENS[@]}"; do
  for MODEL_ALIAS in "${MODEL_ALIASES[@]}"; do
    if [[ "$MODEL_ALIAS" == "glm45" ]]; then
      LAYER_IDXES="$(seq 4 91)"
      NUM_EXPERTS=160
      START_IDX=1638400
      END_IDX=2031616 # 4096 * 496
    elif [[ "$MODEL_ALIAS" == "qwen3" ]]; then
      LAYER_IDXES="$(seq 1 93)"
      NUM_EXPERTS=128
      START_IDX=1638400
      END_IDX=2031616 # 4096 * 496
    else
      echo "Unknown model alias: $MODEL_ALIAS"
      exit 1
    fi

    for DATASET in "${DATASETS[@]}"; do
      N=8
      L=4
      for EXPT in "${EXPTS[@]}"; do
        TEMT_DIR=$LIBRA_ARTIFACT_DIR/expt_token2expert_results/${MODEL_ALIAS}_${DATASET}_train/${MAX_NUM_BATCHES}
        SAVE_DIR=$LIBRA_ARTIFACT_DIR/imbalance_ratio_fluctuation/${MODEL_ALIAS}/${DATASET}/${SEQ_LEN}/${EXPT}

        echo ">>> Running: $MODEL_ALIAS | $DATASET | $EXPT | SEQ_LEN=$SEQ_LEN"

        python evaluate_libra.py \
          --temt-dir $TEMT_DIR \
          --save-dir $SAVE_DIR \
          --model-alias $MODEL_ALIAS \
          --dataset $DATASET \
          --start-idx $START_IDX \
          --end-idx $END_IDX \
          --expt $EXPT \
          --layer-idxes $LAYER_IDXES \
          --num-experts $NUM_EXPERTS \
          --num-gpus $NUM_GPUS \
          --seq-len $SEQ_LEN \
          --max-dup-per-gpu $N \
          --n-local-dup-per-gpu $L \
          --max-send-len $P \
          --min-dup-experts-tokens $Q \
          > log_${MODEL_ALIAS}_${DATASET}_${EXPT}_${SEQ_LEN}.txt 2>&1 
          # optional flags: --correctness-test --perfect-prediction

      done
    done
  done
done

wait
echo "All jobs finished."
