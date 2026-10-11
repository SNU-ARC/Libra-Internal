#!/bin/bash
# Lina imbalance-ratio fluctuation (paper Fig. 9, bottom row), 2048 tokens per GPU per iteration.
# Run from lina/ (needs a CUDA GPU: Lina's planner is written in Triton).
# Reads  $LIBRA_ARTIFACT_DIR/expt_token2expert_results/<alias>_<dataset>_train/1000/real/layer_L.feather
#        $LIBRA_ARTIFACT_DIR/lina_epts/<alias>_mixed_ept/layer_L.feather (run_make_expert_path_table_<alias>.sh)
# Writes $LIBRA_ARTIFACT_DIR/imbalance_ratio_fluctuation/<alias>/<dataset>/2048/lina/lina_layer_L.csv
LIBRA_ARTIFACT_DIR=${LIBRA_ARTIFACT_DIR:-$HOME/libra_artifacts}
LIBRA_ARTIFACT_DIR=${LIBRA_ARTIFACT_DIR/#\~/$HOME}
MODEL_ALIASES=("glm45" "qwen3")
DATASETS=("bookcorpus" "codeforces" "deepseek_prover" "fineweb" "gsm8k" "hellaswag" "humanevalplus" "lmsys_chat_1m")
MAX_NUM_BATCHES=1000

SEQ_LENS=(2048)


for SEQ_LEN in "${SEQ_LENS[@]}"; do
  for MODEL_ALIAS in "${MODEL_ALIASES[@]}"; do
    # set layer indexes depending on model
    if [[ "$MODEL_ALIAS" == "glm45" ]]; then
      NUM_EXPERTS=160
      START_IDX=1638400
      END_IDX=2048000
    elif [[ "$MODEL_ALIAS" == "qwen3" ]]; then
      NUM_EXPERTS=128
      START_IDX=1638400
      END_IDX=2048000
    else
      echo "Unknown model alias: $MODEL_ALIAS"
      exit 1
    fi

    for DATASET in "${DATASETS[@]}"; do
      N=6
      TEMT_DIR=$LIBRA_ARTIFACT_DIR/expt_token2expert_results/${MODEL_ALIAS}_${DATASET}_train/${MAX_NUM_BATCHES}
      EPT_DIR=$LIBRA_ARTIFACT_DIR/lina_epts/${MODEL_ALIAS}_mixed_ept
      SAVE_DIR=$LIBRA_ARTIFACT_DIR/imbalance_ratio_fluctuation/${MODEL_ALIAS}/${DATASET}/${SEQ_LEN}/lina

      echo ">>> Running: $MODEL_ALIAS | $DATASET | SEQ_LEN=$SEQ_LEN"

      python evaluate_lina.py \
          --data-dir $TEMT_DIR \
          --ept-dir $EPT_DIR \
          --save-dir $SAVE_DIR \
          --model-alias $MODEL_ALIAS \
          --dataset $DATASET \
          --start-idx $START_IDX \
          --end-idx $END_IDX \
          --seq-len $SEQ_LEN \
          -N $N
    done
  done
done

wait
echo "All jobs finished."
