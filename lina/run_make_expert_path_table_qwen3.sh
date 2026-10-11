#!/bin/bash
# Build Lina's expert-selection-path tables for qwen3 from the first 204,800 TEMT rows (real routing)
# of each of the 8 datasets. Run from lina/. The per-layer jobs run in parallel in the background.
# Reads  $LIBRA_ARTIFACT_DIR/expt_token2expert_results/qwen3_<dataset>_train/1000/real/layer_L.feather
# Writes $LIBRA_ARTIFACT_DIR/lina_epts/qwen3_mixed_ept/layer_L.feather (LINA_EPT_DIR of the Libra-Core lina branch)
LIBRA_ARTIFACT_DIR=${LIBRA_ARTIFACT_DIR:-$HOME/libra_artifacts}
LIBRA_ARTIFACT_DIR=${LIBRA_ARTIFACT_DIR/#\~/$HOME}

TEMT_BASE_DIR=$LIBRA_ARTIFACT_DIR/expt_token2expert_results
MODEL_ALIAS=qwen3
TEMT_DIRS="${TEMT_BASE_DIR}/${MODEL_ALIAS}_bookcorpus_train/1000/real ${TEMT_BASE_DIR}/${MODEL_ALIAS}_codeforces_train/1000/real ${TEMT_BASE_DIR}/${MODEL_ALIAS}_deepseek_prover_train/1000/real ${TEMT_BASE_DIR}/${MODEL_ALIAS}_fineweb_train/1000/real ${TEMT_BASE_DIR}/${MODEL_ALIAS}_gsm8k_train/1000/real ${TEMT_BASE_DIR}/${MODEL_ALIAS}_hellaswag_train/1000/real ${TEMT_BASE_DIR}/${MODEL_ALIAS}_humanevalplus_train/1000/real ${TEMT_BASE_DIR}/${MODEL_ALIAS}_lmsys_chat_1m_train/1000/real"

SAVE_DIR=$LIBRA_ARTIFACT_DIR/lina_epts/${MODEL_ALIAS}_mixed_ept

START_IDX=0
END_IDX=204800
if [[ "$MODEL_ALIAS" == "glm45" ]]; then
    LAYER_IDXES=$(seq 6 90)
    NUM_EXPERTS=160
elif [[ "$MODEL_ALIAS" == "qwen3" ]]; then
    NUM_EXPERTS=128
    LAYER_IDXES=$(seq 3 92)
else
    echo "Unknown model alias: $MODEL_ALIAS"
    exit 1
fi
MOE_LAYER_FREQ=1

for i in $LAYER_IDXES; do
    python make_expert_path_table.py \
        -i $i \
        --save-dir $SAVE_DIR \
        --temt-dirs $TEMT_DIRS \
        --start-idx $START_IDX \
        --end-idx $END_IDX \
        --num-experts $NUM_EXPERTS \
        --moe-layer-freq $MOE_LAYER_FREQ \
        &
    sleep 4
done
wait
echo "All jobs finished."
