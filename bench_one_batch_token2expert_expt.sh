#!/bin/bash
# Dump the Token-to-Expert Mapping Table (TEMT) of one (model, dataset) pair.
# Run from the Libra-Internal root; generate_temt_{qwen3,glm45}.sh call this script.
#
# Usage: bash bench_one_batch_token2expert_expt.sh DEVICES MODEL MODEL_ALIAS DATASET EXPT \
#            START_PORTION END_PORTION TARGET_SEQ_LEN MINI_BATCH_SIZE
#
# Reads  $LIBRA_MODEL_DIR/MODEL        (default $HOME/MODEL)
#        $LIBRA_DATA_DIR               (default $HOME/hf_datasets)
# Writes $LIBRA_ARTIFACT_DIR/expt_token2expert_results/MODEL_ALIAS_DATASET_train/1000/{real,pred1}/layer_L.feather
#        (LIBRA_ARTIFACT_DIR defaults to $HOME/libra_artifacts)
DEVICE=$1
model=$2
model_alias=$3
dataset=$4
mode=train
expt=$5
start_portion=$6
end_portion=$7
target_seq_len=$8
mini_batch_size=$9

LIBRA_MODEL_DIR=${LIBRA_MODEL_DIR:-$HOME}
LIBRA_DATA_DIR=${LIBRA_DATA_DIR:-$HOME/hf_datasets}
LIBRA_ARTIFACT_DIR=${LIBRA_ARTIFACT_DIR:-$HOME/libra_artifacts}
dataset_dir=${LIBRA_DATA_DIR/#\~/$HOME}
result_dir=${LIBRA_ARTIFACT_DIR/#\~/$HOME}/expt_token2expert_results

model=${LIBRA_MODEL_DIR/#\~/$HOME}/${model}

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

CUDA_VISIBLE_DEVICES=$DEVICE python3 -m sglang.bench_one_batch_token2expert \
    --mem-fraction-static 0.8 \
    --model-path "$model" \
    --trust-remote-code \
    --tp-size 8 \
    --ep-size 8 \
    --disable-cuda-graph \
    --enable-ep-moe \
    --correctness-test \
    --model-alias $model_alias \
    --dataset $dataset \
    --dataset-dir "$dataset_dir" \
    --result-dir "$result_dir" \
    --mode $mode \
    --expt $expt \
    --start-portion $start_portion \
    --end-portion $end_portion \
    --target-seq-len $target_seq_len \
    --mini-batch-size $mini_batch_size \
    --batch-size 1 \
    --max-num-batches 1000
