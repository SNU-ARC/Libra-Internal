#!/bin/bash
# Generate the TEMTs of GLM-4.5 (real + 1-layer-lookahead predicted routing) for the 8 evaluation
# datasets and for the 'mixed' dataset. Run from the Libra-Internal root.
# Output: $LIBRA_ARTIFACT_DIR/expt_token2expert_results/glm45_<dataset>_train/1000/{real,pred1}/layer_L.feather
# GPUs: $CUDA_VISIBLE_DEVICES if set, otherwise 0-7 (8 GPUs, TP=EP=8).
GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

bash bench_one_batch_token2expert_expt.sh $GPUS GLM-4.5 glm45 bookcorpus step1 0 1 1024 2
bash bench_one_batch_token2expert_expt.sh $GPUS GLM-4.5 glm45 codeforces step1 0 100 1024 2
bash bench_one_batch_token2expert_expt.sh $GPUS GLM-4.5 glm45 deepseek_prover step1 0 100 1024 2
bash bench_one_batch_token2expert_expt.sh $GPUS GLM-4.5 glm45 fineweb step1 0 1 1024 2
bash bench_one_batch_token2expert_expt.sh $GPUS GLM-4.5 glm45 gsm8k step1 0 100 1024 2
bash bench_one_batch_token2expert_expt.sh $GPUS GLM-4.5 glm45 hellaswag step1 0 100 1024 2
bash bench_one_batch_token2expert_expt.sh $GPUS GLM-4.5 glm45 humanevalplus step1 0 100 1024 2
bash bench_one_batch_token2expert_expt.sh $GPUS GLM-4.5 glm45 lmsys_chat_1m step1 0 1 1024 2

# 'mixed' TEMT: read by Libra-Core's eplb/find_hot_experts_and_cold_gpus_glm45.sh (rows 0..40960 of
# glm45_mixed_train/1000/real). mixed_to_texts() takes 1000 // 8 = 125 batches from each of the 8 datasets
# above (with the same portions as above: 0-1% for bookcorpus/fineweb/lmsys_chat_1m, 0-100% for the others,
# which it sets itself) and interleaves them batch by batch, so START/END_PORTION below are ignored.
bash bench_one_batch_token2expert_expt.sh $GPUS GLM-4.5 glm45 mixed step1 0 100 1024 2
