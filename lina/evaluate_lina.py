# Lina Imbalance Ratio Fluctuation Measurement Code
import argparse
import time
from pathlib import Path
from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from typing import Optional, Any

from lina_prefetch_route import Lina

NUM_GPUS = 8
L = 3

def load_topk_ids(
    df: pd.DataFrame,
    batch_idx: int,
) -> np.ndarray:    
    df_slice = df.iloc[batch_idx*SEQ_LEN:(batch_idx+NUM_GPUS)*SEQ_LEN]
    return df_slice.drop(['req_idx', 'token_idx'], axis=1).values.reshape(NUM_GPUS, SEQ_LEN, -1)

def calculate_load_stats(
    real_topk_ids: np.ndarray,  # Shape: (num_gpus, seq_len, experts_per_token) - true expert IDs for each token
    expert_gpu_mapping: np.ndarray,  # Shape: (num_experts, num_gpus) - binary matrix of expert placement
    gpu_expert_info: Optional[np.ndarray] = None,  # Shape: (num_gpus, num_experts, num_gpus*2) - (start_idx_1, n_1, start_idx_2, n_2, ...) for remote tokens processed by expert j on GPU i
    gpu_load: Optional[np.ndarray] = None,
    gpu_expert_summary: Optional[np.ndarray] = None,
):
    """Calculate load balancing statistics."""

    num_experts = expert_gpu_mapping.shape[0]
    experts_per_token = real_topk_ids.shape[2]
    experts_per_gpu = num_experts // NUM_GPUS

    if gpu_expert_info is not None:
        n_tokens = np.maximum(gpu_expert_info[..., 1], 0).sum(axis=(1, 2))
    else:
        expert_load = np.bincount(real_topk_ids.ravel(), minlength=num_experts)
        n_tokens = expert_load.reshape(-1, experts_per_gpu).sum(axis=1)

    assert n_tokens.sum() == NUM_GPUS * SEQ_LEN * experts_per_token, n_tokens.sum()

    imbalance_ratio = n_tokens.max() / (SEQ_LEN * experts_per_token)

    return {
        'imbalance_ratio': imbalance_ratio,
        'n_tokens': n_tokens,
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--ept-dir', type=str, required=True)
    parser.add_argument('--seq-len', type=int, required=True)
    parser.add_argument('--start-idx', type=int, required=True)
    parser.add_argument('--end-idx', type=int, required=True)
    parser.add_argument('--model-alias', type=str, required=True)
    parser.add_argument('--dataset', type=str, default="bookcorpus")
    parser.add_argument('--perfect-prediction', '-p', action='store_true')
    parser.add_argument('--max-dup-per-gpu', '-N', type=int, default=8)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.perfect_prediction:
        print('Enabled perfect prediction')

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    EPT_DIR = Path(args.ept_dir)
    SEQ_LEN = args.seq_len

    
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.model_alias == 'qwen3':
        start_layer = 0
        num_layers = 94
    elif args.model_alias == 'glm45':
        start_layer = 3
        num_layers = 92

    cur_rank = 0

    percentiles = [1, 10, 25, 50, 75, 90, 99]
    str_percentiles = list(map(lambda x: f'p{x}', percentiles))
    columns = ['layer', 'mean', 'std', 'min'] + str_percentiles + ['max']

    dfs = [pd.read_feather(data_dir / 'real' / f'layer_{i}.feather') for i in tqdm(range(start_layer, num_layers))]
    print('Loaded real topk ids')

    lina = Lina(num_layers=num_layers, L=L, num_gpus=NUM_GPUS, max_dup_per_gpu=args.max_dup_per_gpu, verbose=True)
    lina.load_expert_path_tables(EPT_DIR, start_layer)
    print('Initialized Lina')

    num_experts = lina.num_experts
    assert num_experts is not None
    experts_per_gpu = num_experts // NUM_GPUS
    baseline_expert_gpu_mapping = np.eye(NUM_GPUS, dtype=bool).repeat(experts_per_gpu, axis=0)

    results = [defaultdict(list) for _ in range(num_layers)]

    gpu_expert_info = torch.empty(
        (NUM_GPUS, num_experts, NUM_GPUS, 2),
        dtype=torch.int64,
        device=lina.device,
    )
    gpu_load = torch.empty(
        (NUM_GPUS, 2),
        dtype=torch.int64,
        device=lina.device,
    )
    gpu_expert_summary = torch.empty(
        (NUM_GPUS, num_experts, 2),
        dtype=torch.int64,
        device=lina.device,
    )
    send_map_local = torch.empty(
        (NUM_GPUS, NUM_GPUS, lina.max_dup_per_gpu),
        dtype=torch.int64,
        device=lina.device,
    )
    send_map_local_len = torch.empty(
        (NUM_GPUS, NUM_GPUS),
        dtype=torch.int64,
        device=lina.device,
    )
    recv_map_local_len = torch.empty(
        (NUM_GPUS, NUM_GPUS),
        dtype=torch.int64,
        device=lina.device,
    )

    # One iteration = NUM_GPUS consecutive chunks of SEQ_LEN TEMT rows (one chunk per GPU)
    for batch_idx in tqdm(range(args.start_idx // SEQ_LEN, args.end_idx // SEQ_LEN, NUM_GPUS)):
        lina.new_batch(NUM_GPUS * SEQ_LEN)
        expert_gpu_mapping = None
        prefetch_time = None
        for i in range(start_layer, num_layers):
            topk_ids = load_topk_ids(dfs[i - start_layer], batch_idx)
            topk_ids_torch = torch.tensor(topk_ids, device='cuda').reshape(NUM_GPUS * SEQ_LEN, -1)
            lina.record(i, topk_ids_torch)

            if expert_gpu_mapping is not None:
                t1 = time.perf_counter_ns()
                lina.route(topk_ids_torch, expert_gpu_mapping, gpu_expert_info, gpu_load, gpu_expert_summary)
                t2 = time.perf_counter_ns()
                route_time = (t2 - t1) / 10**6

                baseline_stats = calculate_load_stats(topk_ids, baseline_expert_gpu_mapping)
                lina_stats = calculate_load_stats(topk_ids, expert_gpu_mapping.cpu().numpy(), gpu_expert_info.cpu().numpy(), gpu_load.cpu().numpy(), gpu_expert_summary.cpu().numpy())
                max_dup_per_gpu = int(expert_gpu_mapping.sum(dim=0).max() - experts_per_gpu)

                stats = {
                    'baseline_imbalance_ratio': baseline_stats['imbalance_ratio'],
                    'lina_imbalance_ratio': lina_stats['imbalance_ratio'],
                    'prefetch_time': prefetch_time,
                    'route_time': route_time,
                    'max_dup_per_gpu': max_dup_per_gpu,
                }

                for k, v in stats.items():
                    results[i][k].append(v)

            if L + start_layer <= i < num_layers-1:
                t1 = time.perf_counter_ns()
                expert_gpu_mapping, _, _ = lina.prefetch(
                    i,
                    cur_rank,
                    send_map_local,
                    send_map_local_len,
                    recv_map_local_len,
                )
                t2 = time.perf_counter_ns()
                prefetch_time = (t2 - t1) / 10**6


    def write_one(layer_idx):
        if not results[layer_idx]:
            return {}

        df = pd.DataFrame({"imbalance_ratio": results[layer_idx]['lina_imbalance_ratio']})
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(save_dir / f'lina_layer_{layer_idx}.csv')
        df.to_csv(out_path, index=False)
        print(f"wrote {out_path}", flush=True)

    summaries = [(i, write_one(i)) for i in range(start_layer, num_layers)]
