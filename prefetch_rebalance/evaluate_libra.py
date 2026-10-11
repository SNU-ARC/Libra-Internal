import argparse
import time
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import torch


class Evaluator:
    def __init__(
        self,
        temt_dir: Path,
        prefetch,
        rebalance,
        num_experts: int,
        num_gpus: int,
        seq_len: int,
        max_dup_per_gpu: int,
        n_local_dup_per_gpu: int,
        max_send_len: int,
        min_dup_experts_tokens : int,
        correctness_test: bool = False,
        perfect_prediction: bool = False,
    ):
        self.temt_dir = temt_dir
        self.prefetch = prefetch
        self.rebalance = rebalance
        self.num_experts = num_experts
        self.num_gpus = num_gpus
        self.seq_len = seq_len
        self.max_dup_per_gpu = max_dup_per_gpu
        self.n_local_dup_per_gpu = n_local_dup_per_gpu
        self.max_send_len = max_send_len
        self.min_dup_experts_tokens = min_dup_experts_tokens
        self.correctness_test = correctness_test
        self.perfect_prediction = perfect_prediction

        self.send_map_local = torch.empty((num_gpus, num_gpus, max_dup_per_gpu), dtype=torch.int64)
        self.send_map_local_len = torch.empty((num_gpus, num_gpus), dtype=torch.int64)
        self.recv_map_local_len = torch.empty((num_gpus, num_gpus), dtype=torch.int64)

    def eval_batch(self, real_topk_ids: np.ndarray, pred_topk_ids: np.ndarray):
        experts_per_gpu = self.num_experts // self.num_gpus

        # Baseline
        baseline_expert_gpu_mapping = np.eye(self.num_gpus, dtype=bool).repeat(experts_per_gpu, axis=0)
        baseline_stats = self.calculate_load_stats(
            real_topk_ids,
            baseline_expert_gpu_mapping,
        )

        # Prefetch
        pred_topk_ids_torch = torch.tensor(pred_topk_ids).reshape(self.num_gpus * self.seq_len, -1).contiguous()
        t1 = time.perf_counter_ns()
        expert_gpu_mapping, init_remote_gpu_expert_load, _ = self.prefetch(
            pred_topk_ids_torch,
            self.num_experts,
            self.seq_len,
            self.num_gpus,
            0,
            self.send_map_local,
            self.send_map_local_len,
            self.recv_map_local_len,
            self.max_dup_per_gpu,
            self.n_local_dup_per_gpu,
            self.max_send_len,
        )
        t2 = time.perf_counter_ns()
        prefetch_time = (t2 - t1) / 10**6
        assert torch.all(self.send_map_local_len.count_nonzero(dim=1) <= self.max_send_len)

        # Rebalance
        real_topk_ids_torch = torch.tensor(real_topk_ids).reshape(self.num_gpus * self.seq_len, -1).contiguous()
        topk_ids_masks = torch.empty((self.num_gpus, self.num_gpus, self.seq_len, real_topk_ids.shape[-1]), dtype=torch.int64)
        for cur_rank in range(self.num_gpus):
            init_remote_gpu_expert_load_clone = init_remote_gpu_expert_load.clone()
            t1 = time.perf_counter_ns()
            rebalance(
                real_topk_ids_torch,
                expert_gpu_mapping,
                init_remote_gpu_expert_load_clone,
                self.num_experts,
                self.seq_len,
                self.num_gpus,
                cur_rank,
                topk_ids_masks[cur_rank],
                self.min_dup_experts_tokens,
            )
            t2 = time.perf_counter_ns()
        rebalance_time = (t2 - t1) / 10**6

        # Correctness test
        if self.correctness_test:
            self.verify(
                real_topk_ids.reshape(self.num_gpus, self.seq_len, -1),
                expert_gpu_mapping.numpy(),
                topk_ids_masks.numpy(),
                self.num_gpus,
                self.seq_len, self.min_dup_experts_tokens
            )

        # Stats after rebalance
        rebalance_stats = self.calculate_load_stats(
            real_topk_ids,
            expert_gpu_mapping.numpy(),
            topk_ids_masks.numpy(),
        )

        # Number of duplicated experts on the GPU that holds the most duplicated experts
        max_dup_per_gpu = expert_gpu_mapping.sum(axis=0).max() - experts_per_gpu

        return {
            'baseline_imbalance_ratio': baseline_stats['imbalance_ratio'],
            'rebalance_imbalance_ratio': rebalance_stats['imbalance_ratio'],
            'baseline_remote_local_ratio': baseline_stats['remote_local_ratio'],
            'rebalance_remote_local_ratio': rebalance_stats['remote_local_ratio'],
            'prefetch_time': prefetch_time,
            'rebalance_time': rebalance_time,
            'max_dup_per_gpu': max_dup_per_gpu,
        }

    def eval_layer(self, layer_idx, pred_type, start_idx, end_idx):
        # Load real, pred TEMT
        filename = f'layer_{layer_idx}.feather'
        real = pd.read_feather(self.temt_dir / 'real' / filename).drop(['req_idx', 'token_idx'], axis=1).values
        if not self.perfect_prediction:
            pred = pd.read_feather(self.temt_dir / f'pred{pred_type}' / filename).drop(['req_idx', 'token_idx'], axis=1).values

        results = defaultdict(list)

        size = self.num_gpus * self.seq_len

        # Evaluate each batch of size
        for i in range(start_idx, end_idx, size):
            real_topk_ids = real[i:i+size, :].reshape(self.num_gpus, self.seq_len, -1)
            if self.perfect_prediction:
                pred_topk_ids = real_topk_ids
            else:
                pred_topk_ids = pred[i:i+size, :].reshape(self.num_gpus, self.seq_len, -1)

            stats = self.eval_batch(real_topk_ids, pred_topk_ids)

            # Collect each metric
            for k, v in stats.items():
                results[k].append(v)

        return results
    
    def calculate_load_stats(
        self,
        real_topk_ids: np.ndarray,       # Shape: (num_gpus, seq_len, experts_per_token) - true expert IDs for each token
        expert_gpu_mapping: np.ndarray,  # Shape: (num_experts, num_gpus) - binary matrix of expert placement
        topk_ids_masks: Optional[np.ndarray] = None,  # Shape: (num_gpus, num_gpus, seq_len, experts_per_token)
    ):
        """Calculate load balancing statistics."""

        num_experts = expert_gpu_mapping.shape[0]
        experts_per_token = real_topk_ids.shape[2]
        experts_per_gpu = num_experts // self.num_gpus

        # Shape: (num_gpus, 1, 1) - r = [[[0]], [[1]], ..., [[num_gpus-1]]]
        r = np.arange(self.num_gpus)[:, None, None]

        # Shape: (num_gpus, seq_len, experts_per_token) - boolean mask indicating whether each token is a local token or not
        local_mask = expert_gpu_mapping[real_topk_ids, r]

        # Shape: (num_gpus,) - # local tokens processed per GPU
        n_local_tokens = local_mask.sum(axis=(1, 2))

        if topk_ids_masks is not None:
            # Shape: (num_gpus,) - # remote tokens processed per GPU
            n_remote_tokens = topk_ids_masks.sum(axis=(1, 2, 3))
        else:
            # Shape: (num_experts,) - # remote tokens for each expert
            remote_expert_load = np.bincount(real_topk_ids[~local_mask], minlength=num_experts)
            # Shape: (num_gpus,) - # remote tokens processed per GPU
            n_remote_tokens = remote_expert_load.reshape(-1, experts_per_gpu).sum(axis=1)
        
        # Shape: (num_gpus,) - # total tokens processed per GPU
        n_tokens = n_local_tokens + n_remote_tokens
        assert n_tokens.sum() == self.num_gpus * self.seq_len * experts_per_token, n_tokens.sum()

        imbalance_ratio = n_tokens.max() / (self.seq_len * experts_per_token)

        return {
            'imbalance_ratio': imbalance_ratio,
            'n_local_tokens': n_local_tokens,
            'n_remote_tokens': n_remote_tokens,
            'n_tokens': n_tokens,
            'remote_local_ratio': (n_remote_tokens/n_local_tokens).mean() if np.all(n_local_tokens > 0) else float('inf'),
        }

    def verify(
        self,
        real_topk_ids,  # Shape: (num_gpus, seq_len, experts_per_token) - true expert IDs for each token
        expert_gpu_mapping,  # Shape: (num_experts, num_gpus) - binary matrix of expert placement
        topk_ids_masks,
        num_gpus,
        seq_len,
        min_dup_experts_tokens,  # Q
    ):
        import numpy as np
        assert real_topk_ids.dtype == np.int64
        assert expert_gpu_mapping.dtype == np.bool
        assert topk_ids_masks.dtype == np.int64

        num_experts = expert_gpu_mapping.shape[0]
        experts_per_token = real_topk_ids.shape[2]
        experts_per_gpu = num_experts // num_gpus

        topk_ids_masks = topk_ids_masks.reshape(num_gpus, num_gpus, seq_len, experts_per_token)

        assert 0 < num_experts <= 256
        assert 1 <= experts_per_token <= 8

        n_total_tokens = num_gpus * seq_len * experts_per_token

        r = np.arange(num_gpus)[:, None, None]
        local_mask = expert_gpu_mapping[real_topk_ids, r]
        n_local_tokens = local_mask.sum(axis=(1, 2))

        n_remote_tokens = topk_ids_masks.sum(axis=(1, 2, 3))
        assert topk_ids_masks.sum(axis=0).max() <= 1, topk_ids_masks.sum(axis=0).max()
        assert np.all(topk_ids_masks.sum(axis=0) + local_mask == 1)

        dup_experts_tokens = np.zeros((num_gpus,), dtype=int)
        for g in range(num_gpus):
            dup_mask = real_topk_ids // experts_per_gpu != g
            dup_experts_tokens[g] = topk_ids_masks[g][dup_mask].sum()
        assert np.all((dup_experts_tokens >= min_dup_experts_tokens) | (dup_experts_tokens == 0)), dup_experts_tokens

        expert_load = np.zeros((num_gpus, num_experts), dtype=int)
        for g_src in range(num_gpus):
            expert_load[g_src] = np.bincount(real_topk_ids[g_src].ravel(), minlength=num_experts)

        n_tokens = n_local_tokens + n_remote_tokens
        assert n_tokens.sum() == n_total_tokens, n_tokens.sum()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temt-dir', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--model-alias', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--start-idx', type=int, default=0, help='Start index for TEMT slicing')
    parser.add_argument('--end-idx', type=int, required=True, help='End index for TEMT slicing')
    parser.add_argument('--expt', type=str, choices=['step1', 'step2'], required=True, help="Choice the experiment ran.")
    parser.add_argument('--layer-idxes', type=int, nargs='+', required=True, help='List of layer indices')
    parser.add_argument('--num-experts', type=int, required=True, help='Number of experts')
    parser.add_argument('--num-gpus', type=int, default=8, help='Number of GPUs')
    parser.add_argument('--seq-len', type=int, default=1024, help='Sequence length')
    parser.add_argument('--correctness-test', '-T', action='store_true')
    parser.add_argument('--perfect-prediction', '-p', action='store_true')
    parser.add_argument('--max-dup-per-gpu', '-N', type=int, default=6)
    parser.add_argument('--n-local-dup-per-gpu', '-L', type=int, default=4)
    parser.add_argument('--max-send-len', '-P', type=int, default=6)
    parser.add_argument('--min-dup-experts-tokens', '-Q', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    from prefetch_rebalance_v8 import prefetch, rebalance

    args = parse_args()

    if args.perfect_prediction:
        print('Enabled perfect prediction')

    temt_dir = Path(args.temt_dir)
    save_dir = Path(args.save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)

    pred_type = int(args.expt[-1])
    layer_idxes = list(map(int, args.layer_idxes))

    # Columns of the result dataframe / csv
    percentiles = [1, 10, 25, 50, 75, 90, 99]
    str_percentiles = list(map(lambda x: f'p{x}', percentiles))
    columns = ['layer', 'mean', 'std', 'min'] + str_percentiles + ['max']

    evaluator = Evaluator(
        temt_dir,
        prefetch,
        rebalance,
        args.num_experts,
        args.num_gpus,
        args.seq_len,
        args.max_dup_per_gpu,
        args.n_local_dup_per_gpu,
        args.max_send_len,
        args.min_dup_experts_tokens,
        args.correctness_test,
        args.perfect_prediction,
    )

    # Evaluate and aggregate results for a layer
    def eval_and_write(layer_idx):
        results = evaluator.eval_layer(layer_idx, pred_type, args.start_idx, args.end_idx)

        def write_one(key: str, prefix: str):
            if key not in results:
                print(f"[warn] key '{key}' missing for layer {layer_idx}")
                return
            x = np.asarray(results[key])

            # 1-D → single column; 2-D → generic column names
            if x.ndim == 1:
                df = pd.DataFrame({"imbalance_ratio": x})
            else:
                cols = [f"c{i}" for i in range(x.shape[1])]
                df = pd.DataFrame(x, columns=cols)

            out_path = save_dir / f"{prefix}_layer_{layer_idx}.csv"
            df.to_csv(out_path, index=False)
            print(f"wrote {out_path}", flush=True)

        # write the two requested files
        write_one("baseline_imbalance_ratio",  "baseline")
        write_one("rebalance_imbalance_ratio", "rebalance")

        print(f"layer {layer_idx} done", flush=True)

    # run for all layers
    for i in layer_idxes:
        eval_and_write(i)
