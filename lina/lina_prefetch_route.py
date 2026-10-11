import numpy as np
import pandas as pd
import torch
import triton
import triton.language as tl
from pathlib import Path
from typing import Optional
from tqdm import tqdm


# Only supports L=3
@triton.jit
def compute_num_devices_kernel(
    # inputs
    paths_ptr,
    ept_keys_ptr,
    ept_expert_ids_ptr,
    ept_values_ptr, 
    # outputs
    num_devices_float_ptr,
    # scalars
    seq_len,
    num_experts,
    num_gpus,
    max_dup_per_gpu,
    ept_values_len,
    topk: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    expert_id = tl.program_id(0)

    total_seq_len = seq_len * num_gpus
    s = 0.0

    base_offsets = tl.arange(0, BLOCK_SIZE)

    # For all tokens, binary search expert path table and sum probabilities for this expert
    for i in range(0, total_seq_len, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < total_seq_len

        # Load expert path for each token
        e0 = tl.load(paths_ptr + offsets, mask=mask, other=0)
        e1 = tl.load(paths_ptr + total_seq_len + offsets, mask=mask, other=0)
        e2 = tl.load(paths_ptr + total_seq_len * 2 + offsets, mask=mask, other=0)
        e3 = tl.load(paths_ptr + total_seq_len * 3 + offsets, mask=mask, other=0)

        # Pack path for lookup
        packed_path = (e0 << 48) | (e1 << 32) | (e2 << 16) | e3
        low = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        high = tl.full((BLOCK_SIZE,), ept_values_len-1, dtype=tl.int64)

        # Binary search to find matching path in expert path table
        for _ in range(30):
            mid = (low + high) // 2
            idx = tl.load(ept_keys_ptr + mid, mask=mask)
            cond = packed_path > idx
            low = tl.where(cond, mid + 1, low)
            high = tl.where(cond, high, mid)

        # For each top-k expert, accumulate probability if matches expert_id
        for k in range(topk):
            expert_ids = tl.load(ept_expert_ids_ptr + low * topk + k, mask=mask, other=-1)
            expert_mask = expert_ids == expert_id
            value = tl.load(ept_values_ptr + low * topk + k, mask=mask & expert_mask, other=0.0)
            s += tl.sum(value)
    
    # Normalize and store result
    num_devices_float = s / seq_len * max_dup_per_gpu
    tl.store(num_devices_float_ptr + expert_id, num_devices_float)


@triton.jit
def prefetch_kernel(
    # inputs
    num_devices_ptr,
    # outputs
    next_expert_gpu_mapping_ptr,
    # temp buffers
    n_dups_ptr,
    # scalars
    num_gpus: tl.constexpr,
    num_experts: tl.constexpr,
    max_dup_per_gpu,
):
    experts_per_gpu = num_experts // num_gpus
    # For each expert, assign to GPUs in round-robin fashion up to n times
    for e in range(num_experts):
        n = tl.load(num_devices_ptr + e)
        for i in range(num_gpus):
            g = (e + i) % num_gpus
            # Assign expert to GPU if not original and under duplication limit
            if n > 0 and (g != e // experts_per_gpu and tl.atomic_add(n_dups_ptr + g, 0) < max_dup_per_gpu):
                tl.store(next_expert_gpu_mapping_ptr + e * num_gpus + g, True)
                tl.atomic_add(n_dups_ptr + g, 1)
                n -= 1
        # Always assign expert to its original GPU
        tl.store(next_expert_gpu_mapping_ptr + e * num_gpus + e // experts_per_gpu, True)


@triton.jit
def prefetch_postprocess_kernel(
    # inputs
    next_expert_gpu_mapping_ptr,
    # outputs
    send_map_ptr,
    send_len_ptr,
    recv_len_ptr,
    recv_experts_ptr,
    recv_num_ptr,
    # scalars
    num_gpus,
    cur_rank,
    experts_per_gpu,
    max_dup_per_gpu,
):
    e = tl.program_id(0)
    g_dst = tl.program_id(1)

    g_src = e // experts_per_gpu
    map_val = tl.load(next_expert_gpu_mapping_ptr + e * num_gpus + g_dst)

    cond = (g_src != g_dst) & map_val

    sidx = tl.atomic_add(send_len_ptr + g_src * num_gpus + g_dst, 1, mask=cond)
    tl.store(send_map_ptr + ((g_src * num_gpus + g_dst) * max_dup_per_gpu) + sidx, e % experts_per_gpu, mask=cond)
    tl.atomic_add(recv_len_ptr + g_dst * num_gpus + g_src, 1, mask=cond)

    cond2 = cond & (g_dst == cur_rank)

    ridx = tl.atomic_add(recv_num_ptr, 1, mask=cond2)
    tl.store(recv_experts_ptr + ridx, e, mask=cond2)


@triton.jit
def compute_expert_gpu_load_kernel(
    # inputs
    topk_ids_ptr,
    # outputs
    expert_gpu_load_ptr,
    # scalars
    num_gpus,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    gpu_id = tl.program_id(0)
    base_offsets = tl.arange(0, BLOCK_SIZE)
    # For each token, count how many times each expert is used on each GPU
    for i in range(0, M, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < M
        expert_ids = tl.load(topk_ids_ptr + gpu_id * M + offsets, mask=mask)
        tl.atomic_add(expert_gpu_load_ptr + expert_ids * num_gpus + gpu_id, 1, mask=mask)


@triton.jit
def route_kernel(
    # inputs
    expert_gpu_mapping_ptr,
    expert_gpu_load_ptr,
    expert_gpu_count_ptr,
    start_indices_ptr,
    # outputs
    gpu_expert_info_ptr,
    gpu_load_out_ptr,
    gpu_expert_summary_ptr,
    # scalars
    num_experts,
    num_gpus: tl.constexpr,
    experts_per_gpu,
):
    e = tl.program_id(0)
    g_src = tl.program_id(1)
    offsets = tl.arange(0, num_gpus)

    # mask of GPUs that have expert e
    g_dst_mask = tl.load(expert_gpu_mapping_ptr + e * num_gpus + offsets)

    # number of tokens that GPU g_src have and go to expert e
    l = tl.load(expert_gpu_load_ptr + e * num_gpus + g_src)

    # number of GPUs that have expert e
    n = tl.load(expert_gpu_count_ptr + e)

    # start index of tokens that GPU g_src have and go to expert e in the layout
    start_idx = tl.load(start_indices_ptr + e * num_gpus + g_src)

    # Compute how many tokens to send to each GPU (evenly)
    cnt = tl.where(g_dst_mask, l // n, 0)
    g_dst_mask_psum = tl.cumsum(tl.cast(g_dst_mask, tl.int64))
    cnt += g_dst_mask & (g_dst_mask_psum <= (l % n))
    mask = cnt > 0

    # Compute indices of tokens to send to each GPU
    idx = start_idx + tl.cumsum(cnt) - cnt

    # Write routing info
    tl.store(gpu_expert_info_ptr + ((offsets * num_experts + e) * num_gpus + g_src) * 2    , idx, mask=mask)
    tl.store(gpu_expert_info_ptr + ((offsets * num_experts + e) * num_gpus + g_src) * 2 + 1, cnt, mask=mask)
    ori_mask = e // experts_per_gpu == offsets
    tl.atomic_add(gpu_load_out_ptr + offsets * 2    , cnt, mask= ori_mask & mask)
    tl.atomic_add(gpu_load_out_ptr + offsets * 2 + 1, cnt, mask=~ori_mask & mask)
    tl.atomic_add(gpu_expert_summary_ptr + (offsets * num_experts + e) * 2    , cnt, mask=mask)
    tl.atomic_add(gpu_expert_summary_ptr + (offsets * num_experts + e) * 2 + 1,   1, mask=mask)


@triton.jit
def route_postprocess_kernel(
    # inputs
    expert_gpu_mapping_ptr,
    # outputs
    gpu_expert_info_ptr,
    gpu_expert_summary_ptr,
    # scalars
    num_experts,
    num_gpus: tl.constexpr,
):
    e = tl.program_id(0)
    g = tl.program_id(1)

    # if expert_gpu_mapping[e, g]:                      cond1
    #     if gpu_expert_summary_out[g, e, 0] == 0:      cond2
    #         gpu_expert_info_out[g, e, 0] = 0          a
    #         gpu_expert_summary_out[g, e, 1] = 1       b
    # else:
    #     gpu_expert_summary_out[g, e] = -1             b

    v = tl.arange(0, 2)

    cond1 = tl.load(expert_gpu_mapping_ptr + e * num_gpus + g)
    cond2 = tl.load(gpu_expert_summary_ptr + (g * num_experts + e) * 2) == 0

    a_ptr = gpu_expert_info_ptr + ((g * num_experts + e) * num_gpus) * 2 + v
    b_ptr = gpu_expert_summary_ptr + (g * num_experts + e) * 2 + v

    a = tl.load(a_ptr)
    b = tl.load(b_ptr)

    new_a = tl.where(cond1 & cond2, 0, a)
    new_b = tl.where(cond1, tl.where(cond2, v, b), -1)

    tl.store(a_ptr, new_a)
    tl.store(b_ptr, new_b)


class Lina:
    def __init__(
        self,
        num_layers: int,
        L: int = 3,
        num_gpus: int = 8,
        topk: int = 8,
        max_dup_per_gpu: int = 6,
        device='cuda',
        verbose: bool = False,
    ):
        self.num_layers = num_layers
        self.L = L
        self.num_gpus = num_gpus
        self.topk = topk
        self.max_dup_per_gpu = max_dup_per_gpu
        self.device = device
        self.verbose = verbose

        assert L == 3

        self.expert_path_tables = None
        self.paths = None
        self.num_experts = None

        self.fields = [f'f{i}' for i in range(L+1)]
        self.dtype_key = np.dtype([(f, np.int64) for f in self.fields])
        self.gpu_indices = torch.arange(num_gpus, dtype=torch.int64, device=device)

    def load_expert_path_tables(
        self,
        expert_path_table_dir: str | Path,
        start_layer: int,
        cache_dir: Optional[str | Path] = None,
        step=1,
    ):
        # Load the expert path tables and precompute, per path, the top-k next-layer experts and
        # their probabilities, sorted by the packed path key (for binary search).
        expert_path_table_dir = Path(expert_path_table_dir)
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

        if type(expert_path_table_dir) is str:
            expert_path_table_dir = Path(expert_path_table_dir)
        
        self.expert_path_tables = [None] * self.num_layers

        for i in tqdm(range(self.L+start_layer, self.num_layers-1), disable=not self.verbose):
            ept_data = pd.read_feather(expert_path_table_dir / f'layer_{i}.feather').values

            keys, expert_ids, values = None, None, None

            ept = torch.tensor(ept_data, device=self.device)
            assert ept.dtype == torch.int64
            assert len(ept) == len(set(map(tuple, ept[:, :self.L+1])))

            keys = ept[:, :self.L+1]
            values = ept[:, self.L+1:].float()
            values /= values.sum(axis=1, keepdims=True)
            
            packed_keys = (keys[:, 0] << 48) | (keys[:, 1] << 32) | (keys[:, 2] << 16) | keys[:, 3]
            values, expert_ids = torch.topk(values, self.topk, dim=1)
            
            sorted_indices = torch.argsort(packed_keys)
            keys = packed_keys[sorted_indices]
            expert_ids = expert_ids[sorted_indices]
            values = values[sorted_indices]
            self.expert_path_tables[i] = (keys, expert_ids, values)
        
        self.num_experts = ept_data.shape[1] - self.L - 1

    def new_batch(self, tot_seq_len: int):
        assert tot_seq_len % self.num_gpus == 0
        self.seq_len = tot_seq_len // self.num_gpus
        self.paths = torch.empty((self.num_layers, tot_seq_len), dtype=torch.int64, device=self.device)

    def record(self, layer_id: int, topk_ids: torch.Tensor):
        assert self.paths is not None
        assert self.paths.shape[1] == topk_ids.shape[0], (self.paths.shape, topk_ids.shape)
        self.paths[layer_id, :] = topk_ids[:, 0]

    def prefetch(
        self,
        layer_id: int,
        cur_rank: int,
        send_map_local_out: torch.Tensor,
        send_map_local_len_out: torch.Tensor,
        recv_map_local_len_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.expert_path_tables is not None, 'Call load_expert_path_tables first'
        assert self.num_experts is not None
        assert self.paths is not None
        assert layer_id >= self.L
        experts_per_gpu = self.num_experts // self.num_gpus

        paths = self.paths[layer_id-self.L:layer_id+1, :]
        ept_keys, ept_expert_ids, ept_values = self.expert_path_tables[layer_id]

        assert paths.is_contiguous()
        assert ept_keys.is_contiguous()
        assert ept_values.is_contiguous()

        # 1. Compute num_devices_float (n_e in the Lina paper)
        num_devices_float = torch.empty((self.num_experts,), dtype=torch.float32, device=self.device)
        grid = (self.num_experts,)
        compute_num_devices_kernel[grid](
            paths,
            ept_keys,
            ept_expert_ids,
            ept_values,
            num_devices_float,
            self.seq_len,
            self.num_experts,
            self.num_gpus,
            self.max_dup_per_gpu,
            len(ept_values),
            self.topk,
            BLOCK_SIZE=1024,
        )

        # 2. Compute num_devices (integer number of replicas per expert)
        num_devices = torch.floor(num_devices_float).int()
        sorted_num_devices_float = torch.argsort(torch.fmod(num_devices_float, 1), descending=True, stable=True)
        mask = sorted_num_devices_float < self.num_gpus*self.max_dup_per_gpu - num_devices.sum()
        num_devices.add_(mask.int())
        
        # 3. prefetch
        next_expert_gpu_mapping = torch.zeros((self.num_experts, self.num_gpus), dtype=torch.bool, device=self.device)
        n_dups = torch.zeros((self.num_gpus,), dtype=torch.int64, device=self.device)
        prefetch_kernel[(1,)](
            num_devices,
            next_expert_gpu_mapping,
            n_dups,
            self.num_gpus,
            self.num_experts,
            self.max_dup_per_gpu,
        )

        # 4. Postprocess: build the send/recv maps on the GPU (no host round-trip)
        recv_experts_next_cur_rank = torch.full((self.max_dup_per_gpu + 1,), 0x3f3f3f3f, dtype=torch.int64, device=self.device)
        recv_num_experts_next = torch.zeros((1,), dtype=torch.int64, device=self.device)
        send_map_local_out.fill_(0x3f3f3f3f)
        send_map_local_len_out.fill_(0)
        recv_map_local_len_out.fill_(0)

        grid = (self.num_experts, self.num_gpus)
        prefetch_postprocess_kernel[grid](
            next_expert_gpu_mapping,
            send_map_local_out,
            send_map_local_len_out,
            recv_map_local_len_out,
            recv_experts_next_cur_rank,
            recv_num_experts_next,
            self.num_gpus,
            cur_rank,
            experts_per_gpu,
            self.max_dup_per_gpu,
        )

        res, _ = send_map_local_out.sort()
        send_map_local_out.copy_(res)

        res, _ = recv_experts_next_cur_rank.sort()
        recv_experts_next_cur_rank.copy_(res)

        return next_expert_gpu_mapping, recv_experts_next_cur_rank, recv_num_experts_next


    def route(
        self,
        topk_ids: torch.Tensor,
        expert_gpu_mapping: torch.Tensor,
        gpu_expert_info_out: torch.Tensor,
        gpu_load_out: torch.Tensor,
        gpu_expert_summary_out: torch.Tensor,
    ) -> None:
        assert self.num_experts is not None
        assert expert_gpu_mapping.is_contiguous()
        experts_per_gpu = self.num_experts // self.num_gpus
        topk_ids = topk_ids.reshape(self.num_gpus, -1).contiguous()

        gpu_expert_info_out.fill_(-1)
        gpu_load_out.fill_(0)
        gpu_expert_summary_out.fill_(0)

        # 1. Compute expert_gpu_load
        expert_gpu_load = torch.zeros((self.num_experts, self.num_gpus), dtype=torch.int64, device=self.device)
        grid = (self.num_gpus,)
        compute_expert_gpu_load_kernel[grid](
            topk_ids,
            expert_gpu_load,
            self.num_gpus,
            topk_ids.size(1),
            BLOCK_SIZE=1024,
        )

        # 2. route
        expert_gpu_count = expert_gpu_mapping.count_nonzero(dim=1)
        expert_gpu_load_flat = expert_gpu_load.reshape(-1)
        start_indices = expert_gpu_load_flat.cumsum(dim=0) - expert_gpu_load_flat

        grid = (self.num_experts, self.num_gpus)
        route_kernel[grid](
            expert_gpu_mapping,
            expert_gpu_load,
            expert_gpu_count,
            start_indices,
            gpu_expert_info_out,
            gpu_load_out,
            gpu_expert_summary_out,
            self.num_experts,
            self.num_gpus,
            experts_per_gpu,
        )

        route_postprocess_kernel[grid](
            expert_gpu_mapping,
            gpu_expert_info_out,
            gpu_expert_summary_out,
            self.num_experts,
            self.num_gpus,
        )