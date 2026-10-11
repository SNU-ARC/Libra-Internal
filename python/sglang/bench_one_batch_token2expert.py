# Modified from SGLang v0.4.10 (https://github.com/sgl-project/sglang) for Libra (ICLR 2026).
"""
Dump the Token-to-Expert Mapping Table (TEMT) of a MoE model on one dataset.

Derived from SGLang's bench_one_batch.py: no server is launched; the model is loaded on all TP/EP ranks
with the low-level ModelRunner API and every dataset batch (mini_batch_size x input_len tokens) is run
as one prefill. For every MoE layer, rank 0 records for each token the real top-k expert ids ("real") and
the top-k ids of the lookahead predictor ("pred1" for --expt step1, "pred2" for --expt step2), and saves

    {result_dir}/{model_alias}_{dataset}_{mode}/{max_num_batches}/{real,predN}/layer_{L}.feather

with int64 columns req_idx (batch index // tp_size), token_idx (input token id) and e1..e{top_k}.
Only the TEMT dump (--correctness-test) is supported.

Usage: bench_one_batch_token2expert_expt.sh and generate_temt_{qwen3,glm45}.sh in the repository root.
"""

import argparse
import dataclasses
import logging
import multiprocessing
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    configure_logger,
    kill_process_tree,
    require_mlp_tp_gather,
    suppress_other_loggers,
)
from sglang.srt.models.mllama4 import Llama4ForConditionalGeneration

from typing import Iterable, List
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import gc

@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    mini_batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    result_filename: str = "result.jsonl"
    correctness_test: bool = False
    # This is only used for correctness test
    cut_len: int = 0
    log_decode_step: int = 0
    profile: bool = False
    profile_filename_prefix: str = "profile"

    # Libra TEMT options
    model_alias: str = ""
    dataset: str = ""
    dataset_dir: str = ""
    result_dir: str = ""
    mode: str = ""
    expt: str = ""
    start_portion: str = "25"
    end_portion: str = "25"
    target_seq_len: int = 1024
    max_num_batches: int = 0

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--mini-batch-size", type=int, nargs="+", default=BenchArgs.mini_batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument("--correctness-test", action="store_true")
        parser.add_argument("--cut-len", type=int, default=BenchArgs.cut_len)
        parser.add_argument(
            "--log-decode-step",
            type=int,
            default=BenchArgs.log_decode_step,
            help="Log decode latency by step, default is set to zero to disable.",
        )
        parser.add_argument(
            "--profile", action="store_true", help="Use Torch Profiler."
        )
        parser.add_argument(
            "--profile-filename-prefix",
            type=str,
            default=BenchArgs.profile_filename_prefix,
            help="Prefix of the profiling file names. The full profiling result file(s) be "
            '"[profile_filename_prefix]_batch[batch_size]_input[input_len]_output[output_len].trace.json.gz"',
        )

        # Libra TEMT options
        parser.add_argument(
            "--model-alias",
            type=str,
            required=True,
            help="Short alias for model name. Used in the result directory name",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            choices=[
                "bookcorpus",
                "codeforces",
                "deepseek_prover",
                "fineweb",
                "gsm8k",
                "hellaswag",
                "humanevalplus",
                "lmsys_chat_1m",
                "mixed",
            ],
            required=True,
            help="Dataset to dump the TEMT for ('mixed' interleaves batches of the other 8).",
        )
        parser.add_argument(
            "--dataset-dir",
            type=str,
            default=os.environ.get("LIBRA_DATA_DIR", "~/hf_datasets"),
            help="Directory with local copies of the HF datasets (default: $LIBRA_DATA_DIR or ~/hf_datasets)",
        )
        parser.add_argument(
            "--result-dir",
            type=str,
            required=True,
            help="Root directory of the TEMT output",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            help="train, test, ...",
        )
        parser.add_argument(
            "--expt",
            type=str,
            choices=["step1", "step2"],
            required=True,
            help="Lookahead distance of the predicted routing: step1 -> pred1 (used by the paper), step2 -> pred2.",
        )
        parser.add_argument(
            "--start-portion",
            type=str,
            default=BenchArgs.start_portion,
            help="Start of the HF split slice, in percent.",
        )
        parser.add_argument(
            "--end-portion",
            type=str,
            default=BenchArgs.end_portion,
            help="End of the HF split slice, in percent.",
        )
        parser.add_argument(
            "--target-seq-len",
            type=int,
            default=BenchArgs.target_seq_len,
            help="Unused (kept for the launcher scripts); the sequence length is --input-len.",
        )
        parser.add_argument(
            "--max-num-batches",
            type=int,
            default=BenchArgs.max_num_batches,
            help="Maximum number of batches to run (the TEMT scripts use 1000).",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


def load_model(server_args, port_args, bench_args, tp_rank):
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
    moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)

    model_config = ModelConfig.from_server_args(server_args)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        moe_ep_rank=moe_ep_rank,
        moe_ep_size=server_args.ep_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    if server_args.tp_size > 1:
        dist.barrier()
    return model_runner, tokenizer


def prepare_inputs_for_correctness_test(total_batches, mini_batch_size, input_len, tp_rank):
    input_ids = np.array(total_batches).reshape(mini_batch_size, input_len)
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(input_ids)):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(input_ids[i]),
            sampling_params=sampling_params,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)

    return reqs


@torch.no_grad
def extend(reqs, model_runner):
    if len(reqs) == 0:
        return None, None, None
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=None,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        enable_custom_logit_processor=False,
    )
    batch.prepare_for_extend()
    _maybe_prepare_dp_attn_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    # This fork's ModelRunner.forward returns only the logits output (no can_run_cuda_graph flag)
    logits_output = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)

    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()
    return next_token_ids, logits_output.next_token_logits, batch


def _maybe_prepare_dp_attn_batch(batch: ScheduleBatch, model_runner):
    if model_runner.server_args.enable_dp_attention:
        Scheduler.prepare_dp_attn_batch_raw(
            batch,
            dp_size=model_runner.server_args.dp_size,
            attn_tp_size=1,
            tp_group=model_runner.tp_group,
            get_idle_batch=None,
            disable_cuda_graph=model_runner.server_args.disable_cuda_graph,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            speculative_num_draft_tokens=None,
            require_mlp_tp_gather=require_mlp_tp_gather(model_runner.server_args),
            disable_overlap_schedule=model_runner.server_args.disable_overlap_schedule,
        )


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def gsm8k_to_texts(bench_args, tokenizer, mini_batch_size, input_len, tp_rank, max_num_batches):
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
    rank_print("Loading GSM8K dataset...")

    # Load GSM8K using Hugging Face dataset library
    hf_dataset_dir = Path(bench_args.dataset_dir)
    cache_path = hf_dataset_dir / "gsm8k"

    dataset = load_dataset(str(cache_path), 'default', split=f"{bench_args.mode}[{bench_args.start_portion}%:{bench_args.end_portion}%]", trust_remote_code=True)

    dataset_lines = [
        {"question": item["question"], "answer": item["answer"]} for item in dataset
    ]

    # Create prompts
    num_shots = 2
    max_num_questions = len(dataset_lines)

    total_batches = []
    input_ids = []
    for i in range(0, max_num_questions, num_shots):
        if len(total_batches) >= max_num_batches:
            break
        if i + num_shots >= max_num_questions:
            excess_num = i + num_shots - max_num_questions
            few_shot_examples = get_few_shot_examples(
                dataset_lines[i:], num_shots - excess_num
            )
        else:
            few_shot_examples = get_few_shot_examples(dataset_lines[i:], num_shots)
        
        for text in few_shot_examples:
            if len(total_batches) >= max_num_batches:
                break
            encoded = tokenizer.encode(text)
            input_ids.extend(encoded)
            if len(input_ids) > input_len * mini_batch_size:
                total_batches.append(input_ids[: input_len * mini_batch_size])
                input_ids.clear()

    return total_batches

def _batchify_stream(
    tokenizer,
    texts: Iterable[str],
    mini_batch_size: int,
    input_len: int,
    max_num_batches: int,
    *,
    carry_over: bool = True,
) -> List[List[int]]:
    """Tokenize a stream of `texts` and pack the tokens into batches.

    Each batch is a flat list of token ids of length exactly `mini_batch_size * input_len`.
    If `carry_over` is True, leftover tokens from one text continue filling the next
    batch; otherwise they are discarded (to mimic the original LMSYS example more
    closely). Defaults to carry-over packing to avoid wasting tokens.
    """
    cap = mini_batch_size * input_len
    total_batches: List[List[int]] = []
    buf: List[int] = []

    def _flush_if_ready() -> None:
        nonlocal buf
        while len(buf) >= cap:
            total_batches.append(buf[:cap])
            buf = buf[cap:]

    for text in texts:
        encoded = tokenizer.encode(text)
        if carry_over:
            buf.extend(encoded)
            _flush_if_ready()
            if len(total_batches) >= max_num_batches:
                break
        else:
            buf.extend(encoded)
            if len(buf) >= cap:
                total_batches.append(buf[:cap])
                buf.clear()
                if len(total_batches) >= max_num_batches:
                    break
            else:
                # drop remainder
                buf.clear()

    return total_batches

def bookcorpus_to_texts(bench_args, tokenizer, mini_batch_size, input_len, tp_rank, max_num_batches):
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
    rank_print("Loading BookCorpus dataset...")

    hf_dataset_dir = Path(bench_args.dataset_dir)
    cache_path = hf_dataset_dir / "bookcorpus" / "bookcorpus"

    dataset = load_dataset(str(cache_path), split=f"{bench_args.mode}[{bench_args.start_portion}%:{bench_args.end_portion}%]", trust_remote_code=True)
    def _texts():
        for sample in dataset:
            txt = sample.get("text") or ""
            if txt:
                yield txt

    return _batchify_stream(tokenizer, _texts(), mini_batch_size, input_len, max_num_batches, carry_over=True)

def lmsyschat_to_texts(bench_args, tokenizer, mini_batch_size, input_len, tp_rank, max_num_batches):
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
    rank_print("Loading LMSYS-Chat-1M dataset for 1K sequence batches...")

    hf_dataset_dir = Path(bench_args.dataset_dir)
    cache_path = hf_dataset_dir / "lmsys-chat-1m"

    # Load dataset
    dataset = load_dataset(str(cache_path), split=f"{bench_args.mode}[{bench_args.start_portion}%:{bench_args.end_portion}%]", trust_remote_code=True)

    total_batches = []
    input_ids = []

    for sample in dataset:
        if len(total_batches) >= max_num_batches:
            break
        for turn in sample["conversation"]:
            if len(total_batches) >= max_num_batches:
                break
            cur_text = f"{turn['role']}: {turn['content']}"
            for text in cur_text:
                if len(total_batches) >= max_num_batches:
                    break
                encoded = tokenizer.encode(text)
                input_ids.extend(encoded)
                if len(input_ids) > input_len * mini_batch_size:
                    total_batches.append(input_ids[: input_len * mini_batch_size])
                    input_ids.clear()

    return total_batches

def fineweb_to_texts(bench_args, tokenizer, mini_batch_size, input_len, tp_rank, max_num_batches):
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
    rank_print("Loading Fineweb dataset...")

    hf_dataset_dir = Path(bench_args.dataset_dir)
    cache_path = hf_dataset_dir / "fineweb"

    dataset = load_dataset(
        "parquet", 
        data_files={"train":f"{cache_path}/sample/10BT/000_00000.parquet"}, 
        split=f"{bench_args.mode}[{bench_args.start_portion}%:{bench_args.end_portion}%]", 
        trust_remote_code=True
    )

    total_batches = []
    input_ids = []
    for sample in dataset:
        if len(total_batches) >= max_num_batches:
            break
        for text in sample['text']:
            if len(total_batches) >= max_num_batches:
                break
            encoded = tokenizer.encode(text)
            input_ids.extend(encoded)
            if len(input_ids) > input_len * mini_batch_size:
                total_batches.append(input_ids[: input_len * mini_batch_size])
                input_ids.clear()
    
    return total_batches

def hellaswag_to_texts(bench_args, tokenizer, mini_batch_size, input_len, tp_rank, max_num_batches):
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
    rank_print("Loading Hellaswag dataset...")

    hf_dataset_dir = Path(bench_args.dataset_dir)
    cache_path = hf_dataset_dir / "hellaswag"

    dataset = load_dataset(
        "parquet", 
        data_files={"train":f"{cache_path}/data/train-00000-of-00001.parquet"}, 
        split=f"{bench_args.mode}[{bench_args.start_portion}%:{bench_args.end_portion}%]", 
        trust_remote_code=True
    )

    total_batches = []
    input_ids = []
    for sample in dataset:
        if len(total_batches) >= max_num_batches:
            break
        for text in sample['ctx']:
            if len(total_batches) >= max_num_batches:
                break
            encoded = tokenizer.encode(text)
            input_ids.extend(encoded)
            if len(input_ids) > input_len * mini_batch_size:
                total_batches.append(input_ids[: input_len * mini_batch_size])
                input_ids.clear()
    
    return total_batches

def codeforces_to_texts(bench_args, tokenizer, mini_batch_size, input_len, tp_rank, max_num_batches):
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
    rank_print("Loading Codeforces dataset...")

    hf_dataset_dir = Path(bench_args.dataset_dir)
    cache_path = hf_dataset_dir / "codeforces"

    dataset = load_dataset(str(cache_path), split=f"{bench_args.mode}[{bench_args.start_portion}%:{bench_args.end_portion}%]", trust_remote_code=True)

    total_batches = []
    input_ids = []
    for sample in dataset:
        if len(total_batches) >= max_num_batches:
            break
        if sample['description'] is None:
            continue
        for text in sample['description']:
            if len(total_batches) >= max_num_batches:
                break
            encoded = tokenizer.encode(text)
            input_ids.extend(encoded)
            if len(input_ids) > input_len * mini_batch_size:
                total_batches.append(input_ids[: input_len * mini_batch_size])
                input_ids.clear()
    
    return total_batches

def humanevalplus_to_texts(bench_args, tokenizer, mini_batch_size, input_len, tp_rank, max_num_batches):
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
    rank_print("Loading Humanevalplus dataset...")

    hf_dataset_dir = Path(bench_args.dataset_dir)
    cache_path = hf_dataset_dir / "humanevalplus"

    dataset = load_dataset(str(cache_path), split=f"test[{bench_args.start_portion}%:{bench_args.end_portion}%]", trust_remote_code=True)
    
    total_batches = []
    input_ids = []
    for sample in dataset:
        if len(total_batches) >= max_num_batches:
            break
        for text in f"prompt: {sample['prompt']}, test: {sample['test']}":
            if len(total_batches) >= max_num_batches:
                break
            encoded = tokenizer.encode(text)
            input_ids.extend(encoded)
            if len(input_ids) > input_len * mini_batch_size:
                total_batches.append(input_ids[: input_len * mini_batch_size])
                input_ids.clear()
    
    return total_batches

def deepseek_prover_v1_to_texts(bench_args, tokenizer, mini_batch_size, input_len, tp_rank, max_num_batches):
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
    rank_print("Loading DeepSeek-Prover-V1 dataset...")

    hf_dataset_dir = Path(bench_args.dataset_dir)
    cache_path = hf_dataset_dir / "DeepSeek-Prover-V1"

    dataset = load_dataset(str(cache_path), 'default', split=f"{bench_args.mode}[{bench_args.start_portion}%:{bench_args.end_portion}%]", trust_remote_code=True)

    texts = [sample['formal_statement'] for sample in dataset]
    
    total_batches = []
    input_ids = []
    for sample in dataset:
        if len(total_batches) >= max_num_batches:
            break
        for text in sample['formal_statement']:
            if len(total_batches) >= max_num_batches:
                break
            encoded = tokenizer.encode(text)
            input_ids.extend(encoded)
            if len(input_ids) > input_len * mini_batch_size:
                total_batches.append(input_ids[: input_len * mini_batch_size])
                input_ids.clear()
    
    return total_batches

def mixed_to_texts(bench_args, tokenizer, mini_batch_size, input_len, tp_rank, max_num_batches):
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
    rank_print("Loading mixed dataset...")

    num_datasets = 8
    bench_args.start_portion = 0
    bench_args.end_portion = 1
    dataset1 = bookcorpus_to_texts(bench_args, tokenizer, mini_batch_size, input_len, tp_rank, max_num_batches // num_datasets)
    bench_args.end_portion = 100
    dataset2 = codeforces_to_texts(bench_args, tokenizer, mini_batch_size, input_len, tp_rank, max_num_batches // num_datasets)
    dataset3 = deepseek_prover_v1_to_texts(bench_args, tokenizer, mini_batch_size, input_len, tp_rank, max_num_batches // num_datasets)
    bench_args.end_portion = 1
    dataset4 = fineweb_to_texts(bench_args, tokenizer, mini_batch_size, input_len, tp_rank, max_num_batches // num_datasets)
    bench_args.end_portion = 100
    dataset5 = gsm8k_to_texts(bench_args, tokenizer, mini_batch_size, input_len, tp_rank, max_num_batches // num_datasets)
    dataset6 = hellaswag_to_texts(bench_args, tokenizer, mini_batch_size, input_len, tp_rank, max_num_batches // num_datasets)
    dataset7 = humanevalplus_to_texts(bench_args, tokenizer, mini_batch_size, input_len, tp_rank, max_num_batches // num_datasets)
    bench_args.end_portion = 1
    dataset8 = lmsyschat_to_texts(bench_args, tokenizer, mini_batch_size, input_len, tp_rank, max_num_batches // num_datasets)

    N = min(len(dataset1), len(dataset2), len(dataset3), len(dataset4), len(dataset5), len(dataset6), len(dataset7), len(dataset8))

    from itertools import chain

    return list(chain.from_iterable(zip(dataset1[:N], dataset2[:N], dataset3[:N], dataset4[:N], dataset5[:N], dataset6[:N], dataset7[:N], dataset8[:N])))

def correctness_test(
    server_args,
    port_args,
    bench_args: BenchArgs,
    tp_rank,
):
    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, bench_args, tp_rank)


    total_batches = []
    if bench_args.dataset == "gsm8k":
        total_batches = gsm8k_to_texts(bench_args, tokenizer, bench_args.mini_batch_size[0], bench_args.input_len[0], tp_rank, bench_args.max_num_batches)
    elif bench_args.dataset == "bookcorpus":
        total_batches = bookcorpus_to_texts(bench_args, tokenizer, bench_args.mini_batch_size[0], bench_args.input_len[0], tp_rank, bench_args.max_num_batches)
    elif bench_args.dataset == "lmsys_chat_1m":
        total_batches = lmsyschat_to_texts(bench_args, tokenizer, bench_args.mini_batch_size[0], bench_args.input_len[0], tp_rank, bench_args.max_num_batches)
    elif bench_args.dataset == "fineweb":
        total_batches = fineweb_to_texts(bench_args, tokenizer, bench_args.mini_batch_size[0], bench_args.input_len[0], tp_rank, bench_args.max_num_batches)
    elif bench_args.dataset == "hellaswag":
        total_batches = hellaswag_to_texts(bench_args, tokenizer, bench_args.mini_batch_size[0], bench_args.input_len[0], tp_rank, bench_args.max_num_batches)
    elif bench_args.dataset == "codeforces":
        total_batches = codeforces_to_texts(bench_args, tokenizer, bench_args.mini_batch_size[0], bench_args.input_len[0], tp_rank, bench_args.max_num_batches)
    elif bench_args.dataset == "humanevalplus":
        total_batches = humanevalplus_to_texts(bench_args, tokenizer, bench_args.mini_batch_size[0], bench_args.input_len[0], tp_rank, bench_args.max_num_batches)
    elif bench_args.dataset == "deepseek_prover":
        total_batches = deepseek_prover_v1_to_texts(bench_args, tokenizer, bench_args.mini_batch_size[0], bench_args.input_len[0], tp_rank, bench_args.max_num_batches)
    elif bench_args.dataset == "mixed":
        total_batches = mixed_to_texts(bench_args, tokenizer, bench_args.mini_batch_size[0], bench_args.input_len[0], tp_rank, bench_args.max_num_batches)

    total_batches = total_batches[:bench_args.max_num_batches]
    rank_print(f"Number of prompts: {len(total_batches)}")

    if isinstance(model_runner.model, Llama4ForConditionalGeneration):
        model = model_runner.model.language_model.model
    else:
        model = model_runner.model.model

    if bench_args.expt == 'step1':
        model.num_prediction_layers = 1
    elif bench_args.expt == 'step2':
        model.num_prediction_layers = 2


    if bench_args.expt in ("token2expert", "step1", "step2"):
        main_token_experts_mapping_table = [
            []
            for _ in range(len(model.layers))
        ]
    elif bench_args.expt == "expert2token":
        main_token_experts_mapping_table = [
            defaultdict(lambda: defaultdict(list))
            for _ in range(len(model.layers))
        ]
    else:
        main_token_experts_mapping_table = [
            defaultdict(list) for _ in range(len(model.layers))
        ]
    if bench_args.expt in ("step1", "step2"):
        baseline_token_experts_mapping_table = [
            []
            for _ in range(len(model.layers))
        ]
    expert_loads = [[] for _ in range(len(model.layers))]
    num_layer = 256

    if tp_rank == 0:
        # The total number of steps:
        total_steps = len(total_batches)
        pbar = tqdm(total=total_steps, desc="Processing prompts")

    start_idx = 0
    if tp_rank == 0:
        for _ in range(start_idx):
            pbar.update(1)

    try:
        for i in range(start_idx, len(total_batches)):
            if i % 100 == 0:
                props      = torch.cuda.get_device_properties(server_args.device)
                total_mem  = props.total_memory
                used_mem   = torch.cuda.memory_allocated(server_args.device)
                used_ratio = used_mem / total_mem
                if used_ratio > 0.95:
                    gc.collect()

            cur_batch = total_batches[i:i + 1]

            # Prepare inputs
            reqs = prepare_inputs_for_correctness_test(
                cur_batch, bench_args.mini_batch_size[0], bench_args.input_len[0], tp_rank
            )

            # Extend (prefill w/ KV cache)
            next_token_ids, next_token_logits, batch = extend(reqs, model_runner)
            del next_token_ids, next_token_logits, batch
            torch.cuda.empty_cache()

            if tp_rank == 0:
                # Access TEMT
                curr_input_ids = model.curr_input_ids.cpu().tolist()
                del model.curr_input_ids
                for layer_idx, layer in enumerate(model.layers):
                    if layer.token_experts_mapping_table is not None:
                        # Collect TEMT
                        for j, expert_ids in enumerate(
                            layer.token_experts_mapping_table
                        ):
                            if bench_args.expt in ("token2expert", "step1", "step2"):
                                main_token_experts_mapping_table[layer_idx].append(
                                    [i // server_args.tp_size, curr_input_ids[j], *expert_ids]
                                )
                            elif bench_args.expt == "expert2token":
                                pass
                    if layer.baseline_token_experts_mapping_table is not None:
                        for j, expert_ids in enumerate(
                            layer.baseline_token_experts_mapping_table
                        ):
                            if bench_args.expt in ("step1", "step2"):
                                baseline_token_experts_mapping_table[layer_idx].append(
                                    [i // server_args.tp_size, curr_input_ids[j], *expert_ids]
                                )
                del curr_input_ids

                # Update the progress bar after each iteration
                pbar.update(1)
    finally:
        if tp_rank == 0:
            pbar.close()
            print(f'req_idx = {i / server_args.tp_size}')
            if bench_args.expt in ("token2expert", "step1", "step2", "all"):
                base_dir = Path(bench_args.result_dir) / f"{bench_args.model_alias}_{bench_args.dataset}_{bench_args.mode}"
                if bench_args.max_num_batches == 0:
                    portion = f"{bench_args.start_portion}_{bench_args.end_portion}"
                else:
                    portion = str(bench_args.max_num_batches)
                baseline_dir = base_dir / portion / "real"
                baseline_dir.mkdir(parents=True, exist_ok=True)
                if bench_args.expt == "step1":
                    base_dir = base_dir / portion / "pred1"
                elif bench_args.expt == "step2":
                    base_dir = base_dir / portion / "pred2"
                else:
                    base_dir = base_dir / portion / "real"
                base_dir.mkdir(parents=True, exist_ok=True)

                print(f"Save Token-to-Expert mapping table...")
                for layer_idx, layer_info in enumerate(main_token_experts_mapping_table):
                    if not layer_info:
                        continue
                    file_path = base_dir / f"layer_{layer_idx}.feather"
                    print(f"Saving {file_path}...")
                    columns = ['req_idx', 'token_idx', *[f'e{i}' for i in range(1, len(layer_info[0])-1)]]
                    pd.DataFrame(layer_info, columns=columns).to_feather(str(file_path))
                for layer_idx, layer_info in enumerate(baseline_token_experts_mapping_table):
                    if not layer_info:
                        continue
                    file_path = baseline_dir / f"layer_{layer_idx}.feather"
                    print(f"Saving {file_path}...")
                    columns = ['req_idx', 'token_idx', *[f'e{i}' for i in range(1, len(layer_info[0])-1)]]
                    pd.DataFrame(layer_info, columns=columns).to_feather(str(file_path))

def main(server_args, bench_args):
    server_args.cuda_graph_max_bs = max(bench_args.batch_size)

    _set_envs_and_config(server_args)

    if server_args.model_path:
        if bench_args.correctness_test:
            work_func = correctness_test
        else:
            raise ValueError(
                "Only the TEMT dump is supported; pass --correctness-test "
                "(see bench_one_batch_token2expert_expt.sh)."
            )
    else:
        raise ValueError(
            "Provide --model-path for running the tests or "
            "provide --result-filename for plotting the results"
        )

    port_args = PortArgs.init_new(server_args)

    if server_args.tp_size == 1:
        work_func(server_args, port_args, bench_args, 0)
    else:
        workers = []
        for tp_rank in range(server_args.tp_size):
            proc = multiprocessing.Process(
                target=work_func,
                args=(
                    server_args,
                    port_args,
                    bench_args,
                    tp_rank,
                ),
            )
            proc.start()
            workers.append(proc)

        for proc in workers:
            proc.join()

        proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)
    bench_args.dataset_dir = os.path.expanduser(bench_args.dataset_dir)
    bench_args.result_dir = os.path.expanduser(bench_args.result_dir)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)
