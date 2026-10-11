import pandas as pd
from pathlib import Path
import numpy as np
import argparse

def make_expert_path_table(expert_selection, L, num_experts):
    bincounts = expert_selection \
        .groupby([f'e{i}' for i in range(L+1)])[f'e{L+1}'] \
        .apply(lambda arr: np.bincount(arr, minlength=num_experts)) \
        .reset_index().values

    assert bincounts.shape[1] == L+2
    ept = np.hstack([bincounts[:, :L+1], np.vstack(bincounts[:, L+1])])
    assert ept.shape[1] == L+1 + num_experts

    return ept

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--layer-idx', type=int, required=True, help='Layer index')
    parser.add_argument('--save-dir', type=str, required=True, help='Directory to save expert path table')
    parser.add_argument('--temt-dirs', type=str, nargs='+', required=True, help='List of TEMT directories')
    parser.add_argument('--start-idx', type=int, default=0, help='Start index for TEMT slicing')
    parser.add_argument('--end-idx', type=int, required=True, help='End index for TEMT slicing')
    parser.add_argument('--num-experts', type=int, required=True, help='Number of experts')
    parser.add_argument('-L', type=int, default=3, help='L, length of path')
    parser.add_argument('--moe-layer-freq', type=int, default=1, help='MoE layer frequency (stride)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    layer_idx = args.layer_idx
    L = args.L
    num_experts = args.num_experts
    moe_layer_freq = args.moe_layer_freq
    assert layer_idx >= L * moe_layer_freq

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    expert_selections = []
    for base_dir in args.temt_dirs:
        base_dir = Path(base_dir)

        dfs = [pd.read_feather(base_dir / f'layer_{i}.feather') for i in range(layer_idx-L*moe_layer_freq, layer_idx+moe_layer_freq+1, moe_layer_freq)]
        dfs = [df.iloc[args.start_idx:args.end_idx] for df in dfs]

        expert_selection = pd.concat([df['e1'].rename(f'e{i}') for i, df in enumerate(dfs)], axis=1)
        expert_selections.append(expert_selection)

    data = make_expert_path_table(pd.concat(expert_selections, axis=0, ignore_index=True), L, num_experts)

    layer_cols = [f'layer_{j}' for j in range(layer_idx-L*moe_layer_freq, layer_idx+1, moe_layer_freq)]
    column_names = [*layer_cols, *[f'e{i}' for i in range(num_experts)]]
    df = pd.DataFrame(data, columns=column_names)
    df.to_feather(str(save_dir / f'layer_{layer_idx}.feather'))

    print(f'layer {layer_idx} done')

