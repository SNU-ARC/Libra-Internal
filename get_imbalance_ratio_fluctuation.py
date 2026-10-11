#!/usr/bin/env python3
"""Combine the simulated per-layer imbalance ratios into the paper's Fig. 9 (bottom row) series.

Reads $LIBRA_ARTIFACT_DIR/imbalance_ratio_fluctuation/{model}/{dataset}/{seq_len}/... (written by
prefetch_rebalance/evaluate_fluctuation_libra_*.sh and lina/evaluate_fluctuation_lina_*.sh; LIBRA_ARTIFACT_DIR
defaults to ~/libra_artifacts). For each model, seq_len and scheme it averages the per-iteration imbalance
ratio over layers, concatenates the 8 datasets in alphabetical order, applies a fixed permutation (the same
one Libra-Core's get_throughput_fluctuation.py applies to the throughput series) and prints the result as a
Python list on stdout. Schemes: baseline (SGLang), rebalance (Libra), lina (Lina).
"""
import os, re, csv, glob, sys
import numpy as np

IMBALANCE_RATIO_ROOT = os.path.join(
    os.path.expanduser(os.environ.get("LIBRA_ARTIFACT_DIR", "~/libra_artifacts")),
    "imbalance_ratio_fluctuation",
)

# Extract layer index from ".../{scheme}_layer_{layer_idx}.csv"
LAYER_RE = re.compile(r"_layer_(\d+)\.csv$")

def read_imbalance_column(csv_path):
    vals = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # header may be ['imbalance_ratio'] or similar; ignore content, just read floats
        for row in reader:
            if not row: 
                continue
            vals.append(float(row[0]))
    return vals

def build_dataset_matrix(root, model, dataset, seq_len, scheme):
    """
    Returns:
      layers_sorted (list[int]),
      matrix: dict[layer_idx] -> list[float] (per-row 'imbalance_ratio')
    Searches scheme directory; if scheme == 'eplb', we also recurse into eplb/*/.
    """
    if scheme == 'rebalance' or scheme == 'baseline':
        base_dir = os.path.join(root, model, dataset, str(seq_len), 'step1')
    elif scheme[:4] == 'eplb':
        assert(len(scheme) > 4)
        base_dir = os.path.join(root, model, dataset, str(seq_len), scheme[:4], scheme[4:])
    else:
        base_dir = os.path.join(root, model, dataset, str(seq_len), scheme)
    if not os.path.isdir(base_dir):
        return [], {}

    # eplb may have an extra depth (e.g., eplb/{D}); search recursively
    if scheme[:4] == "eplb":
        pattern = os.path.join(base_dir, f"eplb_layer_*.csv")
    else:
        pattern = os.path.join(base_dir, f"{scheme}_layer_*.csv")
    paths = glob.glob(pattern, recursive=True)
    layer_to_path = {}
    for p in paths:
        m = LAYER_RE.search(p)
        if m:
            layer_to_path[int(m.group(1))] = p

    layers_sorted = sorted(layer_to_path.keys())
    matrix = {}
    for li in layers_sorted:
        matrix[li] = read_imbalance_column(layer_to_path[li])

    return layers_sorted, matrix

def main():
    random_array_2048 = [13, 14, 134, 169, 186, 29, 54, 196, 75, 30, 84, 173, 45, 71, 176, 77, 95, 15, 70, 142, 152, 171, 105, 140, 181, 190, 124, 22, 170, 172, 195, 80, 154, 82, 120, 3, 88, 137, 83, 159, 63, 157, 2, 115, 133, 73, 61, 178, 91, 198, 131, 101, 87, 136, 100, 161, 38, 188, 189, 34, 19, 153, 127, 148, 112, 37, 28, 65, 21, 162, 102, 143, 119, 103, 184, 163, 145, 146, 6, 47, 104, 68, 193, 86, 158, 64, 166, 132, 39, 125, 164, 1, 56, 16, 138, 31, 81, 32, 128, 174, 114, 49, 180, 8, 149, 93, 144, 96, 72, 107, 187, 50, 12, 46, 130, 147, 141, 94, 67, 5, 135, 79, 53, 185, 117, 55, 109, 139, 177, 51, 90, 192, 92, 89, 108, 27, 122, 48, 58, 0, 17, 167, 25, 168, 9, 36, 69, 197, 76, 85, 123, 44, 110, 42, 62, 155, 99, 4, 106, 18, 191, 66, 20, 116, 121, 129, 165, 33, 194, 24, 156, 59, 151, 175, 98, 160, 97, 182, 179, 35, 78, 118, 111, 150, 41, 57, 40, 7, 60, 74, 199, 52, 11, 26, 43, 183, 10, 113, 126, 23]
    random_array_4096 = [i for i in random_array_2048 if i < 96]
    for model in ['qwen3', 'glm45']:
        ds_list = ['bookcorpus', 'codeforces', 'deepseek_prover', 'fineweb', 'gsm8k', 'hellaswag', 'humanevalplus', 'lmsys_chat_1m']

        for seq_len in [2048, 4096]:
            for scheme in ['baseline', 'rebalance', 'lina']:
                ds_mats = {}
                ds_layers = {}
                num_rows = None

                for ds in ds_list:
                    layers, mat = build_dataset_matrix(IMBALANCE_RATIO_ROOT, model, ds, seq_len, scheme)
                    if not layers:
                        continue
                    # ensure all layers have same row length
                    lengths = {len(v) for v in mat.values()}
                    if len(lengths) != 1:
                        print(f"[WARN] Row-length mismatch in {model}/{ds}/{seq_len}/{scheme}; skipping this dataset.", file=sys.stderr)
                        continue
                    rows_here = lengths.pop()
                    if num_rows is None:
                        num_rows = rows_here
                    else:
                        num_rows = min(num_rows, rows_here)  # keep common minimum length

                    ds_mats[ds] = mat
                    ds_layers[ds] = layers

                if not ds_mats or num_rows is None:
                    print(f"# {model} | seq_len={seq_len} | scheme={scheme}")
                    print("[]")  # nothing to show
                    continue

                # Use per-dataset layer sets (no global intersection);
                # the number of iterations is taken from bookcorpus
                num_rows = len(ds_mats['bookcorpus'][list(ds_mats['bookcorpus'].keys())[0]])

                # Build result per layer
                result = [0 for _ in range(len(ds_list) * num_rows)]

                for L in ds_layers['codeforces']:
                    chunks = []
                    for ds in ds_list:
                        arr = ds_mats[ds][L]
                        N = len(arr)
                        chunks.append(arr)
                    # concat in dataset order ⇒ length = sum(N_by_ds[ds]) which is N*D if all N equal
                    chunks = list(np.array(chunks).reshape(-1))
                    for i in range(len(chunks)):
                        result[i] += chunks[i]
                for i in range(len(ds_list) * num_rows):
                    result[i] /= len(ds_layers['codeforces'])

                if seq_len == 2048:
                    final_result = [result[i] for i in random_array_2048]
                else:
                    final_result = [result[i] for i in random_array_4096]

                print(f"# {model} | seq_len={seq_len} | scheme={scheme}")
                print("[" + ", ".join(f"{v:.9f}" for v in final_result) + "]")

if __name__ == "__main__":
    main()