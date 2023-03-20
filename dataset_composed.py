import os
from pathlib import Path
import argparse

from utils import load_from_file, save_to_file
from dataset_symbolic import save_split_dataset


def construct_composed_dataset(data_fpath, composed_fpath):
    """
    Add composed symbolic dataset to original symbolic dataset.
    """
    dataset = load_from_file(data_fpath)
    composed = load_from_file(composed_fpath)

    dataset["train_iter"].extend(dataset["valid_iter"])
    dataset["train_meta"].extend(dataset["valid_meta"])

    if "zeroshot" in composed_fpath:
        holdout_type = "zeroshot"
        # Train on all base utts, ltls; test on composed utts
        dataset["valid_iter"] = composed["data"]
        dataset["valid_meta"] = composed["meta"]
        dataset["holdout_type"] = holdout_type

        save_dpatph = os.path.join("data", f"composed_{holdout_type}")
        os.makedirs(save_dpatph, exist_ok=True)
        save_fname = f"{Path(composed_fpath).stem}.pkl"
        save_fpath = os.path.join(save_dpatph, save_fname)
        save_to_file(dataset, save_fpath)
    elif "formula" in composed_fpath:
        holdout_type = "formula"
        save_dpatph = os.path.join("data", f"composed_{holdout_type}")
        os.makedirs(save_dpatph, exist_ok=True)

        for train_data, train_meta, valid_data, valid_meta, size, seed, fold_idx in composed:
            save_fname = f"{Path(composed_fpath).stem}_{size}_{seed}_fold{fold_idx}.pkl"
            split_fpath = os.path.join(save_dpatph, save_fname)
            save_split_dataset(split_fpath, train_data, train_meta, valid_data, valid_meta, size, seed)
    elif "utt" in composed_fpath:
        holdout_type = "utt"
        save_dpatph = os.path.join("data", f"composed_{holdout_type}")
        os.makedirs(save_dpatph, exist_ok=True)

        for train_data, train_meta, valid_data, valid_meta, size, seed in composed:
            save_fname = f"{Path(composed_fpath).stem}_{size}_{seed}.pkl"
            split_fpath = os.path.join(save_dpatph, save_fname)
            save_split_dataset(split_fpath, train_data, train_meta, valid_data, valid_meta, size, seed)
    else:
        raise ValueError(f"ERROR: unrecognized holdout type in compose_fpath: {composed_fpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fpath", type=str, default="data/holdout_split_batch12_perm/symbolic_batch12_perm_utt_0.2_0.pkl", help="original symbolic dataset.")
    parser.add_argument("--composed_fpath", type=str, default="data/composed_formula_symbolic_batch12_noperm.pkl", help="composed dataset.")
    args = parser.parse_args()

    construct_composed_dataset(args.data_fpath, args.composed_fpath)
