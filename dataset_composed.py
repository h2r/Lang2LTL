import os
from pathlib import Path
import argparse

from utils import load_from_file, save_to_file
from dataset_symbolic import save_split_dataset_new


def construct_composed_dataset(data_fpath, composed_fpath):
    """
    Construct composed split datasets for zero shot transfer, utterance and formula holdout.
    """
    dataset = load_from_file(data_fpath)
    dataset["train_iter"].extend(dataset["valid_iter"])  # all base dataset used for train
    dataset["train_meta"].extend(dataset["valid_meta"])

    composed = load_from_file(composed_fpath)

    if "zeroshot" in composed_fpath:
        holdout_type = "zeroshot"
        # Train on all base utts, ltls; test on composed utts
        dataset["valid_iter"] = composed["data"]
        dataset["valid_meta"] = composed["meta"]
        dataset["holdout_type"] = holdout_type

        save_dpatph = os.path.join("data", f"composed_{holdout_type}")
        os.makedirs(save_dpatph, exist_ok=True)
        save_fpath = os.path.join(save_dpatph, f"{Path(composed_fpath).stem}.pkl")
        save_to_file(dataset, save_fpath)
    elif "formula" in composed_fpath:
        holdout_type = "formula"
        save_dpatph = os.path.join("data", f"composed_{holdout_type}")
        os.makedirs(save_dpatph, exist_ok=True)

        for train_data, train_meta, valid_data, valid_meta, info in composed:
            size, seed, fold_idx = info["size"], info["seed"], info["fold_idx"]
            save_fname = f"{Path(composed_fpath).stem}_{size}_{seed}_fold{fold_idx}.pkl"
            split_fpath = os.path.join(save_dpatph, save_fname)
            save_split_dataset_new(split_fpath, train_data, train_meta, valid_data, valid_meta, info)
    elif "utt" in composed_fpath:
        holdout_type = "utt"
        save_dpatph = os.path.join("data", f"composed_{holdout_type}")
        os.makedirs(save_dpatph, exist_ok=True)

        for train_data, train_meta, valid_data, valid_meta, info in composed:
            size, seed = info["size"], info["seed"]
            save_fname = f"{Path(composed_fpath).stem}_{size}_{seed}.pkl"
            split_fpath = os.path.join(save_dpatph, save_fname)
            save_split_dataset_new(split_fpath, train_data, train_meta, valid_data, valid_meta, info)
    else:
        raise ValueError(f"ERROR: unrecognized holdout type in compose_fpath: {composed_fpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fpath", type=str, default="data/holdout_split_batch12_perm/symbolic_batch12_perm_utt_0.2_0.pkl", help="original symbolic dataset.")
    parser.add_argument("--composed_fpath", type=str, default="data/composed_utt_symbolic_batch12_noperm.pkl", help="composed dataset.")
    args = parser.parse_args()

    construct_composed_dataset(args.data_fpath, args.composed_fpath)
