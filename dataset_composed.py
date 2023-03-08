import os
from pathlib import Path
import argparse

from utils import load_from_file, save_to_file


def construct_composed_dataset(data_fpath, composed_fpath, holdout_type):
    """
    Add composed symbolic dataset to original symbolic dataset.
    """
    dataset = load_from_file(data_fpath)
    composed = load_from_file(composed_fpath)

    if holdout_type == "utt_holdout":
        dataset = None
    elif holdout_type == "utt_holdout":
        dataset = None
    elif holdout_type == "zero_shot":
        # Train on all base utts, ltls; test on composed utts
        dataset["train_iter"].extend(dataset["valid_iter"])
        dataset["train_meta"].extend(dataset["valid_meta"])
        dataset["valid_iter"] = composed["data"]
        dataset["valid_meta"] = composed["meta"]
        dataset["holdout_type"] = holdout_type
    else:
        raise ValueError(f"ERROR: unrecognized holdout type: {holdout_type}")

    save_dname = os.path.basename(os.path.dirname(data_fpath))
    save_dpatph = os.path.join("data", f"composed_{holdout_type}_{save_dname}")
    os.makedirs(save_dpatph, exist_ok=True)
    save_fname = f"composed_{holdout_type}_{Path(data_fpath).stem}.pkl"
    save_fpath = os.path.join(save_dpatph, save_fname)
    save_to_file(dataset, save_fpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fpath", type=str, default="data/holdout_split_batch12_perm/symbolic_batch12_perm_utt_0.2_0.pkl", help="original symbolic dataset.")
    parser.add_argument("--composed_fpath", type=str, default="data/composed_symbolic_batch12_noperm.pkl", help="composed dataset.")
    parser.add_argument("--holdout_type", type=str, default="zeroshot", choices=["utt", "formula", "zeroshot"], help="new holdout type for new dataset.")
    args = parser.parse_args()

    construct_composed_dataset(args.data_fpath, args.composed_fpath, args.holdout_type)
