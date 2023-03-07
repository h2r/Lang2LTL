import os
from pathlib import Path
import argparse

from utils import load_from_file, save_to_file


def construct_composed_dataset(data_fpath, composed_fpath):
    """
    Add composed symbolic dataset to original symbolic dataset.
    """
    dataset = load_from_file(data_fpath)
    composed = load_from_file(composed_fpath)

    breakpoint()

    dataset["valid_iter"].extend(composed["data"])
    dataset["valid_meta"].extend(composed["meta"])

    save_dname = os.path.basename(os.path.dirname(data_fpath))
    save_dpatph = os.path.join("data", f"composed_{save_dname}")
    os.makedirs(save_dpatph, exist_ok=True)
    save_fname = f"composed_{Path(data_fpath).stem}.pkl"
    save_fpath = os.path.join(save_dpatph, save_fname)
    save_to_file(dataset, save_fpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fpath", type=str, default="data/holdout_split_batch12_perm/symbolic_batch12_perm_utt_0.2_0.pkl", help="original symbolic dataset.")
    parser.add_argument("--composed_fpath", type=str, default="data/composed_symbolic_batch12_noperm.pkl", help="composed dataset.")
    args = parser.parse_args()

    construct_composed_dataset(args.data_fpath, args.composed_fpath)
